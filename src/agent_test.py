"""
NASA Procedural Recall Assistant — main agent entry point.

Usage:
    # Interactive mode
    python src/agent_test.py

    # Single query from command line
    python src/agent_test.py "What is the ISS emergency oxygen procedure?"

Pipeline per query:
    1. classify_query  — route to safety_critical / procedural / informational / prohibited
    2. pageindex_search — retrieve relevant manual pages (local, no network)
    3. Fail-safe gate  — if evidence is insufficient, escalate immediately
    4. ollama_generate — generate answer grounded in evidence only
    5. scan_output     — verify citations, block patterns, length
    6. write_audit_event — append structured JSONL record
"""

import hashlib
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running from project root or from src/
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, _HERE)
# Note: os.chdir is intentionally NOT done at module level so this file
# can be safely imported by the Streamlit app without side-effects.

from audit import write_audit_event
from classify_query import classify_query
from llm_ollama import ollama_generate, render_page_b64
from policy import load_policy
from retrieve_pageindex import pageindex_search
from safety_scan import scan_output

PDF_DIR = Path(_ROOT) / "data" / "raw_pdfs"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POLICY_PATH = "policies/policy.yaml"

FAILSAFE_RESPONSE = (
    "Not found in provided documentation. "
    "Escalate to flight rules, ground support, or crew medical officer."
)

_QUERY_TYPE_PREAMBLE = {
    "safety_critical": (
        "This is a SAFETY-CRITICAL medical or emergency query from a NASA crew member "
        "who needs immediate guidance. "
        "You MUST provide all relevant information from the evidence — do NOT withhold "
        "information or escalate if relevant evidence is present. "
        "State the guidance clearly, then add a note to confirm with a crew medical officer "
        "or ground support as available. "
        "Only output the escalation message if there is absolutely no relevant evidence at all."
    ),
    "procedural": (
        "This is an operational procedure query. "
        "Summarize the relevant steps or guidance from the evidence clearly and completely."
    ),
    "informational": (
        "This is an informational query. "
        "Explain what the evidence says about the topic."
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _build_evidence(pages: list, max_chars: int) -> str:
    parts, total = [], 0
    for p in pages:
        block = (
            f"[EVIDENCE] Doc:{p['doc_title']} "
            f"Pages:{p['page_number']}-{p.get('end_page', p['page_number'])} "
            f"Score:{p['score']:.3f}\n"
            f"{p['text']}\n\n"
        )
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "".join(parts)


def _make_prompt(
    query: str,
    evidence: str,
    query_type: str,
    has_images: bool = False,
    citation_hints: list[str] | None = None,
) -> str:
    preamble = _QUERY_TYPE_PREAMBLE.get(query_type, "")

    inline_cite_instruction = (
        "IMPORTANT: After every numbered step or bullet point, append the page number "
        "it came from in this exact format: (p.N) — where N is the PDF page number from "
        "the EVIDENCE block above. For example: '1. Remove the EpiPen from its tubing. (p.4)'. "
        "If a step draws from multiple pages, list all: (p.4, p.5). "
        "Only omit (p.N) for introductory sentences or section headings."
    )

    if has_images:
        evidence_intro = (
            "EVIDENCE FROM NASA DOCUMENTS (text extracts below + page images attached):"
        )
        instruction = (
            "Using ONLY the evidence above — including the attached page images — "
            "provide a clear and helpful answer. Do NOT add steps, values, timings, "
            "or recommendations from your training knowledge — only what is explicitly "
            "shown in the evidence text or page images. "
            "If the query asks about a figure, diagram, graph, chart, or table, "
            "describe it thoroughly: what it shows, its axes or labels, key values, "
            "and the main point it communicates. "
            "Always use numbered lists for procedures and sequential steps. "
        )
    else:
        evidence_intro = "EVIDENCE FROM NASA DOCUMENTS:"
        instruction = (
            "Using ONLY the evidence above, provide a clear and helpful answer. "
            "Do NOT add steps, values, timings, or recommendations from your training "
            "knowledge — only what is explicitly stated in the evidence. "
            "Always use numbered lists for procedures and sequential steps. "
            "Use bullet points only for non-sequential items such as equipment or symptoms. "
        )

    instruction += inline_cite_instruction + " "

    # Always end with a sources line using the correct citation format
    if citation_hints:
        cite_str = "; ".join(citation_hints)
        instruction += (
            f"End with a 'Sources:' line. "
            f"Use EXACTLY this citation format (copy these strings): {cite_str}"
        )
    else:
        instruction += (
            "End with a 'Sources:' line naming each document and page range referenced."
        )

    parts = [f"{evidence_intro}\n{evidence}", f"QUERY: {query}"]
    if preamble:
        parts.append(preamble)
    parts.append(instruction)
    return "\n\n".join(parts)


def _evidence_sufficient(pages: list, policy: dict) -> bool:
    """True if at least one page meets the minimum confidence threshold."""
    if not pages:
        return False
    return max(p["score"] for p in pages) >= policy["retrieval"]["min_score"]


def compute_quality_score(
    pages: list,
    raw_response: str,
    scan: dict,
    policy: dict,
    failsafe_triggered: bool,
) -> dict:
    """
    Composite answer quality score built from four dimensions:

      1. Retrieval relevance  (40%) — how well retrieved pages matched the query
      2. LLM answer quality   (30%) — did the LLM produce a real, cited answer?
      3. Response completeness(20%) — is the answer substantive enough to be useful?
      4. Coverage             (10%) — how many relevant pages were found?

    Returns a dict with the composite score and each component so the UI can
    surface a breakdown.
    """
    # ------------------------------------------------------------------
    # 1. Retrieval relevance — weighted average of page scores
    #    (top result is weighted most heavily via 1/rank weighting)
    # ------------------------------------------------------------------
    if not pages:
        retrieval = 0.0
    else:
        scores  = [p["score"] for p in pages]
        weights = [1.0 / (i + 1) for i in range(len(scores))]
        retrieval = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    # ------------------------------------------------------------------
    # 2. LLM answer quality — penalise failsafe, missing/bad citations
    # ------------------------------------------------------------------
    if failsafe_triggered or not (raw_response or "").strip():
        llm_quality = 0.0
    else:
        llm_quality = 1.0
        if scan.get("missing_citations"):
            llm_quality -= 0.15
        if scan.get("bad_citations"):
            llm_quality -= 0.30
        if scan.get("blocked"):
            llm_quality -= 0.50
        llm_quality = max(llm_quality, 0.0)

    # ------------------------------------------------------------------
    # 3. Response completeness — proxy: response length
    #    ~500 chars expected for a useful answer; 1 000 chars = full marks
    # ------------------------------------------------------------------
    if failsafe_triggered or not (raw_response or "").strip():
        completeness = 0.0
    else:
        completeness = min(len(raw_response.strip()) / 600, 1.0)

    # ------------------------------------------------------------------
    # 4. Coverage — pages retrieved vs top_k cap
    # ------------------------------------------------------------------
    top_k    = policy["retrieval"].get("top_k", 4)
    coverage = min(len(pages) / top_k, 1.0) if top_k > 0 else 0.0

    # ------------------------------------------------------------------
    # Composite (weighted sum)
    # ------------------------------------------------------------------
    composite = round(
        retrieval    * 0.40 +
        llm_quality  * 0.30 +
        completeness * 0.20 +
        coverage     * 0.10,
        3,
    )

    return {
        "composite":    composite,
        "retrieval":    round(retrieval,    3),
        "llm_quality":  round(llm_quality,  3),
        "completeness": round(completeness, 3),
        "coverage":     round(coverage,     3),
    }


def _render_page_images(pages: list, policy: dict) -> list[str]:
    """
    Render the top retrieved PDF pages as base64 PNGs for VLM visual input.
    Skips pages where the PDF cannot be found or rendering fails.
    Returns an empty list when VLM is disabled in policy.
    """
    vlm_cfg = policy.get("vlm", {})
    if not vlm_cfg.get("enabled", False):
        return []

    dpi      = vlm_cfg.get("render_dpi", 150)
    max_imgs = vlm_cfg.get("max_images_per_query", 4)
    images: list[str] = []

    for page in pages[:max_imgs]:
        doc_name   = page.get("doc_name", "")
        page_num   = page.get("page_number", 1)
        pdf_path   = PDF_DIR / doc_name
        if not pdf_path.exists():
            continue
        try:
            b64 = render_page_b64(pdf_path, page_num, dpi=dpi)
            images.append(b64)
        except Exception:
            continue

    return images


# ---------------------------------------------------------------------------
# Core run function (importable for testing)
# ---------------------------------------------------------------------------

def run_query(query: str, policy: dict, progress_callback=None) -> dict:
    """
    Execute the full pipeline for one query and return the result dict.
    The result is also written to the audit log.

    progress_callback: optional callable(pct: float, text: str) called at each
                       pipeline stage so callers can drive a progress bar.
    """
    def _cb(pct: float, text: str) -> None:
        if progress_callback:
            progress_callback(pct, text)

    ts_start = time.time()
    _cb(0.05, "Classifying query…")
    query_type, classifier_confidence = classify_query(query)

    # --- Prohibited query: refuse immediately, no retrieval ---
    if query_type == "prohibited":
        event = {
            "ts_start": ts_start,
            "ts_end":   time.time(),
            "query":              query,
            "query_type":         query_type,
            "classifier_confidence": classifier_confidence,
            "retrieved_pages":    [],
            "llm_prompt_sha256":  None,
            "llm_output_raw":     None,
            "safety_scan":        {
                "blocked": True,
                "reasons": ["prohibited_query"],
                "missing_citations": False,
                "bad_citations": [],
                "is_failsafe": True,
            },
            "final_response":     FAILSAFE_RESPONSE,
            "failsafe_triggered": True,
            "failsafe_reason":    "prohibited_query",
            "retrieval_type":     "none",
        }
        write_audit_event(policy["audit"]["log_path"], event)
        return event

    # --- Retrieve relevant pages ---
    _cb(0.20, "Retrieving relevant pages…")
    pages, retrieval_type = pageindex_search(
        query,
        policy["retrieval"]["top_k"],
        policy["retrieval"]["min_score"],
    )

    # --- Fail-safe: insufficient evidence ---
    if not _evidence_sufficient(pages, policy):
        event = {
            "ts_start": ts_start,
            "ts_end":   time.time(),
            "query":              query,
            "query_type":         query_type,
            "classifier_confidence": classifier_confidence,
            "retrieved_pages":    pages,
            "llm_prompt_sha256":  None,
            "llm_output_raw":     None,
            "safety_scan":        {
                "blocked": False,
                "reasons": [],
                "missing_citations": False,
                "bad_citations": [],
                "is_failsafe": True,
            },
            "final_response":     FAILSAFE_RESPONSE,
            "failsafe_triggered": True,
            "failsafe_reason":    "insufficient_evidence",
            "retrieval_type":     retrieval_type,
        }
        write_audit_event(policy["audit"]["log_path"], event)
        return event

    # --- Build prompt and call LLM ---
    _cb(0.45, "Building evidence context…")
    evidence = _build_evidence(pages, policy["retrieval"]["max_total_chars_evidence"])

    _cb(0.55, "Rendering page images for visual context…")
    images = _render_page_images(pages, policy)

    # Build exact citation strings so the VLM can copy them (prevents bad_citations)
    citation_hints = [
        f"(Doc:{p['doc_title']}, Page:{p['page_number']})"
        for p in pages
    ]

    prompt      = _make_prompt(query, evidence, query_type,
                               has_images=bool(images),
                               citation_hints=citation_hints)
    prompt_hash = _sha256(prompt)

    _cb(0.65, f"Generating answer via Ollama ({policy['llm']['model']})…")
    raw = ollama_generate(
        prompt,
        model=policy["llm"]["model"],
        timeout=policy["llm"]["timeout_seconds"],
        images=images if images else None,
    )

    # If VLM returned empty (image payload too large or context exceeded),
    # retry once with fewer/no images so we always return something useful.
    if not raw.strip() and images:
        # First retry: half the images at the same prompt
        fewer = images[:max(1, len(images) // 2)]
        raw = ollama_generate(
            prompt,
            model=policy["llm"]["model"],
            timeout=policy["llm"]["timeout_seconds"],
            images=fewer,
        )
    if not raw.strip():
        # Second retry: text-only prompt so at least the text evidence is used
        raw = ollama_generate(
            prompt,
            model=policy["llm"]["model"],
            timeout=policy["llm"]["timeout_seconds"],
            images=None,
        )

    _cb(0.90, "Scanning output for safety…")
    scan = scan_output(raw, policy, pages, is_visual_response=bool(images))

    if scan["blocked"] or scan["is_failsafe"]:
        final             = FAILSAFE_RESPONSE
        failsafe_triggered = True
        failsafe_reason   = "safety_scan_blocked" if scan["blocked"] else "llm_returned_failsafe"
    else:
        final             = raw.strip()
        failsafe_triggered = False
        failsafe_reason   = None

    quality = compute_quality_score(pages, raw, scan, policy, failsafe_triggered)

    event = {
        "ts_start":              ts_start,
        "ts_end":                time.time(),
        "query":                 query,
        "query_type":            query_type,
        "classifier_confidence": classifier_confidence,
        "retrieved_pages":       pages,
        "llm_prompt_sha256":     prompt_hash,
        "llm_output_raw":        raw,
        "safety_scan":           scan,
        "final_response":        final,
        "failsafe_triggered":    failsafe_triggered,
        "failsafe_reason":       failsafe_reason,
        "retrieval_type":        retrieval_type,
        "quality_score":         quality,
    }
    _cb(1.0, "Complete")
    write_audit_event(policy["audit"]["log_path"], event)
    return event


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_result(result: dict) -> None:
    status = "FAILSAFE" if result["failsafe_triggered"] else "OK"
    n_pages = len(result["retrieved_pages"])

    print()
    print(f"  Type   : {result['query_type']} (confidence {result['classifier_confidence']:.2f})")
    print(f"  Status : {status}", end="")
    if result["failsafe_triggered"]:
        print(f"  [{result['failsafe_reason']}]")
    else:
        print(f"  [{n_pages} page(s) retrieved]")

    if result["retrieved_pages"]:
        print("  Sources:")
        for p in result["retrieved_pages"]:
            end = p.get("end_page", p["page_number"])
            print(f"    • {p['doc_title']}  pp.{p['page_number']}–{end}  (score {p['score']:.3f})")

    if result.get("safety_scan", {}).get("reasons"):
        print(f"  Safety flags: {result['safety_scan']['reasons']}")

    print()
    print(result["final_response"])
    print()


if __name__ == "__main__":
    os.chdir(_ROOT)  # only set working dir when running directly
    policy = load_policy(POLICY_PATH)

    if len(sys.argv) > 1:
        # Query passed as command-line argument
        query  = " ".join(sys.argv[1:])
        result = run_query(query, policy)
        print(f"\n[Query] {query}")
        _print_result(result)
    else:
        # Interactive mode
        print("=" * 60)
        print(" NASA Procedural Recall Assistant")
        print(" Type a query and press Enter.  Ctrl-C or blank line to exit.")
        print("=" * 60)
        try:
            while True:
                try:
                    query = input("\nQuery > ").strip()
                except EOFError:
                    break
                if not query:
                    break
                result = run_query(query, policy)
                _print_result(result)
        except KeyboardInterrupt:
            pass
        print("\nSession ended.  Audit log:", policy["audit"]["log_path"])
