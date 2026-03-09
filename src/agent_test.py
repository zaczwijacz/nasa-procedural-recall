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
from llm_ollama import ollama_generate
from policy import load_policy
from retrieve_pageindex import pageindex_search
from safety_scan import scan_output

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
        "This is a SAFETY-CRITICAL query. "
        "If you are not fully certain from the evidence, output the escalation message. "
        "Do NOT speculate beyond what is explicitly stated in the evidence."
    ),
    "procedural": (
        "This is an operational procedure query. "
        "Summarize the relevant steps or guidance from the evidence. "
        "Do not write numbered steps or give direct action instructions."
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


def _make_prompt(query: str, evidence: str, query_type: str) -> str:
    preamble = _QUERY_TYPE_PREAMBLE.get(query_type, "Answer using only the evidence.")
    return (
        f"You are an assistant for NASA flight crew. {preamble}\n\n"
        f"Rules:\n"
        f'- Use ONLY the EVIDENCE below. Never add information not present in the evidence.\n'
        f'- Every factual claim must include an inline citation: (Doc:<title>, Page:<number>).\n'
        f'- If the answer is not in the evidence, respond EXACTLY:\n'
        f'  "{FAILSAFE_RESPONSE}"\n'
        f"- Do not write numbered steps or give direct action commands.\n\n"
        f"QUERY: {query}\n\n"
        f"EVIDENCE:\n{evidence}\n\n"
        f"Answer:"
    )


def _evidence_sufficient(pages: list, policy: dict) -> bool:
    """True if at least one page meets the minimum confidence threshold."""
    if not pages:
        return False
    return max(p["score"] for p in pages) >= policy["retrieval"]["min_score"]


# ---------------------------------------------------------------------------
# Core run function (importable for testing)
# ---------------------------------------------------------------------------

def run_query(query: str, policy: dict) -> dict:
    """
    Execute the full pipeline for one query and return the result dict.
    The result is also written to the audit log.
    """
    ts_start = time.time()
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
        }
        write_audit_event(policy["audit"]["log_path"], event)
        return event

    # --- Retrieve relevant pages ---
    pages = pageindex_search(
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
        }
        write_audit_event(policy["audit"]["log_path"], event)
        return event

    # --- Build prompt and call LLM ---
    evidence    = _build_evidence(pages, policy["retrieval"]["max_total_chars_evidence"])
    prompt      = _make_prompt(query, evidence, query_type)
    prompt_hash = _sha256(prompt)

    raw  = ollama_generate(prompt, model=policy["llm"]["model"])
    scan = scan_output(raw, policy, pages)

    if scan["blocked"] or scan["is_failsafe"]:
        final             = FAILSAFE_RESPONSE
        failsafe_triggered = True
        failsafe_reason   = "safety_scan_blocked" if scan["blocked"] else "llm_returned_failsafe"
    else:
        final             = raw.strip()
        failsafe_triggered = False
        failsafe_reason   = None

    event = {
        "ts_start":           ts_start,
        "ts_end":             time.time(),
        "query":              query,
        "query_type":         query_type,
        "classifier_confidence": classifier_confidence,
        "retrieved_pages":    pages,
        "llm_prompt_sha256":  prompt_hash,
        "llm_output_raw":     raw,
        "safety_scan":        scan,
        "final_response":     final,
        "failsafe_triggered": failsafe_triggered,
        "failsafe_reason":    failsafe_reason,
    }
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
