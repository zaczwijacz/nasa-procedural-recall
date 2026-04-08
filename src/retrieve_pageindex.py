"""
Vision-RAG retriever for the NASA procedural recall assistant.

Pipeline (mirrors the PageIndex Vision RAG notebook):
  1. Load pre-built PageIndex tree JSONs from indexes/.
  2. Build a compact tree (node_id, title, pages, summary) for all documents.
  3. Send the compact tree + query to Qwen2.5-VL-7B, which REASONS over the
     tree and returns the most relevant node IDs (JSON).
  4. Resolve those node IDs to PDF page ranges.
  5. Extract page text AND render page images (JPEG) from the source PDFs.
  6. Return results in the agent's retrieval contract format.

Fallback chain:
  Figure/table caption scan  →  VLM tree reasoning  →  keyword scoring  →  full-page RAG scan
"""

import json
import logging
import re
import requests
import yaml
from pathlib import Path

import pymupdf
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SRC_DIR  = Path(__file__).parent
_ROOT     = _SRC_DIR.parent
INDEX_DIR     = _ROOT / "indexes"
PDF_DIR       = _ROOT / "data" / "raw_pdfs"
_SYNONYMS_PATH = _ROOT / "data" / "synonyms.yaml"

OLLAMA_BASE_URL = "http://localhost:11434"
VLM_MODEL       = "qwen2.5vl:7b"

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
_MAX_CHARS_PER_SECTION      = 2000
_MIN_PAGE_TEXT_CHARS        = 80
_FULLTEXT_FALLBACK_THRESHOLD = 0.30
_VLM_SEARCH_TIMEOUT         = 150   # seconds — VLM reasoning over the tree
_SUMMARY_TRUNCATE           = 220   # chars of summary sent to VLM per node


# ---------------------------------------------------------------------------
# VLM tree-search prompt (adapted from vision_RAG_pageindex notebook)
# ---------------------------------------------------------------------------

_TREE_SEARCH_PROMPT = """\
You are a retrieval assistant for NASA operational manuals.
You are given a query and the hierarchical index of all indexed documents.
Each section has a node_id, title, page range, and a brief summary.

Your task: identify which sections most likely contain the answer to the query.
Consider both section titles and summaries. Think carefully before deciding.

QUERY: {query}

DOCUMENT INDEX:
{tree}

Reply with ONLY this JSON object (no markdown, no extra text):
{{
  "thinking": "<brief reasoning>",
  "results": [
    {{"node_id": "<id>", "doc_name": "<filename.pdf>"}},
    ...
  ]
}}

Order results by relevance (most relevant first). Return up to 6 results.
"""


# ---------------------------------------------------------------------------
# VLM tree search
# ---------------------------------------------------------------------------

def _build_vlm_tree(indexes: list[dict]) -> str:
    """
    Build a compact JSON string describing all indexed sections across all
    documents.  Sent to the VLM so it can reason over the full document space.
    """
    tree = []
    for doc in indexes:
        doc_name  = doc.get("doc_name", "unknown.pdf")
        doc_title = Path(doc_name).stem
        nodes     = _flatten_nodes(doc.get("structure", []))
        sections  = [
            {
                "node_id": n["node_id"],
                "title":   n["title"],
                "pages":   f"{n['start_index']}-{n['end_index']}",
                "summary": (n["summary"] or "")[:_SUMMARY_TRUNCATE],
            }
            for n in nodes
            if n.get("node_id")
        ]
        if sections:
            tree.append({
                "doc_name":  doc_name,
                "doc_title": doc_title,
                "sections":  sections,
            })
    return json.dumps(tree, indent=2)


def _vlm_tree_search(query: str, indexes: list[dict], top_k: int) -> list[dict] | None:
    """
    Ask Qwen2.5-VL to reason over the document tree and return relevant nodes.
    Returns a list of {"node_id", "doc_name"} dicts, or None on failure.
    """
    tree_json = _build_vlm_tree(indexes)
    prompt    = _TREE_SEARCH_PROMPT.format(query=query, tree=tree_json)

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model":    VLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
                "format":   "json",
                "options":  {"temperature": 0},
            },
            timeout=_VLM_SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        data    = json.loads(content)
        results = data.get("results", [])
        if not isinstance(results, list):
            return None
        # Validate and normalise
        valid = [
            r for r in results
            if isinstance(r, dict) and r.get("node_id") and r.get("doc_name")
        ]
        return valid[:top_k] if valid else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def _load_indexes() -> list[dict]:
    indexes = []
    for path in INDEX_DIR.glob("*_structure.json"):
        with open(path, "r", encoding="utf-8") as f:
            indexes.append(json.load(f))
    return indexes


# ---------------------------------------------------------------------------
# Tree flattening
# ---------------------------------------------------------------------------

def _flatten_nodes(structure: list, acc: list | None = None) -> list[dict]:
    if acc is None:
        acc = []
    for node in structure:
        acc.append({
            "title":       node.get("title", ""),
            "summary":     node.get("summary", ""),
            "start_index": node.get("start_index") or 1,
            "end_index":   node.get("end_index")   or node.get("start_index") or 1,
            "node_id":     node.get("node_id", ""),
        })
        if node.get("nodes"):
            _flatten_nodes(node["nodes"], acc)
    return acc


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def _find_pdf(doc_name: str) -> Path | None:
    exact = PDF_DIR / doc_name
    if exact.exists():
        return exact
    stem = Path(doc_name).stem
    for pdf in PDF_DIR.glob("*.pdf"):
        if pdf.stem == stem:
            return pdf
    return None


def _extract_text(pdf_path: Path, start_page: int, end_page: int) -> str:
    try:
        doc   = pymupdf.open(str(pdf_path))
        total = len(doc)
        parts = []
        for i in range(start_page - 1, min(end_page, total)):
            parts.append(doc[i].get_text())
        doc.close()
        return "\n".join(parts).strip()[:_MAX_CHARS_PER_SECTION]
    except Exception as exc:
        return f"[text extraction failed: {exc}]"


# ---------------------------------------------------------------------------
# Keyword scoring (fallback when VLM search fails)
# ---------------------------------------------------------------------------

_SYNONYMS: dict[str, set[str]] = {
    "cardiac":      {"als", "aed", "cpr", "resuscitation", "defibrillat", "arrest", "heart"},
    "arrest":       {"als", "aed", "cpr", "resuscitation", "cardiac", "heart"},
    "heart":        {"cardiac", "als", "aed", "arrest", "cpr", "resuscitation"},
    "stopped":      {"cardiac", "arrest", "cpr", "als", "aed", "resuscitation"},
    "stop":         {"cardiac", "arrest", "cpr"},
    "cpr":          {"als", "cardiac", "resuscitation", "arrest", "defibrillat"},
    "defibrillator":{"aed", "als", "cardiac", "arrest"},
    "resuscitate":  {"als", "cpr", "cardiac", "arrest"},
    "unconscious":  {"cardiac", "arrest", "cpr", "resuscitation", "als"},
    "pulse":        {"cardiac", "arrest", "cpr", "heart"},
    "breathing":    {"airway", "respiratory", "cardiac", "cpr"},
    "allergy":      {"allergic", "anaphylaxis", "anaphylax", "reaction", "epipen", "epinephrine", "swelling", "hives"},
    "allergic":     {"allergy", "anaphylaxis", "anaphylax", "reaction", "epipen", "epinephrine", "swelling"},
    "anaphylaxis":  {"allergic", "allergy", "epipen", "epinephrine", "reaction", "swelling", "throat"},
    "anaphylactic": {"allergic", "allergy", "anaphylaxis", "epipen", "swelling"},
    "epipen":       {"epinephrine", "allergic", "anaphylaxis"},
    "swelling":     {"anaphylaxis", "allergic", "allergy", "epipen", "epinephrine", "throat", "airway"},
    "throat":       {"airway", "swelling", "anaphylaxis", "allergic", "choking", "obstruction"},
    "hives":        {"allergic", "allergy", "anaphylaxis", "reaction"},
    "rash":         {"allergic", "allergy", "reaction"},
    "choke":        {"choking", "airway", "obstruction"},
    "choking":      {"airway", "obstruction"},
    "breathe":      {"airway", "respiratory", "breathing"},
    "breathing":    {"airway", "respiratory"},
    "airway":       {"choking", "obstruction", "respiratory"},
    "toxic":        {"exposure", "decontaminate", "contamination"},
    "poison":       {"toxic", "exposure", "decontaminate"},
    "exposure":     {"toxic", "decontaminate"},
    "seizure":      {"convulsion", "epilepsy"},
    "convulsion":   {"seizure"},
    "tooth":        {"dental", "filling"},
    "teeth":        {"dental", "filling"},
    "reentry":      {"entry", "deorbit", "descent"},
    "landing":      {"entry", "deorbit", "approach", "descent"},
    "launch":       {"ascent", "liftoff", "abort"},
}

# ---------------------------------------------------------------------------
# Synonym loading — YAML overlay deep-merged with hardcoded _SYNONYMS above
# ---------------------------------------------------------------------------

def _load_synonyms_yaml() -> dict[str, set[str]]:
    """Load data/synonyms.yaml and return as {key: set_of_strings}. Returns {}
    silently if the file is absent or unreadable."""
    if not _SYNONYMS_PATH.exists():
        logger.info("synonyms.yaml not found — using hardcoded _SYNONYMS only")
        return {}
    try:
        with open(_SYNONYMS_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        result = {k: set(str(v2) for v2 in v) for k, v in raw.items() if isinstance(v, list)}
        logger.info("Loaded synonyms.yaml (%d keys merged into _SYNONYMS)", len(result))
        return result
    except Exception as exc:
        logger.warning("Failed to load synonyms.yaml: %s — using hardcoded fallback", exc)
        return {}


def _deep_merge_synonyms(base: dict[str, set], overlay: dict[str, set]) -> dict[str, set]:
    """Union overlay sets into base; new keys in overlay are added wholesale."""
    merged = {k: set(v) for k, v in base.items()}
    for key, vals in overlay.items():
        if key in merged:
            merged[key] = merged[key] | vals
        else:
            merged[key] = set(vals)
    return merged


_SYNONYMS = _deep_merge_synonyms(_SYNONYMS, _load_synonyms_yaml())

_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "i", "me", "my", "we", "our",
    "you", "your", "he", "she", "it", "they", "them", "this", "that",
    "what", "which", "who", "how", "when", "where", "why",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "and", "but", "or", "if", "not", "no", "so", "then",
    "there", "here", "up", "out", "about",
    "someone", "something", "somebody", "anyone", "anything",
    "crew", "member", "person", "people", "experiencing", "happening",
    "going", "into", "like", "just", "need", "want", "tell", "help",
    "please", "know", "think", "get", "make", "use", "see",
    "iss", "med", "nasa", "page", "pages", "doc", "fin", "all",
    "paper", "htv5", "spx",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _meaningful_tokens(tokens: list[str]) -> list[str]:
    kept = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
    return kept if kept else tokens


def _expand_tokens(tokens: list[str]) -> list[str]:
    expanded, seen = list(tokens), set(tokens)
    for t in tokens:
        for syn in _SYNONYMS.get(t, set()):
            if syn not in seen:
                expanded.append(syn)
                seen.add(syn)
    return expanded


def _score_node(node: dict, query_tokens: list[str], extra_tokens: set | None = None) -> float:
    if not query_tokens:
        return 0.0
    title_tokens = set(_tokenize(node["title"] + " " + node["summary"]))
    title_hits   = sum(1 for t in query_tokens if t in title_tokens)
    title_score  = title_hits / len(query_tokens)
    text_score   = 0.0
    if extra_tokens:
        text_hits  = sum(1 for t in query_tokens if t in extra_tokens)
        text_score = (text_hits / len(query_tokens)) * 0.5
    base        = max(title_score, text_score)
    span        = max(1, node["end_index"] - node["start_index"] + 1)
    specificity = 1.0 / (1.0 + 0.05 * (span - 1))
    return round(base * specificity, 4)


def _keyword_search(query: str, indexes: list[dict], top_k: int, min_score: float) -> list[dict]:
    """Keyword-based fallback when VLM tree search is unavailable."""
    query_tokens = _meaningful_tokens(_expand_tokens(_tokenize(query)))
    candidates   = []

    for doc in indexes:
        doc_name  = doc.get("doc_name", "unknown.pdf")
        doc_title = Path(doc_name).stem
        pdf_path  = _find_pdf(doc_name)
        nodes     = _flatten_nodes(doc.get("structure", []))

        doc_candidates = []
        for node in nodes:
            score = _score_node(node, query_tokens)
            if score >= min_score:
                doc_candidates.append({
                    "score": score, "doc_name": doc_name,
                    "doc_title": doc_title, "pdf_path": pdf_path, "node": node,
                })

        if not doc_candidates and pdf_path:
            for node in nodes:
                snippet      = _extract_text(pdf_path, node["start_index"],
                                             min(node["start_index"] + 2, node["end_index"]))
                extra_tokens = set(_tokenize(snippet))
                score        = _score_node(node, query_tokens, extra_tokens)
                if score >= min_score:
                    doc_candidates.append({
                        "score": score, "doc_name": doc_name,
                        "doc_title": doc_title, "pdf_path": pdf_path, "node": node,
                    })

        candidates.extend(doc_candidates)

    # Full-page RAG fallback
    top_score = candidates[0]["score"] if candidates else 0.0
    if top_score < _FULLTEXT_FALLBACK_THRESHOLD:
        query_tokens_for_ft = _meaningful_tokens(_expand_tokens(_tokenize(query)))
        for pdf_path in PDF_DIR.glob("*.pdf"):
            doc_title = pdf_path.stem
            try:
                doc = pymupdf.open(str(pdf_path))
            except Exception:
                continue
            for i in range(len(doc)):
                page_text = doc[i].get_text().strip()
                if len(page_text) < _MIN_PAGE_TEXT_CHARS:
                    continue
                page_tokens = set(_tokenize(page_text))
                hits  = sum(1 for t in query_tokens_for_ft if t in page_tokens)
                if not hits:
                    continue
                score = round(hits / len(query_tokens_for_ft), 4)
                if score < min_score:
                    continue
                page_num = i + 1
                existing_keys = {(c["doc_name"], c["node"]["start_index"]) for c in candidates}
                if (pdf_path.name, page_num) not in existing_keys:
                    candidates.append({
                        "score": score, "doc_name": pdf_path.name,
                        "doc_title": doc_title, "pdf_path": pdf_path,
                        "node": {
                            "title": f"Page {page_num}", "summary": "",
                            "start_index": page_num, "end_index": page_num,
                            "node_id": f"fulltext_{doc_title}_{page_num:04d}",
                        },
                    })
            doc.close()

    candidates.sort(key=lambda c: c["score"], reverse=True)
    if candidates:
        cutoff     = candidates[0]["score"] * 0.45
        candidates = [c for c in candidates if c["score"] >= cutoff]
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# Figure / table caption search
# ---------------------------------------------------------------------------

# Matches "Figure 2-5", "Fig. 2-5", "Figure 2.5", "Table 3-1", etc.
_FIGURE_CAPTION_RE = re.compile(
    r"\b(fig(?:ure)?\.?\s*|table\s+)([\d]+[-–.][\d]+|[\d]+)",
    re.IGNORECASE,
)


def _figure_reference_search(query: str, top_k: int) -> list[dict]:
    """
    If the query references a specific figure or table (e.g. 'Figure 2-5'),
    scan all PDFs for pages whose text contains that caption and return those
    pages.  Figures are visual — the VLM describes them from rendered images.
    Falls back gracefully when nothing is found.
    """
    # Collect (label, normalised_number, search_pattern) for every reference
    patterns = []
    for m in _FIGURE_CAPTION_RE.finditer(query):
        num = m.group(2)  # e.g. "2-5"
        pat = re.compile(
            r"\b(?:fig(?:ure)?\.?\s*|table\s+)" + re.escape(num) + r"\b",
            re.IGNORECASE,
        )
        patterns.append((num, pat))

    if not patterns:
        return []

    results: list[dict] = []
    seen:    set         = set()

    for pdf_path in PDF_DIR.glob("*.pdf"):
        try:
            doc = pymupdf.open(str(pdf_path))
        except Exception:
            continue

        for i in range(len(doc)):
            page_text = doc[i].get_text()
            for num, pat in patterns:
                if pat.search(page_text):
                    page_num = i + 1
                    key      = (pdf_path.name, page_num)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append({
                        "doc_id":      f"fig_{pdf_path.stem}_{page_num:04d}",
                        "doc_name":    pdf_path.name,
                        "doc_title":   pdf_path.stem,
                        "page_number": page_num,
                        "end_page":    page_num,
                        "score":       0.93,
                        "text":        page_text.strip()[:_MAX_CHARS_PER_SECTION],
                    })
                    # Also include the immediately following page — figures
                    # sometimes span two pages or the caption precedes the image.
                    next_num = page_num + 1
                    if next_num <= len(doc):
                        nkey = (pdf_path.name, next_num)
                        if nkey not in seen:
                            seen.add(nkey)
                            next_text = doc[i + 1].get_text()
                            results.append({
                                "doc_id":      f"fig_{pdf_path.stem}_{next_num:04d}",
                                "doc_name":    pdf_path.name,
                                "doc_title":   pdf_path.stem,
                                "page_number": next_num,
                                "end_page":    next_num,
                                "score":       0.88,
                                "text":        next_text.strip()[:_MAX_CHARS_PER_SECTION],
                            })

        doc.close()

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# BM25 re-ranker (VLM results only — Task C)
# ---------------------------------------------------------------------------

BM25_MIN_SCORE = 0.10  # normalised threshold below which VLM nodes are downgraded


def _bm25_rerank(query: str, results: list[dict]) -> list[dict]:
    """
    Re-rank VLM-retrieved results using BM25 over their extracted page text.

    - If a node's normalised BM25 score < BM25_MIN_SCORE its confidence score
      is downgraded by 0.15 (floor 0.0) and reranked_down=True is set.
    - Nodes are never removed — only scores and flags are adjusted.
    - Results are returned sorted by updated score descending.
    """
    if len(results) <= 1:
        for r in results:
            r.setdefault("reranked_down", False)
        return results

    corpus  = [_tokenize(r.get("text", "")) or [""] for r in results]
    bm25    = BM25Okapi(corpus)
    q_toks  = _tokenize(query) or [""]
    raw     = bm25.get_scores(q_toks)
    max_raw = max(raw) if max(raw) > 0 else 1.0
    normed  = [s / max_raw for s in raw]

    for result, ns in zip(results, normed):
        if ns < BM25_MIN_SCORE:
            result["score"]         = max(round(result["score"] - 0.15, 4), 0.0)
            result["reranked_down"] = True
        else:
            result["reranked_down"] = False

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def pageindex_search(query: str, top_k: int, min_score: float) -> tuple[list[dict], str]:
    """
    Search all indexed documents for sections relevant to `query`.

    Primary path: VLM tree reasoning (Qwen2.5-VL reads the full document
    index and reasons about which nodes answer the query).

    Fallback: keyword scoring + full-page RAG scan.

    Returns (results, retrieval_type) where retrieval_type is one of:
      "figure_caption" | "vlm_tree" | "keyword" | "none"

    Each result dict contains:
      {
        "doc_id":        str,
        "doc_name":      str,
        "doc_title":     str,
        "page_number":   int,
        "end_page":      int,
        "score":         float,
        "text":          str,
        "reranked_down": bool,   # present on vlm_tree results only
      }
    """
    indexes = _load_indexes()
    if not indexes:
        return [], "none"

    # -----------------------------------------------------------------
    # Step 0: Figure / table caption scan (deterministic, no LLM needed)
    # -----------------------------------------------------------------
    figure_hits = _figure_reference_search(query, top_k)
    if figure_hits:
        return figure_hits, "figure_caption"

    # -----------------------------------------------------------------
    # Primary: VLM tree reasoning
    # -----------------------------------------------------------------
    vlm_hits = _vlm_tree_search(query, indexes, top_k)

    if vlm_hits:
        node_map: dict[tuple[str, str], tuple[dict, dict]] = {}
        for doc in indexes:
            doc_name = doc.get("doc_name", "")
            for node in _flatten_nodes(doc.get("structure", [])):
                if node.get("node_id"):
                    node_map[(doc_name, node["node_id"])] = (doc, node)

        results = []
        for rank, hit in enumerate(vlm_hits):
            node_id  = hit["node_id"]
            doc_name = hit["doc_name"]
            key      = (doc_name, node_id)

            entry = node_map.get(key)
            if entry is None:
                for (dn, nid), val in node_map.items():
                    if nid == node_id:
                        entry    = val
                        doc_name = dn
                        break

            if entry is None:
                continue

            doc, node = entry
            pdf_path  = _find_pdf(doc_name)
            start     = node["start_index"]
            end       = node["end_index"]
            text      = _extract_text(pdf_path, start, end) if pdf_path else f"[PDF not found: {doc_name}]"
            score     = round(max(0.95 - rank * 0.08, 0.40), 4)

            results.append({
                "doc_id":      node_id,
                "doc_name":    doc_name,
                "doc_title":   Path(doc_name).stem,
                "page_number": start,
                "end_page":    end,
                "score":       score,
                "text":        text,
            })

        if results:
            return _bm25_rerank(query, results), "vlm_tree"

    # -----------------------------------------------------------------
    # Fallback: keyword scoring
    # -----------------------------------------------------------------
    candidates = _keyword_search(query, indexes, top_k, min_score)
    results    = []
    for c in candidates:
        node     = c["node"]
        start    = node["start_index"]
        end      = node["end_index"]
        pdf_path = c.get("pdf_path")
        text     = _extract_text(pdf_path, start, end) if pdf_path else f"[PDF not found: {c['doc_name']}]"
        results.append({
            "doc_id":      node["node_id"],
            "doc_name":    c["doc_name"],
            "doc_title":   c["doc_title"],
            "page_number": start,
            "end_page":    end,
            "score":       c["score"],
            "text":        text,
        })
    return results, "keyword"
