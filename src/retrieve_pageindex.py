"""
Local tree-walker retriever for the NASA procedural recall assistant.

How it works (Option B design):
  1. Loads pre-built PageIndex tree JSONs from indexes/ (built once with OpenAI).
  2. Flattens every node in the tree into a searchable list.
  3. Scores each node by normalized keyword overlap with the query (no network needed).
  4. For the top-k nodes, extracts actual page text directly from the PDF using pymupdf.
  5. Returns a list of dicts matching the agent's retrieval contract.

No OpenAI or internet connection is required at query time.
"""

import json
import re
from pathlib import Path

import pymupdf  # pip install pymupdf

# ---------------------------------------------------------------------------
# Query synonym expansion
# Many NASA documents use abbreviations or domain-specific titles that won't
# match natural-language queries. This table maps common query terms to the
# abbreviations and section titles actually used in the indexed documents.
# ---------------------------------------------------------------------------
_SYNONYMS: dict[str, set[str]] = {
    # Cardiac / resuscitation
    "cardiac":      {"als", "aed", "cpr", "resuscitation", "defibrillat"},
    "arrest":       {"als", "aed", "cpr", "resuscitation"},
    "heart":        {"cardiac", "als", "aed"},
    "cpr":          {"als", "cardiac", "resuscitation"},
    "defibrillator":{"aed", "als"},
    "resuscitate":  {"als", "cpr", "cardiac"},
    # Airway / breathing
    "choke":        {"choking", "airway", "obstruction"},
    "choking":      {"airway", "obstruction"},
    "breathe":      {"airway", "respiratory", "breathing"},
    "breathing":    {"airway", "respiratory"},
    "airway":       {"choking", "obstruction", "respiratory"},
    # Toxic / exposure
    "toxic":        {"exposure", "decontaminate", "contamination"},
    "poison":       {"toxic", "exposure", "decontaminate"},
    "exposure":     {"toxic", "decontaminate"},
    # Seizure
    "seizure":      {"convulsion", "epilepsy"},
    "convulsion":   {"seizure"},
    # Dental
    "tooth":        {"dental", "filling"},
    "teeth":        {"dental", "filling"},
    # Flight / reentry
    "reentry":      {"entry", "deorbit", "descent"},
    "landing":      {"entry", "deorbit", "approach", "descent"},
    "launch":       {"ascent", "liftoff", "abort"},
}


def _expand_tokens(tokens: list[str]) -> list[str]:
    """Return query tokens expanded with domain synonyms."""
    expanded = list(tokens)
    seen = set(tokens)
    for t in tokens:
        for syn in _SYNONYMS.get(t, set()):
            if syn not in seen:
                expanded.append(syn)
                seen.add(syn)
    return expanded


# Common English words that carry no topical meaning and inflate the
# token denominator, making real keyword matches score too low.
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "i", "me", "my", "we", "our",
    "you", "your", "he", "she", "it", "they", "them", "this", "that",
    "what", "which", "who", "how", "when", "where", "why",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "and", "but", "or", "if", "not", "no", "so", "then",
    "there", "here", "up", "out", "about",
    # Common conversational filler in natural-language queries
    "someone", "something", "somebody", "anyone", "anything",
    "crew", "member", "person", "people", "experiencing", "happening",
    "going", "into", "like", "just", "need", "want", "tell", "help",
    "please", "know", "think", "get", "make", "use", "see",
    # NASA document header/footer tokens that appear on nearly every page
    # and would cause false matches if used for scoring.
    "iss", "med", "nasa", "page", "pages", "doc", "fin", "all",
    "paper", "htv5", "spx",
}


def _meaningful_tokens(tokens: list[str]) -> list[str]:
    """Remove stop words so scoring is driven by topical keywords only."""
    kept = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
    # Fall back to all tokens if filtering removes everything
    return kept if kept else tokens


# ---------------------------------------------------------------------------
# Paths (relative to this file's location: src/)
# ---------------------------------------------------------------------------
_SRC_DIR   = Path(__file__).parent
_ROOT      = _SRC_DIR.parent
INDEX_DIR  = _ROOT / "indexes"
PDF_DIR    = _ROOT / "data" / "raw_pdfs"

# How much page text to extract per matched section (chars).
_MAX_CHARS_PER_SECTION = 3000

# Full-PDF RAG fallback: pages with less text than this are skipped
# (flowchart / image-only pages tend to be very sparse).
_MIN_PAGE_TEXT_CHARS = 80

# Score threshold below which the structured index is considered too weak,
# triggering a full-page RAG scan of every PDF in PDF_DIR.
_FULLTEXT_FALLBACK_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def _load_indexes() -> list[dict]:
    """Load all *_structure.json files from INDEX_DIR."""
    indexes = []
    for path in INDEX_DIR.glob("*_structure.json"):
        with open(path, "r", encoding="utf-8") as f:
            indexes.append(json.load(f))
    return indexes


# ---------------------------------------------------------------------------
# Tree flattening
# ---------------------------------------------------------------------------

def _flatten_nodes(structure: list, acc: list | None = None) -> list[dict]:
    """
    Recursively flatten a PageIndex tree into a flat list of node dicts.
    Each dict keeps: title, summary, start_index, end_index, node_id.
    """
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
        if "nodes" in node and node["nodes"]:
            _flatten_nodes(node["nodes"], acc)
    return acc


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _score_node(node: dict, query_tokens: list[str], extra_tokens: set | None = None) -> float:
    """
    Normalized token-overlap score in [0, 1].
    Primary score against title+summary; optional extra_tokens (page text) used
    as a fallback at 50% weight so content-only matches can still surface.
    """
    if not query_tokens:
        return 0.0

    title_tokens = set(_tokenize(node["title"] + " " + node["summary"]))
    title_hits = sum(1 for t in query_tokens if t in title_tokens)
    title_score = title_hits / len(query_tokens)

    text_score = 0.0
    if extra_tokens:
        text_hits = sum(1 for t in query_tokens if t in extra_tokens)
        text_score = (text_hits / len(query_tokens)) * 0.5

    base = max(title_score, text_score)

    # Slight specificity boost: prefer tighter page ranges.
    span = max(1, node["end_index"] - node["start_index"] + 1)
    specificity = 1.0 / (1.0 + 0.05 * (span - 1))

    return round(base * specificity, 4)


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def _find_pdf(doc_name: str) -> Path | None:
    """
    Locate the source PDF that corresponds to a doc_name from the index JSON.
    Tries exact filename match first, then stem match.
    """
    exact = PDF_DIR / doc_name
    if exact.exists():
        return exact
    stem = Path(doc_name).stem
    for pdf in PDF_DIR.glob("*.pdf"):
        if pdf.stem == stem:
            return pdf
    return None


def _extract_text(pdf_path: Path, start_page: int, end_page: int) -> str:
    """
    Extract text from pages [start_page, end_page] (1-indexed, inclusive).
    Returns at most _MAX_CHARS_PER_SECTION characters.
    """
    try:
        doc = pymupdf.open(str(pdf_path))
        total_pages = len(doc)
        parts = []
        for i in range(start_page - 1, min(end_page, total_pages)):
            parts.append(doc[i].get_text())
        doc.close()
        combined = "\n".join(parts).strip()
        return combined[:_MAX_CHARS_PER_SECTION]
    except Exception as exc:
        return f"[text extraction failed: {exc}]"


# ---------------------------------------------------------------------------
# Public interface (matches the agent's retrieval contract)
# ---------------------------------------------------------------------------

def _fulltext_page_search(query_tokens: list[str], min_score: float) -> list[dict]:
    """
    Full-PDF RAG fallback: score every page of every PDF in PDF_DIR directly
    against query_tokens.  Returns candidate dicts in the same format as Pass 1/2
    so they can be merged and ranked with structured-index results.

    This ensures that any section of any document is reachable even when the
    pre-built structure JSON does not capture it (e.g. documents without a
    machine-readable table of contents, or scanned PDFs where the indexer
    extracted only a subset of sections).
    """
    candidates = []
    for pdf_path in PDF_DIR.glob("*.pdf"):
        doc_title = pdf_path.stem
        try:
            doc = pymupdf.open(str(pdf_path))
        except Exception:
            continue

        for i in range(len(doc)):
            page_text = doc[i].get_text().strip()
            if len(page_text) < _MIN_PAGE_TEXT_CHARS:
                continue  # skip image-only / near-empty pages

            page_tokens = set(_tokenize(page_text))
            hits = sum(1 for t in query_tokens if t in page_tokens)
            if hits == 0:
                continue
            score = round(hits / len(query_tokens), 4)
            if score < min_score:
                continue

            page_num = i + 1  # 1-indexed
            candidates.append({
                "score":     score,
                "doc_name":  pdf_path.name,
                "doc_title": doc_title,
                "pdf_path":  pdf_path,
                # Wrap the page as a synthetic node so the caller can extract text.
                "node": {
                    "title":       f"Page {page_num}",
                    "summary":     "",
                    "start_index": page_num,
                    "end_index":   page_num,
                    "node_id":     f"fulltext_{doc_title}_{page_num:04d}",
                },
            })

        doc.close()

    return candidates


def pageindex_search(query: str, top_k: int, min_score: float) -> list[dict]:
    """
    Search all loaded indexes for the most relevant sections.

    Returns a list of dicts (up to top_k, all with score >= min_score):
      {
        "doc_id":      str,   # node_id from the tree
        "doc_title":   str,   # PDF filename stem
        "page_number": int,   # first page of the matched section (1-indexed)
        "score":       float, # normalized overlap score in [0, 1]
        "text":        str,   # extracted page text for evidence
      }
    """
    indexes = _load_indexes()
    if not indexes:
        return []

    query_tokens = _meaningful_tokens(_expand_tokens(_tokenize(query)))

    candidates = []
    for doc in indexes:
        doc_name  = doc.get("doc_name", "unknown.pdf")
        doc_title = Path(doc_name).stem
        pdf_path  = _find_pdf(doc_name)
        nodes     = _flatten_nodes(doc.get("structure", []))

        # Pass 1: score by title/summary only.
        doc_candidates = []
        for node in nodes:
            score = _score_node(node, query_tokens)
            if score >= min_score:
                doc_candidates.append({
                    "score":     score,
                    "doc_name":  doc_name,
                    "doc_title": doc_title,
                    "pdf_path":  pdf_path,
                    "node":      node,
                })

        # Pass 2: if title scoring found nothing for this doc, fall back to
        # scoring against extracted page text (slower but catches content-only matches).
        if not doc_candidates and pdf_path:
            for node in nodes:
                snippet = _extract_text(pdf_path, node["start_index"],
                                        min(node["start_index"] + 2, node["end_index"]))
                extra_tokens = set(_tokenize(snippet))
                score = _score_node(node, query_tokens, extra_tokens)
                if score >= min_score:
                    doc_candidates.append({
                        "score":     score,
                        "doc_name":  doc_name,
                        "doc_title": doc_title,
                        "pdf_path":  pdf_path,
                        "node":      node,
                    })

        candidates.extend(doc_candidates)

    # Pass 3 — Full-PDF RAG fallback.
    # If no structured result scored above the fallback threshold, scan every
    # page of every PDF directly.  This catches sections that were never added
    # to any structure JSON (missing TOC entries, scanned-only pages in the
    # index, documents without a machine-readable table of contents, etc.).
    top_score = candidates[0]["score"] if candidates else 0.0
    if top_score < _FULLTEXT_FALLBACK_THRESHOLD:
        fulltext_candidates = _fulltext_page_search(query_tokens, min_score)
        # De-duplicate: drop fulltext hits that overlap with structured results
        # already in candidates (same doc + overlapping page range).
        existing = {
            (c["doc_name"], c["node"]["start_index"])
            for c in candidates
        }
        for fc in fulltext_candidates:
            key = (fc["doc_name"], fc["node"]["start_index"])
            if key not in existing:
                candidates.append(fc)
                existing.add(key)

    # Best scores first.
    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Relative threshold: discard results far below the best match.
    # This automatically filters out off-topic documents without hardcoded
    # domain rules — works for any number of indexed manuals.
    # Example: if flight docs score 0.60 and medical docs score 0.10 for a
    # flight query, cutoff = 0.60 * 0.45 = 0.27, so medical docs are dropped.
    if candidates:
        cutoff = candidates[0]["score"] * 0.45
        candidates = [c for c in candidates if c["score"] >= cutoff]

    candidates = candidates[:top_k]

    results = []
    for c in candidates:
        node  = c["node"]
        start = node["start_index"]
        end   = node["end_index"]

        text = ""
        if c["pdf_path"]:
            text = _extract_text(c["pdf_path"], start, end)
        else:
            text = f"[PDF not found for {c['doc_name']}]"

        results.append({
            "doc_id":      node["node_id"],
            "doc_title":   c["doc_title"],
            "page_number": start,
            "end_page":    end,   # inclusive end of the matched section
            "score":       c["score"],
            "text":        text,
        })

    return results
