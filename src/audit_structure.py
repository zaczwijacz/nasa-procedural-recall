"""
Structure auditor — uses an LLM to compare a document's actual TOC/content
against the PageIndex structure JSON and report inaccuracies.

Usage (from project root):
    uv run python src/audit_structure.py <pdf_name_stem>

    # Example:
    uv run python src/audit_structure.py "JSC-11542 - Flight Procedures Handbook Rev E 200504"
    uv run python src/audit_structure.py NASA-ISSmedicalEmergManual_2016

Outputs a correction report to stdout and optionally writes an improved
structure JSON back to indexes/ if --apply is passed.

The two audit prompts used are:
  Pass 1 — Quick accuracy check (section titles, hierarchy, completeness,
            page ranges, missing branches, overstated/collapsed branches).
  Pass 2 — Stronger structural audit: overall accuracy judgment, specific
            corrections list, recommended revised top-level structure.
"""

import json
import sys
import textwrap
from pathlib import Path

import pymupdf
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent.parent.resolve()
INDEX_DIR = ROOT / "indexes"
PDF_DIR   = ROOT / "data" / "raw_pdfs"

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL           = "qwen2.5:7b-instruct"

# How many pages of the PDF to send as "source document" context.
# TOCs are almost always in the first ~20 pages.
_TOC_PAGE_LIMIT = 20
# Max characters of source text to send (keep within context window).
_MAX_SOURCE_CHARS = 6000
# Max characters of current structure JSON to send.
_MAX_STRUCT_CHARS = 4000


# ---------------------------------------------------------------------------
# Prompts (the two the user provided)
# ---------------------------------------------------------------------------

_PROMPT_PASS1 = """\
Review the attached decision tree (Current Structure JSON) against the attached \
source document (PDF Text Extract) and improve the accuracy of the reading. \
Check each node for:
(1) whether the title matches the document's actual section or subsection title,
(2) whether its placement in the hierarchy is correct,
(3) whether any major sections are missing,
(4) whether any branches are overstated, collapsed, or misleading, and
(5) whether any page numbers or ranges are incorrect.

Identify inaccuracies clearly and organize the output into three groups:
- Accurate
- Needs Revision
- Missing Content

For every item that needs revision, provide the corrected label, the corrected \
parent-child placement, and a brief explanation grounded in the document. \
Keep the response succinct, use bullet points, and prioritize structural accuracy \
over visual design.

---
CURRENT STRUCTURE JSON:
{structure}

---
PDF TEXT EXTRACT (first {n_pages} pages):
{source}
"""

_PROMPT_PASS2 = """\
Compare the attached decision tree (Current Structure JSON) to the attached \
handbook (PDF Text Extract) and audit it as a document map, not just a \
conceptual summary. Verify section names, subsection names, hierarchy, \
completeness, and page references. Flag omitted sections, mislabeled branches, \
and places where the tree makes a subsection look like a top-level chapter. \
Then provide:

1. A brief overall accuracy judgment
2. A concise list of specific corrections
3. A recommended revised top-level structure (section titles + page numbers only)
4. Any uncertain items caused by unreadable or missing text

Base every correction only on the attached document. Do not infer content not \
explicitly supported by the text extract.

---
CURRENT STRUCTURE JSON:
{structure}

---
PDF TEXT EXTRACT (first {n_pages} pages):
{source}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_pdf(stem: str) -> Path | None:
    for pdf in PDF_DIR.glob("*.pdf"):
        if pdf.stem == stem:
            return pdf
    return None


def _extract_toc_text(pdf_path: Path) -> tuple[str, int]:
    """
    Extract text from the first _TOC_PAGE_LIMIT pages (or all pages if fewer).
    Returns (text, pages_read).
    """
    doc = pymupdf.open(str(pdf_path))
    n   = min(len(doc), _TOC_PAGE_LIMIT)
    parts = []
    for i in range(n):
        page_text = doc[i].get_text().strip()
        if page_text:
            parts.append(f"[Page {i+1}]\n{page_text}")
    doc.close()
    combined = "\n\n".join(parts)
    return combined[:_MAX_SOURCE_CHARS], n


def _load_structure(stem: str) -> dict | None:
    path = INDEX_DIR / f"{stem}_structure.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _compact_structure(structure: dict) -> str:
    """Compact representation of the structure JSON (titles + page ranges only)."""
    lines: list[str] = []

    def _walk(nodes: list, depth: int = 0) -> None:
        for node in nodes:
            indent = "  " * depth
            title  = node.get("title", "")[:70]
            start  = node.get("start_index", "?")
            end    = node.get("end_index", start)
            pages  = f"pp.{start}-{end}" if start != end else f"p.{start}"
            lines.append(f"{indent}- {title} ({pages})")
            _walk(node.get("nodes", []), depth + 1)

    _walk(structure.get("structure", []))
    full = "\n".join(lines)
    return full[:_MAX_STRUCT_CHARS]


def _call_ollama(prompt: str, label: str) -> str:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model":   MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":  False,
                "options": {"temperature": 0, "num_predict": 2048},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as exc:
        return f"[LLM call failed: {exc}]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def audit(pdf_stem: str) -> None:
    print(f"\nAuditing structure for: {pdf_stem}")

    pdf_path = _find_pdf(pdf_stem)
    if pdf_path is None:
        print(f"ERROR: PDF not found in {PDF_DIR}  (stem={pdf_stem!r})")
        sys.exit(1)

    structure = _load_structure(pdf_stem)
    if structure is None:
        print(f"ERROR: No structure JSON found in {INDEX_DIR}")
        sys.exit(1)

    source_text, n_pages = _extract_toc_text(pdf_path)
    compact              = _compact_structure(structure)

    print(f"PDF: {pdf_path.name}  ({n_pages} pages read for TOC context)")
    print(f"Structure: {len(compact)} chars  |  Source extract: {len(source_text)} chars")

    # --- Pass 1: accuracy check ---
    p1 = _PROMPT_PASS1.format(
        structure=compact, source=source_text, n_pages=n_pages
    )
    result1 = _call_ollama(p1, "PASS 1 — Accuracy check (Accurate / Needs Revision / Missing)")
    print(result1)

    # --- Pass 2: structural audit ---
    p2 = _PROMPT_PASS2.format(
        structure=compact, source=source_text, n_pages=n_pages
    )
    result2 = _call_ollama(p2, "PASS 2 — Structural audit (corrections + revised top-level)")
    print(result2)

    print(f"\n{'='*60}")
    print("  Audit complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    audit(sys.argv[1])
