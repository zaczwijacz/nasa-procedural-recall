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

import base64
import json
import sys
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
MODEL           = "qwen2.5vl:7b"

# How many pages of the PDF to render as images for the VLM audit.
# TOCs are almost always in the first ~20 pages.
_TOC_PAGE_LIMIT  = 20
# DPI for page rendering — 120 keeps images small enough for fast VLM ingestion.
_RENDER_DPI      = 120
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


def _render_toc_images(pdf_path: Path) -> tuple[list[str], int]:
    """
    Render the first _TOC_PAGE_LIMIT pages as base64 PNGs for VLM input.
    Returns (images_b64, pages_rendered).
    """
    doc    = pymupdf.open(str(pdf_path))
    n      = min(len(doc), _TOC_PAGE_LIMIT)
    mat    = pymupdf.Matrix(_RENDER_DPI / 72, _RENDER_DPI / 72)
    images = []
    for i in range(n):
        pix       = doc[i].get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        images.append(base64.b64encode(img_bytes).decode())
    doc.close()
    return images, n


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


def _call_ollama(prompt: str, label: str, images: list[str] | None = None) -> str:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    user_msg: dict = {"role": "user", "content": prompt}
    if images:
        user_msg["images"] = images
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model":    MODEL,
                "messages": [user_msg],
                "stream":   False,
                "options":  {"temperature": 0, "num_predict": 2048},
            },
            timeout=300,
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

    images, n_pages = _render_toc_images(pdf_path)
    compact         = _compact_structure(structure)

    print(f"PDF: {pdf_path.name}  ({n_pages} pages rendered as images for VLM audit)")
    print(f"Structure: {len(compact)} chars  |  Images: {len(images)}")

    # --- Pass 1: accuracy check (VLM reads page images directly) ---
    p1 = _PROMPT_PASS1.format(
        structure=compact, source="[See attached page images]", n_pages=n_pages
    )
    result1 = _call_ollama(
        p1,
        "PASS 1 — Accuracy check (Accurate / Needs Revision / Missing)",
        images=images,
    )
    print(result1)

    # --- Pass 2: structural audit ---
    p2 = _PROMPT_PASS2.format(
        structure=compact, source="[See attached page images]", n_pages=n_pages
    )
    result2 = _call_ollama(
        p2,
        "PASS 2 — Structural audit (corrections + revised top-level)",
        images=images,
    )
    print(result2)

    print(f"\n{'='*60}")
    print("  Audit complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    audit(sys.argv[1])
