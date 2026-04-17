"""
Ollama LLM/VLM wrapper for the NASA Procedural Recall Assistant.

Provides two public interfaces:
  - render_page_b64()  — renders a PDF page to a base64 JPEG for vision input
  - ollama_generate()  — calls Ollama /api/chat and returns the model's reply

The same model (qwen2.5vl:7b) handles all three inference roles in the pipeline:
query classification, VLM tree search, and answer generation.  Using one model
avoids managing multiple Ollama instances and keeps VRAM usage predictable.
"""

import base64
import requests
from pathlib import Path

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL   = "qwen2.5vl:7b"   # Qwen2.5 Vision-Language — handles text AND page images

# ---------------------------------------------------------------------------
# System prompt — injected into every answer-generation call.
# This is the primary guardrail preventing the model from hallucinating
# information not present in the retrieved evidence.  The "ONLY" constraint
# and the exact failsafe phrasing are checked downstream by safety_scan.py.
# ---------------------------------------------------------------------------
_SYSTEM = (
    "You are a NASA procedural reference assistant. Crew members rely on you for accurate, "
    "clear answers drawn directly from official NASA documentation.\n\n"
    "When answering:\n"
    "- Use ONLY information from the EVIDENCE passages and page images provided — "
    "never add facts, steps, or values from your training knowledge\n"
    "- If page images are attached, read them directly — figures, diagrams, charts, and tables "
    "are visual and their content is in the image, not the text extract\n"
    "- Write in clear, helpful language as a knowledgeable assistant would\n"
    "- Always use numbered lists for procedures, steps, and sequential actions — never switch to bullets mid-response\n"
    "- Use bullet points only for non-sequential items such as equipment lists or symptoms\n"
    "- For figures and diagrams, describe what is shown: labels, axes, values, flow, and key takeaways\n"
    "- Keep the answer focused and directly relevant to the query\n"
    "- End with a 'Sources:' line listing the document name(s) and page range(s) you drew from\n"
    "- Only if no evidence at all is available, respond with exactly:\n"
    "  'Not found in provided documentation. "
    "Escalate to flight rules, ground support, or crew medical officer.'"
)


def render_page_b64(pdf_path, page_num: int, dpi: int = 96) -> str:
    """
    Render a single PDF page (1-based index) to a base64-encoded JPEG for VLM input.

    DPI of 96 balances readability for the VLM against payload size — higher DPI
    causes Ollama 500 errors on the local 12 GB GPU.  JPEG at quality 75 is used
    instead of PNG because it is ~4-8x smaller with negligible quality loss for text.
    """
    import pymupdf
    doc  = pymupdf.open(str(pdf_path))
    page = doc[page_num - 1]           # pymupdf pages are 0-indexed
    mat  = pymupdf.Matrix(dpi / 72, dpi / 72)   # 72 pt/inch is the PDF default
    pix  = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("jpeg", jpg_quality=75)
    doc.close()
    return base64.b64encode(img_bytes).decode()


def ollama_generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 300,
    images: list[str] | None = None,
) -> str:
    """
    Call the Ollama /api/chat endpoint and return the model's reply text.

    If `images` is provided (list of base64-encoded JPEGs), they are attached to
    the user message so the VLM can read page content visually — figures, tables,
    and diagrams that are not captured by text extraction.

    temperature=0 enforces deterministic output, which is essential for a
    safety-critical assistant where the same query should always produce the
    same answer.  num_predict=2048 caps response length to avoid runaway generation.
    """
    user_msg: dict = {"role": "user", "content": prompt}
    if images:
        user_msg["images"] = images   # Ollama multimodal: attach page images to the message

    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model":    model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                user_msg,
            ],
            "stream":  False,
            "options": {"temperature": 0, "num_predict": 2048},
        },
        timeout=timeout,
    )
    if r.status_code == 500:
        # Ollama 500 usually means the image payload was too large for the
        # model context window.  Return empty so agent_test.py can retry
        # with fewer or no images rather than surfacing a hard error.
        return ""
    r.raise_for_status()
    return r.json()["message"]["content"]


if __name__ == "__main__":
    print(ollama_generate('Reply only with: {"ok":true}'))
