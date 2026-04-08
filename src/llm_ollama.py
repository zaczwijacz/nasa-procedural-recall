import base64
import requests
from pathlib import Path

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL   = "qwen2.5vl:7b"

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
    """Render a single PDF page (1-based) to a base64-encoded JPEG for VLM input.
    JPEG is used (not PNG) to keep payload size small enough for local Ollama."""
    import pymupdf
    doc  = pymupdf.open(str(pdf_path))
    page = doc[page_num - 1]
    mat  = pymupdf.Matrix(dpi / 72, dpi / 72)
    pix  = page.get_pixmap(matrix=mat)
    # Convert to JPEG at quality 75 — readable by VLM, ~4-8x smaller than PNG
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
    Call Ollama chat API.  If `images` is provided (list of base64-encoded PNGs),
    they are attached to the user message so the VLM can read page content visually.
    """
    user_msg: dict = {"role": "user", "content": prompt}
    if images:
        user_msg["images"] = images

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
        # Ollama 500 usually means the image payload was too large or the model
        # context was exceeded.  Return empty so the caller can retry without images.
        return ""
    r.raise_for_status()
    return r.json()["message"]["content"]


if __name__ == "__main__":
    print(ollama_generate('Reply only with: {"ok":true}'))
