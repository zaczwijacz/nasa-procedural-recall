import requests

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b-instruct"

_SYSTEM = (
    "You are a NASA procedural reference assistant. Crew members rely on you for accurate, "
    "clear answers drawn directly from official NASA documentation.\n\n"
    "When answering:\n"
    "- Use ONLY information from the EVIDENCE passages provided — never add facts from your training data\n"
    "- Write in clear, helpful language as a knowledgeable assistant would\n"
    "- For procedures and steps, use numbered lists; for equipment or conditions, use bullet points\n"
    "- Keep the answer focused and directly relevant to the query\n"
    "- End with a 'Sources:' line listing the document name(s) and page range(s) you drew from\n"
    "- If the EVIDENCE does not contain a clear answer, respond with exactly:\n"
    "  'Not found in provided documentation. "
    "Escalate to flight rules, ground support, or crew medical officer.'"
)

def ollama_generate(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 300) -> str:
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]

if __name__ == "__main__":
    print(ollama_generate('Reply only with: {"ok":true}'))