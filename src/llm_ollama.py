import requests

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b-instruct-q8_0"

def ollama_generate(prompt: str, model: str = DEFAULT_MODEL) -> str:
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["response"]

if __name__ == "__main__":
    print(ollama_generate('Reply only with: {"ok":true}'))