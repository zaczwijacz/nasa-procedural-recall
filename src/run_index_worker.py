"""
Indexing worker — spawned as a background subprocess by the Streamlit app.

Usage:
    python src/run_index_worker.py <pdf_path>

What it does:
    1. Updates data/index_status.json  →  status: "indexing"
    2. Runs PageIndex run_pageindex.py on the PDF (with --toc-check-pages 3)
    3. On success: copies *_structure.json → indexes/  and sets status: "ready"
    4. On failure: sets status: "error" with the error message
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT          = Path(__file__).parent.parent.resolve()
PAGEINDEX_DIR = Path(os.getenv("PAGEINDEX_DIR", r"C:\Users\ZacMa\Documents\PageIndex"))
STATUS_FILE   = ROOT / "data" / "index_status.json"
INDEX_DIR     = ROOT / "indexes"


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def _load() -> dict:
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save(data: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _update(pdf_name: str, **kwargs) -> None:
    data = _load()
    entry = data.get(pdf_name, {})
    entry.update(kwargs)
    data[pdf_name] = entry
    _save(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: run_index_worker.py <pdf_path>", file=sys.stderr)
        sys.exit(1)

    pdf_path = Path(sys.argv[1]).resolve()
    pdf_name = pdf_path.name
    pdf_stem = pdf_path.stem

    _update(pdf_name,
            status="indexing",
            pid=os.getpid(),
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            finished_at=None,
            error=None)

    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(PAGEINDEX_DIR / "run_pageindex.py"),
                "--pdf_path",           str(pdf_path),
                "--toc-check-pages",    "3",
                "--model",              "qwen2.5vl:7b",
                "--max-pages-per-node", "3",
            ],
            cwd=str(PAGEINDEX_DIR),
            timeout=7200,   # 2 h hard limit for large documents
        )

        if proc.returncode != 0:
            _update(pdf_name, status="error",
                    error=f"Indexer exited with code {proc.returncode}")
            return

        result_json = PAGEINDEX_DIR / "results" / f"{pdf_stem}_structure.json"
        if not result_json.exists():
            _update(pdf_name, status="error",
                    error="run_pageindex.py succeeded but output file not found")
            return

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(result_json), str(INDEX_DIR / result_json.name))

        # Run the structure auditor so the LLM cross-checks the extracted TOC
        # against the actual PDF content and logs any inaccuracies.
        audit_log = ROOT / "logs" / f"{pdf_stem}_audit_structure.txt"
        try:
            audit_proc = subprocess.run(
                [sys.executable, str(ROOT / "src" / "audit_structure.py"), pdf_stem],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=300,
            )
            audit_log.parent.mkdir(parents=True, exist_ok=True)
            audit_log.write_text(
                audit_proc.stdout + audit_proc.stderr, encoding="utf-8"
            )
        except Exception as audit_exc:
            # Audit failure must never block the indexing from completing.
            try:
                audit_log.write_text(f"Audit error: {audit_exc}", encoding="utf-8")
            except Exception:
                pass

        _update(pdf_name,
                status="ready",
                audit_log=str(audit_log),
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    except subprocess.TimeoutExpired:
        _update(pdf_name, status="error", error="Indexing timed out (>1 h)")
    except Exception as exc:
        _update(pdf_name, status="error", error=str(exc))


if __name__ == "__main__":
    main()
