# NASA Procedural Recall Assistant

An offline RAG (Retrieval-Augmented Generation) system for querying NASA flight crew procedural manuals. Designed for astronauts and ground support to retrieve specific information from flight procedures, medical emergency protocols, and operational flight rules — without reading entire documents.

**Fully local and offline. No cloud required.**

---

## Features

- **Document indexing** — upload any PDF manual and build a searchable tree index via PageIndex + Ollama
- **Local LLM** — powered by **Qwen2.5 7B Instruct** running entirely on-device via Ollama
- **Smart retrieval** — keyword overlap scoring with query synonym expansion and relative score thresholding to route each query to the most relevant manual automatically
- **Natural language answers** — responses are synthesized in clear, structured language (numbered steps for procedures, bullets for checklists) grounded solely in the indexed documents
- **Safety wrappers** — citation enforcement, length limits, and a fail-safe escalation gate
- **Audit log** — every query, retrieval result, safety decision, and final response is written to `logs/audit.jsonl`
- **Streamlit UI** — upload PDFs, trigger indexing, query, and view source PDF pages side-by-side with answers

---

## Architecture

```
User Query
  → retrieve_pageindex.py   synonym expansion → keyword scoring over all indexed manuals
                            relative score threshold filters off-topic documents
  → Fail-safe gate          if no evidence meets confidence threshold → escalate without LLM
  → llm_ollama.py           Qwen2.5:7b-instruct via Ollama /api/chat
                            strict system prompt: evidence-only, natural formatting, Sources line
  → safety_scan.py          citation checks · length check
  → audit.py                JSONL append to logs/audit.jsonl
```

---

## Models

| Purpose | Model | Notes |
|---|---|---|
| Query answering | `qwen2.5:7b-instruct` | Ollama default model |
| PDF indexing | `qwen2.5-10k` | Custom Modelfile: `num_ctx 10240, num_predict 8192` |

### Creating the indexing model

```bash
ollama create qwen2.5-10k -f - <<'EOF'
FROM qwen2.5:7b-instruct
PARAMETER num_ctx 10240
PARAMETER num_predict 8192
EOF
```

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) running locally with `qwen2.5:7b-instruct` pulled
- [PageIndex](https://github.com/VectifyAI/PageIndex) cloned at `C:/Users/<you>/Documents/PageIndex/` (for indexing only)

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull the LLM model in Ollama
ollama pull qwen2.5:7b-instruct

# 3. Create the high-context indexing model
ollama create qwen2.5-10k -f Modelfile

# 4. Clone PageIndex (for the one-time indexing step)
git clone https://github.com/VectifyAI/PageIndex C:/Users/<you>/Documents/PageIndex
pip install -r C:/Users/<you>/Documents/PageIndex/requirements.txt

# 5. Launch the Streamlit app
uv run streamlit run app.py
```

---

## Usage

### Streamlit UI (recommended)

```bash
uv run streamlit run app.py
```

- **Documents** tab — upload a PDF and click **Build Index**. Indexing runs in the background and takes 10–60 min per manual depending on size.
- **Query** tab — type a question in natural language. The answer appears with source PDF pages rendered beside it.
- **Audit Log** tab — full record of every session, exportable as JSONL.

### Command line

```bash
python src/agent_test.py "What is the ISS emergency oxygen procedure?"
```

---

## Example Queries

| Query | Manual used |
|---|---|
| "A crew member is going into cardiac arrest, what should I do?" | ISS Medical Emergency Manual → ALS Algorithm |
| "Crew member is having a seizure" | ISS Medical Emergency Manual → Seizure |
| "What is the ISS re-entry procedure?" | JSC Flight Procedures Handbook |
| "How do I handle a toxic exposure on station?" | ISS Medical Emergency Manual → Toxic Exposure |

---

## Source Documents

Place PDFs in `data/raw_pdfs/` and index them via the UI. Documents used during development (not included — obtain from official NASA sources):

- JSC-11542 Flight Procedures Handbook Rev E
- NASA ISS Emergency Medical Procedures Manual (2016)
- Space Shuttle Operational Flight Rules Volume A

---

## Project Structure

```
nasa-procedural-recall/
├── app.py                     Streamlit UI
├── requirements.txt
├── policies/
│   └── policy.yaml            Governance config (model, retrieval, safety constraints)
├── data/
│   ├── raw_pdfs/              Place source PDFs here (not tracked in git)
│   └── manifest.json          Document registry
├── indexes/                   PageIndex tree JSONs (generated, not tracked)
├── logs/
│   └── audit.jsonl            Query audit log (generated, not tracked)
└── src/
    ├── agent_test.py          Main agent pipeline
    ├── classify_query.py      Keyword-based query router
    ├── retrieve_pageindex.py  Local tree-walker retriever with synonym expansion
    ├── safety_scan.py         Citation checks and length enforcement
    ├── audit.py               JSONL audit writer
    ├── llm_ollama.py          Ollama /api/chat wrapper (Qwen2.5)
    ├── policy.py              Policy loader
    └── run_index_worker.py    Background indexing subprocess
```

---

## Safety Design

| Layer | Check |
|---|---|
| Retrieval gate | If no pages meet the confidence threshold, escalation response is returned without calling the LLM |
| Citation check | Citations to pages outside the retrieved set are blocked |
| Length check | Responses exceeding 5,000 characters are blocked |
| Fail-safe | If the LLM returns no answer grounded in evidence, a fixed escalation message is shown |
| Audit | All decisions — including blocks and fail-safes — are logged with ISO timestamps |
