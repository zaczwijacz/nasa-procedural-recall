# NASA Procedural Recall Assistant

An agentic RAG (Retrieval-Augmented Generation) system for querying NASA flight crew procedural manuals. Designed for astronauts and ground support to retrieve specific information from flight procedures, medical emergency protocols, and operational flight rules — without reading entire documents.

**Fully local. No cloud required at query time.**

---

## Features

- **Document indexing** — upload any PDF manual and build a searchable tree index
- **Agentic retrieval** — uses PageIndex's reasoning-based tree search to find the most relevant pages
- **Safety wrappers** — three-layer safety scan: pattern blocking, citation enforcement, length limits
- **Query routing** — deterministic keyword classifier routes queries to `safety_critical`, `procedural`, `informational`, or `prohibited` handling modes
- **Fail-safe gate** — if evidence is insufficient or the query is prohibited, the system escalates immediately without calling the LLM
- **Citation traceability** — every answer must cite `(Doc:<title>, Page:<number>)` from retrieved pages
- **Audit log** — every query, retrieval result, safety decision, and final response is written to `logs/audit.jsonl`
- **Streamlit UI** — upload PDFs, trigger indexing, query, and view source PDF pages side-by-side with answers

---

## Architecture

```
User Query
  → classify_query.py     keyword router (prohibited / safety_critical / procedural / informational)
  → retrieve_pageindex.py local tree-walker over pre-built PageIndex JSON indexes
  → Fail-safe gate        if no evidence or low score → escalate, skip LLM
  → llm_ollama.py         Ollama /api/generate (llama3.1:8b-instruct-q8_0)
  → safety_scan.py        pattern blocks · citation checks · length check
  → audit.py              JSONL append to logs/audit.jsonl
```

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) running locally with `llama3.1:8b-instruct-q8_0` pulled
- [PageIndex](https://github.com/VectifyAI/PageIndex) cloned at `C:/Users/<you>/Documents/PageIndex/` (for indexing only)

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull the LLM model in Ollama
ollama pull llama3.1:8b-instruct-q8_0

# 3. Clone PageIndex (for the one-time indexing step)
git clone https://github.com/VectifyAI/PageIndex C:/Users/<you>/Documents/PageIndex
pip install -r C:/Users/<you>/Documents/PageIndex/requirements.txt

# 4. Launch the Streamlit app
streamlit run app.py
```

---

## Usage

### Streamlit UI (recommended)

```bash
streamlit run app.py
```

- **📚 Documents** tab — upload a PDF and click **Build Index**. Indexing runs in the background via Ollama and takes 10–30 min per manual.
- **💬 Query** tab — type a question. The answer appears with source PDF pages rendered beside it.
- **📋 Audit Log** tab — full record of every session, exportable as JSONL.

### Command line

```bash
# Single query
python src/agent_test.py "What is the ISS emergency oxygen procedure?"

# Interactive session
python src/agent_test.py
```

---

## Source Documents

The system is designed for NASA flight crew manuals. Place your PDFs in `data/raw_pdfs/` and index them via the UI. Documents used during development (not included in this repo — obtain from official NASA sources):

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
    ├── agent_test.py          Main agent pipeline + CLI entry point
    ├── classify_query.py      Keyword-based query router
    ├── retrieve_pageindex.py  Local tree-walker retriever
    ├── safety_scan.py         3-layer safety check
    ├── audit.py               JSONL audit writer
    ├── llm_ollama.py          Ollama API wrapper
    ├── policy.py              Policy loader
    └── run_index_worker.py    Background indexing subprocess
```

---

## Safety Design

This system enforces the following constraints on every response:

| Layer | Check |
|---|---|
| Query routing | Prohibited queries are refused before any retrieval |
| Retrieval gate | If no pages meet the confidence threshold, escalation response is returned without calling the LLM |
| Pattern block | Responses containing numbered steps or direct action verbs are blocked |
| Citation check | Every factual claim must cite a retrieved page; citations to non-retrieved pages are blocked |
| Length check | Responses exceeding 2,400 characters are blocked |
| Audit | All decisions — including blocks and fail-safes — are logged with ISO timestamps |
