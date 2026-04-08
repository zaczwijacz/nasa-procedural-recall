# NASA Procedural Recall Assistant

A locally-hosted **Vision-RAG** system for querying NASA flight crew procedural manuals. Designed for astronauts and ground support to retrieve specific information from flight procedures, medical emergency protocols, and operational flight rules — without reading entire documents.

**Fully local and offline. No cloud required.**

---

## Features

- **Vision-RAG pipeline** — four-tier retrieval cascade: figure caption scan → VLM tree search → keyword scoring → full-page scan, each grounded in indexed document structure
- **Vision-Language Model** — powered by **Qwen2.5-VL 7B** (`qwen2.5vl:7b`) running entirely on-device via Ollama; handles both text and embedded page images
- **BM25 re-ranking** — post-VLM re-ranking demotes false-positive retrievals before generation
- **Synonym expansion** — hardcoded medical/aerospace synonym clusters merged with a user-editable `data/synonyms.yaml` overlay at import time
- **Query classification** — routes each query to `safety_critical`, `procedural`, `informational`, or `prohibited` before retrieval, adjusting response rigour accordingly
- **Inline citations** — every numbered step in a response carries a `(p.N)` citation that hyperlinks directly to the PDF page in the browser
- **Quality scoring** — four independent per-query scores (Retrieval, Answer, Completeness, Coverage) displayed alongside every response
- **Safety wrappers** — fail-safe escalation gate, citation enforcement, and length limits; all decisions are logged
- **Audit log** — every query, retrieval result, safety decision, retrieval tier, and final response is written to `logs/audit.jsonl` as structured JSONL
- **Streamlit UI** — upload PDFs, trigger indexing, query with inline page links, toggle dark/light theme, and export audit records

---

## Architecture

```
User Query
  ├─ classify_query.py      LLM classifier → safety_critical / procedural / informational / prohibited
  │
  └─ retrieve_pageindex.py  Four-tier Vision-RAG cascade:
       Tier 0  Figure caption scan   — deterministic match on "Figure N" / "Table N" patterns
       Tier 1  VLM tree search       — qwen2.5vl:7b scores each index node summary against the query
       Tier 2  Keyword scoring       — synonym-expanded token overlap over all node summaries
       Tier 3  Full-page scan        — fallback: raw text search across all indexed page content
               ↓
       BM25 re-ranking               — normalised BM25 score penalises false-positive VLM hits
               ↓
  ├─ Fail-safe gate          if no page meets min_score → escalation response, no LLM call
  │
  ├─ llm_ollama.py           qwen2.5vl:7b via Ollama /api/chat
  │                          page images rendered at 96 DPI and attached as vision payload
  │                          evidence-only system prompt with inline (p.N) citation instruction
  │
  ├─ safety_scan.py          citation hit check · length check · fail-safe pattern detection
  │
  └─ audit.py                JSONL append → logs/audit.jsonl
                             fields: query, query_type, retrieved_pages, retrieval_type,
                                     llm_prompt_sha256, safety_scan, final_response,
                                     quality_score, duration_ms, ISO timestamps
```

---

## Models

| Purpose | Model | Backend | Notes |
|---|---|---|---|
| Query classification | `qwen2.5vl:7b` | Ollama | Classifies intent before retrieval |
| VLM tree search (Tier 1) | `qwen2.5vl:7b` | Ollama | Scores index node summaries against query |
| Answer generation | `qwen2.5vl:7b` | Ollama | Text + page image (vision) input |
| PDF indexing | `qwen2.5vl-10k` | Ollama | Custom Modelfile: `num_ctx 10240, num_predict 8192` |

All inference uses a single unified model: **Qwen2.5-VL 7B Instruct** — a multimodal vision-language model from Alibaba DAMO Academy that handles text, structured JSON output, and page-image understanding in one pass.

### Creating the high-context indexing model

```bash
ollama create qwen2.5vl-10k -f - <<'EOF'
FROM qwen2.5vl:7b
PARAMETER num_ctx 10240
PARAMETER num_predict 8192
EOF
```

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) running locally with `qwen2.5vl:7b` pulled
- [PageIndex](https://github.com/VectifyAI/PageIndex) — see Credits below

---

## Setup

```bash
# 1. Install dependencies
pip install uv
uv sync

# 2. Pull the vision-language model
ollama pull qwen2.5vl:7b

# 3. Create the high-context indexing variant
ollama create qwen2.5vl-10k -f - <<'EOF'
FROM qwen2.5vl:7b
PARAMETER num_ctx 10240
PARAMETER num_predict 8192
EOF

# 4. Clone PageIndex (required for document indexing)
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

- **Documents** tab — upload a PDF and click **Build Index**. Indexing runs as a background subprocess and takes 10–60 min per manual depending on size and page count.
- **Query** tab — type a question in natural language. The answer appears with inline `(p.N)` citations that hyperlink directly to the PDF page. Four quality scores are shown per response.
- **Document Tree** tab — browse the indexed TOC hierarchy for each document.
- **Audit Log** tab — full JSONL record of every query session, exportable.

### Command line

```bash
python src/agent_test.py "What is the ISS emergency oxygen procedure?"
```

---

## Example Queries

| Query | Manual used |
|---|---|
| "A crew member is going into cardiac arrest, what should I do?" | ISS Medical Emergency Manual → ALS Algorithm |
| "Crew member is having a severe allergic reaction" | ISS Medical Emergency Manual → 1.101 Severe Allergic Reaction |
| "What is the ISS re-entry procedure?" | JSC Flight Procedures Handbook |
| "How do I handle a toxic exposure on station?" | ISS Medical Emergency Manual → Toxic Exposure |
| "Please explain Figure 2-5 Simplified TAEM guidance" | JSC Flight Procedures Handbook → figure page image |

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
├── app.py                     Streamlit UI (query, documents, audit, document tree tabs)
├── pyproject.toml
├── policies/
│   └── policy.yaml            Governance config (model, retrieval thresholds, safety constraints)
├── data/
│   ├── raw_pdfs/              Place source PDFs here (not tracked in git)
│   ├── synonyms.yaml          User-editable synonym overlay (merged with hardcoded clusters)
│   └── manifest.json          Document registry
├── indexes/                   PageIndex tree JSONs (generated by indexer)
├── logs/
│   └── audit.jsonl            Per-query audit log (generated, not tracked)
├── static/                    Static PDF copies for in-browser page links (not tracked)
├── .streamlit/
│   └── config.toml            Streamlit theme and static serving config
└── src/
    ├── agent_test.py          Main pipeline: classify → retrieve → generate → scan → audit
    ├── classify_query.py      LLM-based query router (4 classes)
    ├── retrieve_pageindex.py  Four-tier Vision-RAG retriever with synonym expansion and BM25 re-rank
    ├── safety_scan.py         Citation hit check, length enforcement, fail-safe detection
    ├── audit.py               Structured JSONL audit writer
    ├── audit_structure.py     LLM-based post-index structure auditor
    ├── llm_ollama.py          Ollama /api/chat wrapper with vision (base64 image) support
    ├── policy.py              Policy loader
    └── run_index_worker.py    Background indexing subprocess
```

---

## Safety Design

| Layer | Check |
|---|---|
| Query classification | Prohibited queries are refused before retrieval; safety-critical queries receive a stricter response preamble |
| Retrieval gate | If no pages meet `min_score`, a fixed escalation response is returned without calling the LLM |
| BM25 re-ranking | False-positive VLM retrievals are down-weighted before the evidence block is built |
| Citation check | Citations to pages outside the retrieved set are flagged as `bad_citations` and blocked |
| Length check | Responses exceeding 5,000 characters are blocked |
| Fail-safe | If the LLM returns no evidence-grounded answer, a fixed escalation message is shown |
| Audit | All decisions — including blocks, fail-safes, retrieval tier, and quality scores — are logged with ISO timestamps |

---

## Credits

### PageIndex — VectifyAI

The document indexing pipeline is built on top of **[PageIndex](https://github.com/VectifyAI/PageIndex)** by [VectifyAI](https://github.com/VectifyAI).

PageIndex constructs hierarchical tree indexes from PDF documents using a vision-language model, producing structured JSON node summaries with page ranges that this system's retrieval tiers traverse at query time. Without PageIndex, the four-tier retrieval cascade in `retrieve_pageindex.py` would not be possible.

```
@misc{pageindex,
  author       = {VectifyAI},
  title        = {PageIndex: LLM-powered hierarchical PDF index},
  year         = {2024},
  url          = {https://github.com/VectifyAI/PageIndex}
}
```

### Qwen2.5-VL

Inference is powered by **[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)** from Alibaba DAMO Academy, served locally via [Ollama](https://ollama.com/).

---

## Contributors

| GitHub | Role |
|---|---|
| [@zaczwijacz](https://github.com/zaczwijacz) | Project co-lead, system architecture, pipeline development |
| [@mitulj9](https://github.com/mitulj9) | Project co-lead, system architecture, pipeline development |
