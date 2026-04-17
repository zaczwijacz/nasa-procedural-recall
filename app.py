"""
NASA Procedural Recall Assistant — Streamlit UI

Tabs:
  📚 Documents     — upload PDFs, trigger PageIndex builds, inspect index status
  💬 Query         — chat interface: classify → retrieve → generate → scan → display
                     answers include inline (p.N) citations as clickable PDF links
                     and four per-response quality scores (Retrieval, Answer,
                     Completeness, Coverage)
  📋 Audit Log     — full JSONL record of every query, safety decision, and score
  🌲 Decision Tree — live Graphviz diagram of the six-stage pipeline; highlights
                     the path taken by the most recent query
  📑 Document Tree — browsable TOC hierarchy for each indexed document

All inference runs locally via Ollama (qwen2.5vl:7b).  No network calls are made
during query processing.
"""

import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

import pymupdf
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — resolved before any project imports
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent.resolve()
SRC_DIR     = ROOT / "src"
PDF_DIR     = ROOT / "data" / "raw_pdfs"
INDEX_DIR   = ROOT / "indexes"
STATUS_FILE = ROOT / "data" / "index_status.json"
POLICY_PATH = ROOT / "policies" / "policy.yaml"
AUDIT_PATH  = ROOT / "logs" / "audit.jsonl"

sys.path.insert(0, str(SRC_DIR))
os.chdir(ROOT)

from policy import load_policy    # noqa: E402
from agent_test import run_query  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NASA Procedural Recall",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Theme toggle in sidebar
with st.sidebar:
    st.caption("**Display**")
    light_mode = st.toggle("Light mode", value=False)

if light_mode:
    st.markdown("""<style>
        .stApp, [data-testid="stAppViewContainer"] { background-color: #ffffff !important; color: #000000 !important; }
        [data-testid="stSidebar"] { background-color: #f0f2f6 !important; }
        .stMarkdown, .stCaption, p, li { color: #000000 !important; }
    </style>""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ── General layout ── */
[data-testid="stChatMessageContent"] p { margin-bottom: 0.4rem; }
.block-container { padding-top: 3.5rem; }
header[data-testid="stHeader"] { display: none; }

/* ── Tab bar ── */
[data-testid="stTabs"] {
    margin-bottom: 1.5rem;
}
[data-testid="stTabs"] > div:first-child {
    gap: 0.5rem;
    border-bottom: 2px solid #2e2e2e;
    padding-bottom: 0;
}
[data-testid="stTabs"] button[role="tab"] {
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.65rem 1.4rem !important;
    border-radius: 8px 8px 0 0 !important;
    border: 1px solid transparent !important;
    color: #aaaaaa !important;
    background: transparent !important;
    transition: color 0.2s, background 0.2s;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    color: #ffffff !important;
    background: #1e1e1e !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #ffffff !important;
    background: #0e1117 !important;
    border-color: #2e2e2e #2e2e2e #0e1117 !important;
    border-bottom: 2px solid #D4262C !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def get_policy():
    return load_policy(str(POLICY_PATH))


@st.cache_data(show_spinner=False)
def render_page_image(pdf_path: str, page_num: int, zoom: float = 1.5) -> bytes:
    """Render one PDF page to a PNG byte string. Result is cached."""
    doc = pymupdf.open(pdf_path)
    idx = max(0, min(page_num - 1, len(doc) - 1))
    pix = doc[idx].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    doc.close()
    return pix.tobytes("png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_status() -> dict:
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_status(data: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def effective_status(pdf: Path, raw: dict) -> str:
    """
    Determine the current index status for a PDF.

    Priority order:
      1. If the status file says "indexing", trust it — an active run is in
         progress and the old structure JSON on disk is stale.
      2. If a structure JSON exists on disk, the document is ready.
      3. Fall back to whatever the status file records (error / pending / etc.).
    """
    recorded = raw.get(pdf.name, {}).get("status", "pending")
    if recorded == "indexing":
        return "indexing"
    if (INDEX_DIR / f"{pdf.stem}_structure.json").exists():
        return "ready"
    return recorded


def start_indexing(pdf: Path) -> None:
    # Write "indexing" status BEFORE spawning so that effective_status()
    # returns "indexing" on the next rerun even though the old structure
    # JSON still exists on disk.
    data = load_status()
    data[pdf.name] = {
        "status":      "indexing",
        "started_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished_at": None,
        "error":       None,
    }
    save_status(data)

    worker = SRC_DIR / "run_index_worker.py"
    flags  = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    subprocess.Popen(
        [sys.executable, str(worker), str(pdf)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )


def find_pdf(doc_title: str) -> Path | None:
    for pdf in PDF_DIR.glob("*.pdf"):
        if pdf.stem == doc_title:
            return pdf
    return None


# ---------------------------------------------------------------------------
# Inline page-link injection
# ---------------------------------------------------------------------------

# Matches: (p.4)  or  (p.4, p.5)  or  (p.4, p.5, p.6)
_PAGE_REF_RE = re.compile(r'\(p\.(\d+)(?:,\s*p\.(\d+))*\)')

def inject_page_links(text: str, pages: list) -> str:
    """
    Convert LLM-generated (p.N) / (p.N, p.M) inline citations into clickable
    HTML links that open in a new tab.
    """
    # Build page_number → URL map from all retrieved sections
    page_url: dict[int, str] = {}
    for p in pages:
        doc_file = p.get("doc_name", "") or f"{p['doc_title']}.pdf"
        encoded  = quote(doc_file)
        for pg in range(p["page_number"], p.get("end_page", p["page_number"]) + 1):
            page_url[pg] = f"http://localhost:8501/app/static/{encoded}#page={pg}"

    def _replace(m: re.Match) -> str:
        # Extract all page numbers from the full match string
        nums = re.findall(r'\d+', m.group(0))
        parts = []
        for n_str in nums:
            n   = int(n_str)
            url = page_url.get(n)
            if url:
                parts.append(f'<a href="{url}" target="_blank">p.{n}</a>')
            else:
                parts.append(f"p.{n}")
        return "(" + ", ".join(parts) + ")"

    return _PAGE_REF_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Result renderer — defined BEFORE the tab blocks that call it
# ---------------------------------------------------------------------------

def render_result(msg: dict) -> None:
    """Render answer + source pages side-by-side for one query result."""
    col_ans, col_src = st.columns([3, 2], gap="medium")

    with col_ans:
        qtype   = msg["query_type"]
        ts      = msg.get("ts", "")
        icon    = {"safety_critical": "🔴", "procedural": "🟡",
                   "informational": "🟢", "prohibited": "⛔"}.get(qtype, "⚪")
        quality = msg.get("quality", {})

        st.caption(f"{icon} **{qtype}** · {ts}")

        if quality:
            # Color thresholds: ≥70% green, 40–69% amber, <40% red
            def _dot(v: float) -> str:
                return "🟢" if v >= 0.70 else ("🟡" if v >= 0.40 else "🔴")

            # Renders one metric as: <dot> <Label> <ℹ hover tooltip> <value%>
            def _tip(label: str, tooltip: str, value: str) -> str:
                return (
                    f'{_dot(float(value.strip("%")) / 100)} {label} '
                    f'<span title="{tooltip}" style="cursor:help;font-size:0.75em;'
                    f'color:#888;vertical-align:super;">ℹ</span> {value}'
                )

            r  = quality.get("retrieval",    0)
            a  = quality.get("llm_quality",  0)
            co = quality.get("completeness", 0)
            cv = quality.get("coverage",     0)
            st.markdown(
                "<small>" +
                "  ·  ".join([
                    _tip("Retrieval",    "How closely the source pages pulled from the manuals matched your query. Higher = better page match.",                                        f"{r:.0%}"),
                    _tip("Answer",       "Whether the LLM produced a substantive response. Penalised for failsafe escalation, missing or incorrect citations.",                        f"{a:.0%}"),
                    _tip("Completeness", "Whether the response is detailed enough to be actionable. Very short answers score lower.",                                                   f"{co:.0%}"),
                    _tip("Coverage",     "How many relevant manual sections were found relative to the search limit. Low coverage may mean the topic spans more pages than retrieved.", f"{cv:.0%}"),
                ]) +
                "</small>",
                unsafe_allow_html=True,
            )

        # --- Confidence transparency ---
        _classifier_conf = msg.get("confidence", 1.0)
        _top_ret_score   = max((p["score"] for p in msg.get("pages", [])), default=0.0)
        if (_classifier_conf < 0.70 or _top_ret_score < 0.30) and not msg["failsafe"]:
            st.warning(
                "Low confidence result — please verify directly with the source "
                "document before acting on this guidance.",
                icon="⚠️",
            )
        _cc_tip  = "title=\"How certain the classifier was that it correctly identified the query type (safety-critical / procedural / informational). Does not reflect answer accuracy.\""
        _ret_tip = "title=\"The highest match score across all retrieved pages. Below 0.30 means the retrieval may have found the wrong section — verify the source pages.\""
        st.markdown(
            f"<small>"
            f"<span {_cc_tip} style=\"cursor:help;\">Classifier confidence <sup style='color:#888;'>ℹ</sup></span>: {_classifier_conf:.2f}"
            f"  |  "
            f"<span {_ret_tip} style=\"cursor:help;\">Top retrieval score <sup style='color:#888;'>ℹ</sup></span>: {_top_ret_score:.2f}"
            f"</small>",
            unsafe_allow_html=True,
        )

        if msg["failsafe"]:
            st.warning(msg["answer"], icon="⚠️")
            if msg.get("failsafe_reason"):
                st.caption(f"Reason: `{msg['failsafe_reason']}`")
        else:
            linked = inject_page_links(msg["answer"], msg.get("pages", []))
            st.markdown(linked, unsafe_allow_html=True)

        # Inline source page links — one per retrieved section
        pages = msg.get("pages", [])
        if pages and not msg["failsafe"]:
            parts = []
            for p in pages:
                doc_file = p.get("doc_name", "") or f"{p['doc_title']}.pdf"
                start    = p["page_number"]
                end      = p.get("end_page", start)
                page_ref = f"p.{start}" if start == end else f"pp.{start}–{end}"
                url      = f"http://localhost:8501/app/static/{quote(doc_file)}#page={start}"
                parts.append(f'<a href="{url}" target="_blank">📄 {p["doc_title"]} · {page_ref}</a>')
            st.markdown(
                "<small><b>Sources:</b> " + "  ·  ".join(parts) + "</small>",
                unsafe_allow_html=True,
            )

        # Only show safety flags banner when the response was actually blocked.
        if msg["failsafe"] and msg.get("failsafe_reason") == "safety_scan_blocked":
            reasons = msg.get("scan", {}).get("reasons", [])
            if reasons:
                st.error(f"Safety flags: {', '.join(reasons)}", icon="🚨")

    with col_src:
        pages = msg.get("pages", [])
        if not pages:
            st.caption("_No source pages retrieved._")
            return

        st.caption(f"**{len(pages)} source section(s)**")

        for i, p in enumerate(pages[:3]):   # show up to 3
            pdf_path = find_pdf(p["doc_title"])
            start    = p["page_number"]
            end      = p.get("end_page", start)
            label    = (
                f"📄 {p['doc_title']}  ·  "
                f"pp. {start}–{end}  ·  score {p['score']:.2f}"
            )
            with st.expander(label, expanded=(i == 0)):
                if pdf_path:
                    try:
                        img = render_page_image(str(pdf_path), start)
                        st.image(img, use_container_width=True)
                        if end > start:
                            st.caption(
                                f"Section spans pp. {start}–{end}. "
                                "Showing first page."
                            )
                    except Exception as exc:
                        st.caption(f"Could not render page image: {exc}")
                        st.text(p.get("text", "")[:500])
                else:
                    st.caption("PDF not in library — showing text excerpt.")
                    st.text(p.get("text", "")[:500])


# ===========================================================================
# TAB 1 — Documents
# ===========================================================================

tab_docs, tab_chat, tab_audit, tab_tree, tab_doctree = st.tabs(["📚 Documents", "💬 Query", "📋 Audit Log", "🌲 Decision Tree", "📑 Document Tree"])

with tab_docs:
    st.title("Document Library")
    st.caption(
        "Upload NASA PDF manuals. Click **Build Index** to make a manual queryable. "
        "Indexing runs locally via Ollama and typically takes **10–30 minutes** per document."
    )

    # Upload widget
    uploaded = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="Drag and drop a NASA PDF manual, or click to browse.",
    )
    if uploaded is not None:
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        dest = PDF_DIR / uploaded.name
        if not dest.exists():
            dest.write_bytes(uploaded.read())
            st.success(f"Saved **{uploaded.name}** to the library.")
        else:
            st.info(f"**{uploaded.name}** is already in the library.")

    st.divider()

    # Document list
    pdfs       = sorted(PDF_DIR.glob("*.pdf"))
    raw_status = load_status()

    if not pdfs:
        st.info("No PDFs uploaded yet. Use the file uploader above.")
    else:
        hcol = st.columns([5, 2, 2, 2])
        for h, label in zip(hcol, ["Document", "Size", "Index Status", "Action"]):
            h.markdown(f"**{label}**")
        st.divider()

        for pdf in pdfs:
            st_str  = effective_status(pdf, raw_status)
            c1, c2, c3, c4 = st.columns([5, 2, 2, 2])

            with c1:
                st.markdown(f"**{pdf.stem}**")
                st.caption(pdf.name)

            with c2:
                st.write(f"{pdf.stat().st_size // 1024:,} KB")

            with c3:
                badge = {
                    "ready":    "✅ Ready",
                    "indexing": "🔄 Indexing…",
                    "error":    "❌ Error",
                    "pending":  "⏳ Pending",
                }.get(st_str, "⏳ Pending")
                st.markdown(badge)
                entry = raw_status.get(pdf.name, {})
                if st_str == "error":
                    st.caption((entry.get("error") or "")[:80])
                elif st_str == "indexing":
                    started = entry.get("started_at", "")
                    st.caption(f"Started {started[:16].replace('T', ' ')} UTC")
                    try:
                        start_dt = datetime.datetime.strptime(started, "%Y-%m-%dT%H:%M:%SZ")
                        elapsed  = (datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) - start_dt).total_seconds()
                        est_pct  = min(elapsed / 1800.0, 0.95)  # 30 min estimate, cap at 95%
                        st.progress(est_pct, text=f"~{int(est_pct * 100)}% (estimated)")
                    except Exception:
                        pass

            with c4:
                if st_str == "indexing":
                    if st.button("🔄 Refresh", key=f"ref_{pdf.name}"):
                        st.rerun()
                else:
                    btn_label = "Re-index" if st_str == "ready" else "Build Index"
                    btn_type  = "secondary" if st_str == "ready" else "primary"
                    if st.button(btn_label, key=f"idx_{pdf.name}", type=btn_type):
                        start_indexing(pdf)
                        st.rerun()

            # Structure audit report (written by audit_structure.py after indexing)
            audit_log_path = ROOT / "logs" / f"{pdf.stem}_audit_structure.txt"
            if audit_log_path.exists():
                with st.expander(f"🔍 Structure Audit Report — {pdf.stem[:40]}"):
                    st.code(audit_log_path.read_text(encoding="utf-8", errors="replace"),
                            language=None)

            st.divider()


# ===========================================================================
# TAB 2 — Query / Chat
# ===========================================================================

with tab_chat:
    st.title("Procedural Recall Query")

    with st.expander("ℹ️ How to read the response metrics", expanded=False):
        st.markdown(
            """
Each response is scored across **four independent metrics** to help you judge answer quality at a glance:

| Metric | What it measures |
|---|---|
| **Retrieval** | How closely the source pages pulled from the manuals matched your query. Higher = better page match. |
| **Answer** | Whether the LLM produced a substantive response (penalised for failsafe escalation, missing or incorrect citations). |
| **Completeness** | Whether the response is detailed enough to be actionable — very short answers score lower. |
| **Coverage** | How many relevant manual sections were found relative to the search limit. |

Scores are color-coded: 🟢 ≥ 70% · 🟡 40–69% · 🔴 < 40%
            """
        )

    ready_pdfs = [
        p for p in sorted(PDF_DIR.glob("*.pdf"))
        if (INDEX_DIR / f"{p.stem}_structure.json").exists()
    ]

    if not ready_pdfs:
        st.warning(
            "No indexed manuals available. "
            "Go to **📚 Documents** and click **Build Index** first."
        )
        st.stop()

    st.caption(
        f"Searching {len(ready_pdfs)} indexed manual(s): "
        + "  ·  ".join(f"`{p.stem}`" for p in ready_pdfs)
    )

    # ---- Session history ----
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message("user"):
            st.write(msg["query"])
        with st.chat_message("assistant"):
            render_result(msg)

    # ---- New query ----
    query = st.chat_input(
        "Ask about flight procedures, medical emergencies, or operational rules…"
    )

    if query:
        with st.chat_message("user"):
            st.write(query)

        pb = st.progress(0.0, text="Starting…")
        result = run_query(
            query,
            get_policy(),
            progress_callback=lambda pct, txt: pb.progress(pct, text=txt),
        )
        pb.empty()

        entry = {
            "query":          query,
            "answer":         result["final_response"],
            "query_type":     result["query_type"],
            "confidence":     result["classifier_confidence"],
            "quality":        result.get("quality_score", {}),
            "failsafe":       result["failsafe_triggered"],
            "failsafe_reason": result.get("failsafe_reason"),
            "pages":          result["retrieved_pages"],
            "scan":           result.get("safety_scan", {}),
            "ts":             time.strftime("%H:%M:%S"),
        }
        st.session_state.messages.append(entry)

        with st.chat_message("assistant"):
            render_result(entry)


# ===========================================================================
# TAB 3 — Audit Log
# ===========================================================================

with tab_audit:
    st.title("Audit Log")
    st.caption(
        "Immutable record of every query: routing decision, retrieved pages, "
        "raw LLM output, safety scan results, and final response."
    )

    col_left, col_right = st.columns([6, 1])
    with col_right:
        if AUDIT_PATH.exists():
            st.download_button(
                "📥 Export",
                data=AUDIT_PATH.read_bytes(),
                file_name="nasa_recall_audit.jsonl",
                mime="application/jsonl",
            )

    if not AUDIT_PATH.exists() or AUDIT_PATH.stat().st_size == 0:
        st.info("No queries logged yet. Run a query in the **💬 Query** tab first.")
        st.stop()

    raw_lines = AUDIT_PATH.read_text(encoding="utf-8").strip().splitlines()
    events = []
    for line in reversed(raw_lines[-200:]):
        try:
            events.append(json.loads(line))
        except Exception:
            pass

    st.caption(f"Showing {len(events)} most recent entries (newest first)")

    for ev in events:
        ts_raw  = ev.get("ts_iso_start", "")
        ts_disp = ts_raw[:19].replace("T", " ") if ts_raw else "—"
        qtype   = ev.get("query_type", "—")
        query_t = ev.get("query", "—")
        fail    = ev.get("failsafe_triggered", False)
        dur     = ev.get("duration_ms", 0)
        n_pg    = len(ev.get("retrieved_pages", []))
        status_icon = "⚠️" if fail else "✅"

        header = (
            f"{status_icon} `{ts_disp}` · **{qtype}** · "
            f"{dur} ms · {n_pg} page(s)"
        )
        with st.expander(f"{header}  —  _{query_t[:80]}_", expanded=False):
            left, right = st.columns(2)

            with left:
                st.markdown("**Query**")
                st.write(query_t)

                st.markdown("**Final Response**")
                if fail:
                    st.warning(ev.get("final_response", "—"))
                else:
                    st.write(ev.get("final_response", "—"))

                if ev.get("failsafe_reason"):
                    st.caption(f"Fail-safe reason: `{ev['failsafe_reason']}`")

            with right:
                st.markdown("**Safety Scan**")
                scan = ev.get("safety_scan", {})
                st.json({
                    "blocked":    scan.get("blocked"),
                    "reasons":    scan.get("reasons"),
                    "is_failsafe": scan.get("is_failsafe"),
                })

                st.markdown("**Retrieved Pages**")
                retrieved = ev.get("retrieved_pages", [])
                if retrieved:
                    for p in retrieved:
                        st.caption(
                            f"• {p['doc_title']}  "
                            f"pp.{p['page_number']}–{p.get('end_page', p['page_number'])}  "
                            f"score {p['score']:.3f}"
                        )
                else:
                    st.caption("_None_")


# ===========================================================================
# TAB 4 — Decision Tree
# ===========================================================================

def _build_pipeline_dot(highlight: dict | None = None) -> str:
    """
    Build a Graphviz DOT string for the query pipeline.
    If highlight is provided (from an audit log entry), the nodes along the
    actual path taken are outlined in bright white to show the live trace.
    """
    # Determine which path the last query took
    h = highlight or {}
    qtype        = h.get("query_type", "")
    failsafe     = h.get("failsafe_triggered", False)
    failsafe_why = h.get("failsafe_reason", "")
    has_pages    = bool(h.get("retrieved_pages"))
    is_prohibited = qtype == "prohibited"

    classifier_conf = h.get("classifier_confidence", 1.0)
    low_confidence  = classifier_conf < 0.70 and bool(h)

    def hl(node: str) -> str:
        """Return extra attributes if this node is on the highlighted path."""
        if not h:
            return ""
        active_nodes: set[str] = {"query", "classify"}
        if low_confidence:
            active_nodes.add("conf_warn")
        if is_prohibited:
            active_nodes |= {"prohibited", "audit"}
        else:
            active_nodes.add("retrieve")
            if not has_pages or failsafe_why == "insufficient_evidence":
                active_nodes |= {"failsafe_evidence", "audit"}
            else:
                active_nodes.add("generate")
                active_nodes.add("scan")
                if failsafe_why == "safety_scan_blocked":
                    active_nodes |= {"failsafe_scan", "audit"}
                else:
                    active_nodes |= {"quality", "response", "audit"}
        if node in active_nodes:
            return ', penwidth=3, color="#FFFFFF"'
        return ""

    # Classification label with actual type if known
    classify_label = "Stage 1 — LLM Classification\\nqwen2.5vl:7b"
    if qtype:
        classify_label += f"\\nLast query: {qtype}"

    conf_warn_color = '#8b4500' if low_confidence else '#5a4a1a'
    dot = f"""
digraph pipeline {{
    rankdir=TB
    bgcolor="#0e1117"
    node [fontname="Arial", fontsize=12, margin="0.3,0.2"]
    edge [fontcolor="#aaaaaa", fontsize=10, color="#555555"]

    query [label="User Query", shape=parallelogram,
           style="filled,rounded", fillcolor="#1a4a8a", fontcolor="white"{hl("query")}]

    classify [label="{classify_label}",
              style="filled,rounded", fillcolor="#1a6b3a", fontcolor="white"{hl("classify")}]

    conf_warn [label="Confidence Warning\\nClassifier confidence < 0.70\\nUI warning shown · query continues",
               style="filled,rounded,dashed", fillcolor="{conf_warn_color}", fontcolor="white"{hl("conf_warn")}]

    prohibited [label="BLOCKED\\nProhibited query\\nNo retrieval performed",
                style="filled,rounded", fillcolor="#8b0000", fontcolor="white"{hl("prohibited")}]

    retrieve [label="Stage 2 — Vision-RAG Retrieval\\nTier 0: Figure caption scan\\nTier 1: VLM tree search (qwen2.5vl:7b)\\nTier 2: Keyword + synonym expansion (YAML overlay)\\nTier 3: Full-page scan (fallback)\\nBM25 re-ranking",
              style="filled,rounded", fillcolor="#1a6b3a", fontcolor="white"{hl("retrieve")}]

    failsafe_evidence [label="FAIL-SAFE\\nInsufficient evidence\\nEscalate to ground support",
                       style="filled,rounded", fillcolor="#8b0000", fontcolor="white"{hl("failsafe_evidence")}]

    generate [label="Stage 3 — Answer Generation\\nqwen2.5vl:7b · Text + page images (96 DPI)\\nEvidence-only · Temp=0 · Inline (p.N) citations",
              style="filled,rounded", fillcolor="#1a6b3a", fontcolor="white"{hl("generate")}]

    scan [label="Stage 4 — Safety Scan\\nCitation hit check · Length check\\nFail-safe pattern detection",
          style="filled,rounded", fillcolor="#1a6b3a", fontcolor="white"{hl("scan")}]

    failsafe_scan [label="FAIL-SAFE\\nOutput blocked\\nEscalate to ground support",
                   style="filled,rounded", fillcolor="#8b0000", fontcolor="white"{hl("failsafe_scan")}]

    quality [label="Stage 5 — Quality Scoring\\nRetrieval · Answer · Completeness · Coverage\\nColor-coded scores shown in UI",
             style="filled,rounded", fillcolor="#1a5a6a", fontcolor="white"{hl("quality")}]

    response [label="Final Response\\nInline citations · PDF page links\\nShown to crew member",
              shape=parallelogram, style="filled,rounded",
              fillcolor="#1a4a8a", fontcolor="white"{hl("response")}]

    audit [label="Stage 6 — Audit Log\\nQuery · retrieval_type · quality_score\\nTimestamped · SHA-256 prompt hash",
           style="filled,rounded", fillcolor="#4a1a8a", fontcolor="white"{hl("audit")}]

    query     -> classify
    classify  -> conf_warn         [label="confidence < 0.70", style=dashed, color="#aa6600"]
    conf_warn -> retrieve          [style=dashed, color="#aa6600"]
    classify  -> prohibited        [label="prohibited"]
    classify  -> retrieve          [label="safety_critical / procedural / informational"]
    retrieve  -> failsafe_evidence [label="score < min_score"]
    retrieve  -> generate          [label="evidence found"]
    generate  -> scan
    scan      -> failsafe_scan     [label="blocked"]
    scan      -> quality           [label="passed"]
    quality   -> response

    prohibited        -> audit
    failsafe_evidence -> audit
    failsafe_scan     -> audit
    response          -> audit
}}
"""
    return dot


with tab_tree:
    st.title("Pipeline Decision Tree")
    st.caption(
        "Every query passes through six stages. "
        "The highlighted path shows the route taken by the most recent query. "
        "Dashed lines indicate the low-confidence warning path (query still proceeds)."
    )

    # Load last audit event for live trace
    last_event = None
    if AUDIT_PATH.exists() and AUDIT_PATH.stat().st_size > 0:
        for line in reversed(AUDIT_PATH.read_text(encoding="utf-8").strip().splitlines()[-50:]):
            try:
                last_event = json.loads(line)
                break
            except Exception:
                pass

    col_graph, col_info = st.columns([3, 1])

    with col_graph:
        st.graphviz_chart(_build_pipeline_dot(last_event), use_container_width=True)

    with col_info:
        st.markdown("#### Stage Summary")
        st.markdown("""
**🔵 Stage 1 — Classify**
qwen2.5vl:7b reads the query and routes it to one of four modes: safety\\_critical, procedural, informational, or prohibited. Prohibited queries are blocked immediately. If classifier confidence < 0.70, a low-confidence warning is shown in the UI — the query still proceeds.

**🟢 Stage 2 — Vision-RAG Retrieve**
Four-tier cascade: Tier 0 figure caption scan → Tier 1 VLM tree search (qwen2.5vl:7b scores each index node) → Tier 2 keyword scoring with synonym expansion (YAML overlay) → Tier 3 full-page scan fallback. BM25 re-ranking demotes false-positive VLM hits. If no page meets min\\_score, escalates without calling the LLM.

**🟢 Stage 3 — Generate**
qwen2.5vl:7b synthesises an answer using ONLY the retrieved evidence — both extracted text and rendered page images at 96 DPI (vision input). Temperature 0 for determinism. Every numbered step receives an inline (p.N) citation.

**🟢 Stage 4 — Scan**
Checks that every citation points to a retrieved page, enforces the 5,000-character response limit, and detects fail-safe escalation patterns. Blocks the response if any check fails.

**🔵 Stage 5 — Quality Scoring**
Four independent scores computed from the evidence and response: Retrieval (page match quality), Answer (citation and completeness), Completeness (response depth), Coverage (pages found vs. search limit). Displayed as color-coded indicators with hover tooltips.

**🟣 Stage 6 — Audit**
All decisions — query type, classifier confidence, retrieval tier, quality scores, raw LLM output, safety scan result, and final response — are written to the immutable JSONL audit log with ISO timestamps and SHA-256 prompt hash.
""")

        if last_event:
            st.divider()
            st.markdown("#### Last Query Trace")
            qtype    = last_event.get("query_type", "—")
            fail     = last_event.get("failsafe_triggered", False)
            reason   = last_event.get("failsafe_reason", "")
            n_pages  = len(last_event.get("retrieved_pages", []))
            dur      = last_event.get("duration_ms", 0)
            icon     = {"safety_critical": "🔴", "procedural": "🟡",
                        "informational": "🟢", "prohibited": "⛔"}.get(qtype, "⚪")

            tier_label = {
                "figure_caption": "Tier 0 — Figure caption",
                "vlm_tree":       "Tier 1 — VLM tree search",
                "keyword":        "Tier 2 — Keyword scoring",
                "none":           "No evidence found",
            }.get(last_event.get("retrieval_type", ""), last_event.get("retrieval_type", "—"))

            cc       = last_event.get("classifier_confidence", 1.0)
            qs       = last_event.get("quality_score", {})
            q_ret    = qs.get("retrieval", None)
            q_ans    = qs.get("llm_quality", None)
            q_str    = f"{q_ret:.0%} / {q_ans:.0%}" if q_ret is not None else "—"

            st.caption(f"**Query:** {last_event.get('query','')[:60]}…")
            st.caption(f"{icon} **Type:** {qtype}")
            st.caption(f"**Classifier confidence:** {cc:.2f}")
            st.caption(f"**Retrieval tier:** {tier_label}")
            st.caption(f"**Pages found:** {n_pages}")
            st.caption(f"**Quality (Ret/Ans):** {q_str}")
            st.caption(f"**Outcome:** {'⚠️ Fail-safe — ' + reason if fail else '✅ Answer returned'}")
            st.caption(f"**Duration:** {dur} ms")


# ===========================================================================
# TAB 5 — Document Tree
# ===========================================================================

# Depth-based fill colours (dark theme friendly)
_DEPTH_COLORS = [
    "#1a4a8a",  # 0 — root (dark blue)
    "#1a6b3a",  # 1 — top section (dark green)
    "#5a3a8a",  # 2 — sub-section (purple)
    "#7a5a1a",  # 3 — sub-sub-section (amber)
    "#1a6a6a",  # 4 — deep (teal)
]


def _sanitize_dot(s: str) -> str:
    """Escape characters that would break Graphviz DOT string literals."""
    return s.replace('"', "'").replace("\\", "/").replace("\n", " ")


def _section_page_label(n: int, offsets: list[dict]) -> str:
    """Convert a PDF page number to its section-page label (e.g. 88 -> '4-58')."""
    for seg in offsets:
        if seg["pdf_start"] <= n <= seg["pdf_end"]:
            return f"{seg['prefix']}{n - seg['offset']}"
    return str(n)


def _build_doc_tree_dot(
    structure: list,
    doc_label: str,
    section_offsets: list[dict] | None = None,
) -> str:
    """
    Recursively build a Graphviz DOT string representing the indexed
    section hierarchy of one document.

    Each node shows the section title (truncated to 35 chars) and page range.
    Colour encodes depth.  Layout is left-to-right so deep trees stay readable.
    If section_offsets is provided, page numbers are shown as section-page
    labels (e.g. '4-58') instead of raw PDF page numbers.
    """
    lines: list[str] = []
    node_counter = [0]

    def _page_str(start, end) -> str:
        if section_offsets:
            s = _section_page_label(start, section_offsets)
            e = _section_page_label(end, section_offsets)
        else:
            s, e = str(start), str(end)
        return f"pp.{s}–{e}" if s != e else f"p.{s}"

    def _add_node(node: dict, parent_id: str | None, depth: int) -> None:
        nid = f"n{node_counter[0]}"
        node_counter[0] += 1

        title   = node.get("title", "")
        start   = node.get("start_index", 0)
        end     = node.get("end_index", start)
        label   = _sanitize_dot(title[:35] + ("…" if len(title) > 35 else ""))
        pages   = _page_str(start, end)
        color   = _DEPTH_COLORS[min(depth, len(_DEPTH_COLORS) - 1)]

        lines.append(
            f'    {nid} [label="{label}\\n{pages}", '
            f'style="filled,rounded", fillcolor="{color}", fontcolor="white", '
            f'fontname="Arial", fontsize=10, margin="0.2,0.15"]'
        )
        if parent_id is not None:
            lines.append(f"    {parent_id} -> {nid} [color=\"#555555\"]")

        for child in node.get("nodes", []):
            _add_node(child, nid, depth + 1)

    # Virtual root node for the document itself
    root_id = "doc_root"
    safe_label = _sanitize_dot(doc_label[:50])
    lines.append(
        f'    {root_id} [label="{safe_label}", shape=folder, '
        f'style="filled", fillcolor="#2a2a5a", fontcolor="white", '
        f'fontname="Arial Bold", fontsize=11, margin="0.3,0.2"]'
    )

    # If the index has exactly one wrapper node whose children are the real
    # chapters (legacy structure), skip the wrapper so chapters appear at
    # depth 1 (green).  With a properly structured index (multiple top-level
    # chapters) no flattening is needed.
    if len(structure) == 1 and structure[0].get("nodes"):
        top_nodes = structure[0]["nodes"]
    else:
        top_nodes = structure

    for top_node in top_nodes:
        _add_node(top_node, root_id, depth=1)

    dot = (
        "digraph doc_tree {\n"
        "    rankdir=LR\n"
        "    bgcolor=\"#0e1117\"\n"
        "    node [shape=box]\n"
        "    edge [arrowsize=0.6]\n"
        + "\n".join(lines)
        + "\n}"
    )
    return dot


def _load_toc_indexes() -> list[dict]:
    """
    Return structure JSONs that have a meaningful TOC — defined as documents
    whose structure was extracted by PageIndex from the document itself
    (as opposed to manually curated flat lists).  We use a heuristic:
    the tree must contain at least one node that itself has child nodes.
    """
    results = []
    for path in INDEX_DIR.glob("*_structure.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        structure = data.get("structure", [])
        # Check if any node has children → real nested TOC
        def _has_children(nodes: list) -> bool:
            for n in nodes:
                if n.get("nodes"):
                    return True
                if _has_children(n.get("nodes", [])):
                    return True
            return False
        if _has_children(structure):
            results.append(data)
    return results


with tab_doctree:
    st.title("Indexed Document Trees")
    st.caption(
        "Shows the section hierarchy extracted by PageIndex for documents "
        "where a machine-readable Table of Contents was found.  "
        "Flat lists (ISS Medical Manual, Computers Overview) are excluded "
        "as they have no nested structure."
    )

    toc_indexes = _load_toc_indexes()

    if not toc_indexes:
        st.info("No documents with nested TOC structure found in the indexes/ directory.")
    else:
        doc_names = [Path(d["doc_name"]).stem for d in toc_indexes]
        selected_name = st.selectbox("Select document", doc_names)

        selected_data = next(
            d for d in toc_indexes if Path(d["doc_name"]).stem == selected_name
        )
        structure = selected_data.get("structure", [])

        # Count total nodes for info bar
        def _count_nodes(nodes: list) -> int:
            return sum(1 + _count_nodes(n.get("nodes", [])) for n in nodes)

        total_nodes = _count_nodes(structure)
        max_depth_val = [0]

        def _max_depth(nodes: list, d: int = 0) -> None:
            for n in nodes:
                max_depth_val[0] = max(max_depth_val[0], d)
                _max_depth(n.get("nodes", []), d + 1)

        _max_depth(structure)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total sections", total_nodes)
        col_b.metric("Tree depth", max_depth_val[0] + 1)
        col_c.metric("Top-level sections", len(structure))

        dot_src = _build_doc_tree_dot(
            structure,
            selected_name,
            section_offsets=selected_data.get("section_offsets"),
        )
        st.graphviz_chart(dot_src, use_container_width=True)

        with st.expander("View raw structure JSON"):
            st.json(structure)
