"""
NASA Procedural Recall Assistant — Streamlit UI

Tabs:
  📚 Documents  — upload PDFs and trigger index builds
  💬 Query      — chat interface with side-by-side source page rendering
  📋 Audit Log  — full structured log of every query and safety decision
"""

import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

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
    """Index file on disk is ground truth — overrides stale status entries."""
    if (INDEX_DIR / f"{pdf.stem}_structure.json").exists():
        return "ready"
    return raw.get(pdf.name, {}).get("status", "pending")


def start_indexing(pdf: Path) -> None:
    worker = SRC_DIR / "run_index_worker.py"
    flags  = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    subprocess.Popen(
        [sys.executable, str(worker), str(pdf)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )
    data = load_status()
    data[pdf.name] = {
        "status":      "indexing",
        "started_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished_at": None,
        "error":       None,
    }
    save_status(data)


def find_pdf(doc_title: str) -> Path | None:
    for pdf in PDF_DIR.glob("*.pdf"):
        if pdf.stem == doc_title:
            return pdf
    return None


# ---------------------------------------------------------------------------
# Result renderer — defined BEFORE the tab blocks that call it
# ---------------------------------------------------------------------------

def render_result(msg: dict) -> None:
    """Render answer + source pages side-by-side for one query result."""
    col_ans, col_src = st.columns([3, 2], gap="medium")

    with col_ans:
        qtype = msg["query_type"]
        conf  = msg["confidence"]
        ts    = msg.get("ts", "")
        icon  = {"safety_critical": "🔴", "procedural": "🟡",
                 "informational": "🟢", "prohibited": "⛔"}.get(qtype, "⚪")

        st.caption(f"{icon} **{qtype}** · confidence {conf:.0%} · {ts}")

        if msg["failsafe"]:
            st.warning(msg["answer"], icon="⚠️")
            if msg.get("failsafe_reason"):
                st.caption(f"Reason: `{msg['failsafe_reason']}`")
        else:
            st.markdown(msg["answer"])

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

tab_docs, tab_chat, tab_audit = st.tabs(["📚 Documents", "💬 Query", "📋 Audit Log"])

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

            st.divider()


# ===========================================================================
# TAB 2 — Query / Chat
# ===========================================================================

with tab_chat:
    st.title("Procedural Recall Query")

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
