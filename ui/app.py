import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import tempfile
from vector_store.ingest import extract_text_from_pdfs, chunk_pages, embed_and_index
from main import run_pipeline

st.set_page_config(
    page_title="CortexFlo",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global Styles ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne+Mono&family=DM+Sans:wght@300;400;500&family=Syne:wght@700;800&display=swap');

:root {
    --bg:         #0e0e0e;
    --bg2:        #141414;
    --bg3:        #1a1a1a;
    --border:     #242424;
    --border2:    #2e2e2e;
    --amber:      #d4893a;
    --amber-dim:  #7a4e20;
    --amber-glow: rgba(212,137,58,0.12);
    --text:       #e2e2e2;
    --text-dim:   #666;
    --text-muted: #3a3a3a;
    --green:      #3d8b5e;
    --green-dim:  #1e4a32;
    --red:        #8b3d3d;
    --mono:       'Syne Mono', monospace;
    --body:       'DM Sans', sans-serif;
    --display:    'Syne', sans-serif;
}

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section[data-testid="stMain"] > div {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--body) !important;
}

/* subtle scanline */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    background: repeating-linear-gradient(
        0deg,
        transparent, transparent 2px,
        rgba(0,0,0,0.05) 2px, rgba(0,0,0,0.05) 4px
    );
    z-index: 9999;
}

/* hide chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

.block-container {
    max-width: 1020px !important;
    padding: 0 2.5rem 3rem !important;
    margin: 0 auto !important;
}

/* ── HEADER ── */
.cf-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 30px 0 22px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 36px;
}
.cf-logo { display: flex; align-items: baseline; gap: 14px; }
.cf-logo-name {
    font-family: var(--display);
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.05em;
}
.cf-logo-name span { color: var(--amber); }
.cf-logo-tag {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--text-dim);
    letter-spacing: 0.06em;
}
.cf-stack {
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-align: right;
    line-height: 2;
}

/* ── DOCK LABEL ── */
.cf-dock-label {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] { background: transparent !important; }
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploadDropzone"] {
    background: var(--bg2) !important;
    border: 1px dashed var(--border2) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--amber-dim) !important;
    background: var(--amber-glow) !important;
}
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] span {
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
    color: var(--text-dim) !important;
}
[data-testid="stFileUploader"] button {
    background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 6px !important;
    color: var(--amber) !important;
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
}
[data-testid="stFileUploaderFile"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploaderFileName"] {
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
    color: var(--amber) !important;
}

/* ── TEXT INPUT ── */
[data-testid="stTextInput"] > div > div {
    background: var(--bg2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stTextInput"] > div > div:focus-within {
    border-color: var(--amber-dim) !important;
    box-shadow: 0 0 0 3px var(--amber-glow) !important;
}
[data-testid="stTextInput"] input {
    color: var(--text) !important;
    font-family: var(--body) !important;
    font-size: 0.92rem !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--text-muted) !important; }

/* ── RUN BUTTON ── */
[data-testid="stButton"] > button {
    background: var(--bg2) !important;
    border: 1px solid var(--amber-dim) !important;
    border-radius: 10px !important;
    color: var(--amber) !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    height: 46px !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] > button:hover {
    background: var(--amber-glow) !important;
    border-color: var(--amber) !important;
    box-shadow: 0 0 18px var(--amber-glow) !important;
}

/* ── DIVIDER ── */
.cf-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 36px 0;
}

/* ── SECTION HEADING ── */
.cf-section {
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.cf-section::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── AGENT TRACE ── */
.cf-trace {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-left: 2px solid var(--green-dim);
    border-radius: 8px;
    padding: 16px 20px;
    font-family: var(--mono);
    font-size: 0.7rem;
    line-height: 2.1;
    color: var(--text-muted);
}
.cf-trace .ok  { color: var(--green); }
.cf-trace .err { color: var(--red); }

/* ── SOURCE CARDS ── */
.cf-sources { display: flex; flex-direction: column; gap: 8px; }
.cf-source-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
}
.src-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--amber);
    margin-bottom: 7px;
    letter-spacing: 0.04em;
}
.src-text {
    font-size: 0.79rem;
    color: #787878;
    line-height: 1.55;
    max-height: 78px;
    overflow: hidden;
    -webkit-mask-image: linear-gradient(to bottom, black 55%, transparent 100%);
}

/* ── REPORT ── */
.cf-report-wrap {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-top: 2px solid var(--amber-dim);
    border-radius: 12px;
    padding: 36px 40px;
}
.cf-report-wrap h1 {
    font-family: var(--display) !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    letter-spacing: -0.03em !important;
    margin-top: 0 !important;
    margin-bottom: 22px !important;
}
.cf-report-wrap h2 {
    font-family: var(--mono) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--amber) !important;
    margin-top: 30px !important;
    margin-bottom: 10px !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 6px !important;
}
.cf-report-wrap p  { color: #b4b4b4 !important; font-size: 0.88rem !important; margin-bottom: 12px !important; }
.cf-report-wrap li { color: #acacac !important; font-size: 0.87rem !important; margin-bottom: 8px !important; }
.cf-report-wrap strong { color: var(--text) !important; }

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    margin-top: 14px !important;
}
[data-testid="stExpander"] details summary p {
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
    color: var(--text-dim) !important;
    letter-spacing: 0.06em !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] {
    background: var(--bg2) !important;
    border: 1px solid var(--amber-dim) !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    color: var(--amber) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cf-header">
    <div class="cf-logo">
        <div class="cf-logo-name">Cortex<span>Flo</span></div>
        <div class="cf-logo-tag">research terminal · v1.0</div>
    </div>
    <div class="cf-stack">
        LangGraph &nbsp;·&nbsp; FAISS &nbsp;·&nbsp; Groq &nbsp;·&nbsp; Gemini<br>
        RAG &nbsp;·&nbsp; Multi-Agent Pipeline
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input dock ─────────────────────────────────────────────────────────────────
st.markdown('<div class="cf-dock-label">⬡ &nbsp;attach PDFs + enter query</div>', unsafe_allow_html=True)

col_up, col_q, col_btn = st.columns([1.7, 4.0, 0.8])

with col_up:
    uploaded_files = st.file_uploader(
        "pdfs",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

with col_q:
    query = st.text_input(
        "query",
        placeholder="What would you like to research from these documents?",
        label_visibility="collapsed"
    )

with col_btn:
    run_btn = st.button("RUN ↵", use_container_width=True)

# ── Pipeline ───────────────────────────────────────────────────────────────────
if run_btn:
    if not uploaded_files:
        st.warning("Attach at least one PDF to begin.")
    elif not query.strip():
        st.warning("Enter a research query.")
    else:
        st.markdown('<hr class="cf-divider">', unsafe_allow_html=True)

        with st.spinner(f"Indexing {len(uploaded_files)} document(s)…"):
            with tempfile.TemporaryDirectory() as tmpdir:
                for f in uploaded_files:
                    with open(os.path.join(tmpdir, f.name), "wb") as out:
                        out.write(f.read())
                pages = extract_text_from_pdfs(tmpdir)
                chunks = chunk_pages(pages)
                embed_and_index(chunks)

        with st.spinner("Running agent pipeline…"):
            result = run_pipeline(query)

        # ── Trace + Sources ────────────────────────────────────────────────────
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown('<div class="cf-section">⬡ &nbsp;Agent Trace</div>', unsafe_allow_html=True)
            trace_lines = ""
            for step in result.get("agent_trace", []):
                cls = "err" if "ERROR" in step.upper() else "ok"
                icon = "✗" if cls == "err" else "✓"
                trace_lines += f'<div class="{cls}">{icon}&nbsp;&nbsp;{step}</div>'
            st.markdown(f'<div class="cf-trace">{trace_lines or "<span>No trace data.</span>"}</div>',
                        unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="cf-section">⬡ &nbsp;Retrieved Sources</div>', unsafe_allow_html=True)
            docs = result.get("retrieved_docs", [])
            if docs:
                cards = "".join(f"""
                <div class="cf-source-card">
                    <div class="src-label">[{i+1}]&nbsp;&nbsp;{d['filename']}&nbsp;·&nbsp;p.{d['page']}</div>
                    <div class="src-text">{d['text'][:240].replace('<','&lt;').replace('>','&gt;')}…</div>
                </div>""" for i, d in enumerate(docs))
                st.markdown(f'<div class="cf-sources">{cards}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="cf-trace">No sources retrieved.</div>', unsafe_allow_html=True)

        # ── Report ─────────────────────────────────────────────────────────────
        st.markdown('<hr class="cf-divider">', unsafe_allow_html=True)
        st.markdown('<div class="cf-section">⬡ &nbsp;Research Report</div>', unsafe_allow_html=True)
        st.markdown('<div class="cf-report-wrap">', unsafe_allow_html=True)
        st.markdown(result.get("final_report", "_No report generated._"))
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Raw Insights ───────────────────────────────────────────────────────
        with st.expander("▸ &nbsp;Raw Extracted Insights"):
            st.write(result.get("insights", ""))