import sys, os, html, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
os.chdir(ROOT)  # relative index paths resolve regardless of launch dir
import streamlit as st
from vector_store.ingest import extract_text_from_pdfs, chunk_pages, embed_and_index, INDEX_PATH, METADATA_PATH
from main import run_pipeline

st.set_page_config(page_title="CortexFlo", page_icon="⬡", layout="wide",
                   initial_sidebar_state="collapsed")

# ── State ──────────────────────────────────────────────────────────────────────
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("docs", {})          # filename -> bytes
ss.setdefault("indexed_sig", None)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne+Mono&family=DM+Sans:wght@300;400;500&family=Syne:wght@700;800&display=swap');
:root{
 --bg:#0e0e0e;--bg2:#141414;--bg3:#1a1a1a;--border:#242424;--border2:#2e2e2e;
 --amber:#d4893a;--amber-dim:#7a4e20;--amber-glow:rgba(212,137,58,0.12);
 --text:#e2e2e2;--text-dim:#666;--text-muted:#3a3a3a;
 --green:#3d8b5e;--green-dim:#1e4a32;--red:#8b3d3d;
 --mono:'Syne Mono',monospace;--body:'DM Sans',sans-serif;--display:'Syne',sans-serif;
}
*,*::before,*::after{box-sizing:border-box;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],
section[data-testid="stMain"]>div{background:var(--bg)!important;color:var(--text)!important;font-family:var(--body)!important;}
[data-testid="stAppViewContainer"]::before{content:'';position:fixed;inset:0;pointer-events:none;
 background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.05) 2px,rgba(0,0,0,0.05) 4px);z-index:9999;}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"]{display:none!important;}
.block-container{max-width:1020px!important;padding:0 2.5rem 8rem!important;margin:0 auto!important;}

/* header */
.cf-header{display:flex;align-items:center;justify-content:space-between;padding:30px 0 22px;border-bottom:1px solid var(--border);margin-bottom:28px;}
.cf-logo{display:flex;align-items:baseline;gap:14px;}
.cf-logo-name{font-family:var(--display);font-size:1.6rem;font-weight:800;color:var(--text);letter-spacing:-0.05em;}
.cf-logo-name span{color:var(--amber);}
.cf-logo-tag{font-family:var(--mono);font-size:.62rem;color:var(--text-dim);letter-spacing:.06em;}
.cf-stack{font-family:var(--mono);font-size:.6rem;color:var(--text-muted);letter-spacing:.1em;text-align:right;line-height:2;}

/* empty state */
.cf-empty{text-align:center;padding:70px 0 40px;}
.cf-empty .big{font-family:var(--display);font-size:1.25rem;color:var(--text);letter-spacing:-0.03em;margin-bottom:10px;}
.cf-empty .big span{color:var(--amber);}
.cf-empty .sm{font-family:var(--mono);font-size:.68rem;color:var(--text-dim);letter-spacing:.06em;line-height:2;}

/* doc chips */
.cf-chips{display:flex;flex-wrap:wrap;gap:8px;align-items:center;}
.cf-chip{font-family:var(--mono);font-size:.66rem;color:var(--amber);background:var(--bg3);
 border:1px solid var(--border2);border-radius:999px;padding:5px 12px;}
.cf-chip-label{font-family:var(--mono);font-size:.62rem;color:var(--text-muted);letter-spacing:.12em;text-transform:uppercase;margin-right:4px;}

/* user bubble */
.cf-user{display:flex;justify-content:flex-end;margin:26px 0 18px;}
.cf-user div{max-width:75%;background:var(--bg3);border:1px solid var(--border2);border-radius:14px 14px 4px 14px;
 padding:11px 16px;font-size:.92rem;color:var(--text);line-height:1.5;}

/* sections */
.cf-section{font-family:var(--mono);font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:var(--text-dim);
 margin:6px 0 14px;display:flex;align-items:center;gap:10px;}
.cf-section::after{content:'';flex:1;height:1px;background:var(--border);}

/* trace */
.cf-trace{background:var(--bg2);border:1px solid var(--border);border-left:2px solid var(--green-dim);border-radius:8px;
 padding:16px 20px;font-family:var(--mono);font-size:.7rem;line-height:2.1;color:var(--text-muted);}
.cf-trace .ok{color:var(--green);} .cf-trace .err{color:var(--red);}

/* sources */
.cf-sources{display:flex;flex-direction:column;gap:8px;}
.cf-source-card{background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:12px 16px;}
.src-label{font-family:var(--mono);font-size:.65rem;color:var(--amber);margin-bottom:7px;letter-spacing:.04em;}
.src-text{font-size:.79rem;color:#787878;line-height:1.55;max-height:78px;overflow:hidden;
 -webkit-mask-image:linear-gradient(to bottom,black 55%,transparent 100%);}

/* report card (keyed container) */
[class*="st-key-cfreport"]{background:var(--bg2)!important;border:1px solid var(--border)!important;
 border-top:2px solid var(--amber-dim)!important;border-radius:12px!important;padding:32px 38px!important;}
[class*="st-key-cfreport"] h1{font-family:var(--display)!important;font-size:1.3rem!important;font-weight:700!important;
 color:var(--text)!important;letter-spacing:-0.03em!important;margin:0 0 22px!important;padding:0!important;}
[class*="st-key-cfreport"] h2{font-family:var(--mono)!important;font-size:.65rem!important;letter-spacing:.14em!important;
 text-transform:uppercase!important;color:var(--amber)!important;margin:30px 0 10px!important;
 border-bottom:1px solid var(--border)!important;padding:0 0 6px!important;}
[class*="st-key-cfreport"] p{color:#b4b4b4!important;font-size:.88rem!important;margin-bottom:12px!important;}
[class*="st-key-cfreport"] li{color:#acacac!important;font-size:.87rem!important;margin-bottom:8px!important;}
[class*="st-key-cfreport"] strong{color:var(--text)!important;}

/* expander */
[data-testid="stExpander"]{background:var(--bg2)!important;border:1px solid var(--border)!important;border-radius:8px!important;margin-top:14px!important;}
[data-testid="stExpander"] details summary p{font-family:var(--mono)!important;font-size:.68rem!important;color:var(--text-dim)!important;letter-spacing:.06em!important;}

/* buttons */
[data-testid="stButton"]>button{background:var(--bg2)!important;border:1px solid var(--border2)!important;border-radius:8px!important;
 color:var(--text-dim)!important;font-family:var(--mono)!important;font-size:.64rem!important;letter-spacing:.08em!important;transition:all .2s!important;}
[data-testid="stButton"]>button:hover{border-color:var(--amber)!important;color:var(--amber)!important;background:var(--amber-glow)!important;}

/* alerts */
[data-testid="stAlert"]{background:var(--bg2)!important;border:1px solid var(--amber-dim)!important;border-radius:8px!important;
 font-family:var(--mono)!important;font-size:.72rem!important;color:var(--amber)!important;}

/* ── CHAT INPUT (with inline + attach) ── */
[data-testid="stBottom"],[data-testid="stBottom"]>div,[data-testid="stBottomBlockContainer"]{background:var(--bg)!important;}
[data-testid="stChatInput"]{background:var(--bg2)!important;border:1px solid var(--border2)!important;border-radius:16px!important;
 transition:border-color .2s,box-shadow .2s;}
[data-testid="stChatInput"]:focus-within{border-color:var(--amber-dim)!important;box-shadow:0 0 0 3px var(--amber-glow)!important;}
[data-testid="stChatInput"] > div,[data-testid="stChatInput"] textarea{background:transparent!important;}
[data-testid="stChatInput"] textarea{color:var(--text)!important;font-family:var(--body)!important;font-size:.95rem!important;}
[data-testid="stChatInput"] textarea::placeholder{color:#555!important;}
[data-testid="stChatInput"] button{background:transparent!important;border:none!important;color:var(--amber)!important;}
[data-testid="stChatInput"] button:hover{background:var(--amber-glow)!important;}
[data-testid="stChatInput"] button:disabled{color:var(--text-muted)!important;}
[data-testid="stChatInput"] svg{fill:currentColor;}
[data-testid="stChatInput"] [data-testid="stChatInputFile"],
[data-testid="stChatInput"] [data-testid="stChatInputFileName"]{font-family:var(--mono)!important;font-size:.68rem!important;color:var(--amber)!important;}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
esc = lambda s: html.escape(str(s))

def ensure_index() -> bool:
    """Index exactly the attached PDFs (fresh, no stale data). Returns success."""
    sig = tuple(sorted((n, len(b)) for n, b in ss.docs.items()))
    if ss.indexed_sig == sig:
        return True
    for p in (INDEX_PATH, METADATA_PATH):
        if os.path.exists(p):
            os.remove(p)
    with tempfile.TemporaryDirectory() as tmp:
        for n, b in ss.docs.items():
            with open(os.path.join(tmp, n), "wb") as out:
                out.write(b)
        pages = extract_text_from_pdfs(tmp)
        chunks = chunk_pages(pages) if pages else []
    if not chunks:
        return False
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    embed_and_index(chunks)
    ss.indexed_sig = sig
    return True

def render_answer(m: dict, idx: int):
    res = m["result"]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="cf-section">⬡ &nbsp;Agent Trace</div>', unsafe_allow_html=True)
        lines = ""
        for step in res.get("agent_trace", []):
            err = "ERROR" in step.upper()
            lines += f'<div class="{"err" if err else "ok"}">{"✗" if err else "✓"}&nbsp;&nbsp;{esc(step)}</div>'
        st.markdown(f'<div class="cf-trace">{lines or "No trace data."}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="cf-section">⬡ &nbsp;Retrieved Sources</div>', unsafe_allow_html=True)
        docs = m["docs"]
        if docs:
            cards = "".join(
                f'<div class="cf-source-card"><div class="src-label">[{i+1}]&nbsp;&nbsp;{esc(d["filename"])}&nbsp;·&nbsp;p.{d["page"]}</div>'
                f'<div class="src-text">{esc(d["text"][:240])}…</div></div>'
                for i, d in enumerate(docs))
            st.markdown(f'<div class="cf-sources">{cards}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="cf-trace">No sources retrieved.</div>', unsafe_allow_html=True)

    st.markdown('<div class="cf-section" style="margin-top:28px">⬡ &nbsp;Research Report</div>', unsafe_allow_html=True)
    with st.container(key=f"cfreport_{idx}"):
        st.markdown(res.get("final_report") or "_No report generated._")
    with st.expander("▸  Raw Extracted Insights"):
        st.write(res.get("insights", ""))

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cf-header">
  <div class="cf-logo"><div class="cf-logo-name">Cortex<span>Flo</span></div>
  <div class="cf-logo-tag">research terminal · v1.0</div></div>
  <div class="cf-stack">LangGraph &nbsp;·&nbsp; FAISS &nbsp;·&nbsp; Groq &nbsp;·&nbsp; Gemini<br>RAG &nbsp;·&nbsp; Multi-Agent Pipeline</div>
</div>""", unsafe_allow_html=True)

# ── Attached docs strip ────────────────────────────────────────────────────────
if ss.docs:
    cc, cb = st.columns([6, 1])
    with cc:
        chips = "".join(f'<span class="cf-chip">⬡ {esc(n)}</span>' for n in ss.docs)
        st.markdown(f'<div class="cf-chips"><span class="cf-chip-label">attached</span>{chips}</div>', unsafe_allow_html=True)
    with cb:
        if st.button("CLEAR DOCS", use_container_width=True):
            ss.docs, ss.indexed_sig = {}, None
            for p in (INDEX_PATH, METADATA_PATH):
                if os.path.exists(p):
                    os.remove(p)
            st.rerun()

# ── History ────────────────────────────────────────────────────────────────────
if not ss.messages:
    st.markdown("""
    <div class="cf-empty">
      <div class="big">Ask anything about <span>your</span> documents</div>
      <div class="sm">click <b style="color:#d4893a">+</b> in the box below to attach PDFs<br>
      they are indexed live — answers come only from what you attach</div>
    </div>""", unsafe_allow_html=True)

for i, m in enumerate(ss.messages):
    if m["role"] == "user":
        st.markdown(f'<div class="cf-user"><div>{esc(m["text"])}</div></div>', unsafe_allow_html=True)
    else:
        render_answer(m, i)

# ── Prompt box (text + inline "+" attach) ──────────────────────────────────────
prompt = st.chat_input("Ask about your documents…   (use + to attach PDFs)",
                       accept_file="multiple", file_type=["pdf"])

if prompt:
    text = (prompt["text"] or "").strip()
    for f in (prompt["files"] or []):
        ss.docs[f.name] = f.getvalue()

    if not text:
        st.rerun()  # files only → refresh chips
    elif not ss.docs:
        ss.messages.append({"role": "user", "text": text})
        ss.messages.append({"role": "assistant", "docs": [], "result": {
            "agent_trace": ["ERROR: no PDF attached — click + to attach one"],
            "final_report": "**Attach at least one PDF** using the **+** button, then ask again.",
            "insights": ""}})
        st.rerun()
    else:
        st.markdown(f'<div class="cf-user"><div>{esc(text)}</div></div>', unsafe_allow_html=True)
        with st.spinner("Indexing documents and running agents…"):
            if ensure_index():
                result = run_pipeline(text)
                docs = [d for d in result.get("retrieved_docs", []) if d["filename"] in ss.docs]
            else:
                result = {"agent_trace": ["ERROR: no extractable text (scanned PDF?)"],
                          "final_report": "Could not read text from the attached PDFs. They may be scanned images.",
                          "insights": ""}
                docs = []
        ss.messages.append({"role": "user", "text": text})
        ss.messages.append({"role": "assistant", "docs": docs, "result": result})
        st.rerun()