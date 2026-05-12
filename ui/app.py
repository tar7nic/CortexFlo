import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import tempfile
from vector_store.ingest import extract_text_from_pdfs, chunk_pages, embed_and_index
from main import run_pipeline

st.set_page_config(page_title="Agentic Research Assistant", page_icon="🔬", layout="wide")
st.title("🔬 Agentic Research Assistant")
st.caption("Powered by LangGraph · FAISS · Groq · Gemini")

# --- File upload + query in one row ---
st.subheader("🔍 Research Query")

col_upload, col_query, col_btn = st.columns([0.5, 4, 1])

with col_upload:
    uploaded_files = st.file_uploader("📎", type="pdf", accept_multiple_files=True, label_visibility="collapsed")

with col_query:
    query = st.text_input("Query", placeholder="Upload PDFs 📎 then ask your question...", label_visibility="collapsed")

with col_btn:
    run_btn = st.button("🔍 Search", type="primary", use_container_width=True)

# --- Index + Run ---
if run_btn:
    if not uploaded_files:
        st.warning("Please upload at least one PDF first.")
    elif not query.strip():
        st.warning("Please enter a research query.")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in uploaded_files:
                with open(os.path.join(tmpdir, f.name), "wb") as out:
                    out.write(f.read())
            with st.spinner(f"Indexing {len(uploaded_files)} document(s)..."):
                pages = extract_text_from_pdfs(tmpdir)
                chunks = chunk_pages(pages)
                embed_and_index(chunks)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🔎 Agent Trace")
            with st.status("Running agents...", expanded=True) as status:
                result = run_pipeline(query)
                for step in result["agent_trace"]:
                    st.write(f"✅ {step}")
                status.update(label="Pipeline complete!", state="complete")

        with col2:
            st.subheader("📄 Retrieved Sources")
            for i, doc in enumerate(result["retrieved_docs"]):
                with st.expander(f"Source {i+1} — {doc['filename']} | Page {doc['page']}"):
                    st.write(doc["text"])

        st.subheader("📝 Final Report")
        st.markdown(result["final_report"])

        with st.expander("💡 Raw Insights"):
            st.write(result["insights"])