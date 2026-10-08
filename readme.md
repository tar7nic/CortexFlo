# 🔬 CortexFlo — Multi-Agent Research Assistant

A LangGraph-orchestrated RAG pipeline that answers research questions over uploaded PDFs using a 3-agent architecture. Built with free-tier APIs only — no OpenAI.

---

## Architecture

PDF Upload → Ingest → FAISS Index
↓
[LangGraph StateGraph]
↓
Retriever Agent → FAISS semantic search
↓
Extractor Agent → Groq LLM extracts insights
↓
Reporter Agent → Groq LLM writes structured report


**Stack:**
| Layer | Tool |
|---|---|
| Orchestration | LangGraph `StateGraph` |
| Embedding | Google Gemini `gemini-embedding-001` |
| Vector Store | FAISS `IndexFlatL2` |
| LLM | Groq `llama-3.1-8b-instant` |
| UI | Streamlit (chat-style interface) |
| Evaluation | RAGAS v1 |

---

## Setup

### 1. Clone & create environment
```bash
git clone https://github.com/yourusername/CortexFlo.git
cd CortexFlo
conda create -n cortex python=3.11
conda activate cortex
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API keys
Create a `.env` file in the root:
```env
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
```

---

## Run

```bash
streamlit run app.py
```

---

## Project Structure

CortexFlo/
├── agents/
│ ├── retriever_agent.py # FAISS semantic search
│ ├── extractor_agent.py # Insight extraction via Groq
│ └── reporter_agent.py # Report generation via Groq
├── vector_store/
│ ├── faiss.index
│ └── metadata.pkl
├── app.py # Streamlit chat UI
├── main.py # LangGraph pipeline
├── ingest.py # PDF ingestion + chunking + embedding
├── retriever.py # FAISS query logic
├── eval.py # RAGAS evaluation
├── config.py # Model config + API keys
└── .env # API keys (not committed)


---

## Evaluation

```bash
python eval.py
```

Runs RAGAS metrics — **Faithfulness**, **Context Precision**, **Context Recall** — using Groq as the evaluation LLM. Results saved to `eval_results.csv`.

---

## Known Limitations

- FAISS index is in-memory — does not persist across sessions; re-upload PDFs each run
- `AnswerRelevancy` metric excluded — requires an embedding provider with an OpenAI-compatible endpoint

---
