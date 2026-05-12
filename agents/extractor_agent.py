import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_groq import ChatGroq
from rich import print as rprint
from config import GROQ_API_KEY, LLM_MODEL, MAX_TOKENS

def extractor_agent(state: dict) -> dict:
    rprint(f"[bold yellow][Extractor Agent][/bold yellow] Extracting insights...")
    docs = state.get("retrieved_docs", [])
    
    if not docs:
        state["insights"] = "No relevant documents found."
        return state

    context = "\n\n".join([
        f"Source: {d['filename']} | Page {d['page']}\n{d['text']}"
        for d in docs
    ])

    # Token guard
    if len(context) > 3000:
        context = context[:3000] + "...[truncated]"

    llm = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, max_tokens=MAX_TOKENS)
    prompt = f"""You are an expert research analyst. Extract precise, factual insights from the chunks below.

Query: {state['query']}

Document Chunks:
{context}

Extract 3-5 key insights with source attribution as (filename, Page X). 
If chunks are irrelevant, say "Insufficient relevant content found"."""

    response = llm.invoke(prompt)
    state["insights"] = response.content
    rprint(f"[green]Insights extracted[/green]")
    return state