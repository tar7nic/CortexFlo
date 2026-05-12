import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_store.retriever import retrieve
from rich import print as rprint

def retriever_agent(state: dict) -> dict:
    query = state["query"]
    rprint(f"[bold cyan][Retriever Agent][/bold cyan] Searching for: {query}")
    results = retrieve(query)
    state["retrieved_docs"] = results
    rprint(f"[green]Retrieved {len(results)} chunks[/green]")
    return state