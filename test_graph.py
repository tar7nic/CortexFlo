from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GROQ_API_KEY, GOOGLE_API_KEY, LLM_MODEL, EMBEDDING_MODEL
from rich import print as rprint

# --- Test LLM ---
def test_llm():
    llm = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, max_tokens=100)
    response = llm.invoke("Say hello in one sentence.")
    rprint(f"[bold green]LLM OK:[/bold green] {response.content}")

# --- Test Embeddings ---
def test_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_document"
    )
    vector = embeddings.embed_query("test sentence")
    rprint(f"[bold green]Embeddings OK:[/bold green] vector length = {len(vector)}")

# --- Define state ---
class AgentState(TypedDict):
    query: str
    response: str

# --- Define node ---
def researcher_node(state: AgentState) -> AgentState:
    rprint(f"[bold purple][researcher][/bold purple] Received: {state['query']}")
    llm = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, max_tokens=100)
    response = llm.invoke(state["query"])
    state["response"] = response.content
    return state

# --- Build graph ---
def build_test_graph():
    graph = StateGraph(AgentState)
    graph.add_node("researcher", researcher_node)
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", END)
    return graph.compile()

# --- Run all tests ---
if __name__ == "__main__":
    rprint("\n[bold yellow]--- Testing LLM (Groq) ---[/bold yellow]")
    test_llm()

    rprint("\n[bold yellow]--- Testing Embeddings (Gemini) ---[/bold yellow]")
    test_embeddings()

    rprint("\n[bold yellow]--- Testing LangGraph Pipeline ---[/bold yellow]")
    app = build_test_graph()
    result = app.invoke({
        "query": "What is retrieval-augmented generation? Answer in one sentence.",
        "response": ""
    })
    rprint(f"\n[bold blue]--- Result ---[/bold blue]")
    rprint(result["response"])
    rprint("\n[bold green]Day 1 Complete! All systems working.[/bold green]")