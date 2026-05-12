import os
import sys
from typing import TypedDict
from langgraph.graph import StateGraph, END
from agents.retriever_agent import retriever_agent
from agents.extractor_agent import extractor_agent
from agents.reporter_agent import reporter_agent
from rich import print as rprint

# --- Define full state ---
class AgentState(TypedDict):
    query: str
    retrieved_docs: list
    insights: str
    final_report: str
    agent_trace: list

# --- Wrap agents to track trace ---
def retriever_node(state: AgentState) -> AgentState:
    state["agent_trace"].append("retriever_agent: started")
    state = retriever_agent(state)
    state["agent_trace"].append(f"retriever_agent: found {len(state['retrieved_docs'])} chunks")
    return state

def extractor_node(state: AgentState) -> AgentState:
    state["agent_trace"].append("extractor_agent: started")
    state = extractor_agent(state)
    state["agent_trace"].append("extractor_agent: insights extracted")
    return state

def reporter_node(state: AgentState) -> AgentState:
    state["agent_trace"].append("reporter_agent: started")
    state = reporter_agent(state)
    state["agent_trace"].append("reporter_agent: report generated")
    return state

# --- Conditional edge: retry if no docs found ---
def check_retrieval(state: AgentState) -> str:
    if not state["retrieved_docs"]:
        rprint("[red]No docs retrieved, ending pipeline.[/red]")
        return "end"
    return "extract"

# --- Build graph ---
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retriever", retriever_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("reporter", reporter_node)

    graph.set_entry_point("retriever")
    graph.add_conditional_edges("retriever", check_retrieval, {
        "extract": "extractor",
        "end": END
    })
    graph.add_edge("extractor", "reporter")
    graph.add_edge("reporter", END)
    return graph.compile()

# --- Run pipeline ---
def run_pipeline(query: str) -> dict:
    app = build_graph()
    initial_state = {
        "query": query,
        "retrieved_docs": [],
        "insights": "",
        "final_report": "",
        "agent_trace": []
    }
    try:
        result = app.invoke(initial_state)
        return result
    except Exception as e:
        initial_state["final_report"] = f"Pipeline error: {str(e)}"
        initial_state["agent_trace"].append(f"ERROR: {str(e)}")
        return initial_state

if __name__ == "__main__":
    query = "Explain the attention mechanism in transformers"
    rprint(f"\n[bold yellow]Query:[/bold yellow] {query}\n")

    result = run_pipeline(query)

    rprint("\n[bold blue]--- Agent Trace ---[/bold blue]")
    for step in result["agent_trace"]:
        rprint(f"  ✓ {step}")

    rprint("\n[bold green]--- Final Report ---[/bold green]")
    print(result["final_report"])   