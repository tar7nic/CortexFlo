from agents.retriever_agent import retriever_agent
from agents.extractor_agent import extractor_agent
from agents.reporter_agent import reporter_agent
from rich import print as rprint

state = {
    "query": "What is the attention mechanism in transformers?",
    "retrieved_docs": [],
    "insights": "",
    "final_report": ""
}

state = retriever_agent(state)
state = extractor_agent(state)
state = reporter_agent(state)

rprint("\n[bold green]--- Final Report ---[/bold green]")
print(state["final_report"])