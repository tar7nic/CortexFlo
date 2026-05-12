import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_groq import ChatGroq
from rich import print as rprint
from config import GROQ_API_KEY, LLM_MODEL, MAX_TOKENS

def reporter_agent(state: dict) -> dict:
    rprint(f"[bold magenta][Reporter Agent][/bold magenta] Generating report...")
    
    if len(context) > 3000:
        context = context[:3000] + "...[truncated]"
    
    llm = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, max_tokens=MAX_TOKENS)

    prompt = f"""You are a professional research report writer.

Query: {state['query']}

Insights:
{state['insights']}

Write a structured markdown report with these exact sections:
# Research Report: [topic]
## Summary
[2-3 sentence overview]
## Key Findings
[bullet points from insights]
## Conclusion
[1-2 sentences]
## Sources Referenced
[list all cited sources from insights]

Be concise, factual, and professional."""

    response = llm.invoke(prompt)
    state["final_report"] = response.content
    rprint(f"[green]Report generated[/green]")
    return state