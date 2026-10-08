import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import Faithfulness, ContextPrecision, ContextRecall
from ragas.llms import llm_factory
from openai import OpenAI
from main import run_pipeline
from config import GROQ_API_KEY, LLM_MODEL
from rich import print as rprint

EVAL_SAMPLES = [
    {
        "question": "What is the attention mechanism in transformers?",
        "ground_truth": "The attention mechanism allows transformers to weigh the importance of different tokens when encoding a sequence."
    },
    {
        "question": "What is retrieval-augmented generation?",
        "ground_truth": "RAG combines a retrieval system with a language model to ground responses in external documents."
    },
]

def run_eval():
    questions, answers, contexts, ground_truths = [], [], [], []

    for sample in EVAL_SAMPLES:
        rprint(f"[bold cyan]Running pipeline for:[/bold cyan] {sample['question']}")
        result = run_pipeline(sample["question"])
        questions.append(sample["question"])
        answers.append(result.get("final_report", ""))
        contexts.append([d["text"] for d in result.get("retrieved_docs", [])])
        ground_truths.append(sample["ground_truth"])

    dataset = Dataset.from_dict({
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths,
    })

    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    llm = llm_factory(model=LLM_MODEL, client=client)

    metrics = [
        Faithfulness(llm=llm),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    rprint("\n[bold yellow]Running RAGAS evaluation...[/bold yellow]")
    results = evaluate(dataset=dataset, metrics=metrics)

    rprint("\n[bold green]── Evaluation Results ──[/bold green]")
    df = results.to_pandas()
    print(df[["user_input", "faithfulness", "context_precision", "context_recall"]].to_string(index=False))

    df.to_csv("eval_results.csv", index=False)
    rprint("\n[green]Saved to eval_results.csv[/green]")

if __name__ == "__main__":
    run_eval()