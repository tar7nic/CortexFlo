import os
import pickle
import faiss
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rich import print as rprint
from config import GOOGLE_API_KEY, EMBEDDING_MODEL, TOP_K_RETRIEVAL

# --- Paths ---
VECTOR_STORE_DIR = "vector_store"
INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss.index")
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")


# --- Load FAISS index and metadata ---
def load_index():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("FAISS index not found. Run ingest.py first.")
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


# --- Embed a query ---
def embed_query(query: str) -> np.ndarray:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )
    vector = embeddings.embed_query(query)
    return np.array([vector], dtype=np.float32)


# --- Retrieve top-k chunks ---
def retrieve(query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    index, metadata = load_index()
    query_vector = embed_query(query)

    distances, indices = index.search(query_vector, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = metadata[idx].copy()
        chunk["score"] = float(dist)  # L2 distance (lower = more similar)
        results.append(chunk)

    return results


# --- Pretty print results ---
def print_results(results: list[dict]):
    for i, r in enumerate(results):
        rprint(f"\n[bold blue]Result {i+1}[/bold blue]")
        rprint(f"  [cyan]Source:[/cyan] {r['filename']} | Page {r['page']} | Chunk {r['chunk_index']}")
        rprint(f"  [cyan]Score:[/cyan] {r['score']:.4f} (lower = better)")
        rprint(f"  [cyan]Text:[/cyan] {r['text'][:200]}...")


# --- Test retrieval ---
if __name__ == "__main__":
    query = "What is attention mechanism in transformers?"
    rprint(f"\n[bold yellow]Query:[/bold yellow] {query}\n")

    results = retrieve(query)
    print_results(results)

    rprint(f"\n[bold green]Retrieved {len(results)} chunks successfully![/bold green]")