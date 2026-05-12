import os
import ssl
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import faiss
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rich import print as rprint
from config import GOOGLE_API_KEY, EMBEDDING_MODEL, TOP_K_RETRIEVAL

INDEX_PATH = "vector_store/faiss.index"
METADATA_PATH = "vector_store/metadata.pkl"

def load_index():
    index = faiss.read_index(INDEX_PATH)
    metadata = pickle.load(open(METADATA_PATH, "rb"))
    return index, metadata

def retrieve(query, top_k=TOP_K_RETRIEVAL):
    index, metadata = load_index()
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
    vector = np.array([embeddings.embed_query(query)], dtype=np.float32)
    distances, indices = index.search(vector, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            chunk = metadata[idx].copy()
            chunk["score"] = float(dist)
            results.append(chunk)
    return results

if __name__ == "__main__":
    query = "What is attention mechanism in transformers?"
    results = retrieve(query)
    for i, r in enumerate(results):
        rprint(f"\n[blue]Result {i+1}[/blue] | {r['filename']} | Page {r['page']}")
        rprint(f"Score: {r['score']:.4f} | {r['text'][:200]}...")