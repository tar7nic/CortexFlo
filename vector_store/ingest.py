import ssl
import certifi
import os
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
from pypdf import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from rich import print as rprint
from config import GOOGLE_API_KEY, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

DATA_DIR = "data"
VECTOR_STORE_DIR = "vector_store"
INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss.index")
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")

def extract_text_from_pdfs(data_dir):
    all_pages = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            reader = PdfReader(os.path.join(data_dir, filename))
            rprint(f"[cyan]Reading:[/cyan] {filename}")
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    all_pages.append({"filename": filename, "page": i+1, "text": text})
    rprint(f"[green]Extracted {len(all_pages)} pages[/green]")
    return all_pages

def chunk_pages(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for page in pages:
        for i, chunk in enumerate(splitter.split_text(page["text"])):
            chunks.append({"filename": page["filename"], "page": page["page"], "chunk_index": i, "text": chunk})
    rprint(f"[green]Created {len(chunks)} chunks[/green]")
    return chunks

def embed_and_index(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
    texts = [c["text"] for c in chunks]
    all_vectors = []
    for i in range(0, len(texts), 50):
        all_vectors.extend(embeddings.embed_documents(texts[i:i+50]))
        rprint(f"  Embedded {min(i+50, len(texts))}/{len(texts)}")
    vectors = np.array(all_vectors, dtype=np.float32)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, INDEX_PATH)
    pickle.dump(chunks, open(METADATA_PATH, "wb"))
    rprint(f"[bold green]Ingestion complete! {len(chunks)} chunks indexed.[/bold green]")

if __name__ == "__main__":
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    pages = extract_text_from_pdfs(DATA_DIR)
    if pages:
        embed_and_index(chunk_pages(pages))