import os
import pickle
from pypdf import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from rich import print as rprint
from config import GOOGLE_API_KEY, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

# --- Paths ---
DATA_DIR = "data"
VECTOR_STORE_DIR = "vector_store"
INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss.index")
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")


# --- Step 1: Extract text from PDFs ---
def extract_text_from_pdfs(data_dir: str) -> list[dict]:
    all_pages = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            rprint(f"[bold cyan]Reading:[/bold cyan] {filename}")
            reader = PdfReader(filepath)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    all_pages.append({
                        "filename": filename,
                        "page": page_num + 1,
                        "text": text
                    })
    rprint(f"[green]Extracted {len(all_pages)} pages from {data_dir}[/green]")
    return all_pages


# --- Step 2: Chunk text ---
def chunk_pages(pages: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    all_chunks = []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "filename": page["filename"],
                "page": page["page"],
                "chunk_index": i,
                "text": chunk
            })
    rprint(f"[green]Created {len(all_chunks)} chunks[/green]")
    return all_chunks


# --- Step 3: Embed chunks ---
def embed_chunks(chunks: list[dict]) -> np.ndarray:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )
    texts = [chunk["text"] for chunk in chunks]
    rprint(f"[bold cyan]Embedding {len(texts)} chunks with Gemini...[/bold cyan]")

    # Batch in groups of 50 to avoid rate limits
    all_vectors = []
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vectors = embeddings.embed_documents(batch)
        all_vectors.extend(vectors)
        rprint(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.array(all_vectors, dtype=np.float32)


# --- Step 4: Build and save FAISS index ---
def build_faiss_index(vectors: np.ndarray, chunks: list[dict]):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # Save index
    faiss.write_index(index, INDEX_PATH)
    rprint(f"[green]FAISS index saved to {INDEX_PATH}[/green]")

    # Save metadata (chunk text + source info)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)
    rprint(f"[green]Metadata saved to {METADATA_PATH}[/green]")


# --- Main pipeline ---
def ingest():
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    pages = extract_text_from_pdfs(DATA_DIR)
    if not pages:
        rprint("[red]No PDFs found in /data folder. Add PDFs and try again.[/red]")
        return

    chunks = chunk_pages(pages)
    vectors = embed_chunks(chunks)
    build_faiss_index(vectors, chunks)

    rprint("\n[bold green]Ingestion complete! FAISS index is ready.[/bold green]")
    rprint(f"  Total documents indexed: {len(chunks)}")


if __name__ == "__main__":
    ingest()