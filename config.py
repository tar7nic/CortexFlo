from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

LLM_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "models/gemini-embedding-001"
MAX_TOKENS = 500
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RETRIEVAL = 4