# src/shared/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
os.makedirs(DATA_DIR, exist_ok=True)
SQL_DB_PATH = DATA_DIR / "record_manager_cache.sql"
SQL_CONNECTION_STRING = f"sqlite:///{SQL_DB_PATH}"

# --- LLM SWITCH ---
# Options: "cerebras", "groq", "gemini", or "local"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "cerebras").lower()

# --- API KEYS & URLS ---
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 

LOCAL_LLM_URL = os.getenv(
    "LOCAL_LLM_URL",
    "http://127.0.0.1:11434/v1/chat/completions"
)


# --- MICROSERVICES ---
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
EMBEDDING_API_URL = "http://localhost:8001/embed"
LOCAL_EMBEDDING_API_URL = "http://localhost:8001/v1/embeddings"

# --- EMBEDDING SETTINGS ---
# Options: "llama_local" (for llama-server) or "custom" (for python embedding_server.py)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "llama_local").lower()

# Set default URL based on provider, but allow .env override
default_url = LOCAL_EMBEDDING_API_URL if EMBEDDING_PROVIDER == "llama_local" else EMBEDDING_API_URL
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", default_url)

# --- LLM SETTINGS ---
# Default model IDs for each provider
DEFAULT_MODELS = {
    "cerebras": "gpt-oss-120b",
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-2.5-flash", 
    "local": "local-model"
}

# Get model from env or use default based on provider
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", DEFAULT_MODELS.get(LLM_PROVIDER, "gpt-oss-120b"))

# --- VECTOR DB CONFIGURATION ---
COLLECTION_WIKI = "medical_knowledge"
COLLECTION_PATIENTS = "patient_records"
COLLECTION_METADATA = {
    "hnsw:space": "cosine",       
    "hnsw:construction_ef": 200,   
    "hnsw:M": 32,                  
}
INDEXING_BATCH_SIZE = 500
INDEXING_CLEANUP = "incremental"

# --- Postgres Connection String ---
user = os.getenv("POSTGRES_USER")
paswd = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB_URL = os.getenv("POSTGRES_DB_URL", f"postgresql://{user}:{paswd}@localhost:5432/mediflow")

# --- Lightweight Turso ---
TURSO_DB_URL = os.getenv("TURSO_DB_URL", "sqlite+libsql://127.0.0.1:8008")
