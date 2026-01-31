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
# Options: "cerebras" or "local"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "cerebras").lower()

# --- API KEYS & URLS ---
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY") # Ensure this is in your .env
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:8080/completion")

# --- MICROSERVICES ---
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
EMBEDDING_API_URL = "http://localhost:8001/embed"

# --- LLM SETTINGS ---
# Note: For local, this ID is often ignored by the server, but we keep it for logging
LLM_MODEL_ID = "gpt-oss-120b" if LLM_PROVIDER == "cerebras" else "local-model"

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