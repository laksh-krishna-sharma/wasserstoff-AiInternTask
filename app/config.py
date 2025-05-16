import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# ========== API Keys ==========
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ========== Directory Configuration ==========
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
DB_DIR = DATA_DIR / "db"
STATIC_DIR = BASE_DIR / "static"

# Create directories if they don't exist
for dir_path in [DATA_DIR, UPLOAD_DIR, PROCESSED_DIR, DB_DIR, STATIC_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ========== Vector DB Configuration ==========
CHROMA_COLLECTION_NAME = "document_collection"

# ========== Document Processing ==========
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ========== Model Configuration ==========
LLM_MODEL = "llama-3.3-70b-versatile"  # Hosted on Groq
LLM_TEMPERATURE = 0.0
