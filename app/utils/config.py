import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db_v2")
GRAPH_DIR = os.path.join(BASE_DIR, "app", "graph")
MEMORY_DIR = os.path.join(BASE_DIR, "app", "memory")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)

# LLM & Embedding Models
LLM_MODEL = "llama3.2" # Meta's Llama 3.2 3B (highly optimized for low RAM)
EMBEDDING_MODEL = "BAAI/bge-small-en" # HuggingFace embedding model for low RAM

# Chunking Settings (Semantic Chunking targets)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
