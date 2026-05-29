import os
import sys
from pathlib import Path

# Set working directory to the script's directory
script_dir = Path(__file__).parent.resolve()
os.chdir(script_dir)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers package not found. Please install it first in your environment.")
    sys.exit(1)

# 1. Download all-MiniLM-L6-v2
local_path_minilm = script_dir / "models" / "all-MiniLM-L6-v2"
os.makedirs(local_path_minilm.parent, exist_ok=True)

try:
    if not (local_path_minilm / "config.json").exists():
        print("Starting SentenceTransformer('all-MiniLM-L6-v2') model download...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"Saving model weights locally to: {local_path_minilm}")
        model.save(str(local_path_minilm))
    else:
        print("SentenceTransformer('all-MiniLM-L6-v2') already exists locally.")
    
    print("Verifying all-MiniLM-L6-v2 offline load...")
    test_minilm = SentenceTransformer(str(local_path_minilm), local_files_only=True)
    print("SUCCESS: all-MiniLM-L6-v2 verified and loaded successfully offline!")
except Exception as e:
    print(f"An error occurred during all-MiniLM-L6-v2 download: {e}")
    sys.exit(1)

# 2. Download BAAI/bge-small-en
local_path_bge = script_dir / "models" / "bge-small-en"
try:
    if not (local_path_bge / "config.json").exists():
        print("Starting SentenceTransformer('BAAI/bge-small-en') model download...")
        model_bge = SentenceTransformer("BAAI/bge-small-en")
        print(f"Saving model weights locally to: {local_path_bge}")
        model_bge.save(str(local_path_bge))
    else:
        print("SentenceTransformer('BAAI/bge-small-en') already exists locally.")
        
    print("Verifying BAAI/bge-small-en offline load...")
    test_bge = SentenceTransformer(str(local_path_bge), local_files_only=True)
    print("SUCCESS: BAAI/bge-small-en verified and loaded successfully offline!")
except Exception as e:
    print(f"An error occurred during BAAI/bge-small-en download: {e}")
    sys.exit(1)

print("\nAll models have been downloaded and cached locally!")
