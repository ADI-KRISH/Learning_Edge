import os
import sys

# Crucial fixes for PyTorch on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

print("Importing torch in main thread to prevent TLS crashes...")
import torch
import sentence_transformers
print("Torch imported successfully.")

import streamlit.web.cli as stcli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app.py", "--server.headless", "true"]
    sys.exit(stcli.main())
