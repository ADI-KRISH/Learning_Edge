import requests
import json
import sys

OLLAMA_URL = "http://localhost:11434/api"

print("Step 1: Deleting Phi-3...")
try:
    res = requests.delete(f"{OLLAMA_URL}/delete", json={"name": "phi3"})
    if res.status_code == 200:
        print("Successfully deleted Phi-3 to free up space.")
    else:
        print(f"Phi-3 not found or could not be deleted (Status {res.status_code}).")
except Exception as e:
    print(f"Error connecting to Ollama: {e}")

print("\nStep 2: Pulling Llama 3.2 3B (Quantized by default)...")
print("This may take a few minutes depending on your internet connection.")
try:
    res = requests.post(f"{OLLAMA_URL}/pull", json={"name": "llama3.2"}, stream=True)
    for line in res.iter_lines():
        if line:
            data = json.loads(line)
            if "status" in data:
                # print the status, overwrite the line if it's downloading
                sys.stdout.write(f"\rStatus: {data['status']}")
                sys.stdout.flush()
    print("\n\nSuccessfully pulled Llama 3.2 3B!")
except Exception as e:
    print(f"\nError pulling model: {e}")
