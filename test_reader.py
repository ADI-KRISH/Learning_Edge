from llama_index.core import SimpleDirectoryReader
from app.utils.config import DATA_DIR

docs = SimpleDirectoryReader(DATA_DIR).load_data()
print("Loaded docs:", len(docs))
if len(docs) > 0:
    print("Sample content:")
    print(docs[0].text[:500])
