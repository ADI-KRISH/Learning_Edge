import sys
import os

# Set dummy threads for torch compatibility on windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from app.rag.retrieval import HybridRAGRetriever

def run_test():
    query = "What is the architecture used by that specific distributed system you have in your documents?"
    print(f"QUERY: {query}\n")
    
    retriever_sys = HybridRAGRetriever()
    retriever = retriever_sys.get_retriever(num_chunks=2)
    docs = retriever.invoke(query)
    
    for i, doc in enumerate(docs):
        print(f"--- CHUNK {i+1} ---")
        print(doc.page_content)
        print("-" * 40 + "\n")

if __name__ == "__main__":
    run_test()
