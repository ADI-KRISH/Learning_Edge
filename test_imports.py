try:
    from langchain.retrievers.ensemble import EnsembleRetriever
    print("EnsembleRetriever OK from langchain.retrievers.ensemble")
except Exception as e:
    print(f"Error 1: {e}")

try:
    from langchain.retrievers import EnsembleRetriever
    print("EnsembleRetriever OK from langchain.retrievers")
except Exception as e:
    print(f"Error 2: {e}")

try:
    from langchain_community.retrievers import BM25Retriever
    print("BM25Retriever OK")
except Exception as e:
    print(f"Error 3: {e}")
