import chromadb
import os
import traceback
from datetime import datetime

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

from app.utils.config import VECTOR_DB_DIR, EMBEDDING_MODEL, BASE_DIR

# Log file path -- all retrieved text is saved here for inspection
RETRIEVAL_LOG = os.path.join(BASE_DIR, "retrieval_log.txt")

class HybridRAGRetriever:
    def __init__(self):
        # 1. Setup the Embedding Model
        model_path = os.path.join(BASE_DIR, "models", "bge-small-en")
        if os.path.exists(model_path):
            print(f"Loading local embedding model in HybridRAGRetriever (LangChain): {model_path}")
            self.embed_model = HuggingFaceEmbeddings(model_name=model_path)
        else:
            print(f"Local embedding path not found. Loading: {EMBEDDING_MODEL}")
            self.embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # 2. Connect to existing ChromaDB
        self.db = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        
        # 3. LangChain Vector Store
        self.vector_store = Chroma(
            client=self.db,
            collection_name="course_materials",
            embedding_function=self.embed_model
        )

    def get_retriever(self, num_chunks=2, constraints=None):
        """Builds a Hybrid Retriever combining Vector Search and BM25 using LangChain."""
        search_kwargs = {"k": num_chunks}
        if constraints and isinstance(constraints, dict) and len(constraints) > 0:
            search_kwargs["filter"] = constraints
        vector_retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
        
        try:
            collection = self.db.get_or_create_collection("course_materials")
            chroma_data = collection.get()
            
            if chroma_data and "documents" in chroma_data and chroma_data["documents"]:
                documents = chroma_data["documents"]
                metadatas = chroma_data.get("metadatas", [])
                
                # Build BM25
                bm25_retriever = BM25Retriever.from_texts(texts=documents, metadatas=metadatas)
                bm25_retriever.k = num_chunks
                
                print(f"  [OK] BM25 initialized with {len(documents)} nodes")
                
                # Combine using Manual Reciprocal Rank Fusion since Langchain versions conflict
                class ManualEnsembleRetriever:
                    def __init__(self, retrievers, weights=None, k=60):
                        self.retrievers = retrievers
                        self.weights = weights or [1.0] * len(retrievers)
                        self.k = k
                        
                    def invoke(self, query):
                        fused_scores = {}
                        for retriever, weight in zip(self.retrievers, self.weights):
                            docs = retriever.invoke(query)
                            for rank, doc in enumerate(docs):
                                if doc.page_content not in fused_scores:
                                    fused_scores[doc.page_content] = {"doc": doc, "score": 0.0}
                                fused_scores[doc.page_content]["score"] += weight / (rank + self.k)
                        
                        reranked = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
                        return [x["doc"] for x in reranked]
                
                ensemble_retriever = ManualEnsembleRetriever(
                    retrievers=[bm25_retriever, vector_retriever],
                    weights=[0.5, 0.5]
                )
                self._retriever_mode = "hybrid"
                return ensemble_retriever
            else:
                raise ValueError("No documents found in ChromaDB collection.")
                
        except Exception as e:
            print(f"  [WARN] BM25 setup failed. Full error below:")
            traceback.print_exc()
            self._retriever_mode = "vector_only"
            return vector_retriever

    def retrieve(self, query: str, constraints=None):
        """Retrieves relevant chunks and saves them to retrieval_log.txt for inspection."""
        retriever = self.get_retriever(num_chunks=2, constraints=constraints)
        docs = retriever.invoke(query)
        
        mode = getattr(self, '_retriever_mode', 'unknown')
        print(f"  [INFO] Retrieval Mode: {mode.upper()}")
        print(f"  [INFO] Chunks Retrieved: {len(docs)}")
        
        # Deduplicate docs (EnsembleRetriever might return duplicates depending on how it merges)
        seen = set()
        unique_docs = []
        for d in docs:
            if d.page_content not in seen:
                seen.add(d.page_content)
                unique_docs.append(d)
                
        # Write everything to a log file
        try:
            with open(RETRIEVAL_LOG, "a", encoding="utf-8") as f:
                f.write("========================================================\n")
                f.write(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"QUERY: {query}\n")
                f.write(f"MODE: {mode.upper()}\n")
                f.write("--------------------------------------------------------\n")
                for i, doc in enumerate(unique_docs):
                    f.write(f"--- CHUNK {i+1} ---\n")
                    f.write(f"Metadata: {doc.metadata}\n")
                    f.write(f"Content:\n{doc.page_content}\n")
                    f.write("--------------------------------------------------------\n")
                f.write("\n")
        except Exception as e:
            print(f"  [WARN] Could not write to retrieval log: {e}")
            
        return unique_docs

def get_retriever():
    """Singleton pattern to prevent loading models multiple times."""
    global _retriever_instance
    if '_retriever_instance' not in globals():
        _retriever_instance = HybridRAGRetriever()
    return _retriever_instance
