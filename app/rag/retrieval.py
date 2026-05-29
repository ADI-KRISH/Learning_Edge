import chromadb
import os
import traceback
from datetime import datetime

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

from app.utils.config import VECTOR_DB_DIR, EMBEDDING_MODEL, BASE_DIR

# Log file path for retrieval inspection
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
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        self.db = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    def get_retriever(self, subject_id: str = "default_subject", num_chunks=2, constraints=None):
        """Builds a Subject-Specific Hybrid Retriever combining Vector Search and BM25 using LangChain."""
        collection_name = f"subject_{subject_id}"
        
        # Connect to Chroma Vector Store
        vector_store = Chroma(
            client=self.db,
            collection_name=collection_name,
            embedding_function=self.embed_model
        )
        
        search_kwargs = {"k": num_chunks}
        if constraints and isinstance(constraints, dict) and len(constraints) > 0:
            if len(constraints) == 1:
                search_kwargs["filter"] = constraints
            else:
                search_kwargs["filter"] = {
                    "$and": [{k: v} for k, v in constraints.items()]
                }
        vector_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
        
        try:
            collection = self.db.get_or_create_collection(collection_name)
            chroma_data = collection.get()
            
            if chroma_data and "documents" in chroma_data and chroma_data["documents"]:
                documents = chroma_data["documents"]
                metadatas = chroma_data.get("metadatas", [])
                
                # Build BM25
                bm25_retriever = BM25Retriever.from_texts(texts=documents, metadatas=metadatas)
                bm25_retriever.k = num_chunks
                
                print(f"  [OK] BM25 initialized with {len(documents)} nodes in subject '{subject_id}'")
                
                # RRF Ensemble Retriever
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
                raise ValueError(f"No documents found in Chroma collection '{collection_name}'")
                
        except Exception as e:
            print(f"  [WARN] BM25 setup failed for subject '{subject_id}'. Vector search fallback. Error: {e}")
            self._retriever_mode = "vector_only"
            return vector_retriever

    def retrieve(self, query: str, subject_id: str = "default_subject", constraints=None):
        """Retrieves context chunks from subject collection and logs them with strict metadata filtering."""
        retriever = self.get_retriever(subject_id=subject_id, num_chunks=4, constraints=constraints)
        docs = retriever.invoke(query)
        
        mode = getattr(self, '_retriever_mode', 'unknown')
        print(f"  [INFO] Subject: {subject_id} | Retrieval Mode: {mode.upper()} | Raw Chunks: {len(docs)}")
        
        # Strict manual metadata filtering safeguard
        if constraints and isinstance(constraints, dict):
            target_doc_id = constraints.get("doc_id")
            target_part_index = constraints.get("part_index")
            filtered_docs = []
            for doc in docs:
                meta = doc.metadata or {}
                if target_doc_id and meta.get("doc_id") != target_doc_id:
                    continue
                if target_part_index is not None and meta.get("part_index") != target_part_index:
                    continue
                filtered_docs.append(doc)
            docs = filtered_docs
            print(f"  [FILTER] Strictly filtered chunks: {len(docs)} matching doc='{target_doc_id}', part={target_part_index}")
        
        # Deduplicate docs
        seen = set()
        unique_docs = []
        for d in docs:
            if d.page_content not in seen:
                seen.add(d.page_content)
                unique_docs.append(d)
                
        # Write to retrieval log
        try:
            with open(RETRIEVAL_LOG, "a", encoding="utf-8") as f:
                f.write("========================================================\n")
                f.write(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"SUBJECT: {subject_id}\n")
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
    """Singleton pattern to prevent loading embedding models multiple times."""
    global _retriever_instance
    if '_retriever_instance' not in globals():
        _retriever_instance = HybridRAGRetriever()
    return _retriever_instance
