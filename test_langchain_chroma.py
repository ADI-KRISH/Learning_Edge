import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.utils.config import VECTOR_DB_DIR, BASE_DIR
import os

def main():
    model_path = os.path.join(BASE_DIR, "models", "bge-small-en")
    embed_model = HuggingFaceEmbeddings(model_name=model_path)
    db = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    
    try:
        vector_store = Chroma(
            client=db,
            collection_name="course_materials",
            embedding_function=embed_model
        )
        print("Langchain Chroma init OK")
        docs = vector_store.similarity_search("Hadoop")
        print("Retrieved docs:", len(docs))
    except Exception as e:
        print("Error Langchain Chroma:", type(e), e)

if __name__ == "__main__":
    main()
