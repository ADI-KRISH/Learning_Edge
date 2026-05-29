import chromadb
from app.utils.config import VECTOR_DB_DIR
from app.rag.ingestion import DocumentIngestionPipeline

print("Connecting to ChromaDB to clear corrupted data...")
db = chromadb.PersistentClient(path=VECTOR_DB_DIR)

try:
    db.delete_collection("course_materials")
    print("Successfully deleted old corrupted collection.")
except Exception as e:
    print(f"Collection did not exist or could not be deleted: {e}")

print("\nStarting fresh ingestion with PyMuPDFReader...")
pipeline = DocumentIngestionPipeline()
pipeline.ingest_documents()
print("\nAll done! Clean data is now in the database.")
