import chromadb
from chromadb.config import Settings
import os

from app.utils.config import VECTOR_DB_DIR

def main():
    try:
        db = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        print("Got client:", db)
        collection = db.get_collection("course_materials")
        print("Got collection by name:", collection.name)
        print("Documents count:", collection.count())
    except Exception as e:
        print("Error getting collection:", e)

if __name__ == "__main__":
    main()
