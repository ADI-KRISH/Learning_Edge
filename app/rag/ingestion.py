import os
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from app.utils.config import DATA_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL, BASE_DIR

class DocumentIngestionPipeline:
    def __init__(self):
        # Setup local path fallback for BAAI/bge-small-en
        model_path = os.path.join(BASE_DIR, "models", "bge-small-en")
        if os.path.exists(model_path):
            print(f"Loading local embedding model from: {model_path}")
            self.embed_model = HuggingFaceEmbedding(model_name=model_path, local_files_only=True)
        else:
            print(f"Local embedding model path not found. Loading model_name: {EMBEDDING_MODEL}")
            self.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
        Settings.embed_model = self.embed_model

        
        # Initialize ChromaDB
        print(f"Connecting to Local ChromaDB at {VECTOR_DB_DIR}")
        self.db = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        self.chroma_collection = self.db.get_or_create_collection("course_materials")
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # Use a robust SentenceSplitter optimized for academic/slide-based PDFs.
        # This keeps headers, bullet points, and descriptions grouped together.
        print("Initializing SentenceSplitter Node Parser...")
        from llama_index.core.node_parser import SentenceSplitter
        self.node_parser = SentenceSplitter(
            chunk_size=384,
            chunk_overlap=50
        )

    def ingest_documents(self):
        """Reads documents from DATA_DIR, parses, semantically chunks, and stores in VectorDB."""
        if not os.listdir(DATA_DIR):
            print(f"No documents found in {DATA_DIR}. Please add PDFs, DOCX, or TXT files.")
            return

        print(f"Reading documents from {DATA_DIR}...")
        
        # We explicitly use PyMuPDFReader for PDFs to prevent reading the raw binary PDF source code!
        try:
            from llama_index.readers.file import PyMuPDFReader
            file_extractor = {".pdf": PyMuPDFReader()}
        except ImportError:
            file_extractor = None
            
        documents = SimpleDirectoryReader(DATA_DIR, file_extractor=file_extractor).load_data()
        print(f"Loaded {len(documents)} raw document pages/sections.")

        print("Applying Sentence Chunking...")
        raw_nodes = self.node_parser.get_nodes_from_documents(documents)
        print(f"Generated {len(raw_nodes)} initial chunks.")
        
        # Filter out noise chunks (e.g. page numbers, decorative headers, slide footer credits)
        # Chunks less than 80 characters rarely contain useful tutoring context.
        nodes = [n for n in raw_nodes if len(n.get_content().strip()) >= 80]
        print(f"Retained {len(nodes)} high-quality context chunks (filtered {len(raw_nodes) - len(nodes)} short junk chunks).")

        print("Generating Embeddings and storing in ChromaDB...")
        # Create index from nodes which automatically embeds and stores in ChromaDB
        index = VectorStoreIndex(
            nodes, 
            storage_context=self.storage_context
        )
        print("Ingestion Complete! Vector Database updated.")
        
        # Dynamically build and update the Knowledge Graph from newly ingested documents!
        try:
            print("[KG Builder] Extracting concepts for Knowledge Graph...")
            from app.graph.knowledge_graph import EducationalKnowledgeGraph
            kg = EducationalKnowledgeGraph()
            
            unique_docs = {}
            for doc in documents:
                file_name = doc.metadata.get("file_name", "unknown")
                if file_name not in unique_docs:
                    unique_docs[file_name] = []
                unique_docs[file_name].append(doc.get_content())
                
            for file_name, contents in unique_docs.items():
                combined_sample = ""
                if len(contents) > 0:
                    combined_sample += contents[0][:1500] # Start page/section
                if len(contents) > 2:
                    combined_sample += "\n" + contents[len(contents)//2][:1500] # Middle page/section
                if len(contents) > 1:
                    combined_sample += "\n" + contents[-1][:1500] # Last page/section
                    
                print(f"[KG Builder] Extracting key topics from '{file_name}'...")
                kg.extract_from_text(combined_sample, file_name)
                
            # Regenerate the interactive PyVis visualization HTML
            kg.generate_pyvis_html()
            print("[KG Builder] Knowledge Graph and visualization updated successfully.")
        except Exception as e:
            print(f"[KG Builder] Dynamic extraction failed: {e}")
            
        return index

if __name__ == "__main__":
    # Test the ingestion pipeline
    pipeline = DocumentIngestionPipeline()
    pipeline.ingest_documents()
