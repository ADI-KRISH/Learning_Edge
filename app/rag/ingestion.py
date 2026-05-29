import os
import shutil
import chromadb
from tqdm import tqdm
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.utils.config import DATA_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL, BASE_DIR


class ContextualIngestor:
    def __init__(self):
        # Setup local path fallback for BAAI/bge-small-en
        model_path = os.path.join(BASE_DIR, "models", "bge-small-en")
        if os.path.exists(model_path):
            print(f"Loading local embedding model in ContextualIngestor: {model_path}")
            self.embed_model = HuggingFaceEmbeddings(model_name=model_path)
        else:
            print(f"Local embedding model path not found. Loading model_name: {EMBEDDING_MODEL}")
            self.embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # Connect to ChromaDB Persistent Client (No rmtree here so other subject collections are safe!)
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        self.db_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

        # Standard Recursive Character Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        self._llm = None

    def get_llm(self):
        """Lazy loader for Tutor LLM to save startup resources."""
        if self._llm is None:
            from app.agents.orchestrator import get_tutor_llm
            self._llm = get_tutor_llm()
        return self._llm

    def generate_context(self, document_text: str, chunk_text: str) -> str:
        """Stage 2: Generates a 1-2 sentence context wrapper using the local 3B model (Anthropic's proposed method)."""
        prompt = f"""You are an information architect. Review the Entire Document and the specific Chunk isolated from it. 
Generate a concise, 1-2 sentence summary providing the missing context (such as the main subject, chapter, overarching system architecture, or specific algorithms) needed to make this chunk self-contained. 
Output ONLY the 1-2 sentence context wrapper. No intro, no conversational filler.

Entire Document Summary:
{document_text[:3000]}... (truncated for length)

Isolated Chunk:
{chunk_text}

Context Wrapper:"""
        try:
            llm = self.get_llm()
            res = llm.complete(prompt)
            context = str(res).strip()
            # Clean up potential LLM prefixing
            if context.lower().startswith("context wrapper:"):
                context = context[len("context wrapper:"):].strip()
            if context.lower().startswith("here is the context"):
                context = context.split(":", 1)[-1].strip()
            return context
        except Exception as e:
            print(f"Context generation failed: {e}")
            return "General course material context."

    def ingest_subject_documents(self, subject_id: str):
        """Pipeline to read, contextually chunk, and embed documents for a specific subject."""
        subject_data_dir = os.path.join(DATA_DIR, subject_id)
        if not os.path.exists(subject_data_dir) or not os.listdir(subject_data_dir):
            print(f"No documents found in subject-specific folder: {subject_data_dir}. Please add PDFs or TXT files.")
            return

        print(f"Reading subject '{subject_id}' documents from {subject_data_dir}...")
        
        # Load PDF and TXT documents inside the subject directory
        raw_documents = []
        for file in os.listdir(subject_data_dir):
            file_path = os.path.join(subject_data_dir, file)
            if file.lower().endswith(".pdf"):
                try:
                    loader = PyMuPDFLoader(file_path)
                    raw_documents.extend(loader.load())
                except Exception as e:
                    print(f"  Failed to load PDF '{file}': {e}")
            elif file.lower().endswith(".txt"):
                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                    raw_documents.extend(loader.load())
                except Exception as e:
                    print(f"  Failed to load TXT '{file}': {e}")
                    
        if not raw_documents:
            print(f"No valid readable documents found in {subject_data_dir}.")
            return
            
        print(f"Loaded {len(raw_documents)} raw document pages/sections for subject '{subject_id}'.")
        
        # Group by source file to pass "Full Document" context accurately
        docs_by_source = {}
        for doc in raw_documents:
            src = doc.metadata.get("source", "unknown")
            if src not in docs_by_source:
                docs_by_source[src] = ""
            docs_by_source[src] += doc.page_content + "\n"

        enriched_documents = []
        
        print("Starting Contextual Retrieval Pipeline (Splitting & LLM Enrichment)...")
        for src, full_text in docs_by_source.items():
            file_name = os.path.basename(src)
            print(f"\nProcessing File: {file_name}")
            chunks = self.text_splitter.split_text(full_text)
            print(f" -> Generated {len(chunks)} chunks.")
            
            for i, chunk_text in enumerate(tqdm(chunks, desc="Generating Context")):
                if len(chunk_text.strip()) < 80:
                    continue
                    
                # Generate Context Wrapper
                context_wrapper = self.generate_context(full_text, chunk_text)
                
                # Prepend Context Wrapper
                enriched_text = f"[Context: {context_wrapper}]\n\n{chunk_text}"
                
                # Create LangChain Document
                meta = {"source": file_name, "chunk_index": i, "subject_id": subject_id}
                enriched_documents.append(Document(page_content=enriched_text, metadata=meta))

        print(f"\nCompleted enrichment. {len(enriched_documents)} context-aware chunks prepared.")
        
        # Connect to subject-specific collection in ChromaDB
        collection_name = f"subject_{subject_id}"
        vector_store = Chroma(
            client=self.db_client,
            collection_name=collection_name,
            embedding_function=self.embed_model
        )
        
        # Index in subject collection
        print(f"Generating Embeddings and storing in Chroma collection: '{collection_name}'...")
        if enriched_documents:
            batch_size = 50
            for i in range(0, len(enriched_documents), batch_size):
                batch = enriched_documents[i:i+batch_size]
                vector_store.add_documents(batch)
                
            print(f"Subject '{subject_id}' Vector Database Ingestion Complete!")
        
        # Unified Knowledge Graph Builder
        try:
            print(f"[KG Builder] Extracting concepts for unified Knowledge Graph under subject '{subject_id}'...")
            from app.graph.knowledge_graph import EducationalKnowledgeGraph
            kg = EducationalKnowledgeGraph()
            
            for file_path, contents in docs_by_source.items():
                file_name = os.path.basename(file_path)
                print(f"[KG Builder] Extracting key topics from '{file_name}'...")
                kg.extract_from_text(contents[:4000], file_name, subject_id=subject_id)
                
            kg.generate_pyvis_html()
            print("[KG Builder] Unified Knowledge Graph and visualization updated successfully.")
        except Exception as e:
            print(f"[KG Builder] Dynamic extraction failed: {e}")

if __name__ == "__main__":
    import traceback
    try:
        # Create a test mock subject and ingest
        subject = "test_subject"
        os.makedirs(os.path.join(DATA_DIR, subject), exist_ok=True)
        
        test_file = os.path.join(DATA_DIR, subject, "sample_physics.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("mitosis is a process of cell division where a single cell divides into two identical daughter cells.\n")
            f.write("This cell division process contains phases: Prophase, Metaphase, Anaphase, and Telophase.\n")
            f.write("Mitosis is crucial for biological growth and tissue repair in eukaryotic organisms.\n")
            
        pipeline = ContextualIngestor()
        pipeline.ingest_subject_documents(subject)
    except Exception as e:
        print("CRITICAL ERROR DURING SUBJECT INGESTION:")
        traceback.print_exc()
