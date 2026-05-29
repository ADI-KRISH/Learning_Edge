import os
import re
import uuid
import chromadb
from tqdm import tqdm
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from app.memory.user_memory import UserMemory
from app.rag.ingestion import ContextualIngestor
from app.utils.config import DATA_DIR, VECTOR_DB_DIR, BASE_DIR

class CurriculumDivider:
    def __init__(self):
        self.ingestor = ContextualIngestor()
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        self.db_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    def _generate_part_title(self, text_snippet: str, default_title: str) -> str:
        """Invokes the local LLM to generate a nice, premium title for a subtopic chunk."""
        prompt = f"""You are an academic course designer. Analyze this brief text excerpt and generate a highly descriptive, professional title for it (1-4 words).
Do NOT include any introduction, formatting, or quotes. Output ONLY the title.

Text Excerpt:
{text_snippet[:600]}

Descriptive Title:"""
        try:
            llm = self.ingestor.get_llm()
            res = llm.complete(prompt)
            title = str(res).strip().strip('"').strip("'").strip()
            # If the LLM rambles, fallback
            if len(title) > 40 or "\n" in title or not title:
                return default_title
            return title.title()
        except Exception:
            return default_title

    def divide_and_ingest_curriculum(self, subject_id: str, doc_title: str, text_content: str, file_name: str = ""):
        """Core pipeline to divide a document into parts, save to SQLite, context-enrich, and index in Chroma."""
        mem = UserMemory()
        
        # 1. Generate unique Document ID
        doc_id = "doc_" + str(uuid.uuid4())[:8]
        
        # 2. Divide Document into Parts/Subtopics
        print(f"\n[Divider] Slicing document '{doc_title}' into sequential parts...")
        lines = text_content.split("\n")
        headings = []
        
        # Look for explicit markdown headers or section tags
        for idx, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            # Regex match for Module/Chapter/Part
            is_header = line_stripped.startswith("#") or line_stripped.startswith("==") or any(
                re.match(rf"^\b{kw}\b\s+\d+", line_stripped, re.IGNORECASE) for kw in ["chapter", "module", "part", "unit", "section", "lecture"]
            )
            if is_header and len(line_stripped) < 80:
                headings.append((idx, line_stripped.lstrip("#=").strip()))

        raw_parts = []
        if len(headings) >= 3:
            print(f"  Found {len(headings)} heading indicators. Splitting structurally...")
            for i in range(len(headings)):
                start_idx = headings[i][0]
                end_idx = headings[i+1][0] if i + 1 < len(headings) else len(lines)
                part_content = "\n".join(lines[start_idx:end_idx]).strip()
                part_title = headings[i][1]
                
                # Filter out empty sections
                if len(part_content) > 50:
                    raw_parts.append({"title": part_title, "content": part_content})
        else:
            print("  No strong heading indicators found. Splitting semantically into 3 equal parts...")
            char_len = len(text_content)
            chunk_size = char_len // 3
            for i in range(3):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < 2 else char_len
                part_content = text_content[start:end].strip()
                
                # Generate a beautiful dynamic title using LLM
                default_t = f"Part {i+1}: {doc_title}"
                part_title = self._generate_part_title(part_content, default_t)
                
                if len(part_content) > 50:
                    raw_parts.append({"title": part_title, "content": part_content})

        total_parts = len(raw_parts)
        print(f"  Successfully divided into {total_parts} subtopic parts.")

        # 3. Persist Document & Parts in SQLite
        mem.add_curriculum_document(doc_id, subject_id, doc_title, text_content if not file_name else file_name, total_parts)
        
        # Create Chroma Collection for this subject
        collection_name = f"subject_{subject_id}"
        vector_store = Chroma(
            client=self.db_client,
            collection_name=collection_name,
            embedding_function=self.ingestor.embed_model
        )

        for index, part in enumerate(raw_parts):
            part_index = index + 1
            part_id = f"{doc_id}_part_{part_index}"
            part_title = part["title"]
            part_content = part["content"]
            
            # Save part in SQLite
            mem.add_document_part(part_id, doc_id, part_index, part_title, part_content)
            print(f"    Saved SQLite Part {part_index}: '{part_title}'")
            
            # 4. Contextual Chunking & Ingestion in ChromaDB (Anthropic's Contextual Retrieval strategy)
            print(f"    Indexing Part {part_index} contextually in Chroma...")
            part_chunks = self.ingestor.text_splitter.split_text(part_content)
            enriched_docs = []
            
            for chunk_idx, chunk_text in enumerate(tqdm(part_chunks, desc=f"Part {part_index} chunks")):
                if len(chunk_text.strip()) < 50:
                    continue
                # Generate context based on part content
                context = self.ingestor.generate_context(part_content, chunk_text)
                enriched_text = f"[Context: {context}]\n\n{chunk_text}"
                
                # Metadata contains subject_id, doc_id, and part_index!
                meta = {
                    "subject_id": subject_id,
                    "doc_id": doc_id,
                    "part_index": part_index,
                    "chunk_index": chunk_idx,
                    "part_title": part_title
                }
                enriched_docs.append(Document(page_content=enriched_text, metadata=meta))
                
            if enriched_docs:
                batch_size = 50
                for j in range(0, len(enriched_docs), batch_size):
                    batch = enriched_docs[j:j+batch_size]
                    vector_store.add_documents(batch)
            print(f"    Indexed Part {part_index} chunks successfully.")

        # 5. Initialize active study state for this subject in SQLite
        mem.update_study_state(subject_id, doc_id, 1, "studying")
        
        # 6. Extract dynamic prerequisite connections for unified Knowledge Graph
        try:
            print(f"[KG Builder] Registering concepts to unified Knowledge Graph...")
            from app.graph.knowledge_graph import EducationalKnowledgeGraph
            kg = EducationalKnowledgeGraph()
            
            for index, part in enumerate(raw_parts):
                part_index = index + 1
                part_title = part["title"]
                # Add node representing this part
                kg.add_topic(part_title, prerequisites=[], subject_id=subject_id)
                
                # If there is a previous part, add it as a prerequisite!
                if part_index > 1:
                    prev_title = raw_parts[index - 1]["title"]
                    kg.add_topic(part_title, prerequisites=[prev_title], subject_id=subject_id)
                    
            kg.generate_pyvis_html()
            print("[KG Builder] Unified Knowledge Graph updated with sequential syllabus dependencies.")
        except Exception as e:
            print(f"[KG Builder] Unified graph update failed: {e}")

        print(f"\n[OK] Document '{doc_title}' ingested sequentially for subject '{subject_id}'!")
        return {"doc_id": doc_id, "total_parts": total_parts}

    def divide_and_ingest_file(self, subject_id: str, file_path: str):
        """Helper to upload a curriculum file, read it, and trigger sequential ingestion."""
        file_name = os.path.basename(file_path)
        doc_title = file_name.rsplit(".", 1)[0].replace("_", " ").title()
        
        text_content = ""
        if file_name.lower().endswith(".pdf"):
            try:
                loader = PyMuPDFLoader(file_path)
                pages = loader.load()
                text_content = "\n".join(p.page_content for p in pages)
            except Exception as e:
                print(f"Failed to read PDF file: {e}")
                return None
        elif file_name.lower().endswith(".txt"):
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
                text_content = "\n".join(d.page_content for d in docs)
            except Exception as e:
                print(f"Failed to read TXT file: {e}")
                return None
                
        if not text_content.strip():
            print(f"File '{file_name}' contains no readable text.")
            return None
            
        return self.divide_and_ingest_curriculum(subject_id, doc_title, text_content, file_name)

if __name__ == "__main__":
    # Quick standalone test
    divider = CurriculumDivider()
    subject = "physics_test"
    
    # Create mock subject in SQLite
    mem = UserMemory()
    mem.add_subject(subject, "Advanced Physics")
    
    test_text = """
    # Chapter 1: Introduction to Mechanics
    Mechanics is the study of motion and forces. Sir Isaac Newton formulated the three laws of motion.
    First Law: inertia remains active unless forces act.
    Second Law: Force equals mass times acceleration (F=ma).
    Third Law: Action and reaction are equal and opposite.
    
    # Chapter 2: Electromagnetism Basics
    Electromagnetism concerns electric charges and magnetic fields. Faraday discovered electromagnetic induction.
    Maxwell consolidated electromagnetism into four core equations.
    
    # Chapter 3: Quantum Foundations
    Quantum physics explains microscopic particles. Planck introduced the quantum constant h.
    Einstein explained the photoelectric effect using photons.
    Schrodinger formulated the wave equation describing quantum probability densities.
    """
    
    res = divider.divide_and_ingest_curriculum(subject, "Physics Outline Syllabus", test_text)
    print("Ingestion Result:", res)
