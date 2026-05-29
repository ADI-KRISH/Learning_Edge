import os
from llama_index.core import SimpleDirectoryReader
from app.graph.knowledge_graph import EducationalKnowledgeGraph
from app.utils.config import DATA_DIR

def rebuild_graph():
    print("====================================================")
    print("  REBUILDING KNOWLEDGE GRAPH WITH DYNAMIC LLM")
    print("====================================================")
    
    kg = EducationalKnowledgeGraph()
    
    # We clear the existing edges/nodes to start fresh and avoid double edges
    import networkx as nx
    kg.G = nx.DiGraph()
    
    print(f"Reading study materials from {DATA_DIR}...")
    
    # Use PyMuPDFReader to extract raw text safely
    try:
        from llama_index.readers.file import PyMuPDFReader
        file_extractor = {".pdf": PyMuPDFReader()}
    except Exception:
        file_extractor = None
        
    reader = SimpleDirectoryReader(DATA_DIR, file_extractor=file_extractor)
    documents = reader.load_data()
    print(f"Loaded {len(documents)} raw document pages/sections.")
    
    # Group content by file name
    doc_groups = {}
    for doc in documents:
        file_name = doc.metadata.get("file_name", "unknown")
        doc_groups.setdefault(file_name, []).append(doc.get_content())
        
    # Process each document
    for file_name, pages in doc_groups.items():
        print(f"\n--- Processing: {file_name} ({len(pages)} pages) ---")
        
        # Formulate a representative combined sample from the document (start, middle, and end)
        sample_text = ""
        if len(pages) > 0:
            sample_text += "Introduction / Overview:\n" + pages[0][:1200]
        if len(pages) > 2:
            sample_text += "\n\nCore Concepts:\n" + pages[len(pages)//2][:1200]
        if len(pages) > 1:
            sample_text += "\n\nAdvanced / Summary:\n" + pages[-1][:1200]
            
        print(f"Extracting academic topics and prerequisites using LLM fallback...")
        kg.extract_from_text(sample_text, file_name)
        
    # Regenerate pyvis visual graph HTML
    kg.generate_pyvis_html()
    print("\n====================================================")
    print(f"SUCCESS! Knowledge Graph rebuilt with {kg.G.number_of_nodes()} topics.")
    print("Visual interactive HTML graph saved to: graph.html")
    print("====================================================")

if __name__ == "__main__":
    rebuild_graph()
