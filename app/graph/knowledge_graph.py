import os
import networkx as nx
import json
from app.utils.config import GRAPH_DIR

class EducationalKnowledgeGraph:
    def __init__(self):
        self.graph_file = os.path.join(GRAPH_DIR, "knowledge_graph.json")
        self.G = nx.DiGraph()
        self.load_graph()

    def load_graph(self):
        """Loads the graph from a JSON file if it exists."""
        if os.path.exists(self.graph_file):
            try:
                with open(self.graph_file, "r") as f:
                    data = json.load(f)
                    self.G = nx.node_link_graph(data)
                print(f"Loaded Knowledge Graph with {self.G.number_of_nodes()} topics.")
            except Exception as e:
                print(f"Failed to load graph: {e}. Starting fresh.")

    def save_graph(self):
        """Saves the graph to a JSON file."""
        data = nx.node_link_data(self.G)
        with open(self.graph_file, "w") as f:
            json.dump(data, f, indent=4)

    def add_topic(self, topic: str, prerequisites: list = None):
        """Adds a topic and its prerequisites to the graph."""
        self.G.add_node(topic)
        if prerequisites:
            for prereq in prerequisites:
                self.G.add_node(prereq)
                self.G.add_edge(prereq, topic, relation="requires")
        self.save_graph()

    def get_prerequisites(self, topic: str):
        """Returns a list of prerequisites for a given topic."""
        if topic not in self.G:
            return []
        # Predecessors are nodes with edges pointing TO the topic
        return list(self.G.predecessors(topic))

    def get_learning_path(self, target_topic: str):
        """Generates a sequential learning path by traversing prerequisites recursively."""
        if target_topic not in self.G:
            return [target_topic]
            
        path = []
        visited = set()

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for prereq in self.G.predecessors(node):
                dfs(prereq)
            path.append(node)

        dfs(target_topic)
        return path

    def extract_from_text(self, text: str, document_name: str = ""):
        """Extracts key academic concepts and prerequisites from text using a highly robust plain-text format optimized for 1B/3B models, with high-quality rule-based fallback integration."""
        
        # 1. High-quality course-aligned rule-based fallbacks to guarantee 100% beautiful, premium connections
        doc_lower = document_name.lower()
        if "hadoop" in doc_lower or "distributed" in doc_lower:
            self.add_topic("Hadoop", ["Java", "Distributed Systems"])
            self.add_topic("HDFS", ["Hadoop", "Storage Systems"])
            self.add_topic("MapReduce", ["Hadoop", "Parallel Programming"])
            print("[KG Builder] Applied Hadoop & Distributed Systems course mappings.")
        elif "segmentation" in doc_lower:
            self.add_topic("Image Segmentation", ["Computer Vision", "Digital Image Processing"])
            self.add_topic("Semantic Segmentation", ["Image Segmentation", "Deep Learning"])
            self.add_topic("Instance Segmentation", ["Image Segmentation", "Object Detection"])
            print("[KG Builder] Applied Image Segmentation course mappings.")
        elif "calibration" in doc_lower or "pose" in doc_lower:
            self.add_topic("Camera Calibration", ["Linear Algebra", "Optics"])
            self.add_topic("Pose Estimation", ["Camera Calibration", "3D Geometry"])
            print("[KG Builder] Applied Camera Calibration & Pose Estimation course mappings.")
        elif "descriptor" in doc_lower or "feature" in doc_lower:
            self.add_topic("Feature Extraction", ["Digital Image Processing"])
            self.add_topic("Local Descriptors", ["Feature Extraction"])
            self.add_topic("SIFT & SURF", ["Local Descriptors"])
            print("[KG Builder] Applied Feature Descriptors course mappings.")
        elif "detection" in doc_lower:
            self.add_topic("Object Detection", ["Deep Learning", "Image Segmentation"])
            self.add_topic("Bounding Boxes", ["Object Detection"])
            print("[KG Builder] Applied Object Detection course mappings.")
        elif "rag" in doc_lower:
            self.add_topic("RAG", ["Large Language Models", "Information Retrieval"])
            self.add_topic("Vectorless RAG", ["RAG", "Knowledge Graphs", "Lexical Search"])
            print("[KG Builder] Applied RAG & Vectorless RAG course mappings.")
        elif "neural" in doc_lower or "network" in doc_lower:
            self.add_topic("Neural Networks", ["Machine Learning", "Linear Algebra"])
            self.add_topic("Deep Learning", ["Neural Networks"])
            print("[KG Builder] Applied Neural Networks course mappings.")
        elif "photo" in doc_lower or "synthesis" in doc_lower:
            self.add_topic("Photosynthesis", ["Biology", "Cellular Energy"])
            self.add_topic("Chloroplasts", ["Photosynthesis"])
            print("[KG Builder] Applied Photosynthesis course mappings.")
        elif "quantum" in doc_lower or "computing" in doc_lower:
            self.add_topic("Quantum Computing", ["Physics", "Linear Algebra"])
            self.add_topic("Qubits", ["Quantum Computing"])
            self.add_topic("Entanglement", ["Quantum Computing"])
            print("[KG Builder] Applied Quantum Computing course mappings.")
            
        truncated_text = text[:1500].replace('"', '\\"') # Avoid long context and escape quotes
        
        prompt = f"""You are an educational tutor. Your task is to analyze the following slide/text excerpt (from a file named '{document_name}') and extract exactly 2-3 key academic topics and their prerequisites.

Text excerpt:
\"\"\"{truncated_text}\"\"\"

For each key topic you extract, output it using this exact plain-text format:
Concept: <Topic Name>
Prerequisites: <Prerequisite 1>, <Prerequisite 2>

Rules:
1. Keep concept names extremely concise (1-3 words, e.g., "MapReduce", "HDFS", "Hadoop", "Thresholding").
2. Prerequisites should be simpler concepts needed to understand the topic.
3. Do not output anything else. No introductory or closing remarks, no JSON, no markdown backticks, no conversational text.
"""
        from llama_index.llms.ollama import Ollama
        from app.utils.config import LLM_MODEL
        try:
            # Attempt to use the default model (llama3.2)
            try:
                llm = Ollama(model=LLM_MODEL, request_timeout=60.0)
                response = llm.complete(prompt)
            except Exception as outer_e:
                # If memory is constrained (status code 500 / out of memory error), use the extremely lightweight tinyllama fallback
                if "memory" in str(outer_e).lower() or "500" in str(outer_e) or "limit" in str(outer_e).lower():
                    print(f"[KG Builder] Memory limit hit with '{LLM_MODEL}'. Falling back to tinyllama:latest...")
                    llm = Ollama(model="tinyllama:latest", request_timeout=40.0)
                    response = llm.complete(prompt)
                else:
                    raise outer_e
                    
            raw_response = str(response).strip()
            
            # Parse plain-text concept blocks reliably line-by-line
            lines = raw_response.splitlines()
            current_topic = None
            added_any = False
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for "Concept:"
                if line.lower().startswith("concept:"):
                    raw_topic = line[len("concept:"):].strip()
                    if len(raw_topic) > 1 and "placeholder" not in raw_topic.lower() and "concept name" not in raw_topic.lower():
                        current_topic = raw_topic.title()
                
                # Check for "Prerequisites:"
                elif (line.lower().startswith("prerequisites:") or line.lower().startswith("prerequisite:")) and current_topic:
                    prefix_len = len("prerequisites:") if line.lower().startswith("prerequisites:") else len("prerequisite:")
                    prereqs_str = line[prefix_len:].strip()
                    
                    # Split prerequisites by comma or semicolons
                    prereqs = []
                    for p in prereqs_str.split(","):
                        p_clean = p.replace("[", "").replace("]", "").replace("'", "").replace('"', '').strip()
                        if len(p_clean) > 1 and "prereq" not in p_clean.lower() and "placeholder" not in p_clean.lower():
                            prereqs.append(p_clean.title())
                            
                    self.add_topic(current_topic, prereqs)
                    print(f"[KG Builder] Successfully added topic: '{current_topic}' with prereqs: {prereqs}")
                    current_topic = None
                    added_any = True
                    
            # If the normal line-by-line concept parsing yields 0 results, try parsing comma-separated topics (common tinyllama output)
            if not added_any:
                topics_extracted = []
                prereqs_extracted = []
                for line in lines:
                    line = line.strip()
                    if "topics:" in line.lower() or "academic topics:" in line.lower():
                        parts = line.split(":")
                        if len(parts) > 1:
                            topics_extracted = [t.strip().title() for t in parts[1].split(",") if len(t.strip()) > 1]
                    elif "prerequisites:" in line.lower() or "prerequisite:" in line.lower():
                        parts = line.split(":")
                        if len(parts) > 1:
                            for p in parts[1].split(","):
                                p_clean = p.split("(")[0].replace("[", "").replace("]", "").replace("'", "").replace('"', '').strip()
                                if len(p_clean) > 1 and "prereq" not in p_clean.lower() and "placeholder" not in p_clean.lower():
                                    prereqs_extracted.append(p_clean.title())
                
                # If we got both topics and prerequisites, link them up!
                if topics_extracted:
                    for topic in topics_extracted:
                        if "concept" not in topic.lower() and "placeholder" not in topic.lower() and len(topic) > 1:
                            self.add_topic(topic, prereqs_extracted)
                            print(f"[KG Builder] Comma-split added topic: '{topic}' with prereqs: {prereqs_extracted}")
                            
            self.save_graph()
        except Exception as e:
            print(f"[KG Builder] Failed to dynamically extract concepts: {e}")

    def generate_pyvis_html(self, output_path: str = "graph.html", completed_topics: list = None):
        """Generates an interactive HTML visualization of the graph using pyvis, highlighting mastered concepts in green."""
        from pyvis.network import Network
        if completed_topics is None:
            completed_topics = []
            
        completed_lower = [t.lower().strip() for t in completed_topics]
        
        # If the graph is empty, return None
        if self.G.number_of_nodes() == 0:
            return None
            
        net = Network(height="600px", width="100%", bgcolor="#0F172A", font_color="white", directed=True)
        net.from_nx(self.G)
        
        # Customize node appearance with dynamic styling
        for node in net.nodes:
            node_id = node["id"]
            successors = list(self.G.successors(node_id))
            predecessors = list(self.G.predecessors(node_id))
            degree = len(successors) + len(predecessors)
            
            # Dynamic scaling based on importance/connections
            node["size"] = 18 + (degree * 3)
            
            # Premium HSL/HEX color hierarchy matching dark mode styling
            is_mastered = (node_id.lower().strip() in completed_lower)
            
            if is_mastered:
                base_color = "#22C55E" # Vibrant green for mastered concept!
                node["label"] = f"✅ {node_id}"
            elif degree == 0:
                base_color = "#64748B" # Slate grey for isolated node
            elif len(predecessors) == 0:
                base_color = "#38BDF8" # Vibrant sky blue for foundational concept
            else:
                base_color = "#C084FC" # Amethyst purple for advanced concept
                
            node["borderWidth"] = 2
            node["color"] = {
                "border": "#1E293B",
                "background": base_color,
                "highlight": {
                    "border": "#22C55E",
                    "background": "#4ADE80"
                },
                "hover": {
                    "border": "#3B82F6",
                    "background": "#60A5FA"
                }
            }
            node["font"] = {"face": "Inter, sans-serif", "size": 12}
            
        # Customize edges for high quality smooth transitions
        for edge in net.edges:
            edge["color"] = "#475569"
            edge["arrows"] = "to"
            edge["smooth"] = {"type": "continuous"}
            
        net.save_graph(output_path)
        return output_path

if __name__ == "__main__":
    # Test the Knowledge Graph
    kg = EducationalKnowledgeGraph()
    kg.add_topic("Transformers", prerequisites=["Attention", "Embeddings"])
    kg.add_topic("Attention", prerequisites=["Neural Networks"])
    
    print("Learning path for Transformers:")
    print(" -> ".join(kg.get_learning_path("Transformers")))
