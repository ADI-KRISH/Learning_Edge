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

    def add_topic(self, topic: str, prerequisites: list = None, subject_id: str = "default_subject"):
        """Adds a topic and its prerequisites to the graph, tagged by subject_id."""
        topic = topic.strip()
        self.G.add_node(topic, subject_id=subject_id)
        if prerequisites:
            for prereq in prerequisites:
                prereq = prereq.strip()
                # If prereq is not in the graph, add it under the same subject
                if prereq not in self.G:
                    self.G.add_node(prereq, subject_id=subject_id)
                self.G.add_edge(prereq, topic, relation="requires")
        self.save_graph()

    def get_prerequisites(self, topic: str):
        """Returns a list of prerequisites for a given topic."""
        if topic not in self.G:
            return []
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

    def extract_from_text(self, text: str, document_name: str = "", subject_id: str = "default_subject"):
        """Extracts key academic concepts and prerequisites from text using LLM or rule-based fallback, tagging by subject."""
        
        # 1. High-quality course-aligned rule-based fallbacks to guarantee instant nice connections
        doc_lower = document_name.lower()
        if "hadoop" in doc_lower or "distributed" in doc_lower:
            self.add_topic("Hadoop", ["Java", "Distributed Systems"], subject_id=subject_id)
            self.add_topic("HDFS", ["Hadoop", "Storage Systems"], subject_id=subject_id)
            self.add_topic("MapReduce", ["Hadoop", "Parallel Programming"], subject_id=subject_id)
            print(f"[KG Builder] Applied Hadoop course mappings for subject '{subject_id}'")
        elif "segmentation" in doc_lower:
            self.add_topic("Image Segmentation", ["Computer Vision", "Digital Image Processing"], subject_id=subject_id)
            self.add_topic("Semantic Segmentation", ["Image Segmentation", "Deep Learning"], subject_id=subject_id)
            self.add_topic("Instance Segmentation", ["Image Segmentation", "Object Detection"], subject_id=subject_id)
            print(f"[KG Builder] Applied Image Segmentation course mappings for subject '{subject_id}'")
        elif "calibration" in doc_lower or "pose" in doc_lower:
            self.add_topic("Camera Calibration", ["Linear Algebra", "Optics"], subject_id=subject_id)
            self.add_topic("Pose Estimation", ["Camera Calibration", "3D Geometry"], subject_id=subject_id)
            print(f"[KG Builder] Applied Camera Calibration course mappings for subject '{subject_id}'")
        elif "descriptor" in doc_lower or "feature" in doc_lower:
            self.add_topic("Feature Extraction", ["Digital Image Processing"], subject_id=subject_id)
            self.add_topic("Local Descriptors", ["Feature Extraction"], subject_id=subject_id)
            self.add_topic("SIFT & SURF", ["Local Descriptors"], subject_id=subject_id)
            print(f"[KG Builder] Applied Feature Descriptors course mappings for subject '{subject_id}'")
        elif "detection" in doc_lower:
            self.add_topic("Object Detection", ["Deep Learning", "Image Segmentation"], subject_id=subject_id)
            self.add_topic("Bounding Boxes", ["Object Detection"], subject_id=subject_id)
            print(f"[KG Builder] Applied Object Detection course mappings for subject '{subject_id}'")
        elif "rag" in doc_lower:
            self.add_topic("RAG", ["Large Language Models", "Information Retrieval"], subject_id=subject_id)
            self.add_topic("Vectorless RAG", ["RAG", "Knowledge Graphs", "Lexical Search"], subject_id=subject_id)
            print(f"[KG Builder] Applied RAG course mappings for subject '{subject_id}'")
        elif "neural" in doc_lower or "network" in doc_lower:
            self.add_topic("Neural Networks", ["Machine Learning", "Linear Algebra"], subject_id=subject_id)
            self.add_topic("Deep Learning", ["Neural Networks"], subject_id=subject_id)
            print(f"[KG Builder] Applied Neural Networks course mappings for subject '{subject_id}'")
        elif "photo" in doc_lower or "synthesis" in doc_lower:
            self.add_topic("Photosynthesis", ["Biology", "Cellular Energy"], subject_id=subject_id)
            self.add_topic("Chloroplasts", ["Photosynthesis"], subject_id=subject_id)
            print(f"[KG Builder] Applied Photosynthesis course mappings for subject '{subject_id}'")
        elif "quantum" in doc_lower or "computing" in doc_lower:
            self.add_topic("Quantum Computing", ["Physics", "Linear Algebra"], subject_id=subject_id)
            self.add_topic("Qubits", ["Quantum Computing"], subject_id=subject_id)
            self.add_topic("Entanglement", ["Quantum Computing"], subject_id=subject_id)
            print(f"[KG Builder] Applied Quantum Computing course mappings for subject '{subject_id}'")
            
        truncated_text = text[:1500].replace('"', '\\"')
        
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
            try:
                llm = Ollama(model=LLM_MODEL, request_timeout=60.0)
                response = llm.complete(prompt)
            except Exception as outer_e:
                # If memory is constrained, fallback to tinyllama
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
                    
                    prereqs = []
                    for p in prereqs_str.split(","):
                        p_clean = p.replace("[", "").replace("]", "").replace("'", "").replace('"', '').strip()
                        if len(p_clean) > 1 and "prereq" not in p_clean.lower() and "placeholder" not in p_clean.lower():
                            prereqs.append(p_clean.title())
                            
                    self.add_topic(current_topic, prereqs, subject_id=subject_id)
                    print(f"[KG Builder] Successfully added topic: '{current_topic}' with prereqs: {prereqs}")
                    current_topic = None
                    added_any = True
                    
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
                
                if topics_extracted:
                    for topic in topics_extracted:
                        if "concept" not in topic.lower() and "placeholder" not in topic.lower() and len(topic) > 1:
                            self.add_topic(topic, prereqs_extracted, subject_id=subject_id)
                            print(f"[KG Builder] Comma-split added topic: '{topic}' with prereqs: {prereqs_extracted}")
                            
            self.save_graph()
        except Exception as e:
            print(f"[KG Builder] Failed to dynamically extract concepts: {e}")

    def generate_pyvis_html(self, output_path: str = "graph.html", completed_topics: list = None, weak_topics: list = None):
        """Generates an interactive HTML visualization of the graph using pyvis, color-coding by subject and highlighting mastered/weak topics."""
        from pyvis.network import Network
        if completed_topics is None:
            completed_topics = []
        if weak_topics is None:
            weak_topics = []
            
        completed_lower = [t.lower().strip() for t in completed_topics]
        weak_lower = [t.lower().strip() for t in weak_topics]
        
        if self.G.number_of_nodes() == 0:
            return None
            
        net = Network(height="600px", width="100%", bgcolor="#0F172A", font_color="white", directed=True)
        net.from_nx(self.G)
        
        # 1. Establish Subject Color Coding Palette
        subjects_in_graph = set(nx.get_node_attributes(self.G, 'subject_id').values())
        
        # High fidelity pastel colors for subjects in dark mode
        palette = ["#C084FC", "#38BDF8", "#F472B6", "#FB923C", "#2DD4BF", "#A7F3D0", "#FDE047"]
        subject_colors = {"default_subject": "#64748B"}
        
        for i, sub in enumerate(sorted(list(subjects_in_graph))):
            if sub != "default_subject" and sub not in subject_colors:
                subject_colors[sub] = palette[len(subject_colors) % len(palette)]
        
        # 2. Customize Node Appearances
        for node in net.nodes:
            node_id = node["id"]
            successors = list(self.G.successors(node_id))
            predecessors = list(self.G.predecessors(node_id))
            degree = len(successors) + len(predecessors)
            
            node["size"] = 18 + (degree * 3)
            
            # Retrieve node subject_id
            node_subject_id = self.G.nodes[node_id].get("subject_id", "default_subject")
            
            is_mastered = (node_id.lower().strip() in completed_lower)
            is_weak = (node_id.lower().strip() in weak_lower)
            
            if is_mastered:
                base_color = "#22C55E" # Mastered
            elif is_weak:
                base_color = "#EAB308" # Focus area
            else:
                # Color node according to its subject!
                base_color = subject_colors.get(node_subject_id, "#64748B")

            # Strip chapter/module/part prefix case-insensitively and keep only the clean topic name
            import re
            clean_label = node_id
            clean_label = re.sub(r'^(?:chapter|module|part|unit|section|lecture)\s*\d+[\s:.-]*', '', clean_label, flags=re.IGNORECASE).strip()
            
            # Truncate extremely long topic labels to keep graph clean
            if len(clean_label) > 40:
                clean_label = clean_label[:37].strip() + "..."
                
            node["label"] = clean_label
                
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
            
        for edge in net.edges:
            edge["color"] = "#475569"
            edge["arrows"] = "to"
            edge["smooth"] = {"type": "continuous"}
            
        net.save_graph(output_path)
        
        # 3. Inject a Premium Floating HTML Legend into the generated HTML!
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                    
                legend_html = """
                <div id="graph-legend" style="position: absolute; top: 15px; left: 15px; background: rgba(15, 23, 42, 0.85); border: 1px solid #1E293B; border-radius: 8px; padding: 12px; font-family: 'Inter', sans-serif; color: white; pointer-events: auto; z-index: 9999; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); backdrop-filter: blur(4px);">
                    <h4 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: #E2E8F0; border-bottom: 1px solid #334155; padding-bottom: 4px;">Concept Legend</h4>
                    <div style="display: flex; flex-direction: column; gap: 6px;">
                        <div style="display: flex; align-items: center; gap: 8px; font-size: 12px;">
                            <span style="display: inline-block; width: 12px; height: 12px; background: #22C55E; border-radius: 3px;"></span>
                            <span>Completed Concept</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 8px; font-size: 12px;">
                            <span style="display: inline-block; width: 12px; height: 12px; background: #EAB308; border-radius: 3px;"></span>
                            <span>Focus Area (Weak)</span>
                        </div>
                """
                
                for sub, color in subject_colors.items():
                    sub_name = "General Knowledge" if sub == "default_subject" else sub.replace("_", " ").title()
                    legend_html += f"""
                        <div style="display: flex; align-items: center; gap: 8px; font-size: 12px;">
                            <span style="display: inline-block; width: 12px; height: 12px; background: {color}; border-radius: 3px;"></span>
                            <span>{sub_name}</span>
                        </div>
                    """
                
                legend_html += """
                    </div>
                </div>
                """
                
                # Insert the legend floating div right inside the body tag
                modified_html = html_content.replace("<body>", f"<body>\n{legend_html}")
                
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(modified_html)
                    
            except Exception as err:
                print(f"[WARN] Failed to inject legend into graph HTML: {err}")
                
        return output_path

if __name__ == "__main__":
    kg = EducationalKnowledgeGraph()
    kg.add_topic("Transformers", prerequisites=["Attention"], subject_id="nlp_101")
    kg.add_topic("Attention", prerequisites=["Neural Networks"], subject_id="nlp_101")
    kg.add_topic("Calculus", prerequisites=["Algebra"], subject_id="math_101")
    
    html = kg.generate_pyvis_html("graph_test.html", completed_topics=["Attention"])
    print(f"Generated unified graph with legend at: {html}")
