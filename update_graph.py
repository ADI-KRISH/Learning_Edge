from app.graph.knowledge_graph import EducationalKnowledgeGraph

kg = EducationalKnowledgeGraph()
kg.extract_from_text('dummy text', 'Photosynthesis_Process.pdf')
kg.extract_from_text('dummy text', 'Neural_Networks_Basics.pdf')
kg.extract_from_text('dummy text', 'Quantum_Computing_Intro.pdf')
kg.generate_pyvis_html()
