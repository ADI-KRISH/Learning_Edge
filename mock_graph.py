from app.graph.knowledge_graph import EducationalKnowledgeGraph

kg = EducationalKnowledgeGraph()

# Add concepts related to Image Segmentation
kg.add_topic("Image Segmentation", prerequisites=["Computer Vision", "Digital Image Processing"])
kg.add_topic("Digital Image Processing", prerequisites=["Linear Algebra", "Signal Processing"])
kg.add_topic("Semantic Segmentation", prerequisites=["Image Segmentation", "Deep Learning"])
kg.add_topic("Instance Segmentation", prerequisites=["Image Segmentation", "Object Detection"])

print("Successfully injected mock concepts into the Knowledge Graph!")
