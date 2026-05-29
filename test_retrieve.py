from app.rag.retrieval import HybridRAGRetriever

retriever = HybridRAGRetriever()
context = retriever.retrieve("image segmentation")
with open("retrieved_context.txt", "w", encoding="utf-8") as f:
    f.write(context)
print("Context saved to retrieved_context.txt")
