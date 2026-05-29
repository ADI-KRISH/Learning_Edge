from app.rag.retrieval import get_retriever

def main():
    print("Testing LangChain Retriever...")
    retriever = get_retriever()
    print("Got retriever:", retriever)
    print("Testing retrieval with query: 'Hadoop data splitting'")
    docs = retriever.retrieve("Hadoop data splitting")
    print(f"Retrieved {len(docs)} documents.")

if __name__ == "__main__":
    main()
