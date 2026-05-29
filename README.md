# Learning Edge: Offline Agentic AI Tutor

An intelligent, fully offline AI tutor application built with **LangGraph**, **Streamlit**, and **Ollama**. The system is designed to provide hyper-personalized academic tutoring without ever sending your data to the cloud.

By leveraging an embedded vector database (**ChromaDB**) and a structured local Knowledge Graph (**NetworkX**), the tutor can ingest your course materials, textbooks, or syllabus documents to provide accurate, hallucination-free explanations.

## Core Features

- **Agentic Routing System (LangGraph):** The system uses a multi-agent architecture (Supervisor, Researcher, Pedagogue, Assessor, Scribe) to dynamically route user intents between conversational chat, factual research, and interactive assessments.
- **Hybrid RAG Pipeline:** Documents are embedded locally using `all-MiniLM-L6-v2` and stored in ChromaDB. The system fuses semantic vector search with keyword-based sparse retrieval (BM25) to find exact facts from your uploaded course materials.
- **Dynamic Personalization Matrix:** The tutor dynamically reshapes its persona across 21 combinations (3 Academic Levels x 7 Teaching Styles). It explicitly tracks your "Mastered Topics" to generate real-world analogies tailored to concepts you already know.
- **Knowledge Graph Memory:** Instead of linear chat histories, the system tracks conversation branches via a directed graph (NetworkX). It automatically extracts concepts, generates quizzes, and records your passing scores to build a long-term semantic profile.
- **100% Offline & Private:** Uses `Llama-3.2-3B` via Ollama for all generation tasks, ensuring complete data privacy.

## Architecture Highlights

1. **`app/agents/orchestrator.py`**: The brain of the tutor. Contains the LangGraph nodes that manage state, route intents (Chat vs. Research vs. Quiz), and inject dynamic style prompts.
2. **`app/rag/`**: Contains the retrieval and ingestion pipelines. Uses Langchain Ensemble retrievers to merge BM25 and Vector search.
3. **`app/memory/`**: SQLite and NetworkX hybrid storage that tracks both short-term session state and long-term user profiles (weak topics, completed quizzes).
4. **`app/ui/`**: A modern Streamlit interface featuring a persistent sidebar for persona switching, chat history rendering, and debug views.

## Installation & Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- Pull the necessary model: `ollama run llama3.2`

### Quick Start
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```
4. Upload your course materials (PDFs, Markdown) in the UI or place them in the `data/` directory.

## License
MIT License
