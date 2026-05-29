# Offline AI Tutor - Project Accomplishments

## Overview
We have successfully architected and implemented a fully offline, multi-agent educational platform that runs entirely within the constraints of an 8GB RAM environment. 

## Key Implementations

### 1. System Architecture & UI (`app.py`, `config.py`)
- Configured a modular, production-ready directory structure with centralized configuration (`app/utils/config.py`).
- Built an interactive, multi-tab `Streamlit` user interface featuring:
  - **Dashboard:** For viewing progress and memory stats.
  - **Learn:** A chat interface for the Tutor and Quiz agents.
  - **Graph Visualization:** For viewing the educational knowledge graph.
  - **Sidebar:** For seamless document ingestion.

### 2. RAG & Data Ingestion (`app/rag/ingestion.py`, `app/rag/retrieval.py`)
- Built an ingestion pipeline using `LlamaIndex` to parse PDFs, TXTs, and DOCXs.
- Integrated `ChromaDB` for local vector storage.
- Implemented a highly accurate **Hybrid RAG Retriever**, which combines standard Vector Search with BM25 Keyword Search and merges the results.

### 3. State & Memory Management (`app/memory/user_memory.py`, `app/graph/knowledge_graph.py`)
- Created a `SQLite`-backed memory system that persistently tracks chat history and long-term user profiles (strengths, weaknesses, learning style).
- Engineered a `NetworkX` knowledge graph that maps out educational concepts and tracks prerequisite dependencies.

### 4. Agentic Orchestrator (`app/agents/orchestrator.py`)
- Developed a robust `LangGraph` state machine that intelligently routes tasks.
- **Workflow execution:** Automatically pulls from `UserMemory` and `ChromaDB`, determines if the user is asking a question or requesting a quiz, and routes the context to the appropriate AI Agent (Tutor or Quiz).

### 5. Performance Optimization & Bug Fixes
- **Module Resolution:** Resolved local package import errors by standardizing the `__init__.py` structure across the repository.
- **Dependency Fixes:** Identified and installed missing dependencies (`torchvision`) required by internal components of the `transformers` library.
- **RAM Optimization (Crucial Fix):** Fixed a critical Out-of-Memory (OOM) error where Ollama attempted to allocate 50.0 GB of RAM for the `phi3` model's KV Cache. We bypassed `LlamaIndex` defaults and passed a strict `num_ctx=2048` parameter directly to the Ollama backend, forcing the model to comfortably fit inside the user's 8GB RAM limit.
