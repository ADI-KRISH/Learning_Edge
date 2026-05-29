# Offline AI Tutor - Project Requirements

## Core Objective
Develop an 8GB RAM-optimized, locally hosted, agentic AI learning system. The system must operate completely offline to ensure data privacy and avoid external API costs, while providing a personalized, interactive tutoring experience.

## Technical Requirements

### 1. Environment & Constraints
- **Offline-First:** All model inferences, database operations, and data processing must happen locally on the machine.
- **Resource Limits:** Must operate smoothly within an 8GB RAM constraint. This requires careful selection and configuration of local LLMs (e.g., quantization, strict context window limits) and embedding models.
- **Package Management:** Use `uv` for fast, reproducible Python environment management.

### 2. Architecture & Orchestration
- **Agentic Framework:** Utilize `LangGraph` to build a stateful, multi-agent orchestration pipeline.
- **Agent Roles:** 
  - **Tutor Agent:** For explaining concepts and answering questions.
  - **Quiz Agent:** For generating assessments based on user understanding.
- **Workflow:** The orchestrator must sequentially handle Memory Check -> Graph Analysis -> Retrieval -> Agent Generation.

### 3. Data Processing (RAG Pipeline)
- **Document Ingestion:** Support uploading and parsing of academic documents (PDFs, TXT, DOCX).
- **Chunking:** Implement semantic chunking using `LlamaIndex`.
- **Vector Database:** Use `ChromaDB` for local embedding storage.
- **Retrieval:** Implement a Hybrid Retrieval system combining Vector Search (semantic meaning) and BM25 (keyword matching) with Reciprocal Rank Fusion for high accuracy.

### 4. Memory & Knowledge Tracking
- **Persistent Memory:** Use a local `SQLite` database to track short-term conversational context and long-term user profiles (strengths, weaknesses, completed topics).
- **Knowledge Graph:** Maintain a `NetworkX` graph to track topic prerequisites, relationships, and the user's learning path.

### 5. User Interface
- **Framework:** Build the frontend using `Streamlit`.
- **Features:**
  - A main Dashboard for progress tracking.
  - A "Learn" interface for interacting with the Tutor and Quiz agents.
  - A sidebar for easy document uploading and processing.
  - A Graph Visualization tab to view the knowledge structure.
