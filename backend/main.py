import os
# Critical: set these BEFORE any other imports to prevent PyTorch Windows crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import json
import asyncio
import datetime
import re
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Pre-import torch in the main thread BEFORE any Streamlit-style threading
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from app.agents.orchestrator import stream_tutor_pipeline, generate_state_graph_html, OllamaConnectionError
from app.memory.user_memory import UserMemory
from app.utils.ollama_helper import is_ollama_running, ensure_ollama_running
from app.utils.config import DATA_DIR, VECTOR_DB_DIR
import chromadb

# ─── Singleton ChromaDB client ───────────────────────────────────────────────
_chroma_client = None
def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    return _chroma_client

# ─── Lifespan: warm up models at startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Warming up AI models...")
    # Ensure Ollama is running
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, ensure_ollama_running, 15)
    print("[Startup] Ready!")
    yield
    print("[Shutdown] Cleaning up.")

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Offline AI Tutor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

USER_ID = "default_user"  # Single-user local app

# ─── Pydantic models ──────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_session"

class SessionCreate(BaseModel):
    title: str = "New Chat Session"

class SessionRename(BaseModel):
    title: str

class ProfileUpdate(BaseModel):
    preferred_style: Optional[str] = None
    academic_level: Optional[str] = None
    quiz_difficulty: Optional[str] = None
    quiz_questions: Optional[int] = None

class QuizSubmit(BaseModel):
    session_id: str
    topic: str
    questions: list
    user_answers: list

# ─── Helpers ──────────────────────────────────────────────────────────────────
def repair_json_array(json_str: str) -> str:
    json_str = json_str.strip()
    if not json_str.startswith('['):
        return json_str
    if json_str.endswith(','):
        json_str = json_str[:-1].strip()
    in_string = False
    escaped = False
    for char in json_str:
        if char == '\\' and not escaped:
            escaped = True
        else:
            if char == '"' and not escaped:
                in_string = not in_string
            escaped = False
    if in_string:
        json_str += '"'
    open_brackets = 0
    open_braces = 0
    escaped = False
    temp_in_string = False
    for char in json_str:
        if char == '\\' and not escaped:
            escaped = True
            continue
        if char == '"' and not escaped:
            temp_in_string = not temp_in_string
        if not temp_in_string:
            if char == '[':
                open_brackets += 1
            elif char == ']':
                open_brackets -= 1
            elif char == '{':
                open_braces += 1
            elif char == '}':
                open_braces -= 1
        escaped = False
    if open_braces > 0:
        json_str += '}' * open_braces
    if open_brackets > 0:
        json_str += ']' * open_brackets
    return json_str

# ─── Session Routes ───────────────────────────────────────────────────────────
@app.get("/api/sessions")
def list_sessions():
    mem = UserMemory(USER_ID)
    return mem.get_chat_sessions()

@app.post("/api/sessions")
def create_session(body: SessionCreate):
    mem = UserMemory(USER_ID)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    session_id = f"session_{ts}"
    mem.create_chat_session(session_id, body.title)
    return {"session_id": session_id, "title": body.title}

@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id == "default_session":
        raise HTTPException(status_code=400, detail="Cannot delete the default session.")
    mem = UserMemory(USER_ID)
    mem.delete_chat_session(session_id)
    return {"ok": True}

@app.put("/api/sessions/{session_id}/title")
def rename_session(session_id: str, body: SessionRename):
    mem = UserMemory(USER_ID)
    mem.update_session_title(session_id, body.title)
    return {"ok": True}

@app.get("/api/sessions/{session_id}/history")
def get_history(session_id: str):
    mem = UserMemory(USER_ID)
    return mem.get_chat_history(session_id, limit=50)

@app.get("/api/sessions/{session_id}/graph_tree")
def get_graph_tree(session_id: str):
    """Return the non-linear Git-style memory tree as a list of nodes/edges."""
    mem = UserMemory(USER_ID)
    try:
        graph, head, root = mem.load_graph_memory(session_id)
        nodes = []
        for node_id, data in graph.nodes(data=True):
            nodes.append({
                "id": node_id,
                "label": data.get("topic_label", "Untitled"),
                "is_head": (node_id == head),
                "is_root": (node_id == root),
                "timestamp": data.get("timestamp", ""),
            })
        edges = [{"from": u, "to": v} for u, v in graph.edges()]
        return {"nodes": nodes, "edges": edges, "head": head, "root": root}
    except Exception as e:
        return {"nodes": [], "edges": [], "head": "", "root": "", "error": str(e)}

@app.get("/api/memory_graph/{session_id}")
def get_memory_graph_html(session_id: str):
    """Return the non-linear memory DiGraph as an interactive D3.js HTML visualization."""
    mem = UserMemory(USER_ID)
    try:
        from app.agents.orchestrator import generate_memory_graph_html
        graph, head, root = mem.load_graph_memory(session_id)
        html = generate_memory_graph_html(graph, head)
        return HTMLResponse(html)
    except Exception as e:
        return HTMLResponse(f"<html><body>Error generating graph: {e}</body></html>", status_code=500)

# ─── Chat / SSE Streaming Route ───────────────────────────────────────────────
@app.post("/api/chat")
async def chat(body: ChatRequest):
    """
    Streams the agentic pipeline as Server-Sent Events.
    Each event is a JSON object with keys: type, node, content.
    """
    async def event_generator():
        loop = asyncio.get_event_loop()
        query = body.query
        session_id = body.session_id

        # Auto-rename session title from first message
        mem = UserMemory(USER_ID)
        sessions = mem.get_chat_sessions()
        current_title = next((s["title"] for s in sessions if s["session_id"] == session_id), "")
        if current_title in ("Default Chat Session", "New Chat Session"):
            words = query.strip().split()
            clean_title = " ".join(words[:4])
            if len(words) > 4:
                clean_title += "..."
            clean_title = clean_title.strip('?.!,:;()')
            if clean_title:
                mem.update_session_title(session_id, clean_title)
                yield f"data: {json.dumps({'type': 'session_renamed', 'title': clean_title})}\n\n"

        # Persist user message
        mem.add_chat_message("user", query, session_id)

        try:
            # Run the blocking LangGraph pipeline in a thread pool to not block the event loop
            def run_pipeline():
                results = []
                for node_name, state in stream_tutor_pipeline(query, user_id=USER_ID, session_id=session_id):
                    results.append((node_name, state))
                return results

            yield f"data: {json.dumps({'type': 'pipeline_start'})}\n\n"

            results = await loop.run_in_executor(None, run_pipeline)

            tutor_response = None
            quiz_response = None
            active_topic = "General"

            for node_name, state in results:
                if isinstance(state, dict):
                    if state.get("active_topic"):
                        active_topic = state["active_topic"]
                    if state.get("tutor_response"):
                        tutor_response = state["tutor_response"]
                    if state.get("quiz_response"):
                        quiz_response = state["quiz_response"]

                yield f"data: {json.dumps({'type': 'node_complete', 'node': node_name})}\n\n"

            # Send tutor response
            if tutor_response:
                yield f"data: {json.dumps({'type': 'tutor_response', 'content': tutor_response})}\n\n"

            # Parse and send quiz if present
            if quiz_response:
                try:
                    clean = quiz_response.strip()
                    clean = re.sub(r'^```json\s*', '', clean)
                    clean = re.sub(r'^```\s*', '', clean)
                    clean = re.sub(r'\s*```$', '', clean)
                    start_idx = clean.find('[')
                    if start_idx != -1:
                        clean = clean[start_idx:]
                    clean = repair_json_array(clean)
                    questions = json.loads(clean)
                    if isinstance(questions, list) and len(questions) > 0:
                        yield f"data: {json.dumps({'type': 'quiz_ready', 'topic': active_topic, 'questions': questions})}\n\n"
                    else:
                        raise ValueError("Empty question list")
                except Exception:
                    fallback_content = "**Quiz:**\n" + quiz_response
                    yield f"data: {json.dumps({'type': 'tutor_response', 'content': fallback_content})}\n\n"

            yield f"data: {json.dumps({'type': 'pipeline_done', 'topic': active_topic})}\n\n"

        except OllamaConnectionError:
            yield f"data: {json.dumps({'type': 'error', 'code': 'ollama_offline', 'content': 'Ollama is offline. Please start the Ollama server.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'code': 'pipeline_error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ─── Profile Routes ───────────────────────────────────────────────────────────
@app.get("/api/profile")
def get_profile():
    mem = UserMemory(USER_ID)
    return mem.get_semantic_memory()

@app.put("/api/profile")
def update_profile(body: ProfileUpdate):
    mem = UserMemory(USER_ID)
    updates = {}
    if body.preferred_style is not None:
        updates["preferred_style"] = body.preferred_style
    if body.academic_level is not None:
        updates["academic_level"] = body.academic_level
    if body.quiz_difficulty is not None:
        updates["quiz_difficulty"] = body.quiz_difficulty
    if body.quiz_questions is not None:
        updates["quiz_questions"] = body.quiz_questions
        
    if updates:
        mem.update_semantic_memory(updates)
    return mem.get_semantic_memory()

# ─── Quiz Routes ──────────────────────────────────────────────────────────────
@app.get("/api/quiz/history")
def quiz_history():
    mem = UserMemory(USER_ID)
    return mem.get_quiz_history(limit=100)

@app.post("/api/quiz/submit")
def submit_quiz(body: QuizSubmit):
    questions = body.questions
    user_answers = body.user_answers
    topic = body.topic
    score = 0
    total = len(questions)

    for i, q in enumerate(questions):
        correct_idx = q.get("correct", 0)
        if not isinstance(correct_idx, int) or correct_idx < 0 or correct_idx >= len(q["options"]):
            correct_idx = 0
        if i < len(user_answers) and user_answers[i] == q["options"][correct_idx]:
            score += 1

    passed = score >= (total * 2 / 3)

    # Update semantic memory
    mem = UserMemory(USER_ID)
    profile = mem.get_semantic_memory()
    comp_list = profile.get("completed_topics", [])
    weak_list = profile.get("weak_topics", [])

    if passed:
        if topic not in comp_list:
            comp_list.append(topic)
        if topic in weak_list:
            weak_list.remove(topic)
    else:
        if topic not in weak_list:
            weak_list.append(topic)
        if topic in comp_list:
            comp_list.remove(topic)

    profile["completed_topics"] = comp_list
    profile["weak_topics"] = weak_list
    mem._save_semantic_memory(profile)

    # Record attempt
    mem.add_quiz_record(
        topic=topic,
        score=score,
        total=total,
        passed=passed,
        questions=questions,
        user_answers=user_answers
    )

    # Add chat message record
    score_msg = f"Quiz complete on **{topic}**! Score: **{score}/{total}** ({'PASSED' if passed else 'FAILED'})"
    mem.add_chat_message("assistant", score_msg, body.session_id)

    return {"score": score, "total": total, "passed": passed}

# ─── Status Routes ────────────────────────────────────────────────────────────
@app.get("/api/status")
def status():
    ollama_running = is_ollama_running()
    try:
        db = get_chroma_client()
        col = db.get_or_create_collection("course_materials")
        chunk_count = col.count()
    except Exception:
        chunk_count = 0
    existing_files = []
    if os.path.exists(DATA_DIR):
        existing_files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    return {
        "ollama_running": ollama_running,
        "chunk_count": chunk_count,
        "data_files": existing_files
    }

@app.post("/api/status/start_ollama")
async def start_ollama():
    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(None, ensure_ollama_running, 15)
    return {"success": success}

# ─── Document Upload Route ────────────────────────────────────────────────────
@app.post("/api/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    saved = []
    for file in files:
        file_path = os.path.join(DATA_DIR, file.filename)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        saved.append(file.filename)
    return {"saved": saved}

@app.post("/api/documents/ingest")
async def ingest_documents():
    loop = asyncio.get_event_loop()
    def run_ingestion():
        from app.rag.ingestion import DocumentIngestionPipeline
        pipeline = DocumentIngestionPipeline()
        pipeline.ingest_documents()
        db = get_chroma_client()
        col = db.get_or_create_collection("course_materials")
        return col.count()
    try:
        count = await loop.run_in_executor(None, run_ingestion)
        return {"success": True, "chunk_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Knowledge Graph ──────────────────────────────────────────────────────────
@app.get("/api/graph")
def get_knowledge_graph():
    """Returns the knowledge graph HTML for embedding in an iframe."""
    from app.agents.orchestrator import get_kg
    from app.utils.config import GRAPH_DIR
    mem = UserMemory(USER_ID)
    profile = mem.get_semantic_memory()
    completed = profile.get("completed_topics", [])
    html_path = os.path.join(GRAPH_DIR, "graph.html")
    get_kg().generate_pyvis_html(output_path=html_path, completed_topics=completed)
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse("<html><body style='background:#0f172a;color:#94a3b8;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh'><p>Knowledge graph is empty. Upload documents and ask questions to build it.</p></body></html>")

@app.get("/api/agent_graph")
def get_agent_graph(active_node: str = None, completed_nodes: str = ""):
    """Returns the agent pipeline SVG HTML."""
    completed = [n.strip() for n in completed_nodes.split(",") if n.strip()] if completed_nodes else []
    html = generate_state_graph_html(active_node=active_node, completed_nodes=completed)
    return HTMLResponse(html)

# ─── Serve the standalone React frontend ─────────────────────────────────────
FRONTEND_HTML = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")

# If a Vite build exists, serve from dist (production mode)
if os.path.exists(FRONTEND_DIST) and os.path.exists(os.path.join(FRONTEND_DIST, "index.html")):
    _assets_dir = os.path.join(FRONTEND_DIST, "assets")
    if os.path.exists(_assets_dir):
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

    @app.get("/{full_path:path}")
    def serve_react(full_path: str):
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))

# Otherwise serve the standalone CDN-based index.html directly
elif os.path.exists(FRONTEND_HTML):
    @app.get("/")
    def serve_root():
        return FileResponse(FRONTEND_HTML)

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        # Don't intercept /api routes (they're already registered above)
        return FileResponse(FRONTEND_HTML)

