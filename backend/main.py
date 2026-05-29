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
    subject_id: str = "default_subject"

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

class SubjectCreate(BaseModel):
    subject_id: str
    subject_name: str

class IngestRequest(BaseModel):
    subject_id: str
    doc_title: Optional[str] = None
    manual_text: Optional[str] = None
    file_name: Optional[str] = None

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

def generate_and_insert_part_explanation(subject_id: str, session_id: str = None) -> str:
    """
    Directly invokes the LLM (outside the LangGraph pipeline agents) to generate
    an exhaustive, visually stunning, highly tailored explanation of the active
    curriculum segment/chapter according to the user's preferred learning style and level.
    Saves the explanation directly into the SQLite chat history as an assistant message.
    """
    from app.memory.user_memory import UserMemory
    from llama_index.llms.ollama import Ollama
    from app.utils.config import LLM_MODEL
    from app.agents.orchestrator import _get_persona_prompt

    mem = UserMemory(USER_ID)
    
    # 1. Fetch active study state
    study_state = mem.get_active_study_state(subject_id)
    active_doc_id = study_state.get("active_doc_id")
    active_part_index = study_state.get("active_part_index", 1)
    
    if not active_doc_id:
        print("[Intro Explanation] No active document found for subject.")
        return None
        
    part_details = mem.get_document_part(active_doc_id, active_part_index)
    if not part_details:
        print(f"[Intro Explanation] No part details found for doc={active_doc_id}, part={active_part_index}")
        return None
        
    part_title = part_details.get("part_title", f"Part {active_part_index}")
    part_content = part_details.get("part_content", "")
    
    # 2. Get student's style preferences
    profile = mem.get_semantic_memory(subject_id)
    style = profile.get("preferred_style", "default")
    academic_level = profile.get("academic_level", "Intermediate")
    completed_topics = profile.get("completed_topics", [])
    
    # Get the style formatting persona description
    style_persona = _get_persona_prompt(style, academic_level, completed_topics)
    
    # 3. Resolve active chat session
    if not session_id:
        sessions = mem.get_chat_sessions(subject_id)
        if sessions:
            session_id = sessions[0]["session_id"] # Get most recent
        else:
            # Create a new session
            import datetime
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            session_id = f"session_{ts}"
            mem.create_chat_session(session_id, part_title[:40], subject_id)
            
    # 4. Construct prompt for direct explanation synthesis
    prompt = f"""
You are the **Expert Pedagogue Tutor**. Soon after a document is uploaded/divided, or when the student unlocks a new segment, you are responsible for providing an exhaustive, gorgeous, and stunningly styled introduction and detailed explanation of the ENTIRE active syllabus segment before any tutoring conversation or quizzing begins.

### Student Profile:
* **Academic Level**: {academic_level}
* **Preferred Learning Style**: {style.upper()}
* **Styling Constraint Instructions**: {style_persona}

### Active Curriculum Segment:
* **Title**: {part_title}
* **Segment Body/Content**:
---
{part_content}
---

### Your Instructions:
1. Provide a **very comprehensive and exhaustive introduction and full detailed explanation** of all concepts mentioned in the segment content above. Do NOT summarize or skip details—explain everything fully!
2. You MUST strictly adhere to the styling constraints and target audience of the student's profile (e.g. use visual analogies, sequential steps, etc. depending on style).
3. Use premium markdown formatting. Use bold highlights, clear headers, lists, code snippets, or mermaid diagrams if they fit the style to make the explanation look incredibly engaging and top-tier.
4. Conclude your explanation by warmly welcoming the student to this new chapter, summarizing the core subtopics we will cover, and encouraging them to either ask you questions about it or type 'quiz me' whenever they are ready to test their mastery.

Generate your exhaustive, beautifully-formatted explanation now:
"""

    # Simplified prompt for smaller fallback models (tinyllama etc.) that can't handle complex system prompts
    fallback_prompt = f"""Explain the following topic clearly and in detail using markdown formatting.

Topic: {part_title}

Content:
{part_content}

Provide a comprehensive explanation with headers, bullet points, and bold key terms. End by encouraging the student to ask questions or type 'quiz me' to test their understanding."""

    print(f"[Intro Explanation] Generating full explanation for subject={subject_id}, doc={active_doc_id}, part={active_part_index}...")
    try:
        try:
            llm = Ollama(model=LLM_MODEL, request_timeout=120.0)
            response = llm.complete(prompt)
            explanation = str(response.text).strip()
        except Exception as e:
            print(f"[Intro Explanation] Primary model '{LLM_MODEL}' failed: {e}. Falling back to 'tinyllama'...")
            llm = Ollama(model="tinyllama", request_timeout=90.0)
            response = llm.complete(fallback_prompt)
            explanation = str(response.text).strip()
        
        if not explanation or len(explanation) < 20:
            print(f"[Intro Explanation] LLM returned empty/too-short response, using formatted fallback.")
            explanation = f"""## 📖 {part_title}\n\nWelcome to this chapter! Here's what we'll be covering:\n\n{part_content}\n\n---\n\n💡 **Ready to learn?** Ask me any questions about this topic, or type **\'quiz me\'** when you're ready to test your understanding!"""
        
        # 5. Insert into SQLite chat message log
        mem.add_chat_message("assistant", explanation, session_id)
        print(f"[Intro Explanation] Successfully generated and stored explanation in session {session_id}!")
        return explanation
    except Exception as e:
        print(f"[Intro Explanation] Error generating explanation: {e}")
        import traceback
        traceback.print_exc()
        # Ultimate fallback: insert a basic structured explanation so the user always sees something
        try:
            fallback_text = f"""## 📖 {part_title}\n\nWelcome to this chapter! Here's what we'll be covering:\n\n{part_content}\n\n---\n\n💡 **Ready to learn?** Ask me any questions about this topic, or type **\'quiz me\'** when you're ready to test your understanding!"""
            mem.add_chat_message("assistant", fallback_text, session_id)
            print(f"[Intro Explanation] Inserted basic fallback explanation in session {session_id}")
            return fallback_text
        except Exception:
            return None

@app.get("/api/sessions")
def list_sessions(subject_id: Optional[str] = None):
    mem = UserMemory(USER_ID)
    sessions = mem.get_chat_sessions(subject_id)
    
    # Auto-generate the initial session if there is a curriculum but no sessions yet
    if not sessions and subject_id:
        study_state = mem.get_active_study_state(subject_id)
        doc_id = study_state.get("active_doc_id")
        active_part_index = study_state.get("active_part_index", 1)
        if doc_id:
            part_details = mem.get_document_part(doc_id, active_part_index)
            if part_details:
                part_title = part_details.get("part_title", f"Part {active_part_index}")
                import datetime
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                session_id = f"session_{ts}"
                # Create session and generate explanation synchronously
                mem.create_chat_session(session_id, f"Study: {part_title[:30]}", subject_id)
                generate_and_insert_part_explanation(subject_id, session_id)
                sessions = mem.get_chat_sessions(subject_id)
    return sessions

@app.post("/api/sessions")
def create_session(body: SessionCreate):
    mem = UserMemory(USER_ID)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    session_id = f"session_{ts}"
    
    # Resolve premium study title if it's a generic New Chat
    title = body.title
    study_state = mem.get_active_study_state(body.subject_id)
    doc_id = study_state.get("active_doc_id")
    active_part_index = study_state.get("active_part_index", 1)
    
    if doc_id:
        part_details = mem.get_document_part(doc_id, active_part_index)
        if part_details:
            part_title = part_details.get("part_title", f"Part {active_part_index}")
            if not title or title.strip() == "New Chat Session" or title.strip() == "New Chat":
                title = f"Study: {part_title[:30]}"
                
    mem.create_chat_session(session_id, title, body.subject_id)
    
    # Automatically generate explanation for the active curriculum segment in this new session
    if doc_id:
        generate_and_insert_part_explanation(body.subject_id, session_id)
        
    return {"session_id": session_id, "title": title, "subject_id": body.subject_id}

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

        # Look up subject_id using session_id
        import sqlite3
        conn = sqlite3.connect(mem.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT subject_id FROM chat_sessions WHERE session_id=?", (session_id,))
        row = cursor.fetchone()
        conn.close()
        subject_id = row[0] if row else "default_subject"

        try:
            # Run the blocking LangGraph pipeline in a thread pool to not block the event loop
            def run_pipeline():
                results = []
                for node_name, state in stream_tutor_pipeline(query, user_id=USER_ID, session_id=session_id, subject_id=subject_id):
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
def get_profile(subject_id: Optional[str] = None):
    mem = UserMemory(USER_ID)
    return mem.get_semantic_memory(subject_id or "default_subject")

@app.put("/api/profile")
def update_profile(body: ProfileUpdate, subject_id: Optional[str] = None):
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
        mem.update_semantic_memory(updates, subject_id or "default_subject")
    return mem.get_semantic_memory(subject_id or "default_subject")

# ─── Quiz Routes ──────────────────────────────────────────────────────────────
@app.get("/api/quiz/history")
def quiz_history(subject_id: Optional[str] = None):
    mem = UserMemory(USER_ID)
    return mem.get_quiz_history(limit=100, subject_id=subject_id)

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

    passed = score >= (total * 0.7) # 70% threshold as requested!

    # Look up subject_id using session_id
    mem = UserMemory(USER_ID)
    import sqlite3
    conn = sqlite3.connect(mem.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT subject_id FROM chat_sessions WHERE session_id=?", (body.session_id,))
    row = cursor.fetchone()
    conn.close()
    subject_id = row[0] if row else "default_subject"

    # Update semantic memory
    profile = mem.get_semantic_memory(subject_id)
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
    mem._save_semantic_memory(profile, subject_id)

    # Ensure topic is added to the EducationalKnowledgeGraph
    try:
        from app.agents.orchestrator import get_kg
        kg = get_kg()
        if topic not in kg.G.nodes:
            kg.add_topic(topic, prerequisites=[], subject_id=subject_id)
            print(f"[KG Update] Dynamically added quiz topic '{topic}' to Knowledge Graph.")
    except Exception as e:
        print(f"[KG Update] Error adding topic to graph: {e}")

    # Record attempt
    mem.add_quiz_record(
        topic=topic,
        score=score,
        total=total,
        passed=passed,
        questions=questions,
        user_answers=user_answers,
        subject_id=subject_id
    )

    # Sequential study state advancement
    unlocked_msg = ""
    if passed:
        old_state = mem.get_active_study_state(subject_id)
        new_state = mem.advance_active_part(subject_id)
        if new_state["active_part_index"] > old_state["active_part_index"]:
            next_part_index = new_state["active_part_index"]
            part_details = mem.get_document_part(new_state["active_doc_id"], next_part_index)
            next_title = part_details.get("part_title", f"Part {next_part_index}") if part_details else f"Part {next_part_index}"
            unlocked_msg = f"\n\n🔓 **Next Chapter Unlocked:** You have advanced to **Part {next_part_index}: {next_title}**! Let's begin learning this new topic immediately."
        elif new_state["status"] == "completed":
            unlocked_msg = f"\n\n🎉 **Syllabus Completed!** You have mastered all parts of this curriculum document! Excellent work!"

    # Add chat message record with high-quality messaging
    if passed:
        score_msg = f"🏆 **Quiz Passed!** You scored **{score}/{total}** on **{topic}**! You have mastered this subtopic, and your Knowledge Graph has been updated with your new mastery! 🎉{unlocked_msg}"
    else:
        score_msg = f"📖 **Quiz Failed.** You scored **{score}/{total}** on **{topic}**. Don't worry! This topic has been added to your Focus Areas on the Knowledge Graph. You can retake the test anytime to verify your mastery! 💪"

    mem.add_chat_message("assistant", score_msg, body.session_id)
    
    # Generate the detailed intro explanation of the new chapter synchronously AFTER the quiz status has been saved!
    if passed and new_state["active_part_index"] > old_state["active_part_index"]:
        generate_and_insert_part_explanation(subject_id, body.session_id)

    return {"score": score, "total": total, "passed": passed}

# ─── Subject Routes ───────────────────────────────────────────────────────────
@app.get("/api/subjects")
def list_subjects():
    mem = UserMemory(USER_ID)
    return mem.get_subjects()

@app.post("/api/subjects")
def create_subject(body: SubjectCreate):
    mem = UserMemory(USER_ID)
    sanitized_id = re.sub(r'[^a-zA-Z0-9_]', '_', body.subject_id).lower()
    mem.add_subject(sanitized_id, body.subject_name)
    return {"subject_id": sanitized_id, "subject_name": body.subject_name}

@app.delete("/api/subjects/{subject_id}")
def delete_subject(subject_id: str):
    mem = UserMemory(USER_ID)
    mem.delete_subject(subject_id)
    return {"ok": True}

@app.get("/api/subjects/progress")
def get_subjects_progress():
    mem = UserMemory(USER_ID)
    subjects = mem.get_subjects()
    progress_list = []
    
    for sub in subjects:
        sub_id = sub["subject_id"]
        sub_name = sub["subject_name"]
        
        # Fetch active study state
        state = mem.get_active_study_state(sub_id)
        doc_id = state.get("active_doc_id")
        active_part_index = state.get("active_part_index", 1)
        status = state.get("status", "studying")
        
        # Fetch parts list
        parts = mem.get_document_parts(doc_id) if doc_id else []
        total_parts = len(parts)
        
        active_part_title = "None"
        if doc_id and parts:
            active_part_details = mem.get_document_part(doc_id, active_part_index)
            if active_part_details:
                active_part_title = active_part_details.get("part_title", f"Part {active_part_index}")
        
        # Calculate progress percentage
        if total_parts > 0:
            mastered_count = active_part_index - 1
            if status == "completed":
                mastered_count = total_parts
            pct = int((mastered_count / total_parts) * 100)
        else:
            mastered_count = 0
            pct = 0
            
        # Fetch semantic profile (completed/weak topics)
        profile = mem.get_semantic_memory(sub_id)
        completed_topics = profile.get("completed_topics", [])
        weak_topics = profile.get("weak_topics", [])
        
        # Fetch quiz count and average
        quizzes = mem.get_quiz_history(limit=50, subject_id=sub_id)
        quiz_count = len(quizzes)
        quiz_avg = 0
        if quiz_count > 0:
            quiz_avg = round(sum(q["score"]/q["total"]*100 for q in quizzes)/quiz_count, 1)
            
        progress_list.append({
            "subject_id": sub_id,
            "subject_name": sub_name,
            "active_doc_id": doc_id,
            "active_part_index": active_part_index,
            "active_part_title": active_part_title,
            "status": status,
            "total_parts": total_parts,
            "mastered_parts_count": mastered_count,
            "progress_percent": pct,
            "completed_concepts": len(completed_topics),
            "weak_concepts": len(weak_topics),
            "quiz_count": quiz_count,
            "quiz_average": quiz_avg
        })
        
    return progress_list

@app.get("/api/subjects/{subject_id}/study_state")
def get_subject_study_state(subject_id: str):
    mem = UserMemory(USER_ID)
    state = mem.get_active_study_state(subject_id)
    doc_id = state.get("active_doc_id")
    active_part_index = state.get("active_part_index", 1)
    
    parts = mem.get_document_parts(doc_id) if doc_id else []
    
    active_part_title = ""
    active_part_content = ""
    for p in parts:
        if p["part_index"] == active_part_index:
            active_part_title = p["part_title"]
            active_part_content = p["part_content"]
            
    return {
        "subject_id": subject_id,
        "active_doc_id": doc_id,
        "active_part_index": active_part_index,
        "status": state.get("status", "studying"),
        "parts": [{"part_index": p["part_index"], "part_title": p["part_title"], "part_content": p["part_content"]} for p in parts],
        "active_part_title": active_part_title,
        "active_part_content_preview": active_part_content[:600] + "..." if len(active_part_content) > 600 else active_part_content
    }

@app.post("/api/subjects/{subject_id}/advance")
async def advance_subject_study_state(subject_id: str):
    mem = UserMemory(USER_ID)
    res = mem.advance_active_part(subject_id)
    
    # Trigger introductory explanation for the newly unlocked chapter in the background
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, generate_and_insert_part_explanation, subject_id)
    return res

@app.post("/api/subjects/{subject_id}/explain")
async def trigger_part_explanation(subject_id: str, session_id: Optional[str] = None):
    """On-demand endpoint to generate and insert the explanation for the current active part."""
    loop = asyncio.get_event_loop()
    explanation = await loop.run_in_executor(None, generate_and_insert_part_explanation, subject_id, session_id)
    return {"success": explanation is not None, "length": len(explanation) if explanation else 0}

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
async def upload_documents(files: list[UploadFile] = File(...), subject_id: str = Form("default_subject")):
    saved = []
    for file in files:
        file_path = os.path.join(DATA_DIR, file.filename)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        saved.append(file.filename)
    return {"saved": saved, "subject_id": subject_id}

@app.post("/api/documents/ingest")
async def ingest_documents(body: IngestRequest):
    loop = asyncio.get_event_loop()
    def run_divider_ingestion():
        from app.utils.curriculum_divider import CurriculumDivider
        divider = CurriculumDivider()
        if body.file_name:
            file_path = os.path.join(DATA_DIR, body.file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Uploaded file {body.file_name} not found.")
            res = divider.divide_and_ingest_file(body.subject_id, file_path)
            return res
        elif body.manual_text and body.doc_title:
            res = divider.divide_and_ingest_curriculum(
                subject_id=body.subject_id,
                doc_title=body.doc_title,
                text_content=body.manual_text
            )
            return res
        else:
            raise ValueError("Either file_name OR (doc_title AND manual_text) must be provided.")
            
    try:
        res = await loop.run_in_executor(None, run_divider_ingestion)
        if res is None:
            raise HTTPException(status_code=400, detail="Ingestion failed. File may be empty or unreadable.")
            
        # Create a new active chat session for study
        mem = UserMemory(USER_ID)
        import datetime
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        session_id = f"session_{ts}"
        
        # Use Part 1 title as the session title
        part_details = mem.get_document_part(res.get("doc_id"), 1)
        part_title = part_details.get("part_title", "Chapter 1: Intro") if part_details else "Chapter 1: Intro"
        mem.create_chat_session(session_id, f"Study: {part_title[:30]}", body.subject_id)
        
        # Generate the detailed intro explanation of Part 1 synchronously before returning
        await loop.run_in_executor(None, generate_and_insert_part_explanation, body.subject_id, session_id)
        
        return {
            "success": True, 
            "doc_id": res.get("doc_id"), 
            "total_parts": res.get("total_parts"),
            "session_id": session_id
        }
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
    weak = profile.get("weak_topics", [])
    html_path = os.path.join(GRAPH_DIR, "graph.html")
    get_kg().generate_pyvis_html(output_path=html_path, completed_topics=completed, weak_topics=weak)
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

