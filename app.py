import os
# [CRITICAL FIX] Prevent PyTorch/OpenMP from silently crashing the server on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import chromadb

# Import the orchestrator and memory
from app.agents.orchestrator import stream_tutor_pipeline, generate_state_graph_html, OllamaConnectionError
from app.memory.user_memory import UserMemory
from app.utils.ollama_helper import is_ollama_running, ensure_ollama_running

# Set page config
st.set_page_config(
    page_title="Offline AI Tutor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State for Chat
if "user_id" not in st.session_state:
    st.session_state.user_id = "default_user"

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = "default_session"

# Restore memory profile globally
mem = UserMemory(st.session_state.user_id)
semantic = mem.get_semantic_memory()

if "chat_history" not in st.session_state:
    # Restore chat history from the persistent SQLite database for the active session on reload!
    st.session_state.chat_history = mem.get_chat_history(st.session_state.active_session_id, limit=50)

from app.rag.ingestion import DocumentIngestionPipeline
from app.utils.config import DATA_DIR, VECTOR_DB_DIR

# --- Chat Sessions Manager ---
st.sidebar.markdown("### 💬 Chat Sessions")

# Get list of sessions from SQLite
sessions = mem.get_chat_sessions()
session_ids = [s["session_id"] for s in sessions]
session_titles = {s["session_id"]: s["title"] for s in sessions}

# Make sure active_session_id exists in the session database.
if st.session_state.active_session_id not in session_ids:
    mem.create_chat_session(st.session_state.active_session_id, "Default Chat Session")
    sessions = mem.get_chat_sessions()
    session_ids = [s["session_id"] for s in sessions]
    session_titles = {s["session_id"]: s["title"] for s in sessions}

# Callback to handle session changes safely without infinite reruns
def on_session_change():
    new_sid = st.session_state.session_selectbox
    st.session_state.active_session_id = new_sid
    db_mem = UserMemory(st.session_state.user_id)
    st.session_state.chat_history = db_mem.get_chat_history(new_sid, limit=50)
    st.session_state.active_quiz = None

# Render a dropdown selector to choose past chat sessions
selected_session_idx = session_ids.index(st.session_state.active_session_id) if st.session_state.active_session_id in session_ids else 0
selected_session_id = st.sidebar.selectbox(
    "Select Session",
    options=session_ids,
    format_func=lambda sid: session_titles.get(sid, "Untitled Session"),
    index=selected_session_idx,
    key="session_selectbox",
    on_change=on_session_change,
    label_visibility="collapsed"
)

def on_new_chat():
    import datetime
    timestamp_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db_mem = UserMemory(st.session_state.user_id)
    db_mem.create_chat_session(timestamp_id, "New Chat Session")
    st.session_state.active_session_id = timestamp_id
    st.session_state.session_selectbox = timestamp_id
    st.session_state.chat_history = []
    st.session_state.active_quiz = None

def on_delete_chat():
    db_mem = UserMemory(st.session_state.user_id)
    db_mem.delete_chat_session(st.session_state.active_session_id)
    remaining_sessions = db_mem.get_chat_sessions()
    if remaining_sessions:
        st.session_state.active_session_id = remaining_sessions[0]["session_id"]
    else:
        st.session_state.active_session_id = "default_session"
    st.session_state.session_selectbox = st.session_state.active_session_id
    st.session_state.chat_history = db_mem.get_chat_history(st.session_state.active_session_id, limit=50)
    st.session_state.active_quiz = None

# Session Actions: New Chat, Delete active chat in a clean row
col1, col2 = st.sidebar.columns(2)
with col1:
    st.button("➕ New Chat", width="stretch", on_click=on_new_chat)

with col2:
    is_default = (st.session_state.active_session_id == "default_session")
    st.button("🗑️ Delete Chat", width="stretch", disabled=is_default, on_click=on_delete_chat)

# Expander to manually rename the active session title
with st.sidebar.expander("📝 Rename Current Chat"):
    current_title = session_titles.get(st.session_state.active_session_id, "Default Chat Session")
    new_title = st.text_input("New Title", value=current_title, key="rename_session_input")
    if st.button("Save Title", width="stretch"):
        if new_title.strip():
            mem.update_session_title(st.session_state.active_session_id, new_title.strip())
            st.rerun()

# Expander to visualize the Non-Linear Graph Memory Git-tree
try:
    _graph, _head, _root = mem.load_graph_memory(st.session_state.active_session_id)
    with st.sidebar.expander("🕸️ Non-Linear Git-Tree Memory", expanded=True):
        st.caption("Real-time visual tree of branching conversation contexts.")
        
        def build_nested_tree(graph, node, prefix="", is_last=True):
            label = graph.nodes[node].get("topic_label", "Untitled")
            marker = "└── " if is_last else "├── "
            text = f"{prefix}{marker}**{label}**"
            if node == _head:
                text += " 🌟 *(HEAD)*"
            
            tree_nodes = [text]
            children = list(graph.successors(node))
            for i, child in enumerate(children):
                new_prefix = prefix + ("    " if is_last else "│   ")
                tree_nodes.extend(build_nested_tree(graph, child, new_prefix, is_last=(i == len(children) - 1)))
            return tree_nodes
            
        tree_text = "\n".join(build_nested_tree(_graph, _root))
        st.markdown(tree_text)
except Exception as e:
    st.sidebar.error(f"Failed to load memory tree: {e}")

st.sidebar.markdown("---")

# Sidebar Navigation
st.sidebar.title("Navigation")

# Dynamic notification badge for Active Quiz
quiz_opt = "Active Quiz"
if "active_quiz" in st.session_state and st.session_state.active_quiz and not st.session_state.active_quiz["submitted"]:
    quiz_opt = "Active Quiz (NEW 🛑)"

page = st.sidebar.radio("Go to:", ["Dashboard", "Learn", quiz_opt, "Graph Visualization", "Agent Architecture"])


st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Personalization")

# Render select boxes for style and academic level inside the chat sidebar
styles = ["default", "visual", "step-by-step", "analogy-based", "example-heavy", "concise", "detailed"]
levels = ["Beginner", "Intermediate", "Advanced"]

current_style = semantic.get("preferred_style", "default")
current_level = semantic.get("academic_level", "Intermediate")

style_idx = styles.index(current_style) if current_style in styles else 0
level_idx = levels.index(current_level) if current_level in levels else 1

new_style = st.sidebar.selectbox("Learning Style", options=styles, index=style_idx)
new_level = st.sidebar.selectbox("Academic Level", options=levels, index=level_idx)

if new_style != current_style or new_level != current_level:
    updated = semantic.copy()
    updated["preferred_style"] = new_style
    updated["academic_level"] = new_level
    mem._save_semantic_memory(updated)
    st.sidebar.success("Preferences updated!")

st.sidebar.markdown("---")
st.sidebar.subheader("🏆 Quiz Scoreboard")
quiz_history_list = mem.get_quiz_history(limit=5)
if quiz_history_list:
    for qh in quiz_history_list:
        date_str = qh["timestamp"].split(" ")[0] if " " in qh["timestamp"] else qh["timestamp"]
        status_emoji = "🟢" if qh["passed"] else "🔴"
        st.sidebar.caption(f"{status_emoji} **{qh['topic']}**: {qh['score']}/{qh['total']} ({date_str})")
else:
    st.sidebar.caption("No quizzes taken yet. Ask your tutor to quiz you on a topic!")

st.sidebar.markdown("---")
st.sidebar.write("**System Status:**")
st.sidebar.write("🟢 LangGraph Orchestrator: Active")


# Check and manage Ollama status dynamically
if "ollama_status" not in st.session_state:
    st.session_state.ollama_status = "unknown"

if st.session_state.ollama_status == "unknown":
    # Silently probe if it's already running first to avoid lag
    if is_ollama_running():
        st.session_state.ollama_status = "running"
    else:
        st.session_state.ollama_status = "offline"

if st.session_state.ollama_status == "running":
    st.sidebar.write("🟢 **LLM Runtime: Ollama (llama3.2)**")
else:
    st.sidebar.write("🔴 **LLM Runtime: Ollama (Offline)**")
    # Proactively try to auto-start if offline
    if "auto_start_attempted" not in st.session_state:
        st.session_state.auto_start_attempted = True
        with st.sidebar.status("🔌 Waking up Ollama...", expanded=False) as status:
            if ensure_ollama_running(timeout_seconds=8):
                st.session_state.ollama_status = "running"
                status.update(label="Ollama is ready!", state="complete")
            else:
                status.update(label="Ollama not running.", state="error")

    # If it is still offline, show a button to manually start it
    if st.session_state.ollama_status == "offline":
        if st.sidebar.button("🔌 Start Ollama Server", key="start_ollama_btn"):
            with st.sidebar.status("Starting Ollama server...", expanded=True) as status:
                if ensure_ollama_running(timeout_seconds=15):
                    st.session_state.ollama_status = "running"
                    status.update(label="Ollama started successfully!", state="complete")
                else:
                    status.update(label="Failed to start Ollama. Is it installed?", state="error")


@st.cache_resource
def get_chroma_client_singleton():
    return chromadb.PersistentClient(path=VECTOR_DB_DIR)

# Show how many chunks are already in ChromaDB so user knows data persists
try:
    _db = get_chroma_client_singleton()
    _col = _db.get_or_create_collection("course_materials")
    chunk_count = _col.count()
    if chunk_count > 0:
        st.sidebar.success(f"📦 {chunk_count} chunks already in vector DB")
    else:
        st.sidebar.info("📦 No documents ingested yet")
except Exception:
    st.sidebar.info("📦 Vector DB not initialized")

# Show uploaded files already on disk
existing_files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))] if os.path.exists(DATA_DIR) else []
if existing_files:
    st.sidebar.write(f"**📁 {len(existing_files)} file(s) on disk:**")
    for ef in existing_files:
        st.sidebar.caption(f"  • {ef}")

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Upload Study Material")
uploaded_files = st.sidebar.file_uploader("Upload PDFs or Text files", accept_multiple_files=True, type=['pdf', 'txt', 'docx'])

if st.sidebar.button("Process Documents"):
    if uploaded_files:
        with st.sidebar.status("Processing Documents..."):
            for file in uploaded_files:
                # Save the file to our data directory
                file_path = os.path.join(DATA_DIR, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                st.write(f"Saved {file.name}")
            
            # Trigger ingestion
            st.write("Chunking & Embedding...")
            pipeline = DocumentIngestionPipeline()
            pipeline.ingest_documents()
            st.success("Ingestion Complete!")  # Rerun to update the chunk count in the sidebar
    else:
        st.sidebar.warning("Please upload a file first.")

def dashboard_page():
    st.title("📊 Learner Dashboard")
    st.write("Welcome to your offline personalized learning hub. Review your verified academic progress below.")
    
    # Fetch metrics
    mastered = semantic.get('completed_topics', [])
    weak = semantic.get('weak_topics', [])
    all_quizzes = mem.get_quiz_history(limit=100)
    total_quizzes = len(all_quizzes)
    passed_quizzes = sum(1 for q in all_quizzes if q["passed"])
    
    # 1. Summary Metrics
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(label="🏆 Verified Mastered Concepts", value=len(mastered))
    with col_b:
        st.metric(label="📖 Concepts in Review", value=len(weak))
    with col_c:
        pass_rate = f"{int(passed_quizzes / total_quizzes * 100)}%" if total_quizzes > 0 else "N/A"
        st.metric(label="🎯 Quiz Pass Rate", value=pass_rate, delta=f"{passed_quizzes}/{total_quizzes} passed")
        
    st.divider()
    
    # 2. Mastery Lists
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ Verified Mastery List")
        st.caption("These topics have been added automatically by passing academic quizzes.")
        if mastered:
            for t in mastered:
                st.success(f"🎓 **{t}** (Verified Mastered)")
        else:
            st.info("No concepts mastered yet. Ask the tutor to quiz you on a topic to test your knowledge!")
            
    with col2:
        st.subheader("⚠️ Weak Topics (Focus Areas)")
        st.caption("These topics require more study based on quiz scorecards or your learning preferences.")
        if weak:
            for t in weak:
                st.warning(f"📖 **{t}** (Requires Focus)")
        else:
            st.success("All caught up! You have no weak topics on record.")
            
    # 3. Detailed Quiz Log
    st.divider()
    st.subheader("📋 Academic Quiz History Log")
    if all_quizzes:
        import pandas as pd
        # Convert quiz records to a clean dataframe for tabular display
        records = []
        for qh in all_quizzes:
            records.append({
                "Date": qh["timestamp"],
                "Topic": qh["topic"],
                "Score": f"{qh['score']}/{qh['total']}",
                "Status": "🟢 PASSED" if qh["passed"] else "🔴 FAILED"
            })
        df = pd.DataFrame(records)
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.caption("No quiz logs recorded in SQLite database. Start learning to record your attempts!")


def repair_json_array(json_str: str) -> str:
    """Attempts to repair a truncated JSON array of objects by matching quotes, braces, and brackets."""
    json_str = json_str.strip()
    if not json_str.startswith('['):
        return json_str
        
    # If the string ends with a comma, strip it
    if json_str.endswith(','):
        json_str = json_str[:-1].strip()
        
    # Count open quotes first to see if a string is unclosed
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
        
    # Now balance braces and brackets
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
        
    # Close open braces and brackets
    if open_braces > 0:
        json_str += '}' * open_braces
    if open_brackets > 0:
        json_str += ']' * open_brackets
        
    return json_str


def learn_page():
    st.title("📚 Tutor Agent Workspace")
    st.write("Ask questions and get explanations tailored to your level. Type 'quiz me' at the end to get a quiz!")
    
    # Initialize active quiz session state if not exists
    if "active_quiz" not in st.session_state:
        st.session_state.active_quiz = None

    # Display Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    # Input
    user_input = st.chat_input("Ask your tutor a question...")
    if user_input:
        # Display user message
        st.chat_message("user").write(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Persist to SQLite so chat survives reloads
        _mem = UserMemory(st.session_state.user_id)
        
        # Auto-rename session if it currently has a default placeholder title
        current_sessions = _mem.get_chat_sessions()
        session_title = next((s["title"] for s in current_sessions if s["session_id"] == st.session_state.active_session_id), "")
        if session_title in ("Default Chat Session", "New Chat Session"):
            # Construct a clean 3-4 word title from query
            words = user_input.strip().split()
            clean_title = " ".join(words[:4])
            if len(words) > 4:
                clean_title += "..."
            clean_title = clean_title.strip('?.!,:;()')
            if clean_title:
                _mem.update_session_title(st.session_state.active_session_id, clean_title)
                
        _mem.add_chat_message("user", user_input, st.session_state.active_session_id)
        
        # Run LangGraph Pipeline
        # Use a container for the live state graph
        graph_container = st.container()
        graph_placeholder = graph_container.empty()
        
        # Render the initial state graph (all nodes idle)
        with graph_placeholder:
            st.iframe(generate_state_graph_html(), height=580)
        
        completed = []
        active_topic = "General Knowledge"
        try:
            with st.status("Running Agentic Pipeline...", expanded=True) as status:
                tutor_msg = None
                quiz_msg = None
                for node_name, state in stream_tutor_pipeline(user_input, user_id=st.session_state.user_id, session_id=st.session_state.active_session_id):
                    # Extract active topic dynamically
                    if isinstance(state, dict) and state.get("active_topic"):
                        active_topic = state.get("active_topic")
                    
                    # Update the SVG graph: current node glows, completed nodes are green
                    with graph_placeholder:
                        st.iframe(generate_state_graph_html(active_node=node_name, completed_nodes=completed), height=580)
                    
                    if node_name == "supervisor":
                        st.write("🚦 Git Supervisor checking semantic branches...")
                    elif node_name == "researcher":
                        st.write("📚 Micro-Chunk RAG retrieving study materials...")
                    elif node_name == "pedagogue":
                        st.write("🧠 Pedagogue Tutor generating explanation...")
                        tutor_msg = state.get("tutor_response")
                        quiz_msg = state.get("quiz_response")
                    elif node_name == "scribe":
                        st.write("✍️ Scribe distilling memory into NetworkX DiGraph...")
                    
                    completed.append(node_name)
                
                # Final state: all nodes done, END highlighted
                with graph_placeholder:
                    st.iframe(generate_state_graph_html(active_node="__end__", completed_nodes=completed), height=580)
                status.update(label="Agentic Pipeline Complete!", state="complete", expanded=False)
        except OllamaConnectionError as oce:
            st.session_state.ollama_status = "offline"
            status.update(label="Pipeline Failed: Ollama Connection Error", state="error", expanded=True)
            st.error("🔌 **Ollama connection failed!**")
            st.markdown(
                """
                It looks like your local Ollama server is offline or is still starting up. 
                Since this AI Tutor runs **100% offline**, it requires Ollama to be running on your machine.
                
                ### How to Fix:
                1. **Auto-Start:** Click the **🔌 Start Ollama Server** button in the sidebar.
                2. **Manual Start:** Open a terminal and run `ollama serve`, or open the Ollama Desktop app.
                3. Check that the `llama3.2` model is pulled by running `ollama pull llama3.2` in your command line.
                """
            )
            return

            
        # Display Tutor Response
        if tutor_msg:
            st.chat_message("assistant").write(tutor_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": tutor_msg})
            _mem.add_chat_message("assistant", tutor_msg, st.session_state.active_session_id)
            
        # Display Quiz if generated
        if quiz_msg:
            import json
            import re
            
            # Try to parse the JSON quiz from the LLM response
            try:
                # The LLM might wrap JSON in markdown code blocks, strip them
                clean = quiz_msg.strip()
                clean = re.sub(r'^```json\s*', '', clean)
                clean = re.sub(r'^```\s*', '', clean)
                clean = re.sub(r'\s*```$', '', clean)
                
                # Find the start of the JSON array in the response
                start_idx = clean.find('[')
                if start_idx != -1:
                    clean = clean[start_idx:]
                
                # Repair truncated JSON if needed (e.g. missing brackets/braces from Ollama)
                clean = repair_json_array(clean)
                questions = json.loads(clean)
                
                if isinstance(questions, list) and len(questions) > 0:
                    st.session_state.active_quiz = {
                        "topic": active_topic,
                        "questions": questions,
                        "quiz_key": f"quiz_{len(st.session_state.chat_history)}",
                        "graded": False,
                        "user_answers": [],
                        "score": 0,
                        "submitted": False
                    }
                    status_msg = f"Generated an interactive quiz on **{active_topic}**! Navigate to the **Active Quiz (NEW 🛑)** page in the sidebar menu to start."
                    st.session_state.chat_history.append({"role": "assistant", "content": status_msg})
                    _mem.add_chat_message("assistant", status_msg, st.session_state.active_session_id)
                    st.rerun()
                else:
                    raise ValueError("Empty or invalid question list")
                    
            except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
                # Fallback: if JSON parsing fails, show the raw quiz text
                st.session_state.chat_history.append({"role": "assistant", "content": f"**Quiz:**\n{quiz_msg}"})
                _mem.add_chat_message("assistant", f"**Quiz:**\n{quiz_msg}", st.session_state.active_session_id)
                st.rerun()


def active_quiz_page():
    st.title("📝 Active Quiz Workspace")
    st.write("Objective, verified mastery challenges based on your study materials.")

    # Initialize active quiz session state if not exists
    if "active_quiz" not in st.session_state:
        st.session_state.active_quiz = None

    if st.session_state.active_quiz:
        aq = st.session_state.active_quiz
        st.subheader(f"📝 Quiz Challenge: {aq['topic']}")
        
        with st.form(key=aq["quiz_key"]):
            user_answers = []
            for i, q in enumerate(aq["questions"]):
                st.markdown(f"**Q{i+1}: {q['question']}**")
                
                # Setup index for already submitted choices
                saved_ans = aq["user_answers"][i] if aq["submitted"] else None
                saved_idx = q["options"].index(saved_ans) if saved_ans in q["options"] else None
                
                answer = st.radio(
                    f"Select your answer for Q{i+1}:",
                    options=q["options"],
                    key=f"{aq['quiz_key']}_q{i}",
                    index=saved_idx,
                    disabled=aq["submitted"],
                    label_visibility="collapsed"
                )
                user_answers.append(answer)
                st.markdown("---")
                
            if not aq["submitted"]:
                submitted = st.form_submit_button("✅ Submit Answers", width="stretch")
                if submitted:
                    # Grade the quiz
                    score = 0
                    total = len(aq["questions"])
                    for i, q in enumerate(aq["questions"]):
                        # Safeguard correct index check to avoid IndexError crashes!
                        correct_idx = q.get("correct", 0)
                        if not isinstance(correct_idx, int) or correct_idx < 0 or correct_idx >= len(q["options"]):
                            correct_idx = 0
                            
                        correct_answer = q["options"][correct_idx]
                        if user_answers[i] == correct_answer:
                            score += 1
                            
                    passed = score >= (total * 2 / 3) # e.g. 2/3 passed (66%)
                    
                    # Update semantic memory
                    db_mem = UserMemory(st.session_state.user_id)
                    profile = db_mem.get_semantic_memory()
                    
                    topic = aq["topic"]
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
                    db_mem._save_semantic_memory(profile)
                    
                    # Record attempt to SQLite quiz_history table
                    db_mem.add_quiz_record(
                        topic=topic,
                        score=score,
                        total=total,
                        passed=passed,
                        questions=aq["questions"],
                        user_answers=user_answers
                    )
                    
                    # Save results in session state
                    aq["submitted"] = True
                    aq["user_answers"] = user_answers
                    aq["score"] = score
                    aq["passed"] = passed
                    
                    # Append chat log
                    score_msg = f"Quiz complete on **{topic}**! Score: **{score}/{total}** ({'PASSED' if passed else 'FAILED'})"
                    st.session_state.chat_history.append({"role": "assistant", "content": score_msg})
                    db_mem.add_chat_message("assistant", score_msg, st.session_state.active_session_id)
                    st.rerun()
            else:
                # Quiz is graded! Display results and explanation feedback
                score = aq["score"]
                total = len(aq["questions"])
                passed = aq["passed"]
                
                for i, q in enumerate(aq["questions"]):
                    # Safeguard correct index check to avoid IndexError crashes!
                    correct_idx = q.get("correct", 0)
                    if not isinstance(correct_idx, int) or correct_idx < 0 or correct_idx >= len(q["options"]):
                        correct_idx = 0
                        
                    correct_answer = q["options"][correct_idx]
                    explanation = q.get("explanation", "")
                    u_choice = aq["user_answers"][i]
                    
                    if u_choice == correct_answer:
                        st.success(f"Q{i+1}: ✅ Correct! (Selected: *{u_choice}*)")
                    elif u_choice is None:
                        st.warning(f"Q{i+1}: ⚠️ Unanswered. Correct: **{correct_answer}**")
                    else:
                        st.error(f"Q{i+1}: ❌ Wrong. Selected: *{u_choice}*. Correct: **{correct_answer}**")
                        
                    if explanation:
                        st.info(f"💡 **Explanation:** {explanation}")
                        
                st.markdown(f"### Score: {score}/{total}")
                if passed:
                    st.balloons()
                    st.success(f"🎉 Verified Mastery! **{aq['topic']}** has been added to your completed list.")
                    if st.form_submit_button("Dismiss Quiz", width="stretch"):
                        st.session_state.active_quiz = None
                        st.rerun()
                else:
                    st.warning(f"📖 Score too low. **{aq['topic']}** has been flagged as a weak topic to focus on.")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.form_submit_button("🔄 Retake Quiz", width="stretch"):
                            aq["submitted"] = False
                            aq["user_answers"] = []
                            aq["score"] = 0
                            aq["passed"] = False
                            st.rerun()
                    with col2:
                        if st.form_submit_button("Dismiss Quiz", width="stretch"):
                            st.session_state.active_quiz = None
                            st.rerun()
    else:
        st.info("💡 **No active quiz challenge.**")
        st.markdown(
            """
            To start a quiz and verify your mastery:
            1. Navigate to the **Learn** page in the sidebar.
            2. Ask the tutor to **quiz you** on a specific topic (e.g., *"quiz me on Image Segmentation"* or *"test me on HDFS"*).
            3. The tutor will compile your custom challenge, and a notification tag **(NEW 🛑)** will appear on this page in the sidebar.
            4. Return here to take the interactive test and level up!
            """
        )




def graph_page():
    st.title("🕸️ Concept Knowledge Graph")
    st.write("Visualize the prerequisites and dependencies of your study topics. Mastered topics are highlighted in Green ✅!")
    
    from pathlib import Path
    from app.agents.orchestrator import get_kg
    
    # Retrieve current user's mastered topics
    db_mem = UserMemory(st.session_state.user_id)
    profile = db_mem.get_semantic_memory()
    completed = profile.get("completed_topics", [])
    
    # Always regenerate the graph dynamically to highlight newly mastered concepts in green!
    html_path = get_kg().generate_pyvis_html(completed_topics=completed)
        
    if html_path and os.path.exists(html_path):
        st.iframe(Path(html_path), height=650)
    else:
        st.info("The Knowledge Graph is currently empty. Ask the Tutor Agent questions about your topics to start building connections!")

def agent_architecture_page():
    st.title("🤖 Agentic Architecture")
    st.write("This shows the LangGraph state machine orchestrating the AI Tutor pipeline.")
    st.write("Each node represents an agent that processes the student's query sequentially.")
    
    # Show the full pipeline with all nodes in idle state
    graph_html = generate_state_graph_html()
    st.iframe(graph_html, height=580)
    
    st.markdown("---")
    st.subheader("📋 Node Descriptions")
    st.markdown("""
| Node | Role |
|------|------|
| **🔍 Memory Check** | Loads the student's learning history, preferred style, and past weak topics |
| **🕸️ Knowledge Graph** | Checks if the query topic has prerequisite concepts the student should know |
| **📚 RAG Retrieval** | Searches uploaded documents using hybrid Vector + BM25 keyword search |
| **🧠 Tutor Agent** | Generates a personalized explanation using the LLM with all gathered context |
| **📝 Quiz Agent** | Creates quiz questions if the student requests them |
""")

# Routing
route_page = page
if "Active Quiz" in page:
    route_page = "Active Quiz"

if route_page == "Dashboard":
    dashboard_page()
elif route_page == "Learn":
    learn_page()
elif route_page == "Active Quiz":
    active_quiz_page()
elif route_page == "Graph Visualization":
    graph_page()
elif route_page == "Agent Architecture":
    agent_architecture_page()
