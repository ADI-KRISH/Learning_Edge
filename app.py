import os
# [CRITICAL FIX] Prevent PyTorch/OpenMP from silently crashing the server on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import chromadb
import datetime

# Import the orchestrator and memory
from app.agents.orchestrator import stream_tutor_pipeline, generate_state_graph_html, OllamaConnectionError
from app.memory.user_memory import UserMemory
from app.utils.ollama_helper import is_ollama_running, ensure_ollama_running
from app.rag.ingestion import ContextualIngestor
from app.utils.config import DATA_DIR, VECTOR_DB_DIR

# Set page config
st.set_page_config(
    page_title="Learning Edge: Multi-Subject AI Tutor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "user_id" not in st.session_state:
    st.session_state.user_id = "default_user"

if "active_subject_id" not in st.session_state:
    st.session_state.active_subject_id = "default_subject"

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = "default_session"

# Restore memory profile globally
mem = UserMemory(st.session_state.user_id)

# Ensure subject default exists and load active semantic memory
subjects = mem.get_subjects()
subject_ids = [s["subject_id"] for s in subjects]
if st.session_state.active_subject_id not in subject_ids:
    st.session_state.active_subject_id = "default_subject"

semantic = mem.get_semantic_memory(st.session_state.active_subject_id)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = mem.get_chat_history(st.session_state.active_session_id, limit=50)

# --- Dynamic Subject Manager Sidebar ---
st.sidebar.markdown("### 📚 Subjects")

def on_subject_change():
    new_sub = st.session_state.subject_selectbox
    st.session_state.active_subject_id = new_sub
    db_mem = UserMemory(st.session_state.user_id)
    
    # Load first chat session of new subject
    sub_sessions = db_mem.get_chat_sessions(new_sub)
    if sub_sessions:
        st.session_state.active_session_id = sub_sessions[0]["session_id"]
    else:
        default_sid = f"session_{new_sub}_default"
        db_mem.create_chat_session(default_sid, "Default Subject Session", new_sub)
        st.session_state.active_session_id = default_sid
        
    st.session_state.chat_history = db_mem.get_chat_history(st.session_state.active_session_id, limit=50)
    st.session_state.active_quiz = None

subject_titles = {s["subject_id"]: s["subject_name"] for s in subjects}
selected_subject_idx = subject_ids.index(st.session_state.active_subject_id) if st.session_state.active_subject_id in subject_ids else 0

selected_subject_id = st.sidebar.selectbox(
    "Select Active Subject",
    options=subject_ids,
    format_func=lambda sid: subject_titles.get(sid, "General Knowledge"),
    index=selected_subject_idx,
    key="subject_selectbox",
    on_change=on_subject_change,
    label_visibility="collapsed"
)

# Expander to Add a New Subject + Upload Syllabus
with st.sidebar.expander("📁 Add New Subject"):
    new_sub_name = st.text_input("Subject Name", placeholder="e.g. Natural Language Processing")
    uploaded_syllabus = st.file_uploader("Upload Subject Syllabus/Material", type=["pdf", "txt"])
    
    if st.button("Initialize Subject", width="stretch"):
        if new_sub_name.strip():
            sub_id = new_sub_name.strip().lower().replace(" ", "_")
            
            # 1. Create subject directories
            sub_data_dir = os.path.join(DATA_DIR, sub_id)
            os.makedirs(sub_data_dir, exist_ok=True)
            
            file_name = ""
            if uploaded_syllabus:
                file_name = uploaded_syllabus.name
                file_path = os.path.join(sub_data_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_syllabus.getbuffer())
            
            # 2. Register subject
            mem.add_subject(sub_id, new_sub_name.strip(), file_name)
            
            # 3. Trigger Contextual Retrieval Ingestion if a document was uploaded
            if uploaded_syllabus:
                with st.status(f"Ingesting {file_name} with Contextual Retrieval..."):
                    ingestor = ContextualIngestor()
                    ingestor.ingest_subject_documents(sub_id)
            
            # 4. Create default chat session
            default_sid = f"session_{sub_id}_default"
            mem.create_chat_session(default_sid, f"Welcome to {new_sub_name}", sub_id)
            
            # 5. Swap State & reload
            st.session_state.active_subject_id = sub_id
            st.session_state.active_session_id = default_sid
            st.session_state.chat_history = mem.get_chat_history(default_sid)
            st.session_state.active_quiz = None
            st.success(f"'{new_sub_name}' initialized!")
            st.rerun()
        else:
            st.warning("Subject name is required.")

st.sidebar.markdown("---")

# --- Subject-Aware Chat Sessions Manager ---
st.sidebar.markdown("### 💬 Chat Sessions")

sessions = mem.get_chat_sessions(st.session_state.active_subject_id)
session_ids = [s["session_id"] for s in sessions]
session_titles = {s["session_id"]: s["title"] for s in sessions}

if st.session_state.active_session_id not in session_ids:
    default_sid = f"session_{st.session_state.active_subject_id}_default"
    if default_sid not in session_ids:
        mem.create_chat_session(default_sid, "Default Subject Session", st.session_state.active_subject_id)
    st.session_state.active_session_id = default_sid
    sessions = mem.get_chat_sessions(st.session_state.active_subject_id)
    session_ids = [s["session_id"] for s in sessions]
    session_titles = {s["session_id"]: s["title"] for s in sessions}

def on_session_change():
    new_sid = st.session_state.session_selectbox
    st.session_state.active_session_id = new_sid
    db_mem = UserMemory(st.session_state.user_id)
    st.session_state.chat_history = db_mem.get_chat_history(new_sid, limit=50)
    st.session_state.active_quiz = None

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
    timestamp_id = f"session_{st.session_state.active_subject_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db_mem = UserMemory(st.session_state.user_id)
    db_mem.create_chat_session(timestamp_id, "New Chat Session", st.session_state.active_subject_id)
    st.session_state.active_session_id = timestamp_id
    st.session_state.session_selectbox = timestamp_id
    st.session_state.chat_history = []
    st.session_state.active_quiz = None

def on_delete_chat():
    db_mem = UserMemory(st.session_state.user_id)
    db_mem.delete_chat_session(st.session_state.active_session_id)
    remaining_sessions = db_mem.get_chat_sessions(st.session_state.active_subject_id)
    if remaining_sessions:
        st.session_state.active_session_id = remaining_sessions[0]["session_id"]
    else:
        st.session_state.active_session_id = f"session_{st.session_state.active_subject_id}_default"
    st.session_state.session_selectbox = st.session_state.active_session_id
    st.session_state.chat_history = db_mem.get_chat_history(st.session_state.active_session_id, limit=50)
    st.session_state.active_quiz = None

col1, col2 = st.sidebar.columns(2)
with col1:
    st.button("➕ New Chat", width="stretch", on_click=on_new_chat)
with col2:
    is_default = (st.session_state.active_session_id == f"session_{st.session_state.active_subject_id}_default")
    st.button("🗑️ Delete Chat", width="stretch", disabled=is_default, on_click=on_delete_chat)

with st.sidebar.expander("📝 Rename Current Chat"):
    current_title = session_titles.get(st.session_state.active_session_id, "Default Subject Session")
    new_title = st.text_input("New Title", value=current_title, key="rename_session_input")
    if st.button("Save Title", width="stretch"):
        if new_title.strip():
            mem.update_session_title(st.session_state.active_session_id, new_title.strip())
            st.rerun()

# Non-Linear tree view
try:
    _graph, _head, _root = mem.load_graph_memory(st.session_state.active_session_id)
    with st.sidebar.expander("🕸️ Conversation Tree", expanded=True):
        st.caption("Active branching conversation history nodes.")
        
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

# Navigation
st.sidebar.title("Navigation")
quiz_opt = "Active Quiz"
if "active_quiz" in st.session_state and st.session_state.active_quiz and not st.session_state.active_quiz["submitted"]:
    quiz_opt = "Active Quiz (NEW 🛑)"

page = st.sidebar.radio("Go to:", ["Dashboard", "Learn", quiz_opt, "Concept Knowledge Graph", "Agent Architecture"])

st.sidebar.markdown("---")

# Personalization
st.sidebar.subheader("🎯 Personalization")
styles = ["default", "step-by-step", "concise", "detailed"]
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
    mem._save_semantic_memory(updated, st.session_state.active_subject_id)
    st.sidebar.success("Preferences updated!")

st.sidebar.markdown("---")

# Quick RAG diagnostics
@st.cache_resource
def get_chroma_client_singleton():
    return chromadb.PersistentClient(path=VECTOR_DB_DIR)

try:
    _db = get_chroma_client_singleton()
    _col = _db.get_or_create_collection(f"subject_{st.session_state.active_subject_id}")
    chunk_count = _col.count()
    if chunk_count > 0:
        st.sidebar.success(f"📦 {chunk_count} RAG chunks loaded")
    else:
        st.sidebar.info("📦 No subject materials ingested")
except Exception:
    st.sidebar.info("📦 Vector DB not initialized")

# System Ollama Runtime Status Check
if "ollama_status" not in st.session_state:
    st.session_state.ollama_status = "unknown"

if st.session_state.ollama_status == "unknown":
    if is_ollama_running():
        st.session_state.ollama_status = "running"
    else:
        st.session_state.ollama_status = "offline"

if st.session_state.ollama_status == "running":
    st.sidebar.write("🟢 **LLM: Ollama (llama3.2)**")
else:
    st.sidebar.write("🔴 **LLM: Ollama (Offline)**")
    if "auto_start_attempted" not in st.session_state:
        st.session_state.auto_start_attempted = True
        with st.sidebar.status("🔌 Waking up Ollama...", expanded=False) as status:
            if ensure_ollama_running(timeout_seconds=8):
                st.session_state.ollama_status = "running"
                status.update(label="Ollama is ready!", state="complete")
            else:
                status.update(label="Ollama offline.", state="error")

st.sidebar.write("🟢 LangGraph Pipeline: Active")

# --- UI PAGES ---

def dashboard_page():
    st.title("📊 Multi-Subject Dashboard")
    st.write("Track your learning progress, concept mastery, and quiz metrics per subject.")
    
    # Global metrics
    all_subjects = mem.get_subjects()
    num_subjects = len(all_subjects)
    
    overall_completed = 0
    overall_weak = 0
    for s in all_subjects:
        sem = mem.get_semantic_memory(s["subject_id"])
        overall_completed += len(sem.get("completed_topics", []))
        overall_weak += len(sem.get("weak_topics", []))
        
    all_quizzes = mem.get_quiz_history(limit=100)
    total_quizzes = len(all_quizzes)
    passed_quizzes = sum(1 for q in all_quizzes if q["passed"])
    
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric(label="📚 Total Subjects Enrolled", value=num_subjects)
    with col_b:
        st.metric(label="🏆 Concepts Mastered", value=overall_completed)
    with col_c:
        st.metric(label="⚠️ Weak Focus Areas", value=overall_weak)
    with col_d:
        pass_rate = f"{int(passed_quizzes / total_quizzes * 100)}%" if total_quizzes > 0 else "N/A"
        st.metric(label="🎯 Quiz Pass Rate", value=pass_rate, delta=f"{passed_quizzes}/{total_quizzes} passed")
        
    st.divider()
    
    # Per-Subject Concept Mastery Progress
    st.subheader("📈 Subject Concept Mastery Progress")
    
    # We load concepts from Knowledge Graph and group them by subject
    from app.graph.knowledge_graph import EducationalKnowledgeGraph
    kg = EducationalKnowledgeGraph()
    
    for s in all_subjects:
        sub_id = s["subject_id"]
        sub_name = s["subject_name"]
        
        # Count concepts in unified KG tagged with this subject_id
        subject_concepts = [node for node in kg.G.nodes if kg.G.nodes[node].get("subject_id") == sub_id]
        total_concepts = len(subject_concepts)
        
        # Load user semantic profile for this subject
        sem_profile = mem.get_semantic_memory(sub_id)
        mastered_concepts = [t for t in sem_profile.get("completed_topics", []) if t in subject_concepts or not subject_concepts]
        
        comp_count = len(sem_profile.get("completed_topics", []))
        weak_count = len(sem_profile.get("weak_topics", []))
        
        progress = 0.0
        if total_concepts > 0:
            progress = min(1.0, float(len(mastered_concepts)) / total_concepts)
            progress_label = f"{int(progress * 100)}% ({len(mastered_concepts)}/{total_concepts} Concepts)"
        else:
            progress_label = f"0% (No concepts extracted yet)"
            
        col_name, col_bar = st.columns([1, 3])
        with col_name:
            st.markdown(f"**{sub_name}**")
        with col_bar:
            st.progress(progress, text=progress_label)
            
    st.divider()
    
    # Subject-by-Subject Mastery Details
    st.subheader("📋 Academic Mastery Summary")
    sub_tabs = st.tabs([s["subject_name"] for s in all_subjects])
    
    for idx, s in enumerate(all_subjects):
        with sub_tabs[idx]:
            sub_id = s["subject_id"]
            sem_profile = mem.get_semantic_memory(sub_id)
            comp_list = sem_profile.get("completed_topics", [])
            weak_list = sem_profile.get("weak_topics", [])
            
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("##### ✅ Mastered Concepts")
                if comp_list:
                    for concept in comp_list:
                        st.success(f"🎓 **{concept}**")
                else:
                    st.caption("No concepts mastered yet. Pass a quiz to unlock!")
            with col_r:
                st.markdown("##### ⚠️ Focus Areas")
                if weak_list:
                    for concept in weak_list:
                        st.warning(f"📖 **{concept}**")
                else:
                    st.success("All concepts verified or up-to-date!")
                    
    st.divider()
    
    # Detailed Quiz Log
    st.subheader("📋 Academic Quiz History Log")
    if all_quizzes:
        import pandas as pd
        records = []
        # Create map of subject ids to names
        sub_name_map = {s["subject_id"]: s["subject_name"] for s in all_subjects}
        for qh in all_quizzes:
            records.append({
                "Date": qh["timestamp"],
                "Subject": sub_name_map.get(qh["subject_id"], "General"),
                "Topic": qh["topic"],
                "Score": f"{qh['score']}/{qh['total']}",
                "Status": "🟢 PASSED" if qh["passed"] else "🔴 FAILED"
            })
        df = pd.DataFrame(records)
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.caption("No quiz logs recorded. Start learning to record attempts!")

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

def learn_page():
    st.title("📚 Tutor Workspace")
    st.write(f"Active Subject Room: **{subject_titles.get(st.session_state.active_subject_id, 'General Knowledge')}**")
    
    if "active_quiz" not in st.session_state:
        st.session_state.active_quiz = None

    # Render Chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    # Inputs
    user_input = st.chat_input("Ask your subject tutor a question...")
    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Persist message
        _mem = UserMemory(st.session_state.user_id)
        
        # Auto rename default session title based on query
        current_sessions = _mem.get_chat_sessions(st.session_state.active_subject_id)
        session_title = next((s["title"] for s in current_sessions if s["session_id"] == st.session_state.active_session_id), "")
        if session_title in ("Default Subject Session", "New Chat Session"):
            words = user_input.strip().split()
            clean_title = " ".join(words[:4]).strip('?.!,:;()')
            if clean_title:
                _mem.update_session_title(st.session_state.active_session_id, clean_title)
                
        _mem.add_chat_message("user", user_input, st.session_state.active_session_id)
        
        # Setup LangGraph live state visual
        graph_container = st.container()
        graph_placeholder = graph_container.empty()
        with graph_placeholder:
            st.iframe(generate_state_graph_html(), height=580)
        
        completed = []
        active_topic = "General Concept"
        try:
            with st.status("Agentic Pipeline running...", expanded=True) as status:
                tutor_msg = None
                quiz_msg = None
                
                # Stream the subject-aware pipeline!
                for node_name, state in stream_tutor_pipeline(
                    user_input, 
                    user_id=st.session_state.user_id, 
                    session_id=st.session_state.active_session_id,
                    subject_id=st.session_state.active_subject_id
                ):
                    if isinstance(state, dict) and state.get("active_topic"):
                        active_topic = state.get("active_topic")
                    
                    with graph_placeholder:
                        st.iframe(generate_state_graph_html(active_node=node_name, completed_nodes=completed), height=580)
                    
                    if node_name == "supervisor":
                        st.write("🚦 Supervisor auditing subject branches...")
                    elif node_name == "researcher":
                        st.write("📚 Retrieval pipeline running with Anthropic Contextual Retrieval...")
                    elif node_name == "pedagogue":
                        st.write("🧠 Pedagogical tutor formatting personalized explanation...")
                        tutor_msg = state.get("tutor_response")
                        quiz_msg = state.get("quiz_response")
                    elif node_name == "scribe":
                        st.write("✍️ Scribing updates to unified history...")
                    
                    completed.append(node_name)
                
                with graph_placeholder:
                    st.iframe(generate_state_graph_html(active_node="__end__", completed_nodes=completed), height=580)
                status.update(label="Response Formulated!", state="complete", expanded=False)
        except OllamaConnectionError:
            st.session_state.ollama_status = "offline"
            status.update(label="Ollama Connection Failure", state="error", expanded=True)
            st.error("🔌 **Ollama connection failed!** Make sure `ollama serve` is active on port 11434.")
            return
            
        if tutor_msg:
            st.chat_message("assistant").write(tutor_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": tutor_msg})
            _mem.add_chat_message("assistant", tutor_msg, st.session_state.active_session_id)
            
        if quiz_msg:
            import json
            import re
            try:
                clean = quiz_msg.strip()
                clean = re.sub(r'^```json\s*', '', clean)
                clean = re.sub(r'^```\s*', '', clean)
                clean = re.sub(r'\s*```$', '', clean)
                
                start_idx = clean.find('[')
                if start_idx != -1:
                    clean = clean[start_idx:]
                
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
                    status_msg = f"Generated quiz on concept **{active_topic}**! Navigate to the active quiz tab in the sidebar to attempt."
                    st.session_state.chat_history.append({"role": "assistant", "content": status_msg})
                    _mem.add_chat_message("assistant", status_msg, st.session_state.active_session_id)
                    st.rerun()
                else:
                    raise ValueError("Quiz array empty")
            except Exception:
                st.session_state.chat_history.append({"role": "assistant", "content": f"**Quiz:**\n{quiz_msg}"})
                _mem.add_chat_message("assistant", f"**Quiz:**\n{quiz_msg}", st.session_state.active_session_id)
                st.rerun()

def active_quiz_page():
    st.title("📝 Active Quiz Workspace")
    st.write("Subject quiz challenges based on your ingested study materials.")
    
    if "active_quiz" not in st.session_state:
        st.session_state.active_quiz = None

    if st.session_state.active_quiz:
        aq = st.session_state.active_quiz
        st.subheader(f"📝 Quiz: {aq['topic']}")
        
        with st.form(key=aq["quiz_key"]):
            user_answers = []
            for i, q in enumerate(aq["questions"]):
                st.markdown(f"**Q{i+1}: {q['question']}**")
                
                saved_ans = aq["user_answers"][i] if aq["submitted"] else None
                saved_idx = q["options"].index(saved_ans) if saved_ans in q["options"] else None
                
                answer = st.radio(
                    f"Select option for Q{i+1}:",
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
                    score = 0
                    total = len(aq["questions"])
                    for i, q in enumerate(aq["questions"]):
                        correct_idx = q.get("correct", 0)
                        if not isinstance(correct_idx, int) or correct_idx < 0 or correct_idx >= len(q["options"]):
                            correct_idx = 0
                        correct_answer = q["options"][correct_idx]
                        if user_answers[i] == correct_answer:
                            score += 1
                            
                    passed = score >= (total * 2 / 3) # e.g. 70% threshold
                    
                    db_mem = UserMemory(st.session_state.user_id)
                    profile = db_mem.get_semantic_memory(st.session_state.active_subject_id)
                    
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
                    db_mem._save_semantic_memory(profile, st.session_state.active_subject_id)
                    
                    # Record to history with subject_id!
                    db_mem.add_quiz_record(
                        topic=topic,
                        score=score,
                        total=total,
                        passed=passed,
                        questions=aq["questions"],
                        user_answers=user_answers,
                        subject_id=st.session_state.active_subject_id
                    )
                    
                    aq["submitted"] = True
                    aq["user_answers"] = user_answers
                    aq["score"] = score
                    aq["passed"] = passed
                    
                    score_msg = f"Quiz complete on **{topic}**! Score: **{score}/{total}** ({'PASSED' if passed else 'FAILED'})"
                    st.session_state.chat_history.append({"role": "assistant", "content": score_msg})
                    db_mem.add_chat_message("assistant", score_msg, st.session_state.active_session_id)
                    st.rerun()
            else:
                score = aq["score"]
                total = len(aq["questions"])
                passed = aq["passed"]
                
                for i, q in enumerate(aq["questions"]):
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
                    st.warning(f"📖 Flagged as a weak topic to review: **{aq['topic']}**")
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
            To take a challenge and verify your mastery:
            1. Select your target subject in the sidebar.
            2. Go to the **Learn** Workspace and ask your tutor to **quiz you** (e.g. *"quiz me on image segmentations"*).
            3. Your challenge will compile and the sidebar will notify you when it's ready.
            """
        )

def graph_page():
    st.title("🕸️ Unified Concept Knowledge Graph")
    st.write("Prerequisites and concept dependencies across all subjects. Green ✅ = Mastered, Yellow ⚠️ = Weak review topics.")
    
    from pathlib import Path
    from app.graph.knowledge_graph import EducationalKnowledgeGraph
    
    # Aggregate completed/weak topics across all subjects
    db_mem = UserMemory(st.session_state.user_id)
    all_subjects = db_mem.get_subjects()
    
    overall_completed = []
    overall_weak = []
    for s in all_subjects:
        sem = db_mem.get_semantic_memory(s["subject_id"])
        overall_completed.extend(sem.get("completed_topics", []))
        overall_weak.extend(sem.get("weak_topics", []))
        
    kg = EducationalKnowledgeGraph()
    html_path = kg.generate_pyvis_html(completed_topics=overall_completed, weak_topics=overall_weak)
    
    if html_path and os.path.exists(html_path):
        st.iframe(Path(html_path), height=650)
    else:
        st.info("Knowledge Graph is currently empty. Upload syllabus curriculum details to get started!")

def agent_architecture_page():
    st.title("🤖 Agentic Architecture")
    st.write("This interactive diagram details the active LangGraph routing state machine.")
    st.iframe(generate_state_graph_html(), height=580)
    st.markdown("---")
    st.subheader("📋 Node Descriptions")
    st.markdown("""
| Node | Role |
|------|------|
| **🚦 Git Supervisor** | Audits conversational graph branch tips & routes query to Researcher, Chat, or Quiz |
| **📚 RAG Researcher** | Contextually chunks curriculum details and retrieves relevant content using vector + BM25 hybrid ranking |
| **🧠 Pedagogue Tutor** | Formulates hyper-personalized detailed lessons tailored to user personalization matrix |
| **📝 Quiz Assessor** | Generates objective Multiple Choice questions directly derived from RAG contexts |
| **✍️ Scribe Memory** | Distills state and commits interaction to SQLite and NetworkX long term profiles |
""")

# Route page
route_page = page
if "Active Quiz" in page:
    route_page = "Active Quiz"

if route_page == "Dashboard":
    dashboard_page()
elif route_page == "Learn":
    learn_page()
elif route_page == "Active Quiz":
    active_quiz_page()
elif route_page == "Concept Knowledge Graph":
    graph_page()
elif route_page == "Agent Architecture":
    agent_architecture_page()
