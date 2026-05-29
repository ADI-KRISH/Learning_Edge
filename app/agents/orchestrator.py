import os
import re
import uuid
import numpy as np
import networkx as nx
import logging
from typing import Dict, Any, List
from typing_extensions import TypedDict
from datetime import datetime
from langgraph.graph import StateGraph, END
from llama_index.llms.ollama import Ollama
import os
import torch
try:
    torch.set_num_threads(1)
except RuntimeError:
    pass
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
from sentence_transformers import SentenceTransformer

# Import our custom modules
from app.rag.retrieval import HybridRAGRetriever
from app.memory.user_memory import UserMemory
from app.utils.config import LLM_MODEL, BASE_DIR

# Set up logging for Memory Routing
log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "memory_routing.log"),
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger("memory_routing")

# ==========================================
# 1. Shared State Schema & Custom Exceptions
# ==========================================
class OllamaConnectionError(Exception):
    """Custom exception raised when connection to the Ollama server fails."""
    pass

class TutorState(TypedDict):
    user_input: str
    user_id: str
    session_id: str
    subject_id: str
    active_part_index: int
    active_doc_id: str
    episodic_history: List[Dict[str, str]]
    graph_memory: nx.DiGraph
    head_pointer: str
    root_pointer: str
    active_rag_context: str
    raw_rag_context: str
    next_action: str
    final_output: str
    search_query: str
    rag_constraints: dict
    # Backward compatibility keys for Streamlit app.py
    active_topic: str
    tutor_response: str
    quiz_response: str

# ==========================================
# 2. Singleton Resources & Lightweight Embedder
# ==========================================
_retriever = None
_router_llm = None
_tutor_llm = None
_quiz_llm = None
_embedder = None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = HybridRAGRetriever()
    return _retriever

def get_router_llm():
    """Fast, low-creativity LLM for strict intent routing/distillation."""
    global _router_llm
    if _router_llm is None:
        _router_llm = Ollama(
            model=LLM_MODEL, request_timeout=60.0, context_window=1024,
            additional_kwargs={"num_ctx": 1024, "temperature": 0.0, "num_predict": 256}
        )
    return _router_llm

def get_tutor_llm():
    """Heavy, creative LLM for contextual RAG-based explanations."""
    global _tutor_llm
    if _tutor_llm is None:
        _tutor_llm = Ollama(
            model=LLM_MODEL, request_timeout=120.0, context_window=4096,
            additional_kwargs={"num_ctx": 4096, "temperature": 0.7, "num_predict": 1024}
        )
    return _tutor_llm

def get_quiz_llm():
    """Strict formatting LLM for generating reliable JSON quizzes."""
    global _quiz_llm
    if _quiz_llm is None:
        _quiz_llm = Ollama(
            model=LLM_MODEL, request_timeout=90.0, context_window=2048,
            additional_kwargs={"num_ctx": 2048, "temperature": 0.1, "num_predict": 1024}
        )
    return _quiz_llm

def get_embedder():
    """Lightweight SentenceTransformer for computing vector embeddings of queries."""
    global _embedder
    if _embedder is None:
        model_path = os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2")
        if os.path.exists(model_path):
            print(f"Loading local SentenceTransformer model from: {model_path}")
            _embedder = SentenceTransformer(model_path, local_files_only=True)
        else:
            print("Local model path not found. Downloading 'all-MiniLM-L6-v2'...")
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

_kg = None
def get_kg():
    """Singleton for the EducationalKnowledgeGraph."""
    global _kg
    if _kg is None:
        from app.graph.knowledge_graph import EducationalKnowledgeGraph
        _kg = EducationalKnowledgeGraph()
    return _kg


def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

# ==========================================
# 3. LangGraph Directed Graph Memory Nodes
# ==========================================

def supervisor_node(state: TutorState) -> TutorState:
    print("[Supervisor] Analyzing Non-Linear Graph Memory...")
    user_input = state["user_input"]
    graph_memory = state["graph_memory"]
    head_pointer = state["head_pointer"]
    root_pointer = state["root_pointer"]
    
    # 1. Embed user query using lightweight SentenceTransformer
    embedder = get_embedder()
    query_vector = embedder.encode(user_input)
    
    # 2. Generate new unique node ID
    new_node_id = "node_" + str(uuid.uuid4())[:8]
    
    # 3. Perform similarity checks against existing leaf branch tips
    tips = [n for n in graph_memory.nodes if graph_memory.out_degree(n) == 0 and n != new_node_id]
    if not tips:
        tips = [root_pointer]
        
    print(f"  Branch Tips in Graph: {tips}")
    
    # Calculate similarity to tips
    similarities = []
    for tip in tips:
        embedding = graph_memory.nodes[tip].get("embedding")
        if embedding is None or len(embedding) != 384:
            embedding = [0.0]*384
        sim = cosine_similarity(query_vector, embedding)
        similarities.append((tip, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    msg_sims = f"Similarities to Tips: {similarities}"
    print(f"  {msg_sims}")
    logger.info(f"Query: '{user_input[:50]}...' | {msg_sims}")
    
    # Add new node to DiGraph (to be fully finalized in Scribe node)
    graph_memory.add_node(
        new_node_id,
        node_id=new_node_id,
        topic_label="New Conversation",
        raw_turn={"user": user_input, "ai": ""},
        distilled_state={"q": user_input[:40], "status": "active"},
        embedding=query_vector.tolist(),
        timestamp=datetime.now().isoformat()
    )
    
    operation_type = "Commit"
    
    if len(graph_memory.nodes) <= 2: # Only ROOT and our new node
        # Initial commit from ROOT
        graph_memory.add_edge(root_pointer, new_node_id)
        head_pointer = new_node_id
        operation_type = "Initial Commit"
    else:
        # Check Merge: similarity > 0.5 to TWO distinct tips
        merge_tips = [tip for tip, sim in similarities if sim > 0.5]
        if len(merge_tips) >= 2:
            tip1, tip2 = merge_tips[:2]
            print(f"  -> [MERGE] Synthesis Query! Merging branch tips: {tip1} and {tip2}")
            graph_memory.add_edge(tip1, new_node_id)
            graph_memory.add_edge(tip2, new_node_id)
            head_pointer = new_node_id
            operation_type = "Merge"
        else:
            # Check Commit vs Checkout
            head_embedding = graph_memory.nodes[head_pointer].get("embedding", [0.0]*384)
            sim_head = cosine_similarity(query_vector, head_embedding)
            print(f"  Similarity to HEAD ({head_pointer}): {sim_head:.4f}")
            
            best_tip, best_sim = similarities[0] if similarities else (None, 0.0)
            
            # Checkout ONLY if an older tip is highly relevant and much better than current HEAD
            if best_tip and best_tip != head_pointer and best_sim > 0.7 and best_sim > (sim_head + 0.2):
                print(f"  -> [CHECKOUT] Jumping back to older topic branch tip ({best_tip}) with sim {best_sim:.4f}")
                graph_memory.add_edge(best_tip, new_node_id)
                head_pointer = new_node_id
                operation_type = "Checkout"
            else:
                # Default behavior: Just continue the conversation thread!
                print(f"  -> [COMMIT] Continuing active thread at HEAD ({head_pointer})")
                graph_memory.add_edge(head_pointer, new_node_id)
                head_pointer = new_node_id
                operation_type = "Commit"
                    
    msg_op = f"Operation: {operation_type.upper()} | New HEAD: {head_pointer}"
    print(f"  {msg_op}")
    logger.info(msg_op)
    
    # 4. Routing Action Intent Classification
    # Get recent episodic history directly from State
    history_str = ""
    recent_history = state.get("episodic_history", [])
    if recent_history:
        history_lines = []
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Tutor"
            # Strip excessive length for the router
            content_snippet = msg["content"][:200].replace('\n', ' ')
            history_lines.append(f"- {role}: {content_snippet}")
        history_str = "\n".join(history_lines)
            
    router_prompt = f"""You are a strict routing agent. Analyze the User Query and the History.
Classify the intent into one of:
1. "research" (for factual questions, explanations, OR if the user is saying "yes/sure" to a question the tutor just asked in the history about explaining/elaborating on a topic)
2. "chat" (for purely unrelated small talk, greetings, or "yes/no" answers that do NOT relate to the lesson)
3. "assess" (for explicit requests to take a quiz or test)

CRITICAL: You MUST generate a fully standalone "search_query" representing the main subject or topic of interest for both "research" and "assess" intents. If the user is saying "yes" to elaborate, the search_query MUST be rewritten to specify the actual topic from the history (e.g. "elaborate on strengths and weaknesses of waterfall model").

User Query: "{user_input}"
History: {history_str if history_str else 'None'}

Output ONLY minified JSON format:
{{"intent": "chat|research|assess", "search_query": "standalone query", "constraints": {{}}}}
"""
    llm = get_router_llm()
    try:
        import json
        res = str(llm.complete(router_prompt)).strip()
        # Clean markdown ticks
        if res.startswith("```"):
            res = res.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        res_data = json.loads(res)
        next_action = res_data.get("intent", "research")
        state["search_query"] = res_data.get("search_query") or user_input
        state["rag_constraints"] = res_data.get("constraints", {})
    except Exception as e:
        print(f"  [WARN] LLM Routing failed: {e}. Defaulting to research.")
        next_action = "research"
        state["search_query"] = user_input
        state["rag_constraints"] = {}
        
    # Heuristic override for obvious explanation requests that got misrouted
    query_clean = user_input.lower().strip()
    quiz_keywords = ["quiz", "test", "assess", "assessment", "examine", "exam", "question me", "questions on", "evaluate me", "grade me", "quizzes", "practice questions", "trivia", "retake"]
    if any(w in query_clean for w in quiz_keywords):
        print("  [WARN] Heuristic override: query looks like a quiz request. Forcing ASSESS.")
        next_action = "assess"
    elif next_action == "chat" and (len(query_clean.split()) > 4 or any(w in query_clean for w in ["what", "how", "why", "explain", "tell", "can you", "elaborate"])):
        print("  [WARN] Heuristic override: query looks like a question/explanation request. Forcing RESEARCH.")
        next_action = "research"

    print(f"  -> Supervisor Next Action: {next_action.upper()} | Query: {state.get('search_query', '')}")
    
    # Update state
    state["graph_memory"] = graph_memory
    state["head_pointer"] = head_pointer
    state["next_action"] = next_action
    
    return state

def _get_persona_prompt(style: str, level: str, mastered_topics: list) -> str:
    topics_str = ", ".join(mastered_topics) if mastered_topics else ""
    
    # Base instructions for levels
    if level == "Beginner":
        target = "Target audience: Middle school student or complete novice. Use simple, everyday language. Strictly avoid dense jargon."
    elif level == "Intermediate":
        target = "Target audience: Undergraduate college student. Use standard academic language. Introduce domain-specific terminology but define it in context."
    else: # Advanced
        target = "Target audience: Graduate student or domain expert. Use sophisticated, highly technical jargon freely. Assume deep prior knowledge."
        
    # Build the specific stylistic instruction
    style_inst = ""
    if style == "step-by-step":
        if level == "Beginner":
            style_inst = "Break explanations into simple 1-2-3 steps. Avoid paragraphs. Example: '1. Sunlight hits the leaf. 2. The leaf makes food.'"
        elif level == "Intermediate":
            style_inst = "Break explanations into logical, numbered sequences. Example: '1. Photons are absorbed by chlorophyll. 2. This excites electrons...'"
        else:
            style_inst = "Provide strict algorithmic or process-driven numbered steps. Example: '1. Photoexcitation of P680. 2. Plastoquinone reduction...'"
            
    elif style == "concise":
        if level == "Beginner":
            style_inst = "Be extremely brief and direct using simple words. Cut all fluff."
        elif level == "Intermediate":
            style_inst = "Be concise and direct. State the core academic facts without conversational filler."
        else:
            style_inst = "Provide hyper-dense, brief answers focusing strictly on technical precision. Zero fluff."
            
    elif style == "detailed":
        if level == "Beginner":
            style_inst = "Be very thorough but keep the language simple. Explain every small detail so a novice won't get lost."
        elif level == "Intermediate":
            style_inst = "Be exhaustively thorough, capturing nuances and context of the concept at a college level."
        else:
            style_inst = "Provide deep, comprehensive coverage including theoretical background, limitations, and advanced edge cases."
            
    else: # default
        if level == "Beginner":
            style_inst = "Provide a balanced, clear, and encouraging explanation suitable for a beginner."
        elif level == "Intermediate":
            style_inst = "Provide a balanced, clear, and encouraging explanation suitable for a college student."
        else:
            style_inst = "Provide a balanced, clear, and highly technical explanation."

    return f"{target}\n{style_inst}"

def researcher_node(state: TutorState) -> TutorState:
    print("[Researcher] Starting Agentic Hybrid RAG Pipeline...")
    user_input = state["user_input"]
    active_topic = state.get("active_topic", "")
    subject_id = state.get("subject_id", "default_subject")
    
    # Retrieve active study state from SQLite
    mem_sys = UserMemory(user_id=state["user_id"])
    study_state = mem_sys.get_active_study_state(subject_id)
    active_doc_id = study_state.get("active_doc_id")
    active_part_index = study_state.get("active_part_index")
    
    # Strictly filter the RAG constraints by active document and part!
    constraints = state.get("rag_constraints", {}) or {}
    if active_doc_id:
        constraints["doc_id"] = active_doc_id
        constraints["part_index"] = active_part_index
        print(f"  [Researcher] Enforcing strict curriculum constraints: doc_id='{active_doc_id}', part={active_part_index}")
    state["rag_constraints"] = constraints
    
    # --- Step 1: KG Query Enrichment ---
    search_query = state.get("search_query") or user_input
    enriched_query = search_query
    try:
        from app.graph.knowledge_graph import EducationalKnowledgeGraph
        kg = EducationalKnowledgeGraph()
        
        prereqs = []
        related = []
        if active_topic and active_topic in kg.G.nodes:
            # Predecessors are prerequisites
            prereqs = list(kg.G.predecessors(active_topic))
            # Successors are related/advanced concepts
            related = list(kg.G.successors(active_topic))
            
        if prereqs or related:
            prereq_str = ", ".join(prereqs)
            enriched_query = f"search_query: [{search_query}] kg_context: [prerequisites: {prereq_str}]"
            print(f"  [Researcher] KG Enriched Query: {enriched_query}")
    except Exception as e:
        print(f"  [Researcher] KG Enrichment failed: {e}")

    # --- Step 2: Hybrid Retrieval (Ensemble) ---
    print(f"  [Researcher] Retrieving Micro-Chunks from ChromaDB...")
    from app.rag.retrieval import get_retriever
    retriever = get_retriever()
    
    constraints = state.get("rag_constraints", {})
    subject_id = state.get("subject_id", "default_subject")
    # The new retrieve function handles Lanchain Ensemble logic and returns docs
    docs = retriever.retrieve(enriched_query, subject_id=subject_id, constraints=constraints)
    raw_chunks = [d.page_content for d in docs]
    
    if not raw_chunks:
        state["active_rag_context"] = '{"error": "not_found"}'
        state["raw_rag_context"] = ""
        return state

    # --- Step 3: LLM Context Compression ---
    print(f"  [Researcher] Compressing context with local LLM (Token Savior)...")
    source_text = "\n\n---\n\n".join(raw_chunks)
    state["raw_rag_context"] = source_text
    
    # Truncate source text just in case it's huge, to ensure we don't blow the 2048 budget
    source_text = source_text[:3000] 
    
    prompt = f"""You are a data extractor. Read the Source Text. Extract ONLY the factual answers relevant to the User Query. Output strictly as minified JSON. Use short key-value pairs. Drop all conversational filler. If the answer is not in the text, output {{"error": "not_found"}}.

User Query: {user_input}

Source Text:
{source_text}

Output ONLY minified JSON:"""

    try:
        llm = get_tutor_llm()
        response = llm.complete(prompt)
        compressed_json = str(response).strip()
        
        # Simple JSON cleanup in case the LLM wrapped it in markdown
        import re
        clean_json = re.sub(r'^```json\s*', '', compressed_json)
        clean_json = re.sub(r'^```\s*', '', clean_json)
        clean_json = re.sub(r'\s*```$', '', clean_json)
        
        state["active_rag_context"] = clean_json
        print(f"  [Researcher] Compression successful! Output: {clean_json[:100]}...")
    except Exception as e:
        print(f"  [Researcher] Compression failed: {e}")
        # Fallback to truncated raw text if the model crashes
        state["active_rag_context"] = source_text[:1000]

    # --- Step 4: State Update ---
    # Do not overwrite next_action so route_post_researcher can route to assessor if needed
    return state

def pedagogue_node(state: TutorState) -> TutorState:
    print("[Pedagogue] Formulating Personalized explanation...")
    user_input = state["user_input"]
    graph_memory = state["graph_memory"]
    head_pointer = state["head_pointer"]
    active_rag_context = state["active_rag_context"]
    next_action = state["next_action"]
    
    # 1. Fetch exact Episodic Chat History directly from State
    history_str = ""
    recent_history = state.get("episodic_history", [])
    if recent_history:
        history_lines = []
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Tutor"
            history_lines.append(f"- {role}: {msg['content']}")
        history_str = "\n".join(history_lines)
            
    if not history_str:
        history_str = "No recent thread history."
        
    # Get profile style guidelines
    memory_sys = UserMemory(user_id=state.get("user_id", "default_user"))
    subject_id = state.get("subject_id", "default_subject")
    profile = memory_sys.get_semantic_memory(subject_id=subject_id)
    style = profile.get("preferred_style", "default")
    academic_level = profile.get("academic_level", "Intermediate")
    completed_topics = profile.get("completed_topics", [])
    
    unified_persona = _get_persona_prompt(style, academic_level, completed_topics)
    
    # Get active part details
    study_state = memory_sys.get_active_study_state(subject_id)
    active_doc_id = study_state.get("active_doc_id")
    active_part_index = study_state.get("active_part_index", 1)
    
    part_details = memory_sys.get_document_part(active_doc_id, active_part_index) if active_doc_id else None
    part_title = part_details.get("part_title", "General Syllabus") if part_details else "General Syllabus"
    
    unified_persona = _get_persona_prompt(style, academic_level, completed_topics)
    
    # Generate response
    if next_action == "chat":
        prompt = f"""You are a friendly offline AI Tutor for subject '{subject_id}'. The student is engaging in small talk, answering "yes/no", or asking a simple follow-up.
        Using the recent thread history (if any), keep your response conversational, polite, and directly address their statement. 
        Limit your response to 2-3 sentences maximum.
        
        CRITICAL RULE: Never invent facts. If the student is asking a factual question, tell them you are not sure because it wasn't routed correctly.
        If their statement is just a greeting, respond warmly.
        If they say "yes" or "no" out of context, ask them what they would like to clarify.
        
        COMPACT THREAD HISTORY:
        {history_str}
        
        User Query: "{user_input}"
        Tutor:"""
        llm = get_tutor_llm()
    else:
        if active_rag_context and "not_found" not in active_rag_context:
            system_prompt = f"""You are an expert offline AI Tutor for subject '{subject_id}'. 
            The student is currently studying: '{part_title}' (Part {active_part_index}).
            Core facts MUST match the RAG source provided below, which represents the exact content of this curriculum part.
            
            RAG Source: {active_rag_context}
            
            CRITICAL INSTRUCTIONS:
            1. You MUST write your explanation following the requested STUDENT PROFILE & PERSONALIZATION style and level.
            2. Explicitly end your response by reminding the student they are studying Part {active_part_index} ('{part_title}') and invite them to type 'quiz me' once they are ready to test their mastery and unlock the next part!"""
        else:
            system_prompt = f"""You are a helpful academic guide for subject '{subject_id}'. 
            The student's query falls outside the scope of the active sequential curriculum part: '{part_title}' (Part {active_part_index}). 
            Answer the question accurately using general concepts. Do not invent specific technical metrics.
            
            CRITICAL INSTRUCTIONS:
            1. You MUST write your explanation following the requested STUDENT PROFILE & PERSONALIZATION style and level.
            2. Answer their question politely, but immediately after, gracefully guide the student back to the active syllabus unit '{part_title}'.
            3. Conclude by asking if they are ready to return to the active lesson: '{part_title}' (Part {active_part_index})."""
            
        # Standard lesson explanation
        prompt = f"""{system_prompt}
        
        STUDENT PROFILE & PERSONALIZATION:
        {unified_persona}
        
        COMPACT THREAD HISTORY (Git-style distilled):
        {history_str}
        
        STUDENT QUESTION:
        "{user_input}"
        
        Tutor:"""
        llm = get_tutor_llm()
        
    # Enforce token budget
    if len(prompt) > 6000:
        print(f"  [WARN] Prompt exceeds length cap ({len(prompt)} chars). Truncating RAG context...")
        if active_rag_context:
            active_rag_context = active_rag_context[:1000]
        if next_action != "chat":
            prompt = f"""You are an expert AI Tutor.
            PERSONALIZATION: {unified_persona}
            HISTORY: {history_str}
            RAG SOURCE: {active_rag_context if active_rag_context else 'None'}
            QUESTION: "{user_input}"
            CRITICAL INSTRUCTION: Base facts on RAG source if available. If topic is missing, answer using general knowledge but state that it is not in their uploaded documents."""
        
    try:
        response = llm.complete(prompt)
        output_text = str(response)
        
        # Manually prepend the missing docs warning if applicable
        if next_action != "chat" and (not active_rag_context or "not_found" in active_rag_context):
            preamble = "Currently there are no documents available in the database that you have uploaded for this topic, but I can explain it to you.\n\n"
            if not output_text.startswith("Currently"):
                output_text = preamble + output_text
        
        # Append RAG showcase if context was used
        if next_action != "chat":
            raw_rag = state.get("raw_rag_context", "")
            if not raw_rag:
                raw_rag = "No relevant documents found in the database."
            
            rag_append = f"\n\n:::rag 🔍 View Retrieved RAG Context (Raw Chunks)\n{raw_rag}\n:::\n\n:::rag 🧠 LLM Compressed JSON Context\n{active_rag_context}\n:::"
            output_text += rag_append
            
    except Exception as e:
        import httpx
        err_msg = str(e)
        if "connect" in err_msg.lower() or "connection" in err_msg.lower() or isinstance(e, (ConnectionError, httpx.ConnectError)):
            raise OllamaConnectionError("Failed to connect to Ollama.") from e
        raise e
        
    state["final_output"] = output_text
    state["tutor_response"] = output_text
        
    return state

def assessor_node(state: TutorState) -> TutorState:
    print("[Assessor] Generating Interactive Quiz from RAG Context...")
    user_input = state["user_input"]
    active_rag_context = state.get("active_rag_context", "")
    session_id = state.get("session_id", "default_session")
    
    # Get profile settings
    memory_sys = UserMemory(user_id=state["user_id"])
    subject_id = state.get("subject_id", "default_subject")
    profile = memory_sys.get_semantic_memory(subject_id=subject_id)
    difficulty = profile.get("quiz_difficulty", "Medium")
    num_questions = profile.get("quiz_questions", 3)
    
    # Get active part details
    study_state = memory_sys.get_active_study_state(subject_id)
    active_doc_id = study_state.get("active_doc_id")
    active_part_index = study_state.get("active_part_index", 1)
    
    part_details = memory_sys.get_document_part(active_doc_id, active_part_index) if active_doc_id else None
    part_title = part_details.get("part_title", "General Syllabus") if part_details else "General Syllabus"
    
    prompt = f"""You are a strict academic Quiz Agent. Generate exactly {num_questions} multiple choice questions at a {difficulty} difficulty level.
    These questions MUST be drawn strictly and exclusively from the study materials for the current active subtopic: '{part_title}' (Part {active_part_index}).
    
    Study Materials (Facts for the active part context): 
    {active_rag_context if active_rag_context else 'No study materials available.'}
    
    CRITICAL ANTI-HALLUCINATION INSTRUCTIONS:
    1. You MUST generate questions ONLY based on facts present in the Study Materials above.
    2. Do NOT invent questions about general science, genetics, or outside topics unless they appear in the text.
    3. If the Study Materials are empty or irrelevant to the topic, output an empty JSON array `[]`.
    
    You MUST output ONLY valid JSON, no other text. Format:
    [
      {{
        "question": "What is ...?",
        "options": ["option A", "option B", "option C", "option D"],
        "correct": 0,
        "explanation": "Brief explanation based on the text."
      }}
    ]
    
    Output ONLY the JSON array. No markdown, no extra text."""
    
    llm = get_quiz_llm()
    
    # Enforce token budget
    if len(prompt) > 6000:
        active_rag_context = active_rag_context[:1000]
        prompt = f"""You are a strict Quiz Agent. Based on these materials, generate {num_questions} questions ({difficulty}) for '{part_title}' (Part {active_part_index}).
        Materials: {active_rag_context}
        CRITICAL INSTRUCTION: You MUST generate questions ONLY based on the facts present in the Materials.
        You MUST output ONLY valid JSON format: [{{"question":"...","options":["A","B","C","D"],"correct":0,"explanation":"..."}}]
        Output ONLY the JSON array."""
        
    try:
        response = llm.complete(prompt)
        output_text = str(response)
        
        # Keep quiz_response as pure JSON
        state["quiz_response"] = output_text
        
        # Create a friendly chat message for the UI
        chat_msg = "🎯 **Quiz Ready!** I have prepared a personalized quiz for you. Please navigate to the **Quiz** tab in the sidebar to start."
        
        # Append RAG showcase to the chat message, NOT the JSON
        if active_rag_context:
            rag_append = f"\n\n:::rag 🔍 View Retrieved RAG Context (Cosine Sim & BM25)\n{active_rag_context}\n:::"
            chat_msg += rag_append
            
        state["tutor_response"] = chat_msg
        state["final_output"] = chat_msg
            
    except Exception as e:
        import httpx
        err_msg = str(e)
        if "connect" in err_msg.lower() or "connection" in err_msg.lower() or isinstance(e, (ConnectionError, httpx.ConnectError)):
            raise OllamaConnectionError("Failed to connect to Ollama.") from e
        raise e
        
    return state

def scribe_node(state: TutorState) -> TutorState:
    print("[Scribe] Generating Distilled Turn State & Persisting Graph...")
    user_input = state["user_input"]
    final_output = state["final_output"]
    graph_memory = state["graph_memory"]
    head_pointer = state["head_pointer"]
    session_id = state["session_id"]
    next_action = state["next_action"]
    
    # 1. Distill turn into minified JSON
    distill_prompt = f"""You are a data minification agent. Distill this tutor-student interaction into a single sentence summary.
    
    Interaction:
    User: {user_input}
    Tutor: {final_output[:200]}...
    
    Output ONLY a JSON object: {{"q": "summary of query", "status": "understood/clarified"}}
    No other text."""
    
    llm = get_router_llm()
    try:
        res = llm.complete(distill_prompt)
        import json
        clean_res = str(res).strip()
        # Clean markdown ticks
        if clean_res.startswith("```"):
            clean_res = clean_res.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        distilled_data = json.loads(clean_res)
    except Exception:
        # Fallback distilled state
        distilled_data = {"q": user_input[:40], "status": "active"}
        
    # 2. Generate clean topic label
    topic_label = "General"
    words = user_input.strip().split()
    if words:
        # Extract a clean topic label
        raw_label = "_".join(words[:2]).strip('?.!,:;()')
        if raw_label:
            topic_label = raw_label
            
    # 3. Save attributes to current HEAD node in NetworkX
    if head_pointer in graph_memory:
        graph_memory.nodes[head_pointer]["topic_label"] = topic_label
        graph_memory.nodes[head_pointer]["raw_turn"] = {"user": user_input, "ai": final_output}
        graph_memory.nodes[head_pointer]["distilled_state"] = distilled_data
        
    # 4. Save to SQLite database so the graph persists!
    mem_sys = UserMemory(user_id=state["user_id"])
    mem_sys.save_graph_memory(
        session_id=session_id,
        graph=graph_memory,
        head_pointer=head_pointer,
        root_pointer=state["root_pointer"]
    )
    
    # Also save to old-style chat_history table so backward compatibility in UI is maintained
    mem_sys.add_chat_message("user", user_input, session_id)
    if next_action == "assess":
        quiz_status = f"Generated quiz on topic {topic_label}"
        mem_sys.add_chat_message("assistant", quiz_status, session_id)
        final_assistant_content = quiz_status
    else:
        mem_sys.add_chat_message("assistant", final_output, session_id)
        final_assistant_content = final_output
        
    # Append the new turn directly to episodic_history inside the State
    history = list(state.get("episodic_history", []))
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": final_assistant_content})
    state["episodic_history"] = history
        
    state["active_topic"] = topic_label
    print(f"  [OK] Saved node {head_pointer} with Topic: '{topic_label}' and Distilled state: {distilled_data}")
    return state

# ==========================================
# 4. Build the LangGraph
# ==========================================
def route_post_supervisor(state: TutorState) -> str:
    action = state.get("next_action", "research")
    if action == "chat":
        return "pedagogue"
    # Both 'research' and 'assess' require RAG context first!
    else:
        return "researcher"

def route_post_researcher(state: TutorState) -> str:
    action = state.get("next_action", "research")
    if action == "assess":
        return "assessor"
    else:
        return "pedagogue"

# Build the Graph
workflow = StateGraph(TutorState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("pedagogue", pedagogue_node)
workflow.add_node("assessor", assessor_node)
workflow.add_node("scribe", scribe_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    route_post_supervisor,
    {
        "researcher": "researcher",
        "pedagogue": "pedagogue"
    }
)

workflow.add_conditional_edges(
    "researcher",
    route_post_researcher,
    {
        "assessor": "assessor",
        "pedagogue": "pedagogue"
    }
)

workflow.add_edge("pedagogue", "scribe")
workflow.add_edge("assessor", "scribe")
workflow.add_edge("scribe", END)

# Compile
tutor_app = workflow.compile()

# ==========================================
# 5. Visualization & Execution
# ==========================================
def generate_state_graph_html(active_node=None, completed_nodes=None):
    """Generates an SVG visualization matching the multi-agent Tutor graph."""
    if completed_nodes is None:
        completed_nodes = []
    
    nodes = [
        {"id": "__start__",   "label": "START", "x": 350, "y": 30},
        {"id": "supervisor",  "label": "🚦 Git Supervisor", "x": 350, "y": 110},
        {"id": "researcher",  "label": "📚 RAG Researcher", "x": 200, "y": 200},
        {"id": "pedagogue",   "label": "🧠 Pedagogue Tutor", "x": 200, "y": 320},
        {"id": "assessor",    "label": "📝 Quiz Assessor", "x": 500, "y": 320},
        {"id": "scribe",      "label": "✍️ Scribe Memory", "x": 350, "y": 440},
        {"id": "__end__",      "label": "END", "x": 350, "y": 520},
    ]
    
    edges = [
        ("__start__", "supervisor"),
        ("supervisor", "researcher"),
        ("supervisor", "pedagogue"),
        ("researcher", "pedagogue"),
        ("researcher", "assessor"),
        ("pedagogue", "scribe"),
        ("assessor", "scribe"),
        ("scribe", "__end__"),
    ]
    
    # Build SVG elements
    svg_edges = ""
    for src_id, dst_id in edges:
        src = next(n for n in nodes if n["id"] == src_id)
        dst = next(n for n in nodes if n["id"] == dst_id)
        color = "#4ade80" if (src_id in completed_nodes) else "#555"
        
        # Draw curved/orthogonal paths or direct lines
        svg_edges += f'''<line x1="{src["x"]}" y1="{src["y"]+20}" x2="{dst["x"]}" y2="{dst["y"]-20}" 
            stroke="{color}" stroke-width="2" marker-end="url(#arrow{'-done' if src_id in completed_nodes else ''})"/>'''
    
    svg_nodes = ""
    for n in nodes:
        is_terminal = n["id"] in ("__start__", "__end__")
        is_active = (n["id"] == active_node)
        is_done = (n["id"] in completed_nodes)
        
        if is_active:
            fill, stroke, text_color, extra_class = "#22c55e", "#16a34a", "#fff", "active-node"
        elif is_done:
            fill, stroke, text_color, extra_class = "#bbf7d0", "#4ade80", "#166534", ""
        else:
            fill, stroke, text_color, extra_class = "#f0eeff", "#a5a0e6", "#333", ""
        
        width = 160 if not is_terminal else 100
        x_offset = width / 2
        
        if is_terminal:
            svg_nodes += f'''<g class="{extra_class}">
                <rect x="{n["x"]-x_offset}" y="{n["y"]-18}" width="{width}" height="36" rx="18" 
                    fill="{fill}" stroke="{stroke}" stroke-width="2"/>
                <text x="{n["x"]}" y="{n["y"]+5}" text-anchor="middle" fill="{text_color}" 
                    font-family="Inter,sans-serif" font-size="13" font-weight="bold">{n["label"]}</text>
            </g>'''
        else:
            svg_nodes += f'''<g class="{extra_class}">
                <rect x="{n["x"]-x_offset}" y="{n["y"]-20}" width="{width}" height="40" rx="8"
                    fill="{fill}" stroke="{stroke}" stroke-width="2"/>
                <text x="{n["x"]}" y="{n["y"]+5}" text-anchor="middle" fill="{text_color}" 
                    font-family="Inter,sans-serif" font-size="13" font-weight="600">{n["label"]}</text>
            </g>'''

    html = f'''<!DOCTYPE html>
<html><head><style>
  body {{ margin:0; background: #1a1a2e; display:flex; justify-content:center; padding: 10px 0; }}
  @keyframes pulse {{ 0%,100% {{ filter: drop-shadow(0 0 6px #22c55e); }} 50% {{ filter: drop-shadow(0 0 18px #4ade80); }} }}
  .active-node {{ animation: pulse 1.2s ease-in-out infinite; }}
</style></head>
<body>
<svg width="700" height="560" viewBox="0 0 700 560">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#555"/></marker>
    <marker id="arrow-done" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#4ade80"/></marker>
  </defs>
  {svg_edges}
  {svg_nodes}
</svg>
</body></html>'''
    return html

def generate_memory_graph_html(graph_memory: nx.DiGraph, head_pointer: str) -> str:
    """Generates an HTML page with D3.js to render the conversation DiGraph memory."""
    if not graph_memory or len(graph_memory.nodes) == 0:
        return "<html><body style='color:#94a3b8;font-family:sans-serif;padding:20px;background:#0b0f1a;'>No conversation history in this session yet. Start chatting!</body></html>"

    nodes_json = []
    for n in graph_memory.nodes:
        d = graph_memory.nodes[n]
        label = d.get("topic_label", "Root")
        is_head = "true" if n == head_pointer else "false"
        color = "#22c55e" if n == head_pointer else "#6366f1"
        
        dist = d.get("distilled_state", {})
        query_sum = dist.get("q", "Initial session start") if isinstance(dist, dict) else "Initial session start"
        status = dist.get("status", "active") if isinstance(dist, dict) else "active"
        timestamp = d.get("timestamp", "")
        # Escape quotes for JS
        query_sum = str(query_sum).replace('"', '\\"')
        status = str(status).replace('"', '\\"')
        
        nodes_json.append(f'{{ id: "{n}", label: "{label}", is_head: {is_head}, color: "{color}", q: "{query_sum}", status: "{status}", ts: "{timestamp}" }}')
    
    edges_json = []
    for u, v in graph_memory.edges:
        edges_json.append(f'{{ source: "{u}", target: "{v}" }}')

    html = f"""<!DOCTYPE html>
<html>
<head>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ background: #0b0f1a; margin: 0; overflow: hidden; font-family: Inter, sans-serif; }}
        svg {{ width: 100vw; height: 100vh; }}
        .node circle {{ stroke: #1a2236; stroke-width: 2px; }}
        .node text {{ font-size: 12px; fill: #f0f4ff; pointer-events: none; text-shadow: 0 1px 3px rgba(0,0,0,0.8); }}
        .link {{ fill: none; stroke: rgba(255,255,255,0.15); stroke-width: 2px; }}
        .head-glow {{ filter: drop-shadow(0 0 8px #22c55e); }}
        #tooltip {{ position: absolute; opacity: 0; background: #1a2236; color: #e2e8f0; padding: 12px; border-radius: 8px; font-size: 12px; pointer-events: none; border: 1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.5); max-width: 250px; line-height: 1.4; z-index: 1000; }}
    </style>
</head>
<body>
<div id="tooltip"></div>
<svg></svg>
<script>
    const nodes = [{",".join(nodes_json)}];
    const links = [{",".join(edges_json)}];
    
    const svg = d3.select("svg"),
          width = window.innerWidth,
          height = window.innerHeight;

    // Define arrowhead
    svg.append("defs").append("marker")
        .attr("id", "arrowhead")
        .attr("viewBox", "-0 -5 10 10")
        .attr("refX", 25)
        .attr("refY", 0)
        .attr("orient", "auto")
        .attr("markerWidth", 8)
        .attr("markerHeight", 8)
        .attr("xoverflow", "visible")
        .append("svg:path")
        .attr("d", "M 0,-5 L 10 ,0 L 0,5")
        .attr("fill", "rgba(255,255,255,0.3)")
        .style("stroke","none");

    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id).distance(100))
        .force("charge", d3.forceManyBody().strength(-400))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("y", d3.forceY(height/2).strength(0.1));

    const link = svg.append("g")
        .selectAll("path")
        .data(links)
        .join("path")
        .attr("class", "link")
        .attr("marker-end", "url(#arrowhead)");

    const tooltip = d3.select("#tooltip");

    const node = svg.append("g")
        .selectAll("g")
        .data(nodes)
        .join("g")
        .attr("class", d => d.is_head ? "node head-glow" : "node")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended))
        .on("mouseover", (event, d) => {{
            tooltip.transition().duration(200).style("opacity", 1);
            const tsFormat = d.ts ? new Date(d.ts).toLocaleString() : "Unknown";
            tooltip.html(`<b>Topic: ${{d.label}}</b><br/><hr style="border-color:#334155;margin:6px 0;" />
                          <span style="color:#94a3b8">Query:</span> ${{d.q}}<br/>
                          <span style="color:#94a3b8">Status:</span> <span style="color:#4ade80">${{d.status}}</span><br/>
                          <span style="color:#94a3b8">Time:</span> ${{tsFormat}}`)
                   .style("left", (event.pageX + 15) + "px")
                   .style("top", (event.pageY - 28) + "px");
        }})
        .on("mousemove", (event) => {{
            tooltip.style("left", (event.pageX + 15) + "px")
                   .style("top", (event.pageY - 28) + "px");
        }})
        .on("mouseout", () => {{
            tooltip.transition().duration(300).style("opacity", 0);
        }});

    node.append("circle")
        .attr("r", d => d.is_head ? 14 : 10)
        .attr("fill", d => d.color);

    node.append("text")
        .attr("dx", 18)
        .attr("dy", 4)
        .text(d => d.label + " | " + (d.q.length > 25 ? d.q.substring(0, 25) + '...' : d.q) + (d.is_head ? " (HEAD)" : ""));

    simulation.on("tick", () => {{
        link.attr("d", d => {{
            const dx = d.target.x - d.source.x,
                  dy = d.target.y - d.source.y,
                  dr = 0; // straight lines
            return `M${{d.source.x}},${{d.source.y}}A${{dr}},${{dr}} 0 0,1 ${{d.target.x}},${{d.target.y}}`;
        }});
        node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
    }});

    function dragstarted(event, d) {{
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
    }}
    function dragged(event, d) {{
        d.fx = event.x; d.fy = event.y;
    }}
    function dragended(event, d) {{
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null; d.fy = null;
    }}
</script>
</body>
</html>"""
    return html

def stream_tutor_pipeline(query: str, user_id: str = "default_user", session_id: str = "default_session", subject_id: str = "default_subject"):
    # Load session state graph memory from SQLite
    mem_sys = UserMemory(user_id=user_id)
    graph_memory, head_pointer, root_pointer = mem_sys.load_graph_memory(session_id)
    episodic_history = mem_sys.get_chat_history(session_id, limit=7)
    
    initial_state = TutorState(
        user_input=query,
        user_id=user_id,
        session_id=session_id,
        subject_id=subject_id,
        episodic_history=episodic_history,
        graph_memory=graph_memory,
        head_pointer=head_pointer,
        root_pointer=root_pointer,
        active_rag_context="",
        next_action="chat",
        final_output="",
        active_topic="",
        tutor_response="",
        quiz_response=""
    )
    
    for output in tutor_app.stream(initial_state):
        for node_name, state in output.items():
            yield node_name, state

def run_tutor_pipeline(query: str, user_id: str = "default_user", session_id: str = "default_session", subject_id: str = "default_subject"):
    # Load session state graph memory from SQLite
    mem_sys = UserMemory(user_id=user_id)
    graph_memory, head_pointer, root_pointer = mem_sys.load_graph_memory(session_id)
    episodic_history = mem_sys.get_chat_history(session_id, limit=7)
    
    initial_state = TutorState(
        user_input=query,
        user_id=user_id,
        session_id=session_id,
        subject_id=subject_id,
        episodic_history=episodic_history,
        graph_memory=graph_memory,
        head_pointer=head_pointer,
        root_pointer=root_pointer,
        active_rag_context="",
        next_action="chat",
        final_output="",
        active_topic="",
        tutor_response="",
        quiz_response=""
    )
    final_state = tutor_app.invoke(initial_state)
    return {
        "tutor_response": final_state["tutor_response"],
        "quiz_response": final_state["quiz_response"]
    }
