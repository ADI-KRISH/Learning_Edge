import os
import re
import logging
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from llama_index.llms.ollama import Ollama

import torch
try:
    torch.set_num_threads(1)
except RuntimeError:
    pass
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# Import our custom modules
from app.rag.retrieval import HybridRAGRetriever
from app.memory.user_memory import UserMemory
from app.utils.config import LLM_MODEL, BASE_DIR

# Set up logging
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
    active_rag_context: str
    raw_rag_context: str
    next_action: str
    final_output: str
    search_query: str
    rag_constraints: dict
    # Backward compatibility keys
    active_topic: str
    tutor_response: str
    quiz_response: str

# ==========================================
# 2. Singleton LLM Resources
# ==========================================
_retriever = None
_router_llm = None
_tutor_llm = None
_quiz_llm = None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = HybridRAGRetriever()
    return _retriever

def get_router_llm():
    """Fast, low-creativity LLM for strict intent routing."""
    global _router_llm
    if _router_llm is None:
        _router_llm = Ollama(
            model=LLM_MODEL, request_timeout=60.0, context_window=1024,
            additional_kwargs={"num_ctx": 1024, "temperature": 0.0, "num_predict": 128}
        )
    return _router_llm

def get_tutor_llm():
    """Creative LLM for contextual RAG-based explanations."""
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

_kg = None
def get_kg():
    """Singleton for the EducationalKnowledgeGraph."""
    global _kg
    if _kg is None:
        from app.graph.knowledge_graph import EducationalKnowledgeGraph
        _kg = EducationalKnowledgeGraph()
    return _kg


# ==========================================
# 3. Persona Prompt Builder (One-Shot Style)
# ==========================================
def _get_persona_prompt(style: str, level: str, mastered_topics: list) -> str:
    """
    Returns a rich one-shot persona instruction block for the Pedagogue.
    Each combination includes the target audience, a stylistic rule, and a
    concrete one-shot example of how the LLM should respond.
    """
    topics_str = ", ".join(mastered_topics) if mastered_topics else ""
    mastery_note = f"\nStudent has already mastered: {topics_str}. Do NOT re-explain these in depth." if topics_str else ""

    # ── Target Audience ──────────────────────────────────────────────────────
    if level == "Beginner":
        target = "Target audience: Middle school student or complete novice. Use simple, everyday language. Strictly avoid dense jargon."
    elif level == "Intermediate":
        target = "Target audience: Undergraduate college student. Use standard academic language. Introduce domain-specific terminology but define it in context."
    else:  # Advanced
        target = "Target audience: Graduate student or domain expert. Use sophisticated, highly technical jargon freely. Assume deep prior knowledge."

    # ── Stylistic Instruction + One-Shot Example ──────────────────────────────
    style_inst = ""

    if style == "step-by-step":
        if level == "Beginner":
            style_inst = (
                "Break explanations into simple numbered 1-2-3 steps. Avoid paragraphs. "
                "Each step must be one plain sentence a child can understand.\n\n"
                "ONE-SHOT EXAMPLE (topic: how plants make food):\n"
                "1. Sunlight hits the leaf.\n"
                "2. The leaf soaks up water from the soil.\n"
                "3. It mixes sunlight and water to make sugar for food.\n"
                "4. Extra oxygen gets released into the air.\n"
                "👉 Follow this exact numbered-list format in your response."
            )
        elif level == "Intermediate":
            style_inst = (
                "Break explanations into logical, numbered sequences with brief technical context per step.\n\n"
                "ONE-SHOT EXAMPLE (topic: photosynthesis):\n"
                "1. **Light absorption**: Chlorophyll in the thylakoid membrane absorbs photons.\n"
                "2. **Water splitting**: H₂O molecules are oxidised, releasing O₂ and electrons.\n"
                "3. **ATP synthesis**: The electron transport chain phosphorylates ADP → ATP.\n"
                "4. **Carbon fixation**: In the Calvin cycle, CO₂ is reduced using ATP and NADPH to form G3P.\n"
                "👉 Follow this numbered, bold-header format in your response."
            )
        else:  # Advanced
            style_inst = (
                "Provide strict algorithmic or process-driven numbered steps using precise technical vocabulary. "
                "Include mechanisms, equations, or complexity where relevant.\n\n"
                "ONE-SHOT EXAMPLE (topic: photosynthesis):\n"
                "1. **Photoexcitation of P680**: Photon absorption raises P680 to P680*, ejecting an electron.\n"
                "2. **Plastoquinone reduction**: The electron reduces PQ → PQH₂, pumping H⁺ across the thylakoid lumen.\n"
                "3. **Cytochrome b6f complex**: Mediates electron transfer from PQH₂ to plastocyanin, driving chemiosmosis.\n"
                "4. **Calvin Cycle (C3)**: RuBisCO catalyses CO₂ fixation; 3 CO₂ + 9 ATP + 6 NADPH → G3P (net).\n"
                "👉 Follow this exact depth and precision in your response."
            )

    elif style == "concise":
        if level == "Beginner":
            style_inst = (
                "Be extremely brief. Use simple words. One or two short sentences max per point. No jargon. No long paragraphs.\n\n"
                "ONE-SHOT EXAMPLE (topic: gravity):\n"
                "Gravity is a force that pulls things toward each other. "
                "The bigger the object, the stronger its pull. That's why we stay on the ground.\n"
                "👉 Match this brevity and simplicity in your response."
            )
        elif level == "Intermediate":
            style_inst = (
                "Be concise and direct. State the core academic facts without conversational filler. "
                "Use bullet points when listing multiple ideas.\n\n"
                "ONE-SHOT EXAMPLE (topic: CAP theorem):\n"
                "**CAP Theorem** states a distributed system can only guarantee two of three: "
                "**Consistency** (all nodes see the same data), **Availability** (every request gets a response), "
                "or **Partition Tolerance** (system survives network splits). In practice, partition tolerance is "
                "required, so the real trade-off is CP vs AP.\n"
                "👉 Match this density and precision in your response."
            )
        else:  # Advanced
            style_inst = (
                "Hyper-dense. Technical precision only. Zero fluff. Assume the reader knows all fundamentals. "
                "State trade-offs, failure modes, and edge cases where relevant.\n\n"
                "ONE-SHOT EXAMPLE (topic: CAP theorem):\n"
                "CAP (Brewer 2000; formalised Gilbert & Lynch 2002): under asynchronous networks, "
                "no system simultaneously satisfies C (linearisability), A (liveness for every non-failing node), "
                "and P (arbitrary message loss). PACELC extends this by quantifying latency/consistency trade-offs "
                "even absent partitions. Practical systems (Spanner: CP via TrueTime; DynamoDB: AP via eventual consistency).\n"
                "👉 Match this density in your response."
            )

    elif style == "detailed":
        if level == "Beginner":
            style_inst = (
                "Be very thorough but keep the language completely simple. "
                "Explain every small detail so a novice won't get lost. Use analogies from everyday life. "
                "Write in a friendly, encouraging tone.\n\n"
                "ONE-SHOT EXAMPLE (topic: how WiFi works):\n"
                "**WiFi** is like an invisible delivery system for information. Imagine your router is a post office. "
                "When you watch a YouTube video, your laptop sends a letter to the post office saying 'I want this video.' "
                "The post office (router) goes to the internet, fetches the video, and sends it back as tiny invisible "
                "radio waves — like magic signals in the air. Your laptop's WiFi card catches those waves and turns "
                "them back into the video you see on screen. The closer you are to the router, the stronger the signal "
                "and the faster your video loads.\n"
                "👉 Use this level of detail and analogy in your response."
            )
        elif level == "Intermediate":
            style_inst = (
                "Be exhaustively thorough, capturing nuances, context, and academic depth at a college level. "
                "Use headers, bullet points, and bold key terms to organise the explanation.\n\n"
                "ONE-SHOT EXAMPLE (topic: MapReduce):\n"
                "## MapReduce\n"
                "**MapReduce** (Dean & Ghemawat, 2004) is a programming model for processing large datasets "
                "on commodity hardware clusters.\n\n"
                "### Phases\n"
                "- **Map**: The master node splits input into independent chunks and assigns them to workers. "
                "Each worker applies a user-defined `map(k1,v1) → list(k2,v2)` function.\n"
                "- **Shuffle & Sort**: The framework groups all intermediate values by key across workers.\n"
                "- **Reduce**: Workers apply `reduce(k2, list(v2)) → list(v3)` to aggregate per-key results.\n\n"
                "### Example — Word Count\n"
                "Input: `'the cat sat on the mat'` → Map: `[(the,1),(cat,1),(sat,1)...]` → "
                "Shuffle: `{the:[1,1], cat:[1]...}` → Reduce: `{the:2, cat:1...}`\n"
                "👉 Follow this structure and depth in your response."
            )
        else:  # Advanced
            style_inst = (
                "Provide deep, comprehensive coverage including theoretical background, algorithmic complexity, "
                "system design implications, limitations, and advanced edge cases. "
                "Reference seminal papers or known benchmarks where appropriate.\n\n"
                "ONE-SHOT EXAMPLE (topic: MapReduce):\n"
                "**MapReduce** (Dean & Ghemawat, OSDI 2004) abstracts distributed computation as two higher-order "
                "functions: `map: (k1,v1) → [(k2,v2)]` and `reduce: (k2,[v2]) → [v3]`. The runtime manages "
                "data partitioning, task scheduling, fault tolerance via task re-execution, and locality optimisation "
                "(co-locating map tasks with HDFS data blocks to minimise network I/O). "
                "**Limitations**: MapReduce is ill-suited for iterative algorithms (ML, graph computation) due to "
                "high disk I/O between stages. This drove Spark (Zaharia et al., NSDI 2012) with in-memory RDDs, "
                "achieving 10–100× speedups on iterative workloads. Combiner functions serve as a local reduce to "
                "reduce shuffle volume (O(n) → O(n/k) where k = cluster size).\n"
                "👉 Follow this theoretical depth and citation style in your response."
            )

    else:  # default / balanced
        if level == "Beginner":
            style_inst = (
                "Provide a balanced, clear, and encouraging explanation. Use simple language. "
                "Briefly use an analogy where it helps understanding.\n\n"
                "ONE-SHOT EXAMPLE (topic: databases):\n"
                "A **database** is like a super-organised filing cabinet for a computer. "
                "Instead of shuffling through messy paper files, a database lets you find any piece of information "
                "instantly. For example, when you log into an app, it checks a database to find your account details "
                "in milliseconds. Databases are used everywhere — from your contacts list to Netflix's movie library.\n"
                "👉 Follow this clarity and warmth in your response."
            )
        elif level == "Intermediate":
            style_inst = (
                "Provide a balanced, clear, and well-structured explanation at a college academic standard. "
                "Define key terms, explain mechanisms, and include a brief concrete example.\n\n"
                "ONE-SHOT EXAMPLE (topic: database indexing):\n"
                "A **database index** is a data structure (typically a B-tree or hash map) that allows the database "
                "engine to locate rows without scanning the entire table. Without an index, a `SELECT` on a 10M-row "
                "table is O(n). With a B-tree index on the queried column, it becomes O(log n). "
                "**Trade-off**: indexes speed up reads but slow down writes (INSERT/UPDATE must also update the index) "
                "and consume additional storage.\n"
                "👉 Follow this balance of clarity and technical precision in your response."
            )
        else:  # Advanced
            style_inst = (
                "Provide a balanced, technically rigorous explanation. Assume expert-level knowledge. "
                "Include design rationale, trade-offs, and practical implications.\n\n"
                "ONE-SHOT EXAMPLE (topic: database indexing):\n"
                "**B-tree indexes** (O(log n) lookup, Bayer & McCreight 1972) remain the default in PostgreSQL/MySQL "
                "due to their support for range queries, unlike hash indexes (O(1) point lookup, no range support). "
                "**LSM-tree** (Log-Structured Merge-tree, Cassandra, RocksDB) trades read amplification for write "
                "performance by batching writes in memory (MemTable) and merging SSTables during compaction. "
                "Index selectivity, covering indexes, and partial indexes are key tuning levers. "
                "Bitmap indexes suit low-cardinality OLAP columns; GIN/GiST indexes handle full-text and geometric data.\n"
                "👉 Follow this rigour and specificity in your response."
            )

    return f"{target}\n\n{style_inst}{mastery_note}"


# ==========================================
# 4. Helper: Build Truncated History String
# ==========================================
MAX_MSG_CHARS = 300  # Truncate individual messages longer than this

def _build_history_str(episodic_history: list) -> str:
    """
    Converts the episodic history list into a formatted string for LLM prompts.
    Truncates individual messages that are too long to keep context window lean.
    """
    if not episodic_history:
        return "No recent conversation history."
    lines = []
    for msg in episodic_history:
        role = "Student" if msg["role"] == "user" else "Tutor"
        content = msg["content"].strip().replace("\n", " ")
        if len(content) > MAX_MSG_CHARS:
            content = content[:MAX_MSG_CHARS] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ==========================================
# 5. LangGraph Agent Nodes
# ==========================================

def supervisor_node(state: TutorState) -> TutorState:
    """
    Routes the user query to the correct next agent using:
    - The user's raw query
    - Recent episodic chat history (truncated)
    No graph memory or embeddings used.
    """
    print("[Supervisor] Classifying intent from query + episodic history...")
    user_input = state["user_input"]
    history_str = _build_history_str(state.get("episodic_history", []))

    router_prompt = f"""You are a strict routing agent for an AI tutoring system.
Analyze the Student Query and the Recent Chat History to classify intent into exactly one of:
- "research"  → student is asking a factual question, requesting an explanation, or saying yes/sure to elaborate on a topic
- "assess"    → student explicitly wants a quiz, test, or to be questioned
- "chat"      → purely casual small talk or a simple yes/no with no learning intent

RULES:
1. Only use "assess" if the student EXPLICITLY asks to be quizzed/tested.
2. Any question about HOW, WHAT, WHY, examples, or explanations → "research".
3. Output ONLY minified JSON. No markdown, no extra text.

Recent Chat History:
{history_str}

Student Query: "{user_input}"

Output ONLY: {{"intent": "chat|research|assess", "search_query": "standalone topic query"}}"""

    llm = get_router_llm()
    next_action = "research"
    search_query = user_input

    try:
        import json
        res = str(llm.complete(router_prompt)).strip()
        if res.startswith("```"):
            res = res.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        res_data = json.loads(res)
        next_action = res_data.get("intent", "research")
        search_query = res_data.get("search_query") or user_input
    except Exception as e:
        print(f"  [WARN] Router LLM failed: {e}. Defaulting to research.")

    # ── Heuristic overrides ────────────────────────────────────────────────
    query_clean = user_input.lower().strip()
    quiz_keywords = [
        "quiz", "test me", "assess", "assessment", "examine", "exam",
        "question me", "questions on", "evaluate me", "grade me",
        "quizzes", "practice questions", "trivia", "quiz me", "retake"
    ]
    explanation_keywords = [
        "what", "how", "why", "explain", "tell", "can you", "elaborate",
        "example", "give me", "show me", "describe", "define",
        "difference", "compare", "summarize", "walkthrough"
    ]
    has_quiz_keyword = any(w in query_clean for w in quiz_keywords)
    has_explanation_keyword = any(w in query_clean for w in explanation_keywords)

    if has_quiz_keyword:
        print("  [Override] Quiz keyword detected → ASSESS")
        next_action = "assess"
    elif next_action == "assess" and not has_quiz_keyword:
        print("  [Override] LLM returned 'assess' but no quiz keyword found → RESEARCH")
        next_action = "research"
    elif next_action == "chat" and (len(query_clean.split()) > 4 or has_explanation_keyword):
        print("  [Override] Long/explanatory query misrouted as chat → RESEARCH")
        next_action = "research"

    logger.info(f"Query: '{user_input[:60]}' → intent={next_action} | search_query='{search_query}'")
    print(f"  → Supervisor: {next_action.upper()} | search_query: '{search_query}'")

    state["next_action"] = next_action
    state["search_query"] = search_query
    state["rag_constraints"] = {}
    return state


def researcher_node(state: TutorState) -> TutorState:
    """
    Performs Hybrid RAG retrieval (BM25 + ChromaDB) strictly filtered
    to the student's active curriculum part, then LLM-compresses the context.
    """
    print("[Researcher] Starting Hybrid RAG Pipeline...")
    user_input = state["user_input"]
    subject_id = state.get("subject_id", "default_subject")

    # Retrieve active study state to enforce strict curriculum filtering
    mem_sys = UserMemory(user_id=state["user_id"])
    study_state = mem_sys.get_active_study_state(subject_id)
    active_doc_id = study_state.get("active_doc_id")
    active_part_index = study_state.get("active_part_index")

    constraints = {}
    if active_doc_id:
        constraints["doc_id"] = active_doc_id
        constraints["part_index"] = active_part_index
        print(f"  [Researcher] Curriculum filter: doc={active_doc_id}, part={active_part_index}")
    state["rag_constraints"] = constraints

    # KG query enrichment
    search_query = state.get("search_query") or user_input
    enriched_query = search_query
    try:
        from app.graph.knowledge_graph import EducationalKnowledgeGraph
        kg = EducationalKnowledgeGraph()
        active_topic = state.get("active_topic", "")
        if active_topic and active_topic in kg.G.nodes:
            prereqs = list(kg.G.predecessors(active_topic))
            if prereqs:
                enriched_query = f"query: [{search_query}] prerequisites: [{', '.join(prereqs)}]"
                print(f"  [Researcher] KG enriched query: {enriched_query}")
    except Exception as e:
        print(f"  [Researcher] KG enrichment skipped: {e}")

    # Hybrid retrieval
    from app.rag.retrieval import get_retriever
    retriever = get_retriever()
    docs = retriever.retrieve(enriched_query, subject_id=subject_id, constraints=constraints)
    raw_chunks = [d.page_content for d in docs]

    if not raw_chunks:
        state["active_rag_context"] = '{"error": "not_found"}'
        state["raw_rag_context"] = ""
        return state

    source_text = "\n\n---\n\n".join(raw_chunks)
    state["raw_rag_context"] = source_text
    source_text = source_text[:3000]

    # LLM context compression
    compress_prompt = f"""You are a data extractor. Read the Source Text. Extract ONLY facts relevant to the User Query.
Output strictly as minified JSON. Use short key-value pairs. If the answer is not in the text, output {{"error": "not_found"}}.

User Query: {user_input}

Source Text:
{source_text}

Output ONLY minified JSON:"""

    try:
        llm = get_tutor_llm()
        response = llm.complete(compress_prompt)
        compressed = str(response).strip()
        compressed = re.sub(r'^```json\s*', '', compressed)
        compressed = re.sub(r'^```\s*', '', compressed)
        compressed = re.sub(r'\s*```$', '', compressed)
        state["active_rag_context"] = compressed
        print(f"  [Researcher] Compressed context: {compressed[:100]}...")
    except Exception as e:
        print(f"  [Researcher] Compression failed: {e}")
        state["active_rag_context"] = source_text[:1000]

    return state


def pedagogue_node(state: TutorState) -> TutorState:
    """
    Formulates a personalized, style-adaptive explanation using:
    - Active RAG context
    - Episodic chat history (truncated)
    - Student profile (style + level)
    - Active curriculum part title
    """
    print("[Pedagogue] Formulating personalized explanation...")
    user_input = state["user_input"]
    active_rag_context = state.get("active_rag_context", "")
    next_action = state["next_action"]
    history_str = _build_history_str(state.get("episodic_history", []))

    memory_sys = UserMemory(user_id=state.get("user_id", "default_user"))
    subject_id = state.get("subject_id", "default_subject")
    profile = memory_sys.get_semantic_memory(subject_id=subject_id)
    style = profile.get("preferred_style", "default")
    academic_level = profile.get("academic_level", "Intermediate")
    completed_topics = profile.get("completed_topics", [])

    persona = _get_persona_prompt(style, academic_level, completed_topics)

    study_state = memory_sys.get_active_study_state(subject_id)
    active_doc_id = study_state.get("active_doc_id")
    active_part_index = study_state.get("active_part_index", 1)
    part_details = memory_sys.get_document_part(active_doc_id, active_part_index) if active_doc_id else None
    part_title = part_details.get("part_title", "General Syllabus") if part_details else "General Syllabus"

    if next_action == "chat":
        prompt = f"""You are a friendly offline AI Tutor for subject '{subject_id}'.
The student sent a casual message. Keep your response conversational and warm. Maximum 2-3 sentences.
Never invent facts. If they ask something factual, gently redirect them to ask properly.

Recent History:
{history_str}

Student: "{user_input}"
Tutor:"""
        llm = get_tutor_llm()

    else:
        if active_rag_context and "not_found" not in active_rag_context:
            system_prompt = f"""You are an expert offline AI Tutor for subject '{subject_id}'.
The student is currently studying: '{part_title}' (Part {active_part_index}).
Base your explanation on the RAG Source below. It contains the exact content of this curriculum part.

RAG Source: {active_rag_context}

CRITICAL INSTRUCTIONS:
1. Write your explanation following the STUDENT PROFILE & PERSONALIZATION style exactly.
2. End your response by reminding the student they are on Part {active_part_index} ('{part_title}') and inviting them to type 'quiz me' when ready."""
        else:
            system_prompt = f"""You are a helpful academic guide for subject '{subject_id}'.
The student's question is outside the active curriculum part: '{part_title}' (Part {active_part_index}).
Answer accurately using general knowledge. Do not invent specific metrics.

CRITICAL INSTRUCTIONS:
1. Write following the STUDENT PROFILE & PERSONALIZATION style.
2. After answering, guide the student back to '{part_title}'."""

        prompt = f"""{system_prompt}

STUDENT PROFILE & PERSONALIZATION:
{persona}

RECENT CHAT HISTORY:
{history_str}

STUDENT QUESTION: "{user_input}"

Tutor:"""
        llm = get_tutor_llm()

    if len(prompt) > 6000:
        print(f"  [WARN] Prompt too long ({len(prompt)} chars). Truncating RAG context.")
        if active_rag_context:
            active_rag_context = active_rag_context[:800]
        if next_action != "chat":
            prompt = f"""You are an expert AI Tutor.
PERSONALIZATION: {persona}
HISTORY: {history_str}
RAG SOURCE: {active_rag_context if active_rag_context else 'None'}
QUESTION: "{user_input}"
Base facts on RAG source if available. If topic is missing, answer from general knowledge but note it is not in their documents."""

    try:
        response = llm.complete(prompt)
        output_text = str(response)

        if next_action != "chat" and (not active_rag_context or "not_found" in active_rag_context):
            preamble = "Currently there are no documents in the database covering this topic, but here's my explanation:\n\n"
            if not output_text.startswith("Currently"):
                output_text = preamble + output_text

        if next_action != "chat":
            raw_rag = state.get("raw_rag_context", "") or "No relevant documents found."
            rag_append = (
                f"\n\n:::rag 🔍 View Retrieved RAG Context (Raw Chunks)\n{raw_rag}\n:::\n\n"
                f":::rag 🧠 LLM Compressed JSON Context\n{active_rag_context}\n:::"
            )
            output_text += rag_append

    except Exception as e:
        import httpx
        if "connect" in str(e).lower() or isinstance(e, (ConnectionError, httpx.ConnectError)):
            raise OllamaConnectionError("Failed to connect to Ollama.") from e
        raise e

    state["final_output"] = output_text
    state["tutor_response"] = output_text
    return state


def assessor_node(state: TutorState) -> TutorState:
    """
    Generates an MCQ quiz from the active curriculum part content.
    Quiz topic is always the real part title, never the raw user input.
    """
    print("[Assessor] Generating quiz from curriculum context...")
    active_rag_context = state.get("active_rag_context", "")

    memory_sys = UserMemory(user_id=state["user_id"])
    subject_id = state.get("subject_id", "default_subject")
    profile = memory_sys.get_semantic_memory(subject_id=subject_id)
    difficulty = profile.get("quiz_difficulty", "Medium")
    num_questions = profile.get("quiz_questions", 3)

    study_state = memory_sys.get_active_study_state(subject_id)
    active_doc_id = study_state.get("active_doc_id")
    active_part_index = study_state.get("active_part_index", 1)
    part_details = memory_sys.get_document_part(active_doc_id, active_part_index) if active_doc_id else None
    part_title = part_details.get("part_title", "Current Chapter") if part_details else "Current Chapter"

    prompt = f"""You are a strict academic Quiz Agent. Generate exactly {num_questions} multiple choice questions at {difficulty} difficulty.
Questions MUST be drawn exclusively from the study materials for the active subtopic: '{part_title}' (Part {active_part_index}).

Study Materials:
{active_rag_context if active_rag_context else 'No study materials available.'}

ANTI-HALLUCINATION RULES:
1. Generate questions ONLY from facts present in the Study Materials.
2. Do NOT invent questions about topics not mentioned in the text.
3. If Study Materials are empty or irrelevant, output an empty JSON array [].

Output ONLY valid JSON. No markdown, no extra text:
[
  {{
    "question": "What is ...?",
    "options": ["option A", "option B", "option C", "option D"],
    "correct": 0,
    "explanation": "Brief explanation from the text."
  }}
]"""

    if len(prompt) > 6000:
        active_rag_context = active_rag_context[:1000]
        prompt = f"""Quiz Agent: generate {num_questions} MCQ ({difficulty}) for '{part_title}' (Part {active_part_index}).
Materials: {active_rag_context}
Output ONLY JSON: [{{"question":"...","options":["A","B","C","D"],"correct":0,"explanation":"..."}}]"""

    llm = get_quiz_llm()
    try:
        response = llm.complete(prompt)
        output_text = str(response)
        state["quiz_response"] = output_text

        chat_msg = "🎯 **Quiz Ready!** I have prepared a personalized quiz for you. Please navigate to the **Quiz** tab in the sidebar to start."
        if active_rag_context:
            chat_msg += f"\n\n:::rag 🔍 View Retrieved RAG Context (Cosine Sim & BM25)\n{active_rag_context}\n:::"

        state["tutor_response"] = chat_msg
        state["final_output"] = chat_msg
        # Store the real topic so scribe can use it
        state["active_topic"] = part_title

    except Exception as e:
        import httpx
        if "connect" in str(e).lower() or isinstance(e, (ConnectionError, httpx.ConnectError)):
            raise OllamaConnectionError("Failed to connect to Ollama.") from e
        raise e

    return state


def scribe_node(state: TutorState) -> TutorState:
    """
    Persists the conversation turn to SQLite chat_history and updates
    the in-state episodic_history. No graph memory involved.
    """
    print("[Scribe] Persisting conversation turn to SQLite...")
    user_input = state["user_input"]
    final_output = state["final_output"]
    session_id = state["session_id"]
    next_action = state["next_action"]

    mem_sys = UserMemory(user_id=state["user_id"])

    # Persist both turns
    mem_sys.add_chat_message("user", user_input, session_id)

    if next_action == "assess":
        # Use the real part title as topic (set by assessor_node)
        topic_label = state.get("active_topic", "Current Chapter")
        quiz_status = f"Generated quiz on topic: {topic_label}"
        mem_sys.add_chat_message("assistant", quiz_status, session_id)
        final_assistant_content = quiz_status
    else:
        mem_sys.add_chat_message("assistant", final_output, session_id)
        final_assistant_content = final_output

    # Update in-state episodic history (truncated)
    history = list(state.get("episodic_history", []))
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": final_assistant_content[:MAX_MSG_CHARS]})
    # Keep last 20 messages in state (10 turns)
    state["episodic_history"] = history[-20:]

    # Derive active_topic for non-quiz turns from first 2 meaningful words
    if next_action != "assess":
        words = user_input.strip().split()
        raw_label = "_".join(words[:2]).strip("?.!,:;()") if words else "General"
        state["active_topic"] = raw_label or "General"

    print(f"  [Scribe] Saved turn. intent={next_action}, topic={state['active_topic']}")
    return state


# ==========================================
# 6. Build the LangGraph
# ==========================================
def route_post_supervisor(state: TutorState) -> str:
    action = state.get("next_action", "research")
    if action == "chat":
        return "pedagogue"
    return "researcher"  # both 'research' and 'assess' need RAG first

def route_post_researcher(state: TutorState) -> str:
    action = state.get("next_action", "research")
    if action == "assess":
        return "assessor"
    return "pedagogue"

workflow = StateGraph(TutorState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("pedagogue", pedagogue_node)
workflow.add_node("assessor", assessor_node)
workflow.add_node("scribe", scribe_node)
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", route_post_supervisor, {"researcher": "researcher", "pedagogue": "pedagogue"})
workflow.add_conditional_edges("researcher", route_post_researcher, {"assessor": "assessor", "pedagogue": "pedagogue"})
workflow.add_edge("pedagogue", "scribe")
workflow.add_edge("assessor", "scribe")
workflow.add_edge("scribe", END)
tutor_app = workflow.compile()


# ==========================================
# 7. Visualization & Execution
# ==========================================
def generate_state_graph_html(active_node=None, completed_nodes=None):
    """Generates an SVG visualization of the multi-agent pipeline."""
    if completed_nodes is None:
        completed_nodes = []

    nodes = [
        {"id": "__start__",  "label": "START",              "x": 350, "y": 30},
        {"id": "supervisor", "label": "🚦 Supervisor",      "x": 350, "y": 110},
        {"id": "researcher", "label": "📚 RAG Researcher",  "x": 200, "y": 200},
        {"id": "pedagogue",  "label": "🧠 Pedagogue Tutor", "x": 200, "y": 320},
        {"id": "assessor",   "label": "📝 Quiz Assessor",   "x": 500, "y": 320},
        {"id": "scribe",     "label": "✍️ Scribe Memory",   "x": 350, "y": 440},
        {"id": "__end__",    "label": "END",                "x": 350, "y": 520},
    ]
    edges = [
        ("__start__", "supervisor"), ("supervisor", "researcher"), ("supervisor", "pedagogue"),
        ("researcher", "pedagogue"), ("researcher", "assessor"),
        ("pedagogue", "scribe"), ("assessor", "scribe"), ("scribe", "__end__"),
    ]

    svg_edges = ""
    for src_id, dst_id in edges:
        src = next(n for n in nodes if n["id"] == src_id)
        dst = next(n for n in nodes if n["id"] == dst_id)
        color = "#4ade80" if src_id in completed_nodes else "#555"
        svg_edges += f'<line x1="{src["x"]}" y1="{src["y"]+20}" x2="{dst["x"]}" y2="{dst["y"]-20}" stroke="{color}" stroke-width="2" marker-end="url(#arrow{"-done" if src_id in completed_nodes else ""})"/>'

    svg_nodes = ""
    for n in nodes:
        is_terminal = n["id"] in ("__start__", "__end__")
        is_active = n["id"] == active_node
        is_done = n["id"] in completed_nodes
        if is_active:
            fill, stroke, text_color, extra_class = "#22c55e", "#16a34a", "#fff", "active-node"
        elif is_done:
            fill, stroke, text_color, extra_class = "#bbf7d0", "#4ade80", "#166534", ""
        else:
            fill, stroke, text_color, extra_class = "#f0eeff", "#a5a0e6", "#333", ""
        width = 160 if not is_terminal else 100
        x_offset = width / 2
        if is_terminal:
            svg_nodes += f'<g class="{extra_class}"><rect x="{n["x"]-x_offset}" y="{n["y"]-18}" width="{width}" height="36" rx="18" fill="{fill}" stroke="{stroke}" stroke-width="2"/><text x="{n["x"]}" y="{n["y"]+5}" text-anchor="middle" fill="{text_color}" font-family="Inter,sans-serif" font-size="13" font-weight="bold">{n["label"]}</text></g>'
        else:
            svg_nodes += f'<g class="{extra_class}"><rect x="{n["x"]-x_offset}" y="{n["y"]-20}" width="{width}" height="40" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"/><text x="{n["x"]}" y="{n["y"]+5}" text-anchor="middle" fill="{text_color}" font-family="Inter,sans-serif" font-size="13" font-weight="600">{n["label"]}</text></g>'

    return f'''<!DOCTYPE html><html><head><style>
  body {{ margin:0; background: #1a1a2e; display:flex; justify-content:center; padding: 10px 0; }}
  @keyframes pulse {{ 0%,100% {{ filter: drop-shadow(0 0 6px #22c55e); }} 50% {{ filter: drop-shadow(0 0 18px #4ade80); }} }}
  .active-node {{ animation: pulse 1.2s ease-in-out infinite; }}
</style></head><body>
<svg width="700" height="560" viewBox="0 0 700 560">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#555"/></marker>
    <marker id="arrow-done" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#4ade80"/></marker>
  </defs>
  {svg_edges}
  {svg_nodes}
</svg></body></html>'''


def stream_tutor_pipeline(query: str, user_id: str = "default_user", session_id: str = "default_session", subject_id: str = "default_subject"):
    mem_sys = UserMemory(user_id=user_id)
    # Load last 10 turns (20 messages) for episodic context
    episodic_history = mem_sys.get_chat_history(session_id, limit=10)

    initial_state = TutorState(
        user_input=query,
        user_id=user_id,
        session_id=session_id,
        subject_id=subject_id,
        active_part_index=1,
        active_doc_id="",
        episodic_history=episodic_history,
        active_rag_context="",
        raw_rag_context="",
        next_action="chat",
        final_output="",
        search_query="",
        rag_constraints={},
        active_topic="",
        tutor_response="",
        quiz_response=""
    )

    for output in tutor_app.stream(initial_state):
        for node_name, state in output.items():
            yield node_name, state


def run_tutor_pipeline(query: str, user_id: str = "default_user", session_id: str = "default_session", subject_id: str = "default_subject"):
    mem_sys = UserMemory(user_id=user_id)
    episodic_history = mem_sys.get_chat_history(session_id, limit=10)

    initial_state = TutorState(
        user_input=query,
        user_id=user_id,
        session_id=session_id,
        subject_id=subject_id,
        active_part_index=1,
        active_doc_id="",
        episodic_history=episodic_history,
        active_rag_context="",
        raw_rag_context="",
        next_action="chat",
        final_output="",
        search_query="",
        rag_constraints={},
        active_topic="",
        tutor_response="",
        quiz_response=""
    )
    final_state = tutor_app.invoke(initial_state)
    return {
        "tutor_response": final_state["tutor_response"],
        "quiz_response": final_state["quiz_response"]
    }
