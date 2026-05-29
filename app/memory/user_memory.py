import sqlite3
import json
import os
from app.utils.config import MEMORY_DIR

class UserMemory:
    def __init__(self, user_id="default_user"):
        self.user_id = user_id
        self.db_path = os.path.join(MEMORY_DIR, "user_memory.db")
        self._init_db()

    def _init_db(self):
        """Initializes the SQLite database with multi-subject, multi-session sequential memory tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 1. Subjects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS subjects (
                subject_id TEXT PRIMARY KEY,
                subject_name TEXT,
                curriculum_file TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 2. Chat sessions table (subject-aware)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                subject_id TEXT,
                title TEXT,
                graph_memory TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE
            )
        ''')
        
        # 3. Short term memory table (chat history)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
            )
        ''')
        
        # 4. Semantic/Long term memory table (per user per subject)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_memory (
                user_id TEXT,
                subject_id TEXT,
                preferred_style TEXT,
                academic_level TEXT, -- Beginner, Intermediate, Advanced
                weak_topics TEXT, -- stored as JSON string
                completed_topics TEXT, -- stored as JSON string
                PRIMARY KEY (user_id, subject_id),
                FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE
            )
        ''')
        
        # 5. Quiz history tracking table (subject-aware)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quiz_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                subject_id TEXT,
                topic TEXT,
                score INTEGER,
                total INTEGER,
                passed INTEGER, -- 1 = true, 0 = false
                questions_json TEXT, -- JSON string representing the questions array
                user_answers_json TEXT, -- JSON string representing selected answers
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE
            )
        ''')

        # 6. NEW TABLE: Curriculum Documents
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS curriculum_documents (
                doc_id TEXT PRIMARY KEY,
                subject_id TEXT,
                doc_title TEXT,
                doc_content TEXT,
                total_parts INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE
            )
        ''')

        # 7. NEW TABLE: Document Parts (sequential subtopics)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_parts (
                part_id TEXT PRIMARY KEY,
                doc_id TEXT,
                part_index INTEGER, -- sequential ordering: 1, 2, 3...
                part_title TEXT,
                part_content TEXT,
                FOREIGN KEY (doc_id) REFERENCES curriculum_documents(doc_id) ON DELETE CASCADE
            )
        ''')

        # 8. NEW TABLE: Subject Study State (Sequential progression tracker)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS subject_study_state (
                subject_id TEXT PRIMARY KEY,
                active_doc_id TEXT,
                active_part_index INTEGER DEFAULT 1,
                status TEXT DEFAULT 'studying', -- studying, completed
                FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE,
                FOREIGN KEY (active_doc_id) REFERENCES curriculum_documents(doc_id) ON DELETE CASCADE
            )
        ''')
        
        # Ensure default subject and default session exists for backward compatibility/initial start
        cursor.execute("SELECT 1 FROM subjects WHERE subject_id = 'default_subject'")
        if not cursor.fetchone():
            cursor.execute("INSERT INTO subjects (subject_id, subject_name, curriculum_file) VALUES ('default_subject', 'General Knowledge', '')")
            
        cursor.execute("SELECT 1 FROM chat_sessions WHERE session_id = 'default_session'")
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO chat_sessions (session_id, user_id, subject_id, title) VALUES ('default_session', ?, 'default_subject', 'Default Chat Session')",
                (self.user_id,)
            )
            
        conn.commit()
        conn.close()

    # --- Subject Management ---
    def add_subject(self, subject_id: str, subject_name: str, curriculum_file: str = ""):
        """Registers a new subject and initializes its semantic memory profile."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO subjects (subject_id, subject_name, curriculum_file) VALUES (?, ?, ?)",
            (subject_id, subject_name, curriculum_file)
        )
        conn.commit()
        conn.close()
        
        # Initialize semantic memory for this subject
        memory = {
            "preferred_style": "default",
            "academic_level": "Intermediate",
            "weak_topics": [],
            "completed_topics": []
        }
        self._save_semantic_memory(memory, subject_id)
        
    def get_subjects(self):
        """Retrieves list of all enrolled subjects."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT subject_id, subject_name, curriculum_file, created_at FROM subjects ORDER BY created_at ASC")
        rows = cursor.fetchall()
        conn.close()
        return [{"subject_id": r[0], "subject_name": r[1], "curriculum_file": r[2], "created_at": r[3]} for r in rows]

    def delete_subject(self, subject_id: str):
        """Cleanly deletes a subject and all associated chat sessions, history, and records."""
        if subject_id == "default_subject":
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM subjects WHERE subject_id=?", (subject_id,))
        cursor.execute("DELETE FROM chat_sessions WHERE subject_id=?", (subject_id,))
        cursor.execute("DELETE FROM semantic_memory WHERE subject_id=?", (subject_id,))
        cursor.execute("DELETE FROM quiz_history WHERE subject_id=?", (subject_id,))
        cursor.execute("DELETE FROM curriculum_documents WHERE subject_id=?", (subject_id,))
        cursor.execute("DELETE FROM subject_study_state WHERE subject_id=?", (subject_id,))
        conn.commit()
        conn.close()

    # --- Sequential Study Curriculum APIs ---
    def add_curriculum_document(self, doc_id: str, subject_id: str, doc_title: str, doc_content: str, total_parts: int):
        """Registers a curriculum document in SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO curriculum_documents (doc_id, subject_id, doc_title, doc_content, total_parts)
            VALUES (?, ?, ?, ?, ?)
        ''', (doc_id, subject_id, doc_title, doc_content, total_parts))
        conn.commit()
        conn.close()

    def add_document_part(self, part_id: str, doc_id: str, part_index: int, part_title: str, part_content: str):
        """Slices and persists a sequential subtopic partition."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO document_parts (part_id, doc_id, part_index, part_title, part_content)
            VALUES (?, ?, ?, ?, ?)
        ''', (part_id, doc_id, part_index, part_title, part_content))
        conn.commit()
        conn.close()

    def get_curriculum_documents(self, subject_id: str):
        """Fetches all documents associated with a subject."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id, doc_title, total_parts, created_at FROM curriculum_documents WHERE subject_id=? ORDER BY created_at DESC", (subject_id,))
        rows = cursor.fetchall()
        conn.close()
        return [{"doc_id": r[0], "doc_title": r[1], "total_parts": r[2], "created_at": r[3]} for r in rows]

    def get_document_parts(self, doc_id: str):
        """Fetches all sequential parts of a document in correct order."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT part_id, part_index, part_title, part_content FROM document_parts WHERE doc_id=? ORDER BY part_index ASC", (doc_id,))
        rows = cursor.fetchall()
        conn.close()
        return [{"part_id": r[0], "part_index": r[1], "part_title": r[2], "part_content": r[3]} for r in rows]

    def get_document_part(self, doc_id: str, part_index: int):
        """Fetches a specific part of a document by index."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT part_id, part_title, part_content FROM document_parts WHERE doc_id=? AND part_index=?", (doc_id, part_index))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {"part_id": row[0], "part_title": row[1], "part_content": row[2]}
        return None

    def get_active_study_state(self, subject_id: str):
        """Gets or initializes the active progression state for a subject."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT active_doc_id, active_part_index, status FROM subject_study_state WHERE subject_id=?", (subject_id,))
        row = cursor.fetchone()
        
        if row:
            conn.close()
            return {"active_doc_id": row[0], "active_part_index": row[1], "status": row[2]}
        else:
            # Look up the most recent doc for this subject to auto-initialize
            cursor.execute("SELECT doc_id FROM curriculum_documents WHERE subject_id=? ORDER BY created_at DESC LIMIT 1", (subject_id,))
            doc_row = cursor.fetchone()
            active_doc_id = doc_row[0] if doc_row else None
            
            cursor.execute(
                "INSERT OR IGNORE INTO subject_study_state (subject_id, active_doc_id, active_part_index, status) VALUES (?, ?, 1, 'studying')",
                (subject_id, active_doc_id)
            )
            conn.commit()
            conn.close()
            return {"active_doc_id": active_doc_id, "active_part_index": 1, "status": "studying"}

    def update_study_state(self, subject_id: str, active_doc_id: str, active_part_index: int, status: str = "studying"):
        """Saves current study progression state to SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO subject_study_state (subject_id, active_doc_id, active_part_index, status)
            VALUES (?, ?, ?, ?)
        ''', (subject_id, active_doc_id, active_part_index, status))
        conn.commit()
        conn.close()

    def advance_active_part(self, subject_id: str) -> dict:
        """Increments active_part_index if parts remain, otherwise marks subject completed."""
        state = self.get_active_study_state(subject_id)
        doc_id = state["active_doc_id"]
        current_idx = state["active_part_index"]
        
        if not doc_id:
            return state

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get total parts in active doc
        cursor.execute("SELECT total_parts FROM curriculum_documents WHERE doc_id=?", (doc_id,))
        total_row = cursor.fetchone()
        total_parts = total_row[0] if total_row else 1
        
        if current_idx < total_parts:
            next_idx = current_idx + 1
            new_status = "studying"
        else:
            next_idx = current_idx
            new_status = "completed"
            
        cursor.execute(
            "UPDATE subject_study_state SET active_part_index=?, status=? WHERE subject_id=?",
            (next_idx, new_status, subject_id)
        )
        conn.commit()
        conn.close()
        
        return {"active_doc_id": doc_id, "active_part_index": next_idx, "status": new_status}

    # --- Chat Session Management ---
    def get_chat_sessions(self, subject_id: str = None):
        """Retrieves active chat sessions. Optional filtering by subject."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if subject_id:
            cursor.execute(
                "SELECT session_id, title, created_at, subject_id FROM chat_sessions WHERE user_id=? AND subject_id=? ORDER BY created_at DESC",
                (self.user_id, subject_id)
            )
        else:
            cursor.execute(
                "SELECT session_id, title, created_at, subject_id FROM chat_sessions WHERE user_id=? ORDER BY created_at DESC",
                (self.user_id,)
            )
        rows = cursor.fetchall()
        conn.close()
        return [{"session_id": r[0], "title": r[1], "created_at": r[2], "subject_id": r[3]} for r in rows]

    def create_chat_session(self, session_id: str, title: str, subject_id: str = "default_subject"):
        """Creates a new chat session linked to a subject."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO chat_sessions (session_id, user_id, title, subject_id) VALUES (?, ?, ?, ?)",
            (session_id, self.user_id, title, subject_id)
        )
        conn.commit()
        conn.close()

    def delete_chat_session(self, session_id: str):
        """Deletes a chat session and all its messages."""
        if session_id == "default_session":
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_sessions WHERE session_id=?", (session_id,))
        cursor.execute("DELETE FROM chat_history WHERE session_id=?", (session_id,))
        conn.commit()
        conn.close()

    def update_session_title(self, session_id: str, new_title: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE chat_sessions SET title=? WHERE session_id=?", (new_title, session_id))
        conn.commit()
        conn.close()

    # --- Non-Linear Branching Graph Memory Serialization ---
    def save_graph_memory(self, session_id: str, graph, head_pointer: str, root_pointer: str):
        import networkx as nx
        import json
        from datetime import datetime
        data = nx.node_link_data(graph)
        graph_json = json.dumps({
            "graph": data,
            "head_pointer": head_pointer,
            "root_pointer": root_pointer
        })
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE chat_sessions SET graph_memory=? WHERE session_id=?", (graph_json, session_id))
        conn.commit()
        conn.close()

        # Log explicitly to text file
        log_dir = os.path.join(os.path.dirname(self.db_path), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "memory_graph_log.txt")
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"\n[{ts}] SESSION: {session_id}\n")
                f.write(f"HEAD: {head_pointer} | ROOT: {root_pointer}\n")
                f.write(json.dumps(data, indent=2))
                f.write("\n" + "="*50 + "\n")
        except Exception as e:
            print(f"Failed to write to memory_graph_log.txt: {e}")

    def load_graph_memory(self, session_id: str):
        import networkx as nx
        import json
        from datetime import datetime
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT graph_memory FROM chat_sessions WHERE session_id=?", (session_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0]:
            try:
                meta = json.loads(row[0])
                graph = nx.node_link_graph(meta["graph"])
                return graph, meta.get("head_pointer", ""), meta.get("root_pointer", "")
            except Exception as e:
                print(f"Error loading graph memory: {e}")
                
        # Blank slate DiGraph
        graph = nx.DiGraph()
        root_id = "root_" + session_id
        graph.add_node(
            root_id,
            node_id=root_id,
            topic_label="ROOT",
            raw_turn={"user": "System initialization", "ai": "Welcome to your offline AI Tutor!"},
            distilled_state={"q": "init", "status": "active"},
            embedding=[0.0] * 384,
            timestamp=datetime.now().isoformat()
        )
        return graph, root_id, root_id

    # --- Short Term Episodic Memory ---
    def add_chat_message(self, role: str, content: str, session_id: str = "default_session"):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history (user_id, session_id, role, content) VALUES (?, ?, ?, ?)",
            (self.user_id, session_id, role, content)
        )
        conn.commit()
        conn.close()

    def get_chat_history(self, session_id: str = "default_session", limit=50):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM chat_history WHERE user_id=? AND session_id=? ORDER BY timestamp DESC LIMIT ?",
            (self.user_id, session_id, limit)
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    # --- Semantic/Long Term Memory (Subject-Aware) ---
    def get_semantic_memory(self, subject_id: str = "default_subject"):
        """Retrieves the learner's subject-specific profile."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT preferred_style, academic_level, weak_topics, completed_topics FROM semantic_memory WHERE user_id=? AND subject_id=?", 
            (self.user_id, subject_id)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "preferred_style": row[0] if row[0] else "default",
                "academic_level": row[1] if row[1] else "Intermediate",
                "weak_topics": json.loads(row[2]) if row[2] else [],
                "completed_topics": json.loads(row[3]) if row[3] else []
            }
        else:
            default_profile = {"preferred_style": "default", "academic_level": "Intermediate", "weak_topics": [], "completed_topics": []}
            self._save_semantic_memory(default_profile, subject_id)
            return default_profile

    def update_weak_topics(self, new_weak_topic: str, subject_id: str = "default_subject"):
        memory = self.get_semantic_memory(subject_id)
        if new_weak_topic not in memory["weak_topics"]:
            memory["weak_topics"].append(new_weak_topic)
            self._save_semantic_memory(memory, subject_id)

    def update_semantic_memory(self, updates: dict, subject_id: str = "default_subject"):
        """Updates specific keys in the subject's semantic memory profile."""
        memory = self.get_semantic_memory(subject_id)
        for k, v in updates.items():
            memory[k] = v
        self._save_semantic_memory(memory, subject_id)

    def _save_semantic_memory(self, memory: dict, subject_id: str = "default_subject"):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO semantic_memory (user_id, subject_id, preferred_style, academic_level, weak_topics, completed_topics)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            self.user_id,
            subject_id,
            memory.get("preferred_style", "default"),
            memory.get("academic_level", "Intermediate"),
            json.dumps(memory.get("weak_topics", [])),
            json.dumps(memory.get("completed_topics", []))
        ))
        conn.commit()
        conn.close()

    # --- Quiz History Log (Subject-Aware) ---
    def add_quiz_record(self, topic: str, score: int, total: int, passed: bool, questions: list, user_answers: list, subject_id: str = "default_subject"):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO quiz_history (user_id, subject_id, topic, score, total, passed, questions_json, user_answers_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.user_id,
            subject_id,
            topic,
            score,
            total,
            1 if passed else 0,
            json.dumps(questions),
            json.dumps(user_answers)
        ))
        conn.commit()
        conn.close()

    def get_quiz_history(self, limit=10, subject_id: str = None):
        """Retrieves quiz records, optionally filtered by subject."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if subject_id:
            cursor.execute('''
                SELECT topic, score, total, passed, timestamp, subject_id 
                FROM quiz_history 
                WHERE user_id=? AND subject_id=? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (self.user_id, subject_id, limit))
        else:
            cursor.execute('''
                SELECT topic, score, total, passed, timestamp, subject_id 
                FROM quiz_history 
                WHERE user_id=? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (self.user_id, limit))
        rows = cursor.fetchall()
        conn.close()
        return [{
            "topic": r[0],
            "score": r[1],
            "total": r[2],
            "passed": bool(r[3]),
            "timestamp": r[4],
            "subject_id": r[5]
        } for r in rows]

if __name__ == "__main__":
    mem = UserMemory()
    mem.add_subject("science_101", "Basic Science", "curriculum.txt")
    mem.add_curriculum_document("doc_bio", "science_101", "Biology Handbook", "Mitosis is a process...", 2)
    mem.add_document_part("doc_bio_p1", "doc_bio", 1, "Mitosis Basics", "Mitosis introduction content...")
    mem.add_document_part("doc_bio_p2", "doc_bio", 2, "Mitosis Phases", "Mitosis phases content...")
    mem.update_study_state("science_101", "doc_bio", 1)
    
    print("Subjects:", mem.get_subjects())
    print("Documents:", mem.get_curriculum_documents("science_101"))
    print("Parts for doc_bio:", mem.get_document_parts("doc_bio"))
    print("Active Study State:", mem.get_active_study_state("science_101"))
    print("Advance active part:", mem.advance_active_part("science_101"))
