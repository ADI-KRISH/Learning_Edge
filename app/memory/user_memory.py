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
        """Initializes the SQLite database with memory tables and multi-session schema migrations."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Chat sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                title TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Safe migration: Add graph_memory to chat_sessions if not exists
        try:
            cursor.execute("ALTER TABLE chat_sessions ADD COLUMN graph_memory TEXT")
        except sqlite3.OperationalError:
            pass # Column already exists
        
        # Short term memory table (chat history)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Safe migration: Add session_id to chat_history if not exists
        try:
            cursor.execute("ALTER TABLE chat_history ADD COLUMN session_id TEXT")
        except sqlite3.OperationalError:
            pass # Column already exists
            
        # Update old messages without session_id to default_session
        cursor.execute("UPDATE chat_history SET session_id = 'default_session' WHERE session_id IS NULL OR session_id = ''")
        
        # Ensure default session exists in chat_sessions
        cursor.execute("SELECT 1 FROM chat_sessions WHERE session_id = 'default_session'")
        if not cursor.fetchone():
            cursor.execute("INSERT INTO chat_sessions (session_id, user_id, title) VALUES ('default_session', ?, 'Default Chat Session')", (self.user_id,))
            
        # Semantic/Long term memory table (weak topics, preferred style, academic level)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_memory (
                user_id TEXT PRIMARY KEY,
                preferred_style TEXT,
                academic_level TEXT, -- beginner, intermediate, advanced
                weak_topics TEXT, -- stored as JSON string
                completed_topics TEXT -- stored as JSON string
            )
        ''')
        
        # Safe migration: Add academic_level to semantic_memory if not exists
        try:
            cursor.execute("ALTER TABLE semantic_memory ADD COLUMN academic_level TEXT")
        except sqlite3.OperationalError:
            pass # Column already exists
            
        # Quiz history tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quiz_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                topic TEXT,
                score INTEGER,
                total INTEGER,
                passed INTEGER, -- 1 = true, 0 = false
                questions_json TEXT, -- JSON string representing the questions array
                user_answers_json TEXT, -- JSON string representing selected answers
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()

    def get_chat_sessions(self):
        """Retrieves the list of active chat sessions, ordered by creation date."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT session_id, title, created_at FROM chat_sessions WHERE user_id=? ORDER BY created_at DESC",
            (self.user_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"session_id": r[0], "title": r[1], "created_at": r[2]} for r in rows]

    def create_chat_session(self, session_id: str, title: str):
        """Creates a new chat session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO chat_sessions (session_id, user_id, title) VALUES (?, ?, ?)",
            (session_id, self.user_id, title)
        )
        conn.commit()
        conn.close()

    def delete_chat_session(self, session_id: str):
        """Deletes a chat session and all associated messages."""
        if session_id == "default_session":
            return # Prevent deleting the default session
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_sessions WHERE session_id=?", (session_id,))
        cursor.execute("DELETE FROM chat_history WHERE session_id=?", (session_id,))
        conn.commit()
        conn.close()

    def update_session_title(self, session_id: str, new_title: str):
        """Updates a chat session title."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE chat_sessions SET title=? WHERE session_id=?", (new_title, session_id))
        conn.commit()
        conn.close()

    def save_graph_memory(self, session_id: str, graph, head_pointer: str, root_pointer: str):
        """Serializes and saves the NetworkX DiGraph memory for a session."""
        import networkx as nx
        import json
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

    def load_graph_memory(self, session_id: str):
        """Loads and deserializes the NetworkX DiGraph memory for a session."""
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
                
        # Return blank slate DiGraph
        graph = nx.DiGraph()
        root_id = "root_" + session_id
        # Root node represents the start anchor
        graph.add_node(
            root_id,
            node_id=root_id,
            topic_label="ROOT",
            raw_turn={"user": "System initialization", "ai": "Welcome to your offline AI Tutor!"},
            distilled_state={"q": "init", "status": "active"},
            embedding=[0.0] * 384, # all-MiniLM-L6-v2 embedding dimension is 384
            timestamp=datetime.now().isoformat()
        )
        return graph, root_id, root_id

    def add_chat_message(self, role: str, content: str, session_id: str = "default_session"):
        """Saves a message to short term memory for a specific chat session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history (user_id, session_id, role, content) VALUES (?, ?, ?, ?)",
            (self.user_id, session_id, role, content)
        )
        conn.commit()
        conn.close()

    def get_chat_history(self, session_id: str = "default_session", limit=50):
        """Retrieves recent conversation history for a specific chat session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM chat_history WHERE user_id=? AND session_id=? ORDER BY timestamp DESC LIMIT ?",
            (self.user_id, session_id, limit)
        )
        rows = cursor.fetchall()
        conn.close()
        # Return in chronological order
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    def get_semantic_memory(self):
        """Retrieves the learner's long-term profile."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT preferred_style, academic_level, weak_topics, completed_topics FROM semantic_memory WHERE user_id=?", (self.user_id,))
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
            return {"preferred_style": "default", "academic_level": "Intermediate", "weak_topics": [], "completed_topics": []}

    def update_weak_topics(self, new_weak_topic: str):
        """Adds a topic to the weak topics list."""
        memory = self.get_semantic_memory()
        if new_weak_topic not in memory["weak_topics"]:
            memory["weak_topics"].append(new_weak_topic)
            self._save_semantic_memory(memory)

    def add_quiz_record(self, topic: str, score: int, total: int, passed: bool, questions: list, user_answers: list):
        """Saves a quiz attempt log in history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO quiz_history (user_id, topic, score, total, passed, questions_json, user_answers_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.user_id,
            topic,
            score,
            total,
            1 if passed else 0,
            json.dumps(questions),
            json.dumps(user_answers)
        ))
        conn.commit()
        conn.close()

    def get_quiz_history(self, limit=10):
        """Retrieves quiz history records in chronological order."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT topic, score, total, passed, timestamp 
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
            "timestamp": r[4]
        } for r in rows]

    def _save_semantic_memory(self, memory: dict):
        """Internal helper to save the full semantic memory dict."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO semantic_memory (user_id, preferred_style, academic_level, weak_topics, completed_topics)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            self.user_id,
            memory.get("preferred_style", "default"),
            memory.get("academic_level", "Intermediate"),
            json.dumps(memory.get("weak_topics", [])),
            json.dumps(memory.get("completed_topics", []))
        ))
        conn.commit()
        conn.close()


if __name__ == "__main__":
    mem = UserMemory()
    mem.create_chat_session("session_1", "Test Session")
    mem.add_chat_message("user", "Explain Backpropagation.", "session_1")
    mem.update_weak_topics("Backpropagation")
    print("Sessions:", mem.get_chat_sessions())
    print("Chat History:", mem.get_chat_history("session_1"))
    print("Semantic Memory:", mem.get_semantic_memory())
