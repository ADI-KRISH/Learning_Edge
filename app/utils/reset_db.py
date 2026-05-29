import os
import shutil
import sqlite3

def reset_all_data():
    print("=== STARTING AI TUTOR NUCLEAR DATA RESET ===")
    
    # 1. Base Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    vector_db_dir = os.path.join(base_dir, "vector_db_v2")
    old_vector_db_dir = os.path.join(base_dir, "vector_db")
    memory_dir = os.path.join(base_dir, "app", "memory")
    graph_dir = os.path.join(base_dir, "app", "graph")
    logs_dir = os.path.join(base_dir, "logs")

    # 2. Delete Vector DB Directories
    for path in [vector_db_dir, old_vector_db_dir]:
        if os.path.exists(path):
            print(f"Removing vector DB folder: {path}")
            shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)

    # 3. Clean uploaded data directory
    if os.path.exists(data_dir):
        print(f"Clearing files in data folder: {data_dir}")
        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    print(f"  Deleted: {file}")
                except Exception as e:
                    print(f"  Failed to delete {file}: {e}")

    # 4. Clean concept graph folder
    if os.path.exists(graph_dir):
        print(f"Clearing knowledge graph files in: {graph_dir}")
        for file in ["knowledge_graph.json", "graph.html"]:
            file_path = os.path.join(graph_dir, file)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"  Deleted: {file}")
                except Exception as e:
                    print(f"  Failed to delete {file}: {e}")

    # 5. Reset SQLite Database
    db_path = os.path.join(memory_dir, "user_memory.db")
    if os.path.exists(db_path):
        print(f"Deleting SQLite memory database: {db_path}")
        try:
            os.remove(db_path)
            print("  Database deleted successfully.")
        except Exception as e:
            print(f"  Failed to delete database: {e}")
            print("  Attempting to clear all tables in database instead...")
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                for table in tables:
                    cursor.execute(f"DROP TABLE IF EXISTS {table[0]}")
                    print(f"    Dropped table: {table[0]}")
                conn.commit()
                conn.close()
            except Exception as inner_e:
                print(f"    Fallback table clearing failed: {inner_e}")

    # 6. Clean log files
    if os.path.exists(logs_dir):
        print(f"Clearing files in logs folder: {logs_dir}")
        for file in os.listdir(logs_dir):
            file_path = os.path.join(logs_dir, file)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
    
    # Clean workspace root logs if any
    for root_log in ["retrieval_log.txt", "run_output.log", "debug_server.log", "streamlit_debug.log", "streamlit_err.log", "streamlit_out.log", "crash_log.txt"]:
        log_path = os.path.join(base_dir, root_log)
        if os.path.exists(log_path):
            try:
                os.remove(log_path)
            except Exception:
                pass

    print("=== NUCLEAR RESET COMPLETE! FRESH SLATE ESTABLISHED ===")

if __name__ == "__main__":
    reset_all_data()
