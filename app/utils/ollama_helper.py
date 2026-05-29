import os
import socket
import subprocess
import time
import shutil

OLLAMA_PORT = 11434
OLLAMA_URL = f"http://localhost:{OLLAMA_PORT}"

def is_ollama_running() -> bool:
    """Checks if the Ollama port is open and responding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(("127.0.0.1", OLLAMA_PORT))
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False

def find_ollama_executable() -> str:
    """Attempts to find the Ollama executable on the system."""
    # 1. Check if 'ollama' is in PATH
    path_exe = shutil.which("ollama")
    if path_exe:
        return path_exe

    # 2. Check typical Windows local appdata installation folder
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        win_path = os.path.join(local_app_data, "Programs", "Ollama", "ollama.exe")
        if os.path.isfile(win_path):
            return win_path

    # 3. Check hardcoded user path as a robust fallback
    fallback_path = r"C:\Users\GS Adithya Krishna\AppData\Local\Programs\Ollama\ollama.exe"
    if os.path.isfile(fallback_path):
        return fallback_path

    return ""

def ensure_ollama_running(timeout_seconds: int = 15) -> bool:
    """Ensures Ollama is running, launching it if necessary.
    
    Returns:
        True if running or successfully started, False otherwise.
    """
    if is_ollama_running():
        return True

    exe_path = find_ollama_executable()
    if not exe_path:
        print("[Ollama Helper] Ollama executable not found on the system.")
        return False

    print(f"[Ollama Helper] Starting Ollama from: {exe_path}")
    try:
        # Start Ollama in background without blocking
        if os.name == "nt":
            # Windows: start with no console window and detached process
            subprocess.Popen(
                [exe_path, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
            )
        else:
            # Unix-like: start in background
            subprocess.Popen(
                [exe_path, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setpgrp
            )
        
        # Poll and wait for server to start
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if is_ollama_running():
                print("[Ollama Helper] Ollama started successfully.")
                return True
            time.sleep(1.0)
            
        print("[Ollama Helper] Ollama startup timed out.")
        return False
    except Exception as e:
        print(f"[Ollama Helper] Failed to launch Ollama: {e}")
        return False

if __name__ == "__main__":
    print("Checking if Ollama is running...")
    running = is_ollama_running()
    print(f"Is running: {running}")
    if not running:
        print("Attempting to start...")
        started = ensure_ollama_running()
        print(f"Started successfully: {started}")
