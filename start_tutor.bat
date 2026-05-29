@echo off
title Offline AI Tutor
color 0a

cd /d "%~dp0"

echo.
echo  ==========================================
echo   Offline AI Tutor - React + FastAPI Stack
echo  ==========================================
echo.

:: Check Python venv
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found!
    echo Please run: uv sync
    pause
    exit /b 1
)

:: Check Ollama
echo [CHECK] Checking Ollama...
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    echo [OK] Ollama found.
) else (
    where ollama >nul 2>&1
    if errorlevel 1 (
        echo [WARN] Ollama not found. Download from: https://ollama.com
    ) else (
        echo [OK] Ollama found in PATH.
    )
)

echo.
echo [START] Launching FastAPI backend...
echo [INFO]  Open your browser to: http://localhost:8000
echo [INFO]  Press Ctrl+C to stop the server.
echo.

:: Open browser after 3 seconds
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8000"

:: Start the FastAPI server (blocking)
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

echo.
echo [STOPPED] Server has stopped.
pause
