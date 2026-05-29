@echo off
title AI Tutor Dev Mode
color 0a

cd /d "%~dp0"

echo.
echo  ==========================================
echo   Offline AI Tutor - Development Mode
echo  ==========================================
echo.
echo  This script runs BOTH servers simultaneously:
echo    - FastAPI backend on http://localhost:8000
echo    - React dev server on http://localhost:5173
echo.

:: Install frontend deps if needed
if not exist "frontend\node_modules" (
    echo [SETUP] Installing React frontend dependencies...
    cd frontend
    call npm install
    cd ..
    echo [SETUP] Done!
)

echo [START] Opening React app in browser...
start "" http://localhost:5173

echo [START] Launching FastAPI backend...
start cmd /k ".venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

echo [START] Launching React dev server...
cd frontend
call npm run dev
