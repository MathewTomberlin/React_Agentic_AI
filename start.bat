@echo off
setlocal

REM === Backend (Python/FastAPI) ===
echo Setting up Python backend...
cd backend

if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

REM Start FastAPI backend in a new window
start "FastAPI Backend" cmd /k "cd /d %cd% && call venv\Scripts\activate && python main.py"

REM === Backend (Node.js/SwarmUI MCP) ===
echo Setting up Node.js SwarmUI MCP server...
if exist package.json (
    npm install
    if exist server.ts (
        npx tsc server.ts
    )
    REM Start Node.js server in a new window
    start "Node Server" cmd /k "cd /d %cd% && npm start"
)

REM === Ollama ===
where ollama >nul 2>nul
if %ERRORLEVEL%==0 (
    echo Starting Ollama...
    start "Ollama" cmd /k "ollama serve"
) else (
    echo Ollama not found. Please install Ollama if you need local LLMs.
)

REM === Frontend (React) ===
cd ..\frontend
echo Setting up React frontend...
if exist package.json (
    npm install
    REM Start React app in a new window
    start "React Frontend" cmd /k "cd /d %cd% && npm start"
)

echo All services started!
pause