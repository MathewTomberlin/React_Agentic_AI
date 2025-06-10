@echo off
setlocal

REM === CONFIGURE PATHS ===
set BACKEND_DIR=%cd%\backend
set FRONTEND_DIR=%cd%\frontend

REM --- 1. Start FastAPI backend if not running (port 8000) ---
echo Checking if FastAPI backend is running...
netstat -ano | findstr ":8000" >nul
if %ERRORLEVEL%==0 (
    echo FastAPI backend is already running.
) else (
    echo Setting up Python backend...
    pushd %BACKEND_DIR%
    if not exist venv (
        python -m venv venv
    )
    call venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo Starting FastAPI backend...
    start "" cmd /k "cd /d %BACKEND_DIR% && call venv\Scripts\activate && python main.py"
    popd
    REM Wait for FastAPI backend to start
    echo Waiting for FastAPI backend to start...
    :wait_fastapi
    timeout /t 2 >nul
    netstat -ano | findstr ":8000" >nul
    if %ERRORLEVEL%==0 (
        echo FastAPI backend started!
    ) else (
        goto wait_fastapi
    )
)

REM --- 3. Start Ollama if not running (port 11434) ---
echo Checking if Ollama is running...
netstat -ano | findstr ":11434" >nul
if %ERRORLEVEL%==0 (
    echo Ollama is already running.
) else (
    WHERE ollama
    IF %ERRORLEVEL%==0 (
        echo Starting Ollama...
        start "" cmd /k "ollama serve"
        REM Wait for Ollama to start
        echo Waiting for Ollama to start...
        :wait_ollama
        timeout /t 2 >nul
        netstat -ano | findstr ":11434" >nul
        if %ERRORLEVEL%==0 (
            echo Ollama started!
        ) else (
            goto wait_ollama
        )
    ) else (
        echo Ollama not found. Please install Ollama if you need local LLMs.
        pause
    )
)

REM --- 4. Start React frontend if not running (port 3000) ---
echo Checking if React frontend is running...
netstat -ano | findstr ":3000" >nul
if %ERRORLEVEL%==0 (
    echo React frontend is already running.
) else (
    echo Setting up React frontend...
    pushd %FRONTEND_DIR%
    if exist package.json (
        npm install
        echo Starting React app...
        start "" cmd /k "cd /d %FRONTEND_DIR% && npm start"
    ) else (
        echo package.json not found in frontend!
        pause
    )
    popd
    REM Wait for React frontend to start
    echo Waiting for React frontend to start...
    :wait_react
    timeout /t 2 >nul
    netstat -ano | findstr ":3000" >nul
    if %ERRORLEVEL%==0 (
        echo React frontend started!
    ) else (
        goto wait_react
    )
)

REM --- 2. Start Node MCP server if not running (port 5001) ---
echo Checking if Node MCP server is running...
netstat -ano | findstr ":5001" >nul
if %ERRORLEVEL%==0 (
    echo Node MCP server is already running.
) else (
    echo Setting up Node MCP server...
    pushd %BACKEND_DIR%
    if exist package.json (
        npm install
        if exist server.ts (
            npx tsc server.ts
        )
        echo Starting Node MCP server...
        start "" cmd /k "cd /d %BACKEND_DIR% && npm start"
    ) else (
        echo Node.js package.json not found in backend!
        pause
    )
    popd
    REM Wait for Node MCP server to start
    echo Waiting for Node MCP server to start...
    :wait_node
    timeout /t 2 >nul
    netstat -ano | findstr ":5001" >nul
    if %ERRORLEVEL%==0 (
        echo Node MCP server started!
    ) else (
        goto wait_node
    )
)

echo All services started!
pause