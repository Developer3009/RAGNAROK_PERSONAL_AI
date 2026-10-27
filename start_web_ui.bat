@echo off
title Ragnarok AI Agent - Web Interface
color 0b

echo ===================================================================
echo     RAGNAROK - PRIVACY-FIRST LOCAL AI AGENT
echo ===================================================================
echo.

:: 1. Verify Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3 is not installed or not in PATH!
    echo Please download and install Python 3.12+ from:
    echo   https://www.python.org/downloads/
    echo *IMPORTANT: Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: 2. Auto-create virtual environment if not present
if not exist ".venv" (
    echo [SETUP] First time setup: creating Python virtual environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: 3. Activate environment
call .venv\Scripts\activate.bat

:: 4. Ensure dependencies are installed
echo [SETUP] Checking and verifying dependencies...
pip install -r requirements.txt --quiet

:: 5. Launch web UI and automatically open default browser
echo.
echo ===================================================================
echo   Ragnarok is launching at: http://localhost:8000
echo   Opening your browser automatically...
echo.
echo   Press CTRL+C in this window at any time to stop the agent.
echo ===================================================================
echo.

start http://localhost:8000
python -m uvicorn ragnarok.ui.app:app --host 127.0.0.1 --port 8000

pause
