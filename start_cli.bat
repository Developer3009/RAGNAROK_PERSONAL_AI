@echo off
title Ragnarok AI Agent - Interactive Terminal
color 0b

echo ===================================================================
echo     RAGNAROK - PRIVACY-FIRST LOCAL AI AGENT (CLI)
echo ===================================================================
echo.

:: 1. Verify Python
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
)

:: 3. Activate environment
call .venv\Scripts\activate.bat

:: 4. Ensure dependencies are installed
echo [SETUP] Checking and verifying dependencies...
pip install -r requirements.txt --quiet

:: 5. Launch interactive CLI
echo.
echo Launching Ragnarok Terminal Interface...
python -m ragnarok

pause
