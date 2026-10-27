#!/usr/bin/env bash
set -e

echo "==================================================================="
echo "    RAGNAROK - PRIVACY-FIRST LOCAL AI AGENT"
echo "==================================================================="
echo ""

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed!"
    echo "Please install Python 3.12+ from https://www.python.org/downloads/"
    exit 1
fi

# 2. Virtual environment
if [ ! -d ".venv" ]; then
    echo "[SETUP] Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# 3. Dependencies
echo "[SETUP] Verifying dependencies..."
pip install -r requirements.txt --quiet

# 4. Run Web UI
echo ""
echo "==================================================================="
echo "  Ragnarok is launching at: http://localhost:8000"
echo "  Press CTRL+C at any time to stop the server."
echo "==================================================================="
echo ""

python3 -m uvicorn ragnarok.ui.app:app --host 127.0.0.1 --port 8000
