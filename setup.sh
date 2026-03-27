#!/bin/bash
# Setup script for ML Model Building Agent

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MIN_PY_MAJOR=3
MIN_PY_MINOR=9

echo "=== ML Model Building Agent Setup ==="
echo ""

# Find Python
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python 3 is required but not found."
    echo "Install Python ${MIN_PY_MAJOR}.${MIN_PY_MINOR}+ from https://python.org"
    exit 1
fi

# Check Python version meets minimum requirement
PY_VERSION=$($PYTHON_CMD --version 2>&1)
PY_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

echo "Found: $PY_VERSION"

if [ "$PY_MAJOR" -lt "$MIN_PY_MAJOR" ] || { [ "$PY_MAJOR" -eq "$MIN_PY_MAJOR" ] && [ "$PY_MINOR" -lt "$MIN_PY_MINOR" ]; }; then
    echo "ERROR: Python ${MIN_PY_MAJOR}.${MIN_PY_MINOR}+ is required (found ${PY_MAJOR}.${PY_MINOR})."
    echo ""
    echo "Options:"
    echo "  - Install a newer Python from https://python.org"
    echo "  - Use pyenv: pyenv install ${MIN_PY_MAJOR}.${MIN_PY_MINOR} && pyenv local ${MIN_PY_MAJOR}.${MIN_PY_MINOR}"
    echo "  - On macOS: brew install python@3.11"
    echo "  - On Ubuntu: sudo apt install python3.11 python3.11-venv"
    exit 1
fi

echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate and install
echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "Dependencies installed."

# Create .env if not exists
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "Created .env from .env.example"
    echo ""
    echo ">>> IMPORTANT: Edit .env with your API credentials <<<"
    echo ""
    echo "Choose a provider:"
    echo "  Option 1 (recommended): Set ANTHROPIC_API_KEY"
    echo "  Option 2: Set AZURE_AI_ENDPOINT + AZURE_AI_API_KEY"
    echo "  Option 3: Set PROVIDER=claude-code (uses Claude Code CLI)"
else
    echo ".env file already exists."
fi

# Create workspaces directory
mkdir -p workspaces

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API credentials (see .env.example for options)"
echo "  2. Run: source venv/bin/activate"
echo "  3. Run: python main.py"
echo ""
