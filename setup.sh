#!/bin/bash
# Setup script for ML Model Building Agent

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== ML Model Building Agent Setup ==="
echo ""

# Check Python version
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python 3 is required but not found."
    exit 1
fi

PY_VERSION=$($PYTHON_CMD --version 2>&1)
echo "Using: $PY_VERSION"

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
    echo ">>> IMPORTANT: Edit .env with your Azure AI Foundry credentials <<<"
else
    echo ".env file already exists."
fi

# Create workspaces directory
mkdir -p workspaces

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your Azure AI Foundry credentials"
echo "  2. Run: source venv/bin/activate"
echo "  3. Run: python main.py"
echo ""
