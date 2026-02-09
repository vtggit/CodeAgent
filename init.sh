#!/bin/bash
set -e

echo "=========================================="
echo "Multi-Agent GitHub Issue Routing System"
echo "Development Environment Setup"
echo "=========================================="
echo ""

# Check if Python 3.11+ is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
elif [ -f "pyproject.toml" ]; then
    echo "Installing dependencies from pyproject.toml..."
    pip install -e .
    echo "✓ Dependencies installed"
else
    echo "Warning: No requirements.txt or pyproject.toml found"
fi

# Check for .env file
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo ""
        echo "⚠️  No .env file found. Creating from .env.example..."
        cp .env.example .env
        echo "✓ .env file created"
        echo ""
        echo "IMPORTANT: Edit .env and add your API keys:"
        echo "  - GITHUB_TOKEN"
        echo "  - ANTHROPIC_API_KEY"
        echo "  - REDIS_URL (if using external Redis)"
        echo ""
    else
        echo ""
        echo "⚠️  Warning: No .env file found. You'll need to configure environment variables."
        echo ""
    fi
fi

# Check if Redis is needed and running
echo "Checking Redis..."
if command -v redis-cli &> /dev/null; then
    if redis-cli ping > /dev/null 2>&1; then
        echo "✓ Redis is running"
    else
        echo "⚠️  Redis is installed but not running"
        echo "   Start with: redis-server"
    fi
else
    echo "⚠️  Redis not found. Install with:"
    echo "   Ubuntu/Debian: sudo apt-get install redis-server"
    echo "   macOS: brew install redis"
    echo "   Or use Docker: docker run -d -p 6379:6379 redis:alpine"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Configure your .env file with API keys"
echo "  2. Start the development server:"
echo "     uvicorn src.api.main:app --reload"
echo ""
echo "  3. Start the worker process (in another terminal):"
echo "     python src/orchestration/worker.py"
echo ""
echo "  4. Access the API documentation:"
echo "     http://localhost:8000/docs"
echo ""
echo "For Docker setup:"
echo "  docker-compose up"
echo ""
