#!/bin/bash
# FraudGuard Backend Startup Script
# Ensures environment is configured and starts the server

set -e

echo "🚀 Starting FraudGuard Backend..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "📝 Creating .env from env.example..."
    cp env.example .env
    echo "✅ Created .env file. Please update it with your configuration."
    echo "   At minimum, set VLLM_SERVER_URL and REDIS_URL"
    read -p "Press Enter to continue or Ctrl+C to exit and configure .env..."
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "🐍 Python version: $python_version"

# Check if dependencies are installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Validate critical environment variables
source .env 2>/dev/null || true

if [ -z "$VLLM_SERVER_URL" ]; then
    echo "⚠️  Warning: VLLM_SERVER_URL not set. Using default: http://localhost:8001/v1"
fi

if [ -z "$REDIS_URL" ]; then
    echo "⚠️  Warning: REDIS_URL not set. Using default: redis://localhost:6379"
fi

# Start the server
echo "🌟 Starting FastAPI server..."
echo "📡 API will be available at http://localhost:8000"
echo "📊 Health check: http://localhost:8000/health"
echo "📚 API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload

