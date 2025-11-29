@echo off
REM FraudGuard Backend Startup Script for Windows
REM Ensures environment is configured and starts the server

echo 🚀 Starting FraudGuard Backend...

REM Check if .env exists
if not exist .env (
    echo ⚠️  Warning: .env file not found!
    echo 📝 Creating .env from env.example...
    copy env.example .env
    echo ✅ Created .env file. Please update it with your configuration.
    echo    At minimum, set VLLM_SERVER_URL and REDIS_URL
    pause
)

REM Check Python version
python --version
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

REM Check if dependencies are installed
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo 📦 Installing dependencies...
    pip install -r requirements.txt
)

REM Start the server
echo 🌟 Starting FastAPI server...
echo 📡 API will be available at http://localhost:8000
echo 📊 Health check: http://localhost:8000/health
echo 📚 API docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

