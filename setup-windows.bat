@echo off
REM FraudGuard Windows Setup Script
REM This script sets up the environment for Windows

echo.
echo ========================================
echo   FraudGuard Windows Setup
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "docker-compose.yml" (
    echo ERROR: docker-compose.yml not found!
    echo Please run this script from the FRAUDGUARD project root directory.
    echo.
    echo Current directory: %CD%
    echo.
    pause
    exit /b 1
)

echo ✅ Found project directory
echo.

REM Step 1: Setup backend .env file
echo [1/4] Setting up backend environment...
if not exist "backend\.env" (
    if exist "backend\env.example" (
        copy "backend\env.example" "backend\.env" >nul
        echo ✅ Created backend\.env from env.example
        echo.
        echo ⚠️  IMPORTANT: Please edit backend\.env and set:
        echo    - VLLM_SERVER_URL=http://localhost:8001/v1
        echo    - REDIS_URL=redis://localhost:6379
        echo.
        echo Opening .env file in Notepad...
        timeout /t 2 /nobreak >nul
        notepad backend\.env
    ) else (
        echo ❌ ERROR: backend\env.example not found!
        pause
        exit /b 1
    )
) else (
    echo ✅ backend\.env already exists
)
echo.

REM Step 2: Check Docker
echo [2/4] Checking Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: Docker not found!
    echo Please install Docker Desktop for Windows
    echo Download from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)
echo ✅ Docker is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: docker-compose not found!
    pause
    exit /b 1
)
echo ✅ docker-compose is available
echo.

REM Step 3: Check if model exists
echo [3/4] Checking model files...
if not exist "inference\model\fraudguard-8b-merged" (
    echo ⚠️  WARNING: Model directory not found at inference\model\fraudguard-8b-merged
    echo    The vLLM server will not work without the model.
    echo    You can skip vLLM for now and test other services.
) else (
    echo ✅ Model directory found
)
echo.

REM Step 4: Start services
echo [4/4] Starting services with Docker Compose...
echo.
echo This will start:
echo   - PostgreSQL database
echo   - Redis cache
echo   - vLLM server (if GPU available)
echo   - Backend API
echo.
set /p start_services="Start services now? (Y/n): "
if /i "%start_services%"=="n" (
    echo Skipping service start.
    echo.
    echo To start services later, run:
    echo   docker-compose up -d
    pause
    exit /b 0
)

echo.
echo Starting services...
docker-compose up -d

if %errorlevel% equ 0 (
    echo.
    echo ✅ Services started successfully!
    echo.
    echo Waiting for services to be ready...
    timeout /t 10 /nobreak >nul
    echo.
    echo Checking backend health...
    curl -f http://localhost:8000/health >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ Backend is healthy!
        echo.
        echo You can now:
        echo   1. Test backend: http://localhost:8000/health
        echo   2. Start frontend: cd frontend ^&^& npm install ^&^& npm run dev
        echo   3. Open dashboard: http://localhost:3000
    ) else (
        echo ⚠️  Backend may still be starting...
        echo    Check logs with: docker-compose logs -f backend
        echo    Or wait a bit longer and try: curl http://localhost:8000/health
    )
) else (
    echo.
    echo ❌ Failed to start services
    echo Check the error messages above
    echo.
    echo Common issues:
    echo   - Docker Desktop not running
    echo   - Ports already in use (8000, 8001, 5432, 6379)
    echo   - GPU not available (vLLM will fail, but other services should work)
)

echo.
echo Setup complete!
echo.
pause



