@echo off
REM RunPod Setup Script for FraudGuard
REM This configures your local backend to use RunPod GPU

echo.
echo ========================================
echo   FraudGuard RunPod Setup
echo ========================================
echo.

REM Check if .env exists
if not exist "backend\.env" (
    echo ❌ ERROR: backend\.env not found!
    echo    Please run setup-windows.bat first
    pause
    exit /b 1
)

echo.
echo 📋 Step 1: Get your RunPod vLLM URL
echo.
echo   1. Go to RunPod Console: https://www.runpod.io/console
echo   2. Start your GPU pod
echo   3. Start vLLM server (see RUNPOD_SETUP.md for instructions)
echo   4. Copy the public URL (e.g., https://xxxxx-8000.proxy.runpod.net)
echo.
set /p runpod_url="Enter your RunPod vLLM URL (with /v1 at end): "

if "%runpod_url%"=="" (
    echo ❌ No URL provided. Exiting.
    pause
    exit /b 1
)

echo.
echo 📝 Step 2: Updating backend\.env...
echo.

REM Update VLLM_SERVER_URL in .env file
powershell -Command "(Get-Content backend\.env) -replace 'VLLM_SERVER_URL=.*', 'VLLM_SERVER_URL=%runpod_url%' | Set-Content backend\.env"

echo ✅ Updated VLLM_SERVER_URL to: %runpod_url%
echo.

echo 📋 Step 3: Starting local services...
echo.

REM Start services (without vLLM)
docker-compose up -d db redis backend

if %errorlevel% equ 0 (
    echo.
    echo ✅ Services started!
    echo.
    echo ⏳ Waiting for services to be ready...
    timeout /t 15 /nobreak >nul
    echo.
    echo 🧪 Testing backend connection...
    curl -f http://localhost:8000/health >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ Backend is running!
        echo.
        echo 📊 Health check:
        curl http://localhost:8000/health
        echo.
        echo.
        echo 🎉 Setup complete!
        echo.
        echo Next steps:
        echo   1. Make sure vLLM is running on RunPod
        echo   2. Test: curl http://localhost:8000/health
        echo   3. Start frontend: cd frontend ^&^& npm install ^&^& npm run dev
    ) else (
        echo ⚠️  Backend may still be starting...
        echo    Check logs: docker-compose logs -f backend
    )
) else (
    echo.
    echo ❌ Failed to start services
    echo    Check Docker is running
)

echo.
pause



