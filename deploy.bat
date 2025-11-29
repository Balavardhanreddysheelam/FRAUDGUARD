@echo off
REM FraudGuard Deployment Script for Windows
REM This script helps deploy FraudGuard to production

echo.
echo 🚀 FraudGuard Deployment Script
echo ================================
echo.

REM Check if .env exists
if not exist "backend\.env" (
    echo ⚠️  .env file not found in backend\
    echo ℹ️  Creating .env from env.example...
    if exist "backend\env.example" (
        copy "backend\env.example" "backend\.env" >nul
        echo ✅ .env file created
        echo ⚠️  Please update backend\.env with your configuration before continuing
    ) else (
        echo ❌ env.example not found!
        exit /b 1
    )
) else (
    echo ✅ .env file exists
)

REM Check if model exists
if not exist "inference\model\fraudguard-8b-merged" (
    echo ⚠️  Model directory not found at inference\model\fraudguard-8b-merged
    echo ℹ️  Please ensure the model is in the correct location
) else (
    echo ✅ Model directory found
)

echo.
echo ℹ️  Deployment Options:
echo 1. Start services with docker-compose
echo 2. Check service health
echo 3. View logs
echo 4. Exit
echo.
set /p option="Select option (1-4): "

if "%option%"=="1" (
    echo ℹ️  Starting services with docker-compose...
    docker-compose up -d
    if %errorlevel% equ 0 (
        echo ✅ Services started
        echo ℹ️  Waiting for services to be healthy...
        timeout /t 10 /nobreak >nul
        call :check_health
    ) else (
        echo ❌ Failed to start services
    )
) else if "%option%"=="2" (
    call :check_health
) else if "%option%"=="3" (
    echo ℹ️  Showing logs (Ctrl+C to exit)...
    docker-compose logs -f
) else if "%option%"=="4" (
    echo ℹ️  Exiting...
    exit /b 0
) else (
    echo ❌ Invalid option
)

exit /b 0

:check_health
echo ℹ️  Checking backend health...
timeout /t 5 /nobreak >nul
curl -f http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Backend is healthy!
    curl http://localhost:8000/health
) else (
    echo ⚠️  Backend health check failed. Services may still be starting...
    echo ℹ️  Check logs with: docker-compose logs -f backend
)
exit /b 0



