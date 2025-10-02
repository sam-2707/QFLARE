@echo off
echo 🏗️  Building QFLARE Production Environment...

echo.
echo 📦 Building Frontend...
cd /d "%~dp0..\frontend\qflare-ui"
call npm run build
if %ERRORLEVEL% neq 0 (
    echo ❌ Frontend build failed!
    pause
    exit /b 1
)

echo.
echo 🐍 Installing Python Dependencies...
cd /d "%~dp0..\server"
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo ❌ Python dependencies installation failed!
    pause
    exit /b 1
)

echo.
echo 🐳 Building Docker Images...
cd /d "%~dp0.."
docker-compose -f docker-compose.prod.yml build
if %ERRORLEVEL% neq 0 (
    echo ❌ Docker build failed!
    pause
    exit /b 1
)

echo.
echo ✅ Production build completed successfully!
echo 🚀 Ready to deploy QFLARE!
echo.
echo Next steps:
echo 1. Run: docker-compose -f docker-compose.prod.yml up -d
echo 2. Access: http://localhost:4000 (Frontend)
echo 3. Access: http://localhost:8000 (Backend API)
echo.
pause