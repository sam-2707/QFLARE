@echo off
REM Start QFLARE FL Dashboard
echo ========================================
echo Starting QFLARE FL Dashboard
echo ========================================
echo.

cd /d "%~dp0frontend\qflare-ui"

echo Checking if port 4000 is available...
netstat -ano | findstr :4000 > nul
if %errorlevel% == 0 (
    echo Port 4000 is in use. Waiting for it to free up...
    timeout /t 5 /nobreak > nul
)

echo.
echo Starting React development server...
echo Frontend will be available at: http://localhost:4000
echo FL Dashboard will be at: http://localhost:4000/federated-learning
echo.
echo Press CTRL+C to stop the server
echo ========================================

npm start
