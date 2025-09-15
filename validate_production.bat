@echo off
REM QFLARE Production Deployment Validation Script for Windows
REM This script validates the production deployment setup

echo 🔍 QFLARE Production Deployment Validator
echo =========================================

REM Colors for output (Windows CMD)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "NC=[0m"

REM Function to print status
:print_status
setlocal
set "status=%~1"
set "message=%~2"
if "%status%"=="success" (
    echo ✅ %message%
) else if "%status%"=="warning" (
    echo ⚠️  %message%
) else (
    echo ❌ %message%
)
goto :eof

REM Check if Docker is running
:check_docker
echo 🐳 Checking Docker...
docker info >nul 2>&1
if errorlevel 1 (
    call :print_status "error" "Docker is not running or not accessible"
    echo    Please start Docker and try again
    exit /b 1
)
call :print_status "success" "Docker is running"
goto :eof

REM Check if Docker Compose is available
:check_docker_compose
echo 📦 Checking Docker Compose...
docker-compose version >nul 2>&1
if errorlevel 1 (
    docker compose version >nul 2>&1
    if errorlevel 1 (
        call :print_status "error" "Docker Compose is not installed"
        exit /b 1
    )
)
call :print_status "success" "Docker Compose is available"
goto :eof

REM Check required files
:check_files
echo 📁 Checking required files...
set "missing_files="
if not exist "Dockerfile.prod" set "missing_files=%missing_files% Dockerfile.prod"
if not exist "docker-compose.prod.yml" set "missing_files=%missing_files% docker-compose.prod.yml"
if not exist "requirements.prod.txt" set "missing_files=%missing_files% requirements.prod.txt"
if not exist ".env" set "missing_files=%missing_files% .env"
if not exist "config\redis.conf" set "missing_files=%missing_files% config\redis.conf"
if not exist "config\nginx.conf" set "missing_files=%missing_files% config\nginx.conf"
if not exist "config\prometheus.yml" set "missing_files=%missing_files% config\prometheus.yml"

if defined missing_files (
    call :print_status "error" "Missing required files:"
    for %%f in (%missing_files%) do echo    - %%f
    exit /b 1
)
call :print_status "success" "All required files present"
goto :eof

REM Validate environment file
:validate_env
echo 🔧 Validating environment configuration...
if not exist ".env" (
    call :print_status "error" ".env file not found"
    goto :eof
)

REM Check for required environment variables
set "missing_vars="
findstr /c:"ENVIRONMENT=" .env >nul 2>&1 || set "missing_vars=%missing_vars% ENVIRONMENT"
findstr /c:"QFLARE_JWT_SECRET=" .env >nul 2>&1 || set "missing_vars=%missing_vars% QFLARE_JWT_SECRET"
findstr /c:"DATABASE_URL=" .env >nul 2>&1 || set "missing_vars=%missing_vars% DATABASE_URL"
findstr /c:"REDIS_URL=" .env >nul 2>&1 || set "missing_vars=%missing_vars% REDIS_URL"

if defined missing_vars (
    call :print_status "warning" "Missing environment variables:"
    for %%v in (%missing_vars%) do echo    - %%v
    echo    Using default values where possible
) else (
    call :print_status "success" "Environment configuration valid"
)
goto :eof

REM Test Docker build
:test_build
echo 🏗️  Testing Docker build...
docker-compose -f docker-compose.prod.yml build --no-cache >nul 2>&1
if errorlevel 1 (
    call :print_status "error" "Docker build failed"
    echo    Check build logs: docker-compose -f docker-compose.prod.yml build
    goto :eof
)
call :print_status "success" "Docker build successful"
goto :eof

REM Test service startup
:test_startup
echo 🚀 Testing service startup...
docker-compose -f docker-compose.prod.yml up -d >nul 2>&1
if errorlevel 1 (
    call :print_status "error" "Service startup failed"
    echo    Check logs: docker-compose -f docker-compose.prod.yml logs
    goto :eof
)

REM Wait for services to be healthy
echo    Waiting for services to be ready...
set /a max_attempts=30
set /a attempt=1

:wait_loop
docker-compose -f docker-compose.prod.yml ps | findstr "Up" >nul 2>&1
if not errorlevel 1 (
    call :print_status "success" "Services started successfully"
    goto :eof
)
echo    Attempt %attempt%/%max_attempts%: Waiting...
timeout /t 2 /nobreak >nul
set /a attempt+=1
if %attempt% leq %max_attempts% goto wait_loop

call :print_status "error" "Services failed to start within timeout"
echo    Check logs: docker-compose -f docker-compose.prod.yml logs
goto :eof

REM Test health endpoints
:test_health
echo 🏥 Testing health endpoints...
curl -f -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    call :print_status "warning" "QFLARE Server health check failed"
) else (
    call :print_status "success" "QFLARE Server health check passed"
)

curl -f -s http://localhost:9090/-/healthy >nul 2>&1
if errorlevel 1 (
    call :print_status "warning" "Prometheus health check failed"
) else (
    call :print_status "success" "Prometheus health check passed"
)
goto :eof

REM Test API endpoints
:test_api
echo 🔗 Testing API endpoints...
curl -f -s http://localhost:8000/docs >nul 2>&1
if errorlevel 1 (
    call :print_status "warning" "API Documentation endpoint failed"
) else (
    call :print_status "success" "API Documentation accessible"
)

curl -f -s http://localhost:8000/openapi.json >nul 2>&1
if errorlevel 1 (
    call :print_status "warning" "OpenAPI Spec endpoint failed"
) else (
    call :print_status "success" "OpenAPI Spec accessible"
)
goto :eof

REM Cleanup function
:cleanup
echo 🧹 Cleaning up test containers...
docker-compose -f docker-compose.prod.yml down -v >nul 2>&1
call :print_status "success" "Cleanup completed"
goto :eof

REM Main validation function
:main
echo.

REM Run all checks
call :check_docker
call :check_docker_compose
call :check_files
call :validate_env

REM Build and test
call :test_build
if not errorlevel 1 (
    call :test_startup
    if not errorlevel 1 (
        call :test_health
        call :test_api
    )
)

REM Cleanup
call :cleanup

echo.
echo 🎯 Validation Complete!
echo ======================
echo If all checks passed, your production deployment is ready!
echo Run 'deploy.bat' to start the production services.
goto :eof

REM Run main function
call :main %*