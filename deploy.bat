@echo off
REM QFLARE Production Deployment Script for Windows

setlocal enabledelayedexpansion

REM Colors for output (Windows CMD)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "RESET=[0m"

REM Configuration
set COMPOSE_FILE=docker-compose.prod.yml
set ENV_FILE=.env.prod

REM Functions
:log_info
echo [94m[INFO][0m %~1
goto :eof

:log_success
echo [92m[SUCCESS][0m %~1
goto :eof

:log_warning
echo [93m[WARNING][0m %~1
goto :eof

:log_error
echo [91m[ERROR][0m %~1
goto :eof

:check_dependencies
call :log_info "Checking dependencies..."

REM Check Docker
docker --version >nul 2>&1
if errorlevel 1 (
    call :log_error "Docker is not installed. Please install Docker first."
    exit /b 1
)

REM Check Docker Compose
docker-compose --version >nul 2>&1
if errorlevel 1 (
    docker compose version >nul 2>&1
    if errorlevel 1 (
        call :log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit /b 1
    )
)
call :log_success "Dependencies check passed"
goto :eof

:setup_environment
call :log_info "Setting up environment..."

REM Copy environment file if it doesn't exist
if not exist ".env" (
    if exist "%ENV_FILE%" (
        copy "%ENV_FILE%" .env >nul
        call :log_warning "Copied %ENV_FILE% to .env"
        call :log_warning "Please edit .env file with your production values before proceeding"
        echo.
        pause
    ) else (
        call :log_error "Environment file %ENV_FILE% not found"
        exit /b 1
    )
)

REM Generate secure JWT secret if not set
findstr "your-super-secure-jwt-secret" .env >nul
if not errorlevel 1 (
    REM Generate a random secret (simplified for Windows)
    set "NEW_SECRET="
    for /l %%i in (1,1,64) do (
        set /a "rand=!random! %% 16"
        if !rand! lss 10 (
            set "NEW_SECRET=!NEW_SECRET!!rand!"
        ) else (
            set /a "letter=!rand! + 87"
            cmd /c exit !letter!
            set "char=!exitcode!"
            for %%c in (!char!) do set "NEW_SECRET=!NEW_SECRET!%%c"
        )
    )

    powershell -Command "(Get-Content .env) -replace 'your-super-secure-jwt-secret-here-change-this', '%NEW_SECRET%' | Set-Content .env"
    call :log_success "Generated secure JWT secret"
)

call :log_success "Environment setup completed"
goto :eof

:create_networks
call :log_info "Creating Docker networks..."

docker network create qflare-network 2>nul

call :log_success "Networks created"
goto :eof

:build_images
call :log_info "Building Docker images..."

docker-compose -f "%COMPOSE_FILE%" build --no-cache
if errorlevel 1 (
    REM Try with new Docker Compose syntax
    docker compose -f "%COMPOSE_FILE%" build --no-cache
)

call :log_success "Images built successfully"
goto :eof

:start_services
call :log_info "Starting QFLARE services..."

docker-compose -f "%COMPOSE_FILE%" up -d
if errorlevel 1 (
    docker compose -f "%COMPOSE_FILE%" up -d
)

call :log_success "Services started"
goto :eof

:wait_for_services
call :log_info "Waiting for services to be ready..."

REM Wait for Redis
call :log_info "Waiting for Redis..."
timeout /t 30 /nobreak >nul
for /l %%i in (1,1,30) do (
    docker exec qflare-redis redis-cli ping 2>nul | findstr PONG >nul
    if not errorlevel 1 goto redis_ready
    timeout /t 2 /nobreak >nul
)
call :log_error "Redis failed to start"
exit /b 1

:redis_ready
REM Wait for QFLARE server
call :log_info "Waiting for QFLARE server..."
for /l %%i in (1,1,30) do (
    powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:8000/health' -TimeoutSec 5 | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
    if not errorlevel 1 goto server_ready
    timeout /t 2 /nobreak >nul
)
call :log_error "QFLARE server failed to start"
exit /b 1

:server_ready
call :log_success "All services are ready"
goto :eof

:show_status
call :log_info "Service Status:"
echo.

docker-compose -f "%COMPOSE_FILE%" ps
if errorlevel 1 (
    docker compose -f "%COMPOSE_FILE%" ps
)

echo.
call :log_success "QFLARE Production Deployment Complete!"
echo.
echo 🌐 QFLARE Dashboard: http://localhost:8000
echo 🔴 Redis Commander:  http://localhost:8081
echo 📊 Prometheus:       http://localhost:9090
echo 📈 Grafana:          http://localhost:3000
echo.
echo To view logs:
echo   docker-compose -f %COMPOSE_FILE% logs -f qflare-server
echo.
echo To stop services:
echo   docker-compose -f %COMPOSE_FILE% down
goto :eof

:cleanup
call :log_info "Cleaning up temporary files..."
if exist ".env.bak" del ".env.bak"
call :log_success "Cleanup completed"
goto :eof

REM Main deployment process
:main
echo 🚀 QFLARE Production Deployment
echo ==============================
echo.

call :check_dependencies
call :setup_environment
call :create_networks
call :build_images
call :start_services
call :wait_for_services
call :show_status
call :cleanup

echo.
call :log_success "🎉 Deployment completed successfully!"
echo.
echo Next steps:
echo 1. Open http://localhost:8000 to access the dashboard
echo 2. Register devices using the device simulator
echo 3. Start federated learning rounds
echo 4. Monitor performance with Grafana dashboards
goto :eof

REM Handle command line arguments
if "%1"=="build" goto build_only
if "%1"=="start" goto start_only
if "%1"=="stop" goto stop_only
if "%1"=="restart" goto restart_only
if "%1"=="logs" goto logs_only
if "%1"=="status" goto show_status
if "%1"=="validate" goto validate_only
if "%1"=="cleanup" goto cleanup_only
if "%1"=="backup" goto backup_only

goto main

:build_only
call :check_dependencies
call :build_images
goto :eof

:start_only
call :check_dependencies
call :start_services
call :wait_for_services
call :show_status
goto :eof

:stop_only
call :log_info "Stopping services..."
docker-compose -f "%COMPOSE_FILE%" down
if errorlevel 1 (
    docker compose -f "%COMPOSE_FILE%" down
)
call :log_success "Services stopped"
goto :eof

:restart_only
call :log_info "Restarting services..."
docker-compose -f "%COMPOSE_FILE%" down
if errorlevel 1 (
    docker compose -f "%COMPOSE_FILE%" down
)
docker-compose -f "%COMPOSE_FILE%" up -d
if errorlevel 1 (
    docker compose -f "%COMPOSE_FILE%" up -d
)
call :wait_for_services
call :show_status
goto :eof

:logs_only
set "service=%2"
if "%service%"=="" set "service=qflare-server"
docker-compose -f "%COMPOSE_FILE%" logs -f "%service%"
if errorlevel 1 (
    docker compose -f "%COMPOSE_FILE%" logs -f "%service%"
)
goto :eof

:validate_only
call :log_info "Running production validation..."
if exist "validate_production.bat" (
    call validate_production.bat
) else (
    call :log_error "Validation script not found. Run validation separately."
    exit /b 1
)
goto :eof

:cleanup_only
call :log_info "Cleaning up containers and volumes..."
docker-compose -f "%COMPOSE_FILE%" down -v --remove-orphans
if errorlevel 1 (
    docker compose -f "%COMPOSE_FILE%" down -v --remove-orphans
)
docker system prune -f
call :log_success "Cleanup completed"
goto :eof

:backup_only
call :log_info "Creating backup..."
set "BACKUP_DIR=backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "BACKUP_DIR=%BACKUP_DIR: =0%"
mkdir "%BACKUP_DIR%" 2>nul

REM Backup database
docker exec qflare-server sqlite3 /app/data/qflare_prod.db ".backup /tmp/backup.db" 2>nul
docker cp qflare-server:/tmp/backup.db "%BACKUP_DIR%/qflare_prod.db" 2>nul

REM Backup configurations
copy .env "%BACKUP_DIR%\" >nul 2>&1
copy docker-compose.prod.yml "%BACKUP_DIR%\" >nul 2>&1

call :log_success "Backup created in %BACKUP_DIR%"
goto :eof