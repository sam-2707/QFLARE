#!/bin/bash
# QFLARE Production Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env.prod"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    log_success "Dependencies check passed"
}

setup_environment() {
    log_info "Setting up environment..."

    # Copy environment file if it doesn't exist
    if [ ! -f ".env" ]; then
        if [ -f "$ENV_FILE" ]; then
            cp "$ENV_FILE" .env
            log_warning "Copied $ENV_FILE to .env"
            log_warning "Please edit .env file with your production values before proceeding"
            echo ""
            read -p "Press Enter after editing .env file..."
        else
            log_error "Environment file $ENV_FILE not found"
            exit 1
        fi
    fi

    # Generate secure JWT secret if not set
    if grep -q "your-super-secure-jwt-secret" .env; then
        NEW_SECRET=$(openssl rand -hex 32)
        sed -i.bak "s/your-super-secure-jwt-secret-here-change-this/$NEW_SECRET/" .env
        log_success "Generated secure JWT secret"
    fi

    log_success "Environment setup completed"
}

create_networks() {
    log_info "Creating Docker networks..."

    docker network create qflare-network 2>/dev/null || true

    log_success "Networks created"
}

build_images() {
    log_info "Building Docker images..."

    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" build --no-cache
    else
        docker compose -f "$COMPOSE_FILE" build --no-cache
    fi

    log_success "Images built successfully"
}

start_services() {
    log_info "Starting QFLARE services..."

    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" up -d
    else
        docker compose -f "$COMPOSE_FILE" up -d
    fi

    log_success "Services started"
}

wait_for_services() {
    log_info "Waiting for services to be ready..."

    # Wait for Redis
    log_info "Waiting for Redis..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker exec qflare-redis redis-cli ping | grep -q PONG; then
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done

    if [ $timeout -le 0 ]; then
        log_error "Redis failed to start"
        exit 1
    fi

    # Wait for QFLARE server
    log_info "Waiting for QFLARE server..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:8000/health &>/dev/null; then
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done

    if [ $timeout -le 0 ]; then
        log_error "QFLARE server failed to start"
        exit 1
    fi

    log_success "All services are ready"
}

show_status() {
    log_info "Service Status:"
    echo ""

    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" ps
    else
        docker compose -f "$COMPOSE_FILE" ps
    fi

    echo ""
    log_success "QFLARE Production Deployment Complete!"
    echo ""
    echo "🌐 QFLARE Dashboard: http://localhost:8000"
    echo "🔴 Redis Commander:  http://localhost:8081"
    echo "📊 Prometheus:       http://localhost:9090"
    echo "📈 Grafana:          http://localhost:3000"
    echo ""
    echo "To view logs:"
    echo "  docker-compose -f $COMPOSE_FILE logs -f qflare-server"
    echo ""
    echo "To stop services:"
    echo "  docker-compose -f $COMPOSE_FILE down"
}

cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f .env.bak 2>/dev/null || true
    log_success "Cleanup completed"
}

# Main deployment process
main() {
    echo "🚀 QFLARE Production Deployment"
    echo "=============================="
    echo ""

    check_dependencies
    setup_environment
    create_networks
    build_images
    start_services
    wait_for_services
    show_status
    cleanup

    echo ""
    log_success "🎉 Deployment completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Open http://localhost:8000 to access the dashboard"
    echo "2. Register devices using the device simulator"
    echo "3. Start federated learning rounds"
    echo "4. Monitor performance with Grafana dashboards"
}

# Handle command line arguments
case "${1:-}" in
    "build")
        check_dependencies
        build_images
        ;;
    "start")
        check_dependencies
        start_services
        wait_for_services
        show_status
        ;;
    "stop")
        log_info "Stopping services..."
        if command -v docker-compose &> /dev/null; then
            docker-compose -f "$COMPOSE_FILE" down
        else
            docker compose -f "$COMPOSE_FILE" down
        fi
        log_success "Services stopped"
        ;;
    "restart")
        log_info "Restarting services..."
        if command -v docker-compose &> /dev/null; then
            docker-compose -f "$COMPOSE_FILE" down
            docker-compose -f "$COMPOSE_FILE" up -d
        else
            docker compose -f "$COMPOSE_FILE" down
            docker compose -f "$COMPOSE_FILE" up -d
        fi
        wait_for_services
        show_status
        ;;
    "logs")
        if command -v docker-compose &> /dev/null; then
            docker-compose -f "$COMPOSE_FILE" logs -f "${2:-qflare-server}"
        else
            docker compose -f "$COMPOSE_FILE" logs -f "${2:-qflare-server}"
        fi
        ;;
    "status")
        show_status
        ;;
    "validate")
        log_info "Running production validation..."
        if [ -f "validate_production.sh" ]; then
            bash validate_production.sh
        else
            log_error "Validation script not found. Run validation separately."
            exit 1
        fi
        ;;
    "cleanup")
        log_info "Cleaning up containers and volumes..."
        if command -v docker-compose &> /dev/null; then
            docker-compose -f "$COMPOSE_FILE" down -v --remove-orphans
        else
            docker compose -f "$COMPOSE_FILE" down -v --remove-orphans
        fi
        docker system prune -f
        log_success "Cleanup completed"
        ;;
    "backup")
        log_info "Creating backup..."
        BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$BACKUP_DIR"

        # Backup database
        if command -v docker-compose &> /dev/null; then
            docker-compose -f "$COMPOSE_FILE" exec qflare-server sqlite3 /app/data/qflare_prod.db ".backup /tmp/backup.db" 2>/dev/null || true
            docker cp "$(docker-compose -f "$COMPOSE_FILE" ps -q qflare-server)":/tmp/backup.db "$BACKUP_DIR/qflare_prod.db" 2>/dev/null || true
        else
            docker compose -f "$COMPOSE_FILE" exec qflare-server sqlite3 /app/data/qflare_prod.db ".backup /tmp/backup.db" 2>/dev/null || true
            docker cp "$(docker compose -f "$COMPOSE_FILE" ps -q qflare-server)":/tmp/backup.db "$BACKUP_DIR/qflare_prod.db" 2>/dev/null || true
        fi

        # Backup configurations
        cp .env "$BACKUP_DIR/" 2>/dev/null || true
        cp docker-compose.prod.yml "$BACKUP_DIR/" 2>/dev/null || true

        log_success "Backup created in $BACKUP_DIR"
        ;;
    *)
        main
        ;;
esac