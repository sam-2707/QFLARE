#!/bin/bash
# QFLARE Production Deployment Validation Script
# This script validates the production deployment setup

set -e

echo "🔍 QFLARE Production Deployment Validator"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "success" ]; then
        echo -e "${GREEN}✅ $message${NC}"
    elif [ "$status" = "warning" ]; then
        echo -e "${YELLOW}⚠️  $message${NC}"
    else
        echo -e "${RED}❌ $message${NC}"
    fi
}

# Check if Docker is running
check_docker() {
    echo "🐳 Checking Docker..."
    if ! docker info >/dev/null 2>&1; then
        print_status "error" "Docker is not running or not accessible"
        echo "   Please start Docker and try again"
        exit 1
    fi
    print_status "success" "Docker is running"
}

# Check if Docker Compose is available
check_docker_compose() {
    echo "📦 Checking Docker Compose..."
    if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
        print_status "error" "Docker Compose is not installed"
        exit 1
    fi
    print_status "success" "Docker Compose is available"
}

# Check required files
check_files() {
    echo "📁 Checking required files..."
    local files=(
        "Dockerfile.prod"
        "docker-compose.prod.yml"
        "requirements.prod.txt"
        ".env"
        "config/redis.conf"
        "config/nginx.conf"
        "config/prometheus.yml"
    )

    local missing_files=()
    for file in "${files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        fi
    done

    if [ ${#missing_files[@]} -ne 0 ]; then
        print_status "error" "Missing required files:"
        for file in "${missing_files[@]}"; do
            echo "   - $file"
        done
        exit 1
    fi
    print_status "success" "All required files present"
}

# Validate environment file
validate_env() {
    echo "🔧 Validating environment configuration..."
    if [ ! -f ".env" ]; then
        print_status "error" ".env file not found"
        return 1
    fi

    # Check for required environment variables
    local required_vars=(
        "ENVIRONMENT"
        "QFLARE_JWT_SECRET"
        "DATABASE_URL"
        "REDIS_URL"
    )

    local missing_vars=()
    for var in "${required_vars[@]}"; do
        if ! grep -q "^${var}=" .env; then
            missing_vars+=("$var")
        fi
    done

    if [ ${#missing_vars[@]} -ne 0 ]; then
        print_status "warning" "Missing environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "   - $var"
        done
        echo "   Using default values where possible"
    else
        print_status "success" "Environment configuration valid"
    fi
}

# Test Docker build
test_build() {
    echo "🏗️  Testing Docker build..."
    if ! docker-compose -f docker-compose.prod.yml build --no-cache >/dev/null 2>&1; then
        print_status "error" "Docker build failed"
        echo "   Check build logs: docker-compose -f docker-compose.prod.yml build"
        return 1
    fi
    print_status "success" "Docker build successful"
}

# Test service startup
test_startup() {
    echo "🚀 Testing service startup..."
    if ! docker-compose -f docker-compose.prod.yml up -d >/dev/null 2>&1; then
        print_status "error" "Service startup failed"
        echo "   Check logs: docker-compose -f docker-compose.prod.yml logs"
        return 1
    fi

    # Wait for services to be healthy
    echo "   Waiting for services to be ready..."
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
            print_status "success" "Services started successfully"
            return 0
        fi
        echo "   Attempt $attempt/$max_attempts: Waiting..."
        sleep 2
        ((attempt++))
    done

    print_status "error" "Services failed to start within timeout"
    echo "   Check logs: docker-compose -f docker-compose.prod.yml logs"
    return 1
}

# Test health endpoints
test_health() {
    echo "🏥 Testing health endpoints..."
    local endpoints=(
        "http://localhost:8000/health:QFLARE Server"
        "http://localhost:9090/-/healthy:Prometheus"
    )

    local failed_endpoints=()
    for endpoint in "${endpoints[@]}"; do
        local url=$(echo "$endpoint" | cut -d: -f1)
        local name=$(echo "$endpoint" | cut -d: -f2)

        if ! curl -f -s "$url" >/dev/null 2>&1; then
            failed_endpoints+=("$name ($url)")
        fi
    done

    if [ ${#failed_endpoints[@]} -ne 0 ]; then
        print_status "warning" "Some health checks failed:"
        for endpoint in "${failed_endpoints[@]}"; do
            echo "   - $endpoint"
        done
    else
        print_status "success" "All health checks passed"
    fi
}

# Test API endpoints
test_api() {
    echo "🔗 Testing API endpoints..."
    local api_endpoints=(
        "http://localhost:8000/docs:API Documentation"
        "http://localhost:8000/openapi.json:OpenAPI Spec"
    )

    local failed_apis=()
    for endpoint in "${api_endpoints[@]}"; do
        local url=$(echo "$endpoint" | cut -d: -f1)
        local name=$(echo "$endpoint" | cut -d: -f2)

        if ! curl -f -s "$url" >/dev/null 2>&1; then
            failed_apis+=("$name ($url)")
        fi
    done

    if [ ${#failed_apis[@]} -ne 0 ]; then
        print_status "warning" "Some API endpoints failed:"
        for api in "${failed_apis[@]}"; do
            echo "   - $api"
        done
    else
        print_status "success" "API endpoints accessible"
    fi
}

# Cleanup function
cleanup() {
    echo "🧹 Cleaning up test containers..."
    docker-compose -f docker-compose.prod.yml down -v >/dev/null 2>&1
    print_status "success" "Cleanup completed"
}

# Main validation function
main() {
    echo ""

    # Run all checks
    check_docker
    check_docker_compose
    check_files
    validate_env

    # Build and test
    if test_build; then
        if test_startup; then
            test_health
            test_api
        fi
    fi

    # Cleanup
    cleanup

    echo ""
    echo "🎯 Validation Complete!"
    echo "======================"
    echo "If all checks passed, your production deployment is ready!"
    echo "Run './deploy.sh' to start the production services."
}

# Run main function
main "$@"