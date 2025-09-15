# QFLARE Deployment Scripts
# Production deployment automation and management

#!/bin/bash
set -euo pipefail

# QFLARE Production Deployment Script
# Usage: ./deploy.sh [environment] [component]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
ENVIRONMENT="${1:-production}"
COMPONENT="${2:-all}"
NAMESPACE="qflare"
REGISTRY="ghcr.io"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check docker
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check helm (optional)
    if ! command -v helm &> /dev/null; then
        log_warning "helm is not installed - some features may not be available"
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Setup namespace and basic resources
setup_namespace() {
    log_info "Setting up namespace: $NAMESPACE"
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        kubectl create namespace "$NAMESPACE"
        log_success "Created namespace: $NAMESPACE"
    else
        log_info "Namespace $NAMESPACE already exists"
    fi
    
    # Label namespace for network policies
    kubectl label namespace "$NAMESPACE" name="$NAMESPACE" --overwrite
    
    log_success "Namespace setup completed"
}

# Deploy secrets
deploy_secrets() {
    log_info "Deploying secrets..."
    
    # Check if secrets file exists
    if [[ ! -f "$PROJECT_ROOT/k8s/secrets.yaml" ]]; then
        log_error "Secrets file not found: $PROJECT_ROOT/k8s/secrets.yaml"
        exit 1
    fi
    
    # Apply secrets
    kubectl apply -f "$PROJECT_ROOT/k8s/secrets.yaml"
    
    log_success "Secrets deployed successfully"
}

# Deploy ConfigMaps
deploy_configmaps() {
    log_info "Deploying ConfigMaps..."
    
    kubectl apply -f "$PROJECT_ROOT/k8s/configmaps.yaml"
    
    log_success "ConfigMaps deployed successfully"
}

# Deploy database components
deploy_database() {
    log_info "Deploying database components..."
    
    # Deploy PostgreSQL
    kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: $NAMESPACE
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: qflare_db
        - name: POSTGRES_USER
          value: qflare
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: qflare-secrets
              key: postgres-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
EOF
    
    # Wait for PostgreSQL to be ready
    log_info "Waiting for PostgreSQL to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres -n "$NAMESPACE" --timeout=300s
    
    log_success "Database components deployed successfully"
}

# Deploy Redis
deploy_redis() {
    log_info "Deploying Redis..."
    
    kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: $NAMESPACE
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
EOF
    
    # Wait for Redis to be ready
    log_info "Waiting for Redis to be ready..."
    kubectl wait --for=condition=ready pod -l app=redis -n "$NAMESPACE" --timeout=120s
    
    log_success "Redis deployed successfully"
}

# Deploy QFLARE server
deploy_server() {
    log_info "Deploying QFLARE server..."
    
    # Update deployment with current image tag
    sed "s|qflare/server:latest|${REGISTRY}/qflare/server:${IMAGE_TAG}|g" \
        "$PROJECT_ROOT/k8s/deployment.yaml" | \
        kubectl apply -f -
    
    # Wait for deployment to complete
    log_info "Waiting for QFLARE server deployment to complete..."
    kubectl rollout status deployment/qflare-server -n "$NAMESPACE" --timeout=600s
    
    log_success "QFLARE server deployed successfully"
}

# Deploy monitoring components
deploy_monitoring() {
    log_info "Deploying monitoring components..."
    
    # Deploy Prometheus
    kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        args:
        - --config.file=/etc/prometheus/prometheus.yml
        - --storage.tsdb.path=/prometheus
        - --web.enable-lifecycle
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus
      volumes:
      - name: prometheus-config
        configMap:
          name: prometheus-config
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: $NAMESPACE
spec:
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090
EOF
    
    # Deploy Grafana
    kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:latest
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          valueFrom:
            secretKeyRef:
              name: qflare-secrets
              key: grafana-password
        ports:
        - containerPort: 3000
        volumeMounts:
        - name: grafana-config
          mountPath: /etc/grafana/provisioning
      volumes:
      - name: grafana-config
        configMap:
          name: grafana-config
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: $NAMESPACE
spec:
  selector:
    app: grafana
  ports:
  - port: 3000
    targetPort: 3000
EOF
    
    log_success "Monitoring components deployed successfully"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Check if all pods are running
    log_info "Checking pod status..."
    kubectl get pods -n "$NAMESPACE"
    
    # Wait for all deployments to be ready
    log_info "Waiting for all deployments to be ready..."
    kubectl wait --for=condition=available deployment --all -n "$NAMESPACE" --timeout=300s
    
    # Test service endpoints
    log_info "Testing service endpoints..."
    
    # Get QFLARE server service IP
    SERVER_IP=$(kubectl get service qflare-server -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    
    # Test health endpoint
    if kubectl run --rm -i --tty --restart=Never test-pod --image=curlimages/curl -- \
        curl -f "http://${SERVER_IP}:8000/health"; then
        log_success "QFLARE server health check passed"
    else
        log_error "QFLARE server health check failed"
        return 1
    fi
    
    # Test metrics endpoint
    if kubectl run --rm -i --tty --restart=Never test-pod --image=curlimages/curl -- \
        curl -f "http://${SERVER_IP}:9090/metrics"; then
        log_success "QFLARE server metrics check passed"
    else
        log_error "QFLARE server metrics check failed"
        return 1
    fi
    
    log_success "All health checks passed"
}

# Display deployment status
show_status() {
    log_info "Deployment Status:"
    echo "===================="
    
    echo -e "\n${BLUE}Namespace:${NC} $NAMESPACE"
    echo -e "${BLUE}Environment:${NC} $ENVIRONMENT"
    echo -e "${BLUE}Image Tag:${NC} $IMAGE_TAG"
    
    echo -e "\n${BLUE}Pods:${NC}"
    kubectl get pods -n "$NAMESPACE" -o wide
    
    echo -e "\n${BLUE}Services:${NC}"
    kubectl get services -n "$NAMESPACE"
    
    echo -e "\n${BLUE}Ingresses:${NC}"
    kubectl get ingresses -n "$NAMESPACE" 2>/dev/null || echo "No ingresses found"
    
    echo -e "\n${BLUE}Persistent Volumes:${NC}"
    kubectl get pvc -n "$NAMESPACE"
    
    # Get external access information
    EXTERNAL_IP=$(kubectl get service qflare-server -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "Pending...")
    
    echo -e "\n${BLUE}Access Information:${NC}"
    echo -e "QFLARE Server: http://${EXTERNAL_IP}:8000"
    echo -e "QFLARE HTTPS: https://${EXTERNAL_IP}:8443"
    echo -e "Metrics: http://${EXTERNAL_IP}:9090/metrics"
    echo -e "Grafana: http://$(kubectl get service grafana -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}'):3000"
    echo -e "Prometheus: http://$(kubectl get service prometheus -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}'):9090"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up deployment..."
    
    read -p "Are you sure you want to delete the entire QFLARE deployment? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kubectl delete namespace "$NAMESPACE"
        log_success "QFLARE deployment cleaned up successfully"
    else
        log_info "Cleanup cancelled"
    fi
}

# Rollback function
rollback() {
    local REVISION="${1:-}"
    
    log_info "Rolling back QFLARE server deployment..."
    
    if [[ -n "$REVISION" ]]; then
        kubectl rollout undo deployment/qflare-server -n "$NAMESPACE" --to-revision="$REVISION"
    else
        kubectl rollout undo deployment/qflare-server -n "$NAMESPACE"
    fi
    
    kubectl rollout status deployment/qflare-server -n "$NAMESPACE" --timeout=300s
    
    log_success "Rollback completed successfully"
}

# Main deployment function
deploy() {
    log_info "Starting QFLARE deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Component: $COMPONENT"
    log_info "Image Tag: $IMAGE_TAG"
    
    check_prerequisites
    setup_namespace
    
    case "$COMPONENT" in
        "all")
            deploy_secrets
            deploy_configmaps
            deploy_database
            deploy_redis
            deploy_server
            deploy_monitoring
            ;;
        "secrets")
            deploy_secrets
            ;;
        "configmaps")
            deploy_configmaps
            ;;
        "database")
            deploy_database
            ;;
        "redis")
            deploy_redis
            ;;
        "server")
            deploy_server
            ;;
        "monitoring")
            deploy_monitoring
            ;;
        *)
            log_error "Unknown component: $COMPONENT"
            log_info "Available components: all, secrets, configmaps, database, redis, server, monitoring"
            exit 1
            ;;
    esac
    
    run_health_checks
    show_status
    
    log_success "QFLARE deployment completed successfully!"
}

# Main script logic
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "status")
        show_status
        ;;
    "cleanup")
        cleanup
        ;;
    "rollback")
        rollback "${2:-}"
        ;;
    "health")
        run_health_checks
        ;;
    *)
        echo "Usage: $0 {deploy|status|cleanup|rollback|health} [options]"
        echo ""
        echo "Commands:"
        echo "  deploy [environment] [component]  - Deploy QFLARE (default: production all)"
        echo "  status                           - Show deployment status"
        echo "  cleanup                          - Clean up entire deployment"
        echo "  rollback [revision]              - Rollback to previous or specific revision"
        echo "  health                          - Run health checks"
        echo ""
        echo "Examples:"
        echo "  $0 deploy production all"
        echo "  $0 deploy staging server"
        echo "  $0 rollback 3"
        echo "  $0 status"
        exit 1
        ;;
esac