# QFLARE Deployment Guide

## Overview

This guide covers the complete deployment process for QFLARE across different environments, from development setup to production deployment with high availability.

## Prerequisites

### System Requirements

#### Minimum Requirements (Development)
- CPU: 2 cores
- RAM: 4GB
- Storage: 50GB
- OS: Linux/macOS/Windows with Docker support

#### Recommended Requirements (Production)
- CPU: 8+ cores per node
- RAM: 16GB+ per node
- Storage: 500GB+ SSD
- Network: Gigabit Ethernet
- OS: Ubuntu 20.04+ LTS or RHEL 8+

### Software Dependencies

- Docker Engine 20.10+
- Docker Compose 2.0+
- Kubernetes 1.24+ (for production)
- kubectl CLI
- Helm 3.8+ (optional)
- Git 2.30+
- Python 3.10+
- Node.js 18+ (for frontend development)
- **Network Plugin**: Calico, Flannel, or Weave Net
- **Storage**: Dynamic provisioning support (e.g., Ceph, AWS EBS)
- **Ingress Controller**: NGINX Ingress Controller

### Software Dependencies

#### Required Tools
```bash
# Kubernetes management
kubectl >= 1.25
helm >= 3.8

# Container management
docker >= 20.10
docker-compose >= 2.0

# Development tools (for custom builds)
git >= 2.30
python >= 3.11
cmake >= 3.20
ninja-build
```

#### Intel SGX Setup (for TEE support)
```bash
# Install Intel SGX SDK and drivers
wget https://download.01.org/intel-sgx/sgx-linux/2.23/distro/ubuntu22.04-server/sgx_linux_x64_sdk_2.23.100.2.bin
chmod +x sgx_linux_x64_sdk_2.23.100.2.bin
echo -e 'no\n/opt/intel' | ./sgx_linux_x64_sdk_2.23.100.2.bin

# Install SGX runtime
echo 'deb [arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu jammy main' | sudo tee /etc/apt/sources.list.d/intel-sgx.list
wget -O - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | sudo apt-key add -
sudo apt update
sudo apt install sgx-aesm-service libsgx-aesm-launch-plugin libsgx-aesm-pce-plugin
```

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/your-org/qflare.git
cd qflare
```

### 2. Development Environment
```bash
# Start development environment
docker-compose -f docker/docker-compose.dev.yml up -d

# Verify services
curl http://localhost:8000/health
```

### 3. Production Deployment
```bash
# Deploy to Kubernetes
./scripts/deploy.sh deploy production all

# Check deployment status
./scripts/deploy.sh status
```

## Detailed Deployment

### Docker Deployment

#### Single-Node Development
```bash
# Copy environment template
cp docker/.env.template docker/.env

# Edit environment variables
vim docker/.env

# Start all services
docker-compose -f docker/docker-compose.dev.yml up -d

# Monitor logs
docker-compose -f docker/docker-compose.dev.yml logs -f
```

#### Multi-Node Production
```bash
# Copy production environment
cp docker/.env.prod.template docker/.env

# Configure production settings
vim docker/.env

# Start production services
docker-compose -f docker/docker-compose.prod.yml up -d

# Verify deployment
curl https://your-domain.com/health
```

### Kubernetes Deployment

#### 1. Prepare Environment
```bash
# Create namespace
kubectl create namespace qflare

# Setup secrets
kubectl apply -f k8s/secrets.yaml

# Apply configuration
kubectl apply -f k8s/configmaps.yaml
```

#### 2. Deploy Database Layer
```bash
# Deploy PostgreSQL
kubectl apply -f k8s/postgres.yaml

# Deploy Redis
kubectl apply -f k8s/redis.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n qflare --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n qflare --timeout=120s
```

#### 3. Deploy QFLARE Components
```bash
# Deploy server
kubectl apply -f k8s/deployment.yaml

# Deploy monitoring
kubectl apply -f k8s/monitoring.yaml

# Setup ingress
kubectl apply -f k8s/ingress.yaml
```

#### 4. Deploy Edge Nodes
```bash
# Deploy edge node DaemonSet
kubectl apply -f k8s/edge-nodes.yaml

# Or deploy individual edge nodes
kubectl apply -f k8s/edge-deployment.yaml
```

### Automated Deployment Script

#### Basic Usage
```bash
# Deploy everything to production
./scripts/deploy.sh deploy production all

# Deploy only server component
./scripts/deploy.sh deploy production server

# Deploy to staging environment
./scripts/deploy.sh deploy staging all
```

#### Advanced Options
```bash
# Deploy with custom image tag
IMAGE_TAG=v1.2.3 ./scripts/deploy.sh deploy production all

# Deploy to custom namespace
NAMESPACE=qflare-test ./scripts/deploy.sh deploy production all

# Deploy with specific registry
REGISTRY=your-registry.com ./scripts/deploy.sh deploy production all
```

## Configuration

### Environment Variables

#### Server Configuration
```bash
# Database
DATABASE_URL=postgresql://qflare:password@postgres:5432/qflare_db
REDIS_URL=redis://redis:6379/0

# Security
QFLARE_JWT_SECRET=your-super-secret-jwt-key
SGX_SPID=your-sgx-spid
SGX_IAS_KEY=your-sgx-ias-key

# TEE Configuration
QFLARE_SGX_MODE=HW  # HW or SIM
QFLARE_TEE_PREFERRED=auto  # auto, sgx, sev, mock

# Monitoring
QFLARE_LOG_LEVEL=INFO
PROMETHEUS_ENABLED=true
```

#### Edge Node Configuration
```bash
# Server Connection
QFLARE_SERVER_URL=https://qflare-server.your-domain.com
QFLARE_DEVICE_ID=edge-node-001
QFLARE_DEVICE_TYPE=edge

# Training Parameters
QFLARE_BATCH_SIZE=16
QFLARE_LOCAL_EPOCHS=5
QFLARE_LEARNING_RATE=0.01
```

### Custom Configuration Files

#### Server Config (config/server-config.yaml)
```yaml
qflare:
  server:
    host: "0.0.0.0"
    port: 8000
    ssl_port: 8443
    workers: 4
  
  federated_learning:
    training:
      min_clients: 2
      max_clients: 100
      rounds: 1000
      client_fraction: 0.3
    
    algorithms:
      default: "fedavg"
      available:
        - "fedavg"
        - "fedprox"
        - "fedbn"
        - "per_fedavg"
  
  security:
    pqc:
      enabled: true
      signature_algorithm: "Dilithium3"
      kem_algorithm: "Kyber768"
```

#### Edge Config (config/edge-config.yaml)
```yaml
qflare:
  edge_node:
    device_type: "edge"
    communication:
      heartbeat_interval: 30
      registration_retry_interval: 60
  
  training:
    local:
      batch_size: 16
      epochs: 5
      learning_rate: 0.01
  
  security:
    pqc:
      enabled: true
      signature_algorithm: "Dilithium3"
```

## Monitoring and Observability

### Metrics Collection

#### Prometheus Configuration
```yaml
# Scrape QFLARE server metrics
- job_name: 'qflare-server'
  static_configs:
    - targets: ['qflare-server:9090']
  metrics_path: '/metrics'
  scrape_interval: 30s

# Scrape edge node metrics
- job_name: 'qflare-edge'
  kubernetes_sd_configs:
    - role: pod
  relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_app]
      action: keep
      regex: qflare-edge
```

#### Key Metrics
- `qflare_fl_rounds_completed_total` - Completed FL rounds
- `qflare_edge_nodes_connected` - Active edge nodes
- `qflare_tee_operations_total` - TEE operations count
- `qflare_fl_training_duration_seconds` - Training duration
- `qflare_model_accuracy` - Current model accuracy

### Logging

#### Centralized Logging Setup
```yaml
# Loki configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: loki-config
data:
  config.yaml: |
    auth_enabled: false
    server:
      http_listen_port: 3100
    ingester:
      lifecycler:
        ring:
          kvstore:
            store: inmemory
```

#### Log Aggregation
```bash
# View server logs
kubectl logs -f deployment/qflare-server -n qflare

# View edge node logs
kubectl logs -f daemonset/qflare-edge -n qflare

# View aggregated logs
kubectl logs -l app=qflare -n qflare --tail=100
```

### Grafana Dashboards

#### QFLARE Dashboard Import
1. Access Grafana: `http://grafana.your-domain.com`
2. Login with admin credentials
3. Import dashboard: `k8s/grafana/qflare-dashboard.json`
4. Configure data sources:
   - Prometheus: `http://prometheus:9090`
   - Loki: `http://loki:3100`

## Security Hardening

### Network Security

#### Network Policies
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: qflare-network-policy
spec:
  podSelector:
    matchLabels:
      app: qflare-server
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: qflare-edge
    ports:
    - protocol: TCP
      port: 8443
```

#### TLS Configuration
```bash
# Generate certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key -out tls.crt \
  -subj "/CN=qflare.your-domain.com"

# Create TLS secret
kubectl create secret tls qflare-tls \
  --cert=tls.crt --key=tls.key -n qflare
```

### Access Control

#### RBAC Setup
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: qflare
  name: qflare-role
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list", "watch"]
```

#### Service Account
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: qflare-service-account
  namespace: qflare
automountServiceAccountToken: false
```

### Secret Management

#### External Secret Management
```bash
# Using sealed-secrets
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.18.0/controller.yaml

# Create sealed secret
echo -n mypassword | kubectl create secret generic mysecret --dry-run=client --from-file=password=/dev/stdin -o yaml | kubeseal -o yaml
```

## Scaling and Performance

### Horizontal Scaling

#### Server Scaling
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: qflare-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: qflare-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

#### Database Scaling
```yaml
# PostgreSQL cluster (using Postgres Operator)
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: postgres-cluster
spec:
  instances: 3
  postgresql:
    parameters:
      max_connections: "200"
      shared_preload_libraries: "pg_stat_statements"
```

### Performance Optimization

#### Resource Allocation
```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
    sgx.intel.com/sgx: "1"
  limits:
    memory: "4Gi"
    cpu: "2000m"
    sgx.intel.com/sgx: "1"
```

#### Node Affinity
```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: sgx.intel.com/sgx
          operator: In
          values: ["true"]
```

## Backup and Recovery

### Database Backup

#### Automated Backup Script
```bash
#!/bin/bash
# Backup script for PostgreSQL

NAMESPACE=qflare
POD_NAME=$(kubectl get pods -n $NAMESPACE -l app=postgres -o jsonpath='{.items[0].metadata.name}')
BACKUP_FILE="/backups/qflare-$(date +%Y%m%d-%H%M%S).sql"

kubectl exec -n $NAMESPACE $POD_NAME -- pg_dump -U qflare qflare_db > $BACKUP_FILE
echo "Backup completed: $BACKUP_FILE"
```

#### Scheduled Backup CronJob
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: qflare-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15-alpine
            command:
            - /bin/bash
            - -c
            - pg_dump -h postgres -U qflare qflare_db > /backup/backup-$(date +%Y%m%d).sql
```

### Disaster Recovery

#### Full System Backup
```bash
# Backup Kubernetes resources
kubectl get all,secrets,configmaps,pv,pvc -n qflare -o yaml > qflare-k8s-backup.yaml

# Backup persistent volumes
velero backup create qflare-backup --include-namespaces qflare
```

#### Recovery Procedure
```bash
# Restore from backup
kubectl apply -f qflare-k8s-backup.yaml

# Restore database
kubectl exec -i postgres-pod -- psql -U qflare qflare_db < backup.sql

# Verify recovery
./scripts/deploy.sh health
```

## Troubleshooting

### Common Issues

#### Pod Startup Issues
```bash
# Check pod status
kubectl get pods -n qflare

# View pod logs
kubectl logs pod-name -n qflare

# Describe pod for events
kubectl describe pod pod-name -n qflare
```

#### SGX TEE Issues
```bash
# Check SGX device availability
ls -la /dev/sgx*

# Verify SGX services
systemctl status aesmd

# Test SGX functionality
sgx-ra-sample
```

#### Network Connectivity
```bash
# Test internal connectivity
kubectl exec -it qflare-server-pod -- curl redis:6379

# Check service endpoints
kubectl get endpoints -n qflare

# Verify ingress
kubectl get ingress -n qflare
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods -n qflare

# Monitor metrics
curl http://qflare-server:9090/metrics

# Check database performance
kubectl exec -it postgres-pod -- psql -U qflare -c "SELECT * FROM pg_stat_activity;"
```

### Debugging Commands

#### Container Debugging
```bash
# Access running container
kubectl exec -it qflare-server-pod -- /bin/bash

# Check container logs
docker logs qflare-server

# Monitor container resources
docker stats qflare-server
```

#### Network Debugging
```bash
# Test DNS resolution
nslookup qflare-server.qflare.svc.cluster.local

# Check port connectivity
telnet qflare-server 8000

# Trace network path
traceroute qflare-server
```

## Maintenance

### Updates and Upgrades

#### Rolling Updates
```bash
# Update server image
kubectl set image deployment/qflare-server \
  qflare-server=qflare/server:v1.2.3 -n qflare

# Monitor rollout
kubectl rollout status deployment/qflare-server -n qflare

# Rollback if needed
kubectl rollout undo deployment/qflare-server -n qflare
```

#### Database Migrations
```bash
# Run database migrations
kubectl exec -it qflare-server-pod -- python -m alembic upgrade head

# Check migration status
kubectl exec -it qflare-server-pod -- python -m alembic current
```

### Health Monitoring

#### Automated Health Checks
```bash
# Server health check
curl -f https://qflare.your-domain.com/health

# Database health check
kubectl exec postgres-pod -- pg_isready -U qflare

# Redis health check
kubectl exec redis-pod -- redis-cli ping
```

#### Alerting Setup
```yaml
# Prometheus AlertManager rule
groups:
- name: qflare.rules
  rules:
  - alert: QFLAREServerDown
    expr: up{job="qflare-server"} == 0
    for: 5m
    annotations:
      summary: "QFLARE Server is down"
```

## Support and Community

### Documentation
- **API Documentation**: `/docs/api_docs.md`
- **Architecture Guide**: `/docs/system_design.md`
- **Troubleshooting**: `/TROUBLESHOOTING.md`

### Community Resources
- **GitHub Issues**: Report bugs and feature requests
- **Discussions**: Community Q&A and discussions
- **Wiki**: Additional documentation and guides

### Commercial Support
For enterprise support and consulting services, contact the QFLARE team.

---

This guide provides comprehensive instructions for deploying QFLARE in production environments. For additional help, refer to the troubleshooting guide or community resources.