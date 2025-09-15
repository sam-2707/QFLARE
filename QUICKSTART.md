# QFLARE Quick Start Guide
# How to run the QFLARE federated learning platform

## 🚀 Quick Start Options

### Option 1: Development Environment (Recommended for testing)
```powershell
# 1. Start development services with Docker Compose
docker-compose -f docker/docker-compose.dev.yml up -d

# 2. Wait for services to start (30-60 seconds)
Start-Sleep 60

# 3. Check if services are running
docker-compose -f docker/docker-compose.dev.yml ps

# 4. Test the server
curl http://localhost:8000/health
```

### Option 2: Local Python Development
```powershell
# 1. Set up Python environment
python -m venv qflare-env
qflare-env\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
pip install -r server/requirements.txt

# 3. Set environment variables
$env:DATABASE_URL = "sqlite:///qflare_dev.db"
$env:REDIS_URL = "redis://localhost:6379/0"
$env:QFLARE_JWT_SECRET = "dev-secret-key"
$env:QFLARE_SGX_MODE = "SIM"

# 4. Start Redis (if not using Docker)
# Install Redis from https://redis.io/download or use Docker:
docker run -d -p 6379:6379 redis:7-alpine

# 5. Initialize database
python -m server.database.init_db

# 6. Start the server
python -m server.main

# 7. In another terminal, start an edge node
python -m edge_node.main
```

### Option 3: Production Docker Deployment
```powershell
# 1. Copy production environment template
copy docker\.env.template docker\.env

# 2. Edit environment variables in docker\.env
# Set your production passwords, secrets, etc.

# 3. Start production services
docker-compose -f docker/docker-compose.prod.yml up -d

# 4. Check deployment status
docker-compose -f docker/docker-compose.prod.yml ps

# 5. View logs
docker-compose -f docker/docker-compose.prod.yml logs -f
```

## 🔧 Prerequisites Setup

### Install Docker Desktop (Windows)
```powershell
# Download and install Docker Desktop from:
# https://www.docker.com/products/docker-desktop

# Verify installation
docker --version
docker-compose --version
```

### Install Python Dependencies (if running locally)
```powershell
# Install Python 3.11+ from https://python.org
python --version

# Install required system packages
# For Windows, you might need Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

## 🏃‍♂️ Step-by-Step Running Instructions

### Method 1: Docker Development (Easiest)

#### Step 1: Start Backend Services
```powershell
# Navigate to project directory
cd D:\QFLARE_Project_Structure

# Start PostgreSQL and Redis
docker-compose -f docker/docker-compose.dev.yml up -d postgres-dev redis-dev

# Wait for databases to be ready
Start-Sleep 30
```

#### Step 2: Start QFLARE Server
```powershell
# Start the QFLARE server
docker-compose -f docker/docker-compose.dev.yml up -d qflare-server-dev

# Check server logs
docker-compose -f docker/docker-compose.dev.yml logs -f qflare-server-dev
```

#### Step 3: Start Edge Node
```powershell
# Start an edge node
docker-compose -f docker/docker-compose.dev.yml up -d qflare-edge-dev

# Check edge node logs
docker-compose -f docker/docker-compose.dev.yml logs -f qflare-edge-dev
```

#### Step 4: Access Services
- **QFLARE Server**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:9090/metrics
- **API Documentation**: http://localhost:8000/docs
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9091

### Method 2: Local Python Development

#### Step 1: Environment Setup
```powershell
# Create virtual environment
python -m venv venv
venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r server/requirements.txt
```

#### Step 2: Start Dependencies
```powershell
# Start Redis with Docker
docker run -d --name qflare-redis -p 6379:6379 redis:7-alpine

# Or start PostgreSQL (optional, SQLite is default for dev)
docker run -d --name qflare-postgres -p 5432:5432 -e POSTGRES_DB=qflare_dev -e POSTGRES_USER=qflare -e POSTGRES_PASSWORD=dev_pass postgres:15-alpine
```

#### Step 3: Configure Environment
```powershell
# Set environment variables
$env:QFLARE_CONFIG_PATH = "config/global_config.yaml"
$env:DATABASE_URL = "sqlite:///data/qflare_dev.db"
$env:REDIS_URL = "redis://localhost:6379/0"
$env:QFLARE_JWT_SECRET = "dev-secret-key-change-in-production"
$env:QFLARE_LOG_LEVEL = "DEBUG"
$env:QFLARE_SGX_MODE = "SIM"
```

#### Step 4: Initialize Database
```powershell
# Create data directory
mkdir -p data

# Initialize database (if using SQLite)
python -c "
from server.database.models import Base
from server.database.connection import engine
Base.metadata.create_all(engine)
print('Database initialized successfully!')
"
```

#### Step 5: Start Server
```powershell
# Start QFLARE server
python -m server.main
```

#### Step 6: Start Edge Node (in new terminal)
```powershell
# Activate environment
venv\Scripts\Activate.ps1

# Set edge node environment
$env:QFLARE_DEVICE_ID = "edge-node-001"
$env:QFLARE_SERVER_URL = "http://localhost:8000"
$env:QFLARE_LOG_LEVEL = "DEBUG"

# Start edge node
python -m edge_node.main
```

## 🧪 Testing the System

### Basic Health Checks
```powershell
# Test server health
curl http://localhost:8000/health

# Test metrics endpoint
curl http://localhost:9090/metrics

# Test edge node health
curl http://localhost:8001/health
```

### Run Federated Learning Test
```powershell
# Run the FL training test
python tests/test_fl_training.py

# Or run specific tests
python -m pytest tests/test_fl_training.py -v
```

### Monitor Training Progress
```powershell
# Watch server logs for FL progress
docker-compose -f docker/docker-compose.dev.yml logs -f qflare-server-dev | findstr "FL"

# Or if running locally:
# Check logs in data/logs/ directory
Get-Content data/logs/qflare-server.log -Wait | Select-String "FL"
```

## 📊 Monitoring and Visualization

### Access Grafana Dashboard
1. Open browser: http://localhost:3000
2. Login: admin / admin
3. Navigate to QFLARE dashboard
4. Monitor FL training metrics, system performance, and TEE operations

### View Prometheus Metrics
1. Open browser: http://localhost:9091
2. Query metrics like:
   - `qflare_fl_rounds_completed_total`
   - `qflare_edge_nodes_connected`
   - `qflare_tee_operations_total`

## 🛠️ Common Commands

### Docker Management
```powershell
# Stop all services
docker-compose -f docker/docker-compose.dev.yml down

# Restart specific service
docker-compose -f docker/docker-compose.dev.yml restart qflare-server-dev

# View logs
docker-compose -f docker/docker-compose.dev.yml logs qflare-server-dev

# Clean up
docker-compose -f docker/docker-compose.dev.yml down -v
docker system prune -f
```

### Development Workflow
```powershell
# Run tests
python -m pytest tests/ -v

# Check code quality
black --check server/ edge_node/ models/
flake8 server/ edge_node/ models/

# Run specific component
python -m server.main --dev
python -m edge_node.main --dev
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Port Already in Use
```powershell
# Check what's using the port
netstat -an | findstr ":8000"

# Kill process using port
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force
```

#### Docker Issues
```powershell
# Restart Docker Desktop
# Or reset Docker to factory defaults

# Clear Docker cache
docker system prune -a -f
docker volume prune -f
```

#### Database Connection Issues
```powershell
# Check if PostgreSQL/Redis containers are running
docker ps | findstr postgres
docker ps | findstr redis

# Restart database containers
docker-compose -f docker/docker-compose.dev.yml restart postgres-dev redis-dev
```

#### Python Environment Issues
```powershell
# Recreate virtual environment
Remove-Item -Recurse -Force venv
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 🔧 Configuration Files

### Key Configuration Files:
- `config/global_config.yaml` - Main configuration
- `docker/server-config.yaml` - Server-specific config
- `docker/edge-config.yaml` - Edge node config
- `docker/.env` - Environment variables
- `docker-compose.dev.yml` - Development services

### Environment Variables:
- `QFLARE_CONFIG_PATH` - Path to configuration file
- `DATABASE_URL` - Database connection string
- `REDIS_URL` - Redis connection string
- `QFLARE_JWT_SECRET` - JWT signing secret
- `QFLARE_SGX_MODE` - SGX mode (HW/SIM)
- `QFLARE_LOG_LEVEL` - Logging level

## 🎯 Next Steps

After getting QFLARE running:

1. **Explore the API**: Visit http://localhost:8000/docs for interactive API documentation
2. **Monitor Training**: Watch federated learning progress in Grafana
3. **Add More Edge Nodes**: Start additional edge nodes with different device IDs
4. **Try Different Algorithms**: Experiment with FedProx, FedBN, Personalized FL
5. **Test TEE Features**: Enable SGX hardware mode for production security
6. **Scale Up**: Deploy to Kubernetes for production workloads

For production deployment, see `docs/DEPLOYMENT_GUIDE.md` for comprehensive instructions.

Happy federated learning! 🚀