# 🚀 QFLARE Production Deployment

This guide covers deploying QFLARE Core in a production environment using Docker containers.

## 📋 Prerequisites

- **Docker** (20.10+)
- **Docker Compose** (2.0+) or Docker Compose V1
- **Git** (for cloning the repository)
- **At least 4GB RAM** available
- **2GB free disk space**

## 🏗️ Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd qflare-project-structure
```

### 2. Configure Environment
```bash
# Copy production environment template
cp .env.prod .env

# Edit with your production values
nano .env  # or your preferred editor
```

### 3. Deploy
```bash
# Linux/Mac
./deploy.sh

# Windows
deploy.bat
```

### 4. Access Services
- **QFLARE Dashboard**: http://localhost:8000
- **Redis Commander**: http://localhost:8081
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

## 📁 Project Structure

```
qflare-project-structure/
├── Dockerfile.prod              # Production container
├── docker-compose.prod.yml      # Production services
├── deploy.sh                    # Linux/Mac deployment script
├── deploy.bat                   # Windows deployment script
├── requirements.prod.txt        # Production dependencies
├── .env.prod                    # Environment template
├── config/
│   ├── redis.conf              # Redis configuration
│   ├── nginx.conf              # Nginx reverse proxy
│   └── prometheus.yml          # Monitoring configuration
├── data/                       # Persistent data (created at runtime)
└── logs/                       # Application logs (created at runtime)
```

## ⚙️ Configuration

### Environment Variables

Edit `.env` with your production values:

```bash
# Application Settings
ENVIRONMENT=production
QFLARE_JWT_SECRET=your-super-secure-jwt-secret-here

# Database
DATABASE_URL=sqlite:///data/qflare_prod.db

# Redis
REDIS_URL=redis://redis:6379/0

# Security
QFLARE_SGX_MODE=SIM
ENABLE_SSL=false

# Monitoring
PROMETHEUS_ENABLED=true
```

### Service Configuration

The production setup includes:

- **QFLARE Core Server**: Main application
- **Redis**: Caching and message broker
- **Redis Commander**: Web UI for Redis management
- **Nginx**: Reverse proxy and load balancer
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards

## 🚀 Deployment Commands

### Full Deployment
```bash
# Deploy all services
./deploy.sh

# Or step by step
./deploy.sh build    # Build images
./deploy.sh start    # Start services
```

### Service Management
```bash
# View status
./deploy.sh status

# View logs
./deploy.sh logs qflare-server
./deploy.sh logs redis

# Restart services
./deploy.sh restart

# Stop services
./deploy.sh stop
```

### Windows Commands
```batch
# Deploy all services
deploy.bat

# Service management
deploy.bat status
deploy.bat logs qflare-server
deploy.bat restart
deploy.bat stop
```

## 🔧 Service Architecture

```
Internet
    ↓
  Nginx (Port 80/443)
    ↓
QFLARE Server (Port 8000)
    ↙        ↘
 Redis       SQLite
 (Cache)    (Database)
    ↓
Prometheus (Port 9090)
    ↓
 Grafana (Port 3000)
```

## 📊 Monitoring & Observability

### Prometheus Metrics
- **QFLARE Server**: http://localhost:8000/metrics
- **Redis**: Built-in Redis metrics
- **System**: Node exporter (optional)

### Grafana Dashboards
1. **QFLARE Overview**: System health and FL metrics
2. **Device Monitoring**: Device registration and activity
3. **FL Rounds**: Training progress and performance
4. **Infrastructure**: System resources and containers

### Health Checks
- **Application**: http://localhost:8000/health
- **Database**: Automatic connection validation
- **Redis**: Built-in health checks
- **Containers**: Docker health checks

## 🔒 Security Considerations

### Production Security
- Change default passwords in `.env`
- Enable SSL/TLS in production
- Use secrets management (Docker secrets, Kubernetes secrets)
- Configure firewall rules
- Regular security updates

### Network Security
- Services communicate over Docker network
- External access through Nginx reverse proxy
- Rate limiting configured
- Security headers enabled

## 📈 Scaling

### Horizontal Scaling
```yaml
# Scale QFLARE servers
docker-compose up -d --scale qflare-server=3
```

### Load Balancing
- Nginx automatically distributes requests
- Redis handles session management
- Database connection pooling

### Resource Limits
```yaml
services:
  qflare-server:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
```

## 🔧 Troubleshooting

### Common Issues

**Services won't start:**
```bash
# Check logs
./deploy.sh logs

# Check Docker status
docker ps -a
docker-compose ps
```

**Port conflicts:**
```bash
# Check port usage
netstat -tulpn | grep :8000

# Change ports in docker-compose.prod.yml
```

**Database connection issues:**
```bash
# Check database file permissions
ls -la data/

# Reset database
rm data/qflare_prod.db
./deploy.sh restart
```

**Memory issues:**
```bash
# Check memory usage
docker stats

# Increase Docker memory limit
# Docker Desktop > Settings > Resources > Memory
```

### Logs and Debugging

```bash
# Application logs
./deploy.sh logs qflare-server

# All service logs
docker-compose logs -f

# Container resource usage
docker stats

# Enter container for debugging
docker exec -it qflare-core /bin/bash
```

## 🔄 Backup & Recovery

### Database Backup
```bash
# Backup SQLite database
docker cp qflare-core:/app/data/qflare_prod.db ./backup/

# Automated backup script
#!/bin/bash
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
docker cp qflare-core:/app/data/qflare_prod.db "$BACKUP_DIR/"
```

### Configuration Backup
```bash
# Backup configurations
cp .env .env.backup
cp docker-compose.prod.yml docker-compose.prod.yml.backup
```

### Recovery
```bash
# Restore from backup
docker cp ./backup/qflare_prod.db qflare-core:/app/data/
./deploy.sh restart
```

## 📚 API Documentation

Once deployed, access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 🎯 Next Steps

After successful deployment:

1. **Register Devices**: Use device simulator or API
2. **Start FL Rounds**: Begin federated learning
3. **Monitor Performance**: Use Grafana dashboards
4. **Scale Services**: Add more server instances
5. **Configure Alerts**: Set up monitoring alerts

## 🆘 Support

### Getting Help
- Check logs: `./deploy.sh logs`
- Health check: `curl http://localhost:8000/health`
- Docker status: `docker ps`

### Common Commands
```bash
# Quick health check
curl -f http://localhost:8000/health

# Check Redis
docker exec qflare-redis redis-cli ping

# View all logs
docker-compose logs --tail=100
```

---

## 🎉 Success!

Your QFLARE production deployment is complete! 🚀

**Access your federated learning platform:**
- 🌐 **Dashboard**: http://localhost:8000
- 📊 **Monitoring**: http://localhost:3000
- 🔍 **Metrics**: http://localhost:9090

**Ready to start federated learning with real devices!** 🎯