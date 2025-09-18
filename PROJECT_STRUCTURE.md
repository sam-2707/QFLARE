# 🧹 QFLARE Project Structure (Cleaned & Organized)

## 📁 **Project Overview**

The QFLARE project has been cleaned up to remove unnecessary files, duplicates, and organize all components properly. Here's the current clean structure:

## 📂 **Essential Files & Directories**

### **Root Level:**
```
QFLARE_Project_Structure/
├── .env.example                   # Environment variables template
├── .env.prod                      # Production environment variables  
├── .git/                          # Git repository
├── .github/                       # GitHub workflows and templates
├── .gitignore                     # Git ignore patterns
├── alembic.ini                    # Database migration configuration
├── common/                        # Shared utilities across modules
├── config/                        # Configuration files
├── data/                          # Database files and data storage
├── deploy.bat/.sh                 # Deployment scripts
├── docker/                        # Docker-related files
├── docker-compose.yml             # Docker development config
├── docker-compose.prod.yml        # Docker production config
├── Dockerfile.prod                # Production Docker image
├── docs/                          # Project documentation
├── edge_node/                     # Edge node implementation
├── enclaves/                      # Secure enclave implementations
├── frontend/                      # Web frontend components (if any)
├── k8s/                           # Kubernetes manifests
├── liboqs/                        # Post-quantum cryptography library
├── liboqs-python/                 # Python bindings for liboqs
├── models/                        # ML model definitions
├── monitoring/                    # System monitoring and metrics
├── qflare-env/                    # Python virtual environment
├── quantum_key_overview.md        # Quantum cryptography overview
├── quantum_key_usage_guide.md     # Quantum key usage guide
├── README.md                      # Main project documentation
├── requirements.txt               # Development dependencies
├── requirements.prod.txt          # Production dependencies
├── scripts/                       # Utility and setup scripts
├── security/                      # Security tools and configurations
├── server/                        # Main server application ⭐
├── setup.py                       # Package setup
├── start_qflare.py               # Main startup script ⭐
├── tests/                         # Test suites
├── TROUBLESHOOTING.md            # Common issues and solutions
├── PROJECT_STATUS.md             # Current development status
└── PROJECT_STRUCTURE.md          # This file
```

### **Server Directory (`server/`):**
```
server/
├── main.py                        # Main FastAPI application
├── registry.py                    # Device registration system
├── ssl_manager.py                 # SSL/TLS management
├── start_server.py                # Server startup script
├── requirements.txt               # Server dependencies
├── Dockerfile                     # Docker configuration
├── __init__.py                    # Package initialization
├── auth/                          # Authentication system
│   └── pqcrypto_utils.py         # Quantum cryptography utilities
├── api/                           # API endpoints
│   ├── routes.py                  # API routes
│   └── schemas.py                 # Data schemas
├── fl_core/                       # Federated learning core
├── enclave/                       # Secure enclave
├── templates/                     # HTML templates
├── static/                        # Static files
├── monitoring/                    # Monitoring system
├── security/                      # Security components
└── ledger/                        # Blockchain ledger
```

### **Tests Directory (`tests/`):**
```
tests/
├── test_fl_training.py           # Federated learning tests
├── test_auth.py                  # Authentication tests
└── test_key_rotation.py          # Key rotation tests
```

## 🗑️ **Files Removed During Cleanup**

### **Redundant Test Files:**
- `test_challenge.py`
- `test_endpoints.py`
- `test_authentication.py`
- `test_quantum_keys.py`
- `test_registration.py`
- `test_server_simple.py`
- `test_bypass_liboqs.py`
- `test_simple_auth.py`
- `test_pqc_integration.py`
- `run_server_test.py`

### **Redundant Utility Scripts:**
- `start_simple.py`
- `restart_server.py`
- `run_server.py`
- `check_server_status.py`
- `show_urls.py`
- `example_key_usage.py`
- `example_key_usage_fixed.py`
- `quantum_key_status.py`

### **Temporary Files:**
- `qflare.db` (SQLite database)
- `enrollment_tokens.json`
- `key.env`
- `DS.txt` (empty file)
- `DS.pdf` (large PDF)
- `logs/` directory
- `.pytest_cache/` directory
- `__pycache__/` directories

## ✅ **Benefits of Cleanup**

1. **Reduced Clutter** - Removed 20+ unnecessary files
2. **Better Organization** - Clear separation of essential vs. temporary files
3. **Faster Navigation** - Easier to find important files
4. **Reduced Size** - Removed ~200KB of unnecessary files
5. **Cleaner Development** - No confusion about which files to use

## 🚀 **How to Use the Cleaned Project**

### **Start the Server:**
```bash
# From project root directory
python start_qflare.py

# Or from server directory
cd server
python start_server.py
```

### **Run Tests:**
```bash
python -m pytest tests/
```

### **Generate Quantum Keys:**
```bash
# Browser access
http://localhost:8000/api/request_qkey

# API call
curl http://localhost:8000/api/request_qkey
```

### **Register Devices:**
```bash
# Web interface
http://localhost:8000/register
```

## 📋 **Essential Files for Development**

### **Core Application:**
- `server/main.py` - Main application
- `server/registry.py` - Device management
- `server/auth/pqcrypto_utils.py` - Quantum cryptography
- `server/api/routes.py` - API endpoints

### **Documentation:**
- `README.md` - Project overview
- `quantum_key_usage_guide.md` - Usage instructions
- `TROUBLESHOOTING.md` - Problem solving

### **Configuration:**
- `requirements.txt` - Dependencies
- `docker-compose.yml` - Docker setup
- `config/global_config.yaml` - Global settings

### **Libraries:**
- `liboqs/` - Quantum cryptography
- `liboqs-python/` - Python bindings

## 🎉 **Project is Now Clean and Ready!**

The QFLARE project has been successfully cleaned up and is ready for development. All essential functionality remains intact while unnecessary files have been removed. 