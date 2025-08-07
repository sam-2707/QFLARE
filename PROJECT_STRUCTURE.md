# 🧹 QFLARE Project Structure (Cleaned)

## 📁 **Project Overview**

The QFLARE project has been cleaned up to remove unnecessary and redundant files. Here's the current structure:

## 📂 **Essential Files & Directories**

### **Root Level:**
```
QFLARE_Project_Structure/
├── .git/                          # Git repository
├── server/                        # Main server application
├── tests/                         # Core test files
├── docs/                          # Project documentation
├── config/                        # Configuration files
├── edge_node/                     # Edge node implementation
├── liboqs/                        # Quantum cryptography library
├── liboqs-python/                 # Python bindings for liboqs
├── scripts/                       # Utility scripts
├── enclaves/                      # Secure enclave implementations
├── models/                        # ML models
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── docker-compose.yml            # Docker configuration
├── LICENSE                        # License file
├── setup.py                       # Package setup
├── TROUBLESHOOTING.md            # Troubleshooting guide
├── quantum_key_usage_guide.md    # Quantum key usage guide
└── quantum_key_overview.md       # Quantum key overview
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