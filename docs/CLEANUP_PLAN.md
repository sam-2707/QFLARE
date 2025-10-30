# 🧹 QFLARE Project Cleanup Plan

## Files to DELETE (Unnecessary/Redundant)

### 1. Temporary/Patch Files
- `server/fl_core/fl_controller_patch.txt` - No longer needed
- `test_fl_implementation.py` - Root level test file, replaced by scripts/

### 2. Duplicate Demo Files  
- `scripts/fl_demo_complete.py` - Duplicate functionality
- `scripts/fl_client_demo.py` - Redundant with fl_edge_simulator.py
- `scripts/start_fl_demo.py` - Not used

### 3. Old Documentation Files
- Files already covered in newer comprehensive docs

### 4. QR Code/Device Files (Test Data)
- `qr_code_*.json` files (test data)
- `device_*_credentials.json` files
- `security_token_*.json` files

### 5. Redundant Config Files
- Old database files that are regenerated

## Files to KEEP (Essential)

### Core Application Files
✅ `server/simple_server.py` - Main backend
✅ `server/fl_core/` - FL implementation
✅ `frontend/qflare-ui/` - React dashboard
✅ `scripts/quick_fl_test.py` - Testing
✅ `scripts/generate_token.py` - Admin tools
✅ `scripts/enroll_device.py` - Admin tools

### Documentation
✅ `README.md` - Main documentation
✅ `PROJECT_COMPLETION_STATUS.md` - Status report
✅ `FL_*.md` - Implementation guides
✅ `TROUBLESHOOTING.md` - Support docs

### Configuration
✅ `requirements.txt` - Dependencies
✅ `package.json` - Frontend deps
✅ `docker-compose.yml` - Containerization
✅ `.env.example` - Config template

### Libraries (Keep)
✅ `liboqs/` - Post-quantum crypto library
✅ `liboqs-python/` - Python bindings