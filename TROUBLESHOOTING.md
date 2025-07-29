# QFLARE Troubleshooting Guide

## Common Issues and Solutions

### 1. liboqs Installation Issues

#### Problem: "No oqs shared libraries found" on Windows

This is a common issue on Windows systems where liboqs cannot find the required shared libraries.

**Solutions:**

**Option A: Use Fallback Mode (Recommended for Development)**
The system is designed to work without liboqs using fallback implementations:

```bash
# The system will automatically detect liboqs issues and use fallbacks
# No additional action needed - just proceed with installation
```

**Option B: Install liboqs with Pre-built Binaries**
```bash
# For Windows, try installing from a wheel
pip install --only-binary=all liboqs-python

# Or install from conda
conda install -c conda-forge liboqs-python
```

**Option C: Build from Source (Advanced)**
```bash
# Clone and build liboqs
git clone https://github.com/open-quantum-safe/liboqs.git
cd liboqs
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../liboqs-install ..
make -j
make install

# Then install Python bindings
cd ../python
pip install -e .
```

#### Problem: ImportError for liboqs

**Solution:**
```bash
# Uninstall and reinstall
pip uninstall liboqs-python
pip install liboqs-python

# Or use conda
conda install -c conda-forge liboqs-python
```

### 2. Docker Issues

#### Problem: Docker containers fail to start

**Solutions:**

**Check Docker is running:**
```bash
docker --version
docker-compose --version
```

**Rebuild containers:**
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

**Check logs:**
```bash
docker-compose logs server
docker-compose logs edge_node
```

### 3. Network Connectivity Issues

#### Problem: Cannot connect to server

**Solutions:**

**Check server is running:**
```bash
# Check if server is accessible
curl http://localhost:8000/health
```

**Check firewall settings:**
- Ensure port 8000 is not blocked
- Check Windows Firewall settings
- Try disabling antivirus temporarily

**Use different port:**
```bash
# Edit .env file
SERVER_PORT=8080
```

### 4. SSL Certificate Issues

#### Problem: SSL certificate verification fails

**Solutions:**

**For development (not recommended for production):**
```bash
# Set environment variable
export VERIFY_SSL=false
# Or on Windows
set VERIFY_SSL=false
```

**Generate self-signed certificate:**
```bash
# Generate certificate for development
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

### 5. Python Version Issues

#### Problem: Incompatible Python version

**Solution:**
QFLARE requires Python 3.8+. Check your version:
```bash
python --version
```

**Upgrade Python if needed:**
- Download from python.org
- Or use conda: `conda create -n qflare python=3.9`

### 6. Missing Dependencies

#### Problem: ModuleNotFoundError

**Solutions:**

**Install all dependencies:**
```bash
pip install -r requirements.txt
pip install -r server/requirements.txt
pip install -r edge_node/requirements.txt
```

**Check virtual environment:**
```bash
# Create virtual environment
python -m venv qflare_env
source qflare_env/bin/activate  # On Windows: qflare_env\Scripts\activate
```

### 7. Permission Issues

#### Problem: Permission denied errors

**Solutions:**

**On Windows:**
- Run as Administrator
- Check folder permissions

**On Linux/Mac:**
```bash
chmod +x scripts/*.py
sudo chown -R $USER:$USER .
```

### 8. Database Issues

#### Problem: SQLite database errors

**Solutions:**

**Check database file permissions:**
```bash
# Ensure write permissions
chmod 666 qflare.db  # If exists
```

**Reset database:**
```bash
# Remove and recreate
rm qflare.db
```

### 9. Rate Limiting Issues

#### Problem: "Too Many Requests" errors

**Solutions:**

**Increase rate limits in development:**
Edit `server/main.py` and increase limits:
```python
@limiter.limit("100/minute")  # Instead of "10/minute"
```

**Check rate limiting configuration:**
```bash
# View current limits
curl http://localhost:8000/api/health
```

### 10. Enrollment Token Issues

#### Problem: Invalid enrollment token

**Solutions:**

**Generate new token:**
```bash
python scripts/generate_token.py --device-id edge_device_001
```

**Check token file:**
```bash
# View enrollment tokens
python scripts/generate_token.py --list
```

**Clear expired tokens:**
```bash
# The system automatically cleans up expired tokens
# Or manually delete enrollment_tokens.json
```

## Development Mode

For development and testing, the system works perfectly without liboqs using fallback implementations:

### Features Available in Fallback Mode:
- ✅ Device enrollment with secure tokens
- ✅ Session establishment (simulated)
- ✅ Model signing and verification (hash-based)
- ✅ Secure enclave simulation
- ✅ Poisoning defense
- ✅ Rate limiting
- ✅ All API endpoints

### Features Requiring liboqs:
- 🔒 Real post-quantum cryptography
- 🔒 Hardware-level security
- 🔒 Production-grade key exchange

## Getting Help

### Check System Status:
```bash
# Check server health
curl http://localhost:8000/health

# Check system status
curl http://localhost:8000/status

# Check enclave status
curl http://localhost:8000/api/enclave/status
```

### View Logs:
```bash
# Server logs
docker-compose logs server

# Edge node logs
docker-compose logs edge_node

# Or if running locally
python server/main.py
```

### Common Commands:
```bash
# Start system
docker-compose up -d

# Stop system
docker-compose down

# Rebuild
docker-compose build --no-cache

# View logs
docker-compose logs -f
```

## Production Considerations

For production deployment:

1. **Install liboqs properly** or use hardware security modules
2. **Use real TEE** instead of mock enclave
3. **Configure proper SSL certificates**
4. **Set up monitoring and logging**
5. **Use secure key storage**
6. **Implement proper backup strategies**

## Still Having Issues?

If you're still experiencing problems:

1. Check the logs for specific error messages
2. Ensure all dependencies are installed
3. Try running in development mode (without liboqs)
4. Check system requirements (Python 3.8+, Docker)
5. Verify network connectivity and firewall settings

The system is designed to be robust and will work in fallback mode for development and testing purposes. 