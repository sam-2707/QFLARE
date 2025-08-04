# QFLARE - Quantum-Resistant Federated Learning Environment

A secure, web-based federated learning server with post-quantum cryptography, device management, and real-time monitoring.

## 🌟 Features

- **Post-Quantum Cryptography**: Kyber768 KEM and Dilithium3 signatures
- **Secure Device Enrollment**: Token-based device registration
- **Real-time Dashboard**: Modern web interface for device management
- **Database Storage**: SQLite-based device and key management
- **Key Rotation**: Automatic server key management
- **Rate Limiting**: Protection against abuse
- **Health Monitoring**: System status and statistics

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python run_server.py
```

The server will start and display connection information:

```
============================================================
🚀 QFLARE Server - Quantum-Resistant Federated Learning
============================================================

📋 Configuration:
   Host: 0.0.0.0
   Port: 8000
   Local IP: 192.168.1.100

🌐 Server URLs:
   Local:     http://localhost:8000
   Network:   http://192.168.1.100:8000

📱 Device Connection:
   Other systems can connect to: http://192.168.1.100:8000
   API Documentation: http://192.168.1.100:8000/docs
   Health Check: http://192.168.1.100:8000/health
```

### 3. Access the Dashboard

Open your browser and navigate to:
- **Dashboard**: `http://your-server-ip:8000`
- **API Docs**: `http://your-server-ip:8000/docs`

## 📱 Device Connection

### For Other Systems

Other systems can connect to your QFLARE server using the provided URL:

```python
import requests

# Server URL (replace with your actual server IP)
SERVER_URL = "http://192.168.1.100:8000"

# Health check
response = requests.get(f"{SERVER_URL}/health")
print(response.json())

# Get server info
response = requests.get(f"{SERVER_URL}/api/server_info")
print(response.json())
```

### Device Enrollment Process

1. **Generate Enrollment Token**:
   - Use the web dashboard or API
   - Provide device ID and expiration time
   - Receive secure enrollment token

2. **Device Registration**:
   - Device presents enrollment token
   - Server validates token and registers device
   - Device receives server public keys

3. **Secure Communication**:
   - Device generates KEM and signature keys
   - Establishes secure session with server
   - Can participate in federated learning

## 🔐 Security Features

### Post-Quantum Cryptography

- **KEM Algorithm**: Kyber768 (NIST PQC candidate)
- **Signature Algorithm**: Dilithium3 (NIST PQC candidate)
- **Key Sizes**: 768-bit security level
- **Forward Secrecy**: Perfect forward secrecy with session keys

### Device Security

- **Enrollment Tokens**: Time-limited, signed tokens for device registration
- **Key Management**: Automatic key generation and rotation
- **Session Management**: Secure session establishment and management
- **Rate Limiting**: Protection against brute force attacks

## 📊 Dashboard Features

### Device Management

- **Real-time Device List**: View all registered devices
- **Device Status**: Active, inactive, suspended status tracking
- **Key Information**: View device public keys and algorithms
- **Last Seen**: Track device activity

### Key Generation

- **Enrollment Tokens**: Generate secure tokens for device registration
- **Token Expiration**: Configurable token lifetime
- **Copy to Clipboard**: Easy token sharing

### System Monitoring

- **Health Status**: Real-time system health monitoring
- **Statistics**: Device counts and activity metrics
- **Security Level**: Current security configuration
- **Uptime**: System availability tracking

## 🗄️ Database Schema

### Tables

- **devices**: Device registration and status
- **key_pairs**: Cryptographic key storage
- **sessions**: Active device sessions
- **model_updates**: Federated learning model updates

### Key Management

- **Server Keys**: KEM and signature key pairs
- **Device Keys**: Per-device public keys
- **Session Keys**: Temporary session encryption keys
- **Key Rotation**: Automatic server key rotation

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000

# Database Configuration
DATABASE_URL=sqlite:///./qflare.db

# Security Configuration
SECURITY_LEVEL=3
KEY_ROTATION_INTERVAL=24
```

### API Endpoints

#### Device Management
- `GET /api/devices` - List all devices
- `POST /api/generate_token` - Generate enrollment token
- `PUT /api/devices/{device_id}/status` - Update device status

#### System Information
- `GET /health` - Health check
- `GET /status` - System status
- `GET /api/server_info` - Server information

#### Key Management
- `POST /api/rotate_keys` - Rotate server keys

## 🛠️ Development

### Project Structure

```
QFLARE_Project_Structure/
├── server/
│   ├── main.py              # Main FastAPI application
│   ├── database.py          # Database models and operations
│   ├── key_manager.py       # Key management system
│   ├── api/
│   │   ├── routes.py        # API endpoints
│   │   └── schemas.py       # Pydantic models
│   ├── auth/
│   │   └── pqcrypto_utils.py # Cryptographic utilities
│   └── templates/
│       └── index.html       # Dashboard template
├── run_server.py            # Startup script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### Running in Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run with auto-reload
python run_server.py

# Or run directly with uvicorn
cd server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Testing

```bash
# Health check
curl http://localhost:8000/health

# Get devices
curl http://localhost:8000/api/devices

# Generate token
curl -X POST http://localhost:8000/api/generate_token \
  -H "Content-Type: application/json" \
  -d '{"device_id": "test-device", "expiration_hours": 24}'
```

## 🔒 Security Considerations

### Production Deployment

1. **HTTPS**: Use SSL/TLS certificates
2. **Firewall**: Configure network security
3. **Rate Limiting**: Adjust rate limits for production
4. **Database**: Use production database (PostgreSQL, MySQL)
5. **Monitoring**: Implement logging and monitoring
6. **Backup**: Regular database backups

### Key Management

- **Key Rotation**: Regular server key rotation
- **Token Expiration**: Short-lived enrollment tokens
- **Session Timeout**: Automatic session expiration
- **Access Control**: Implement proper authentication

## 📈 Monitoring and Logging

### Health Checks

- **System Status**: `/health` endpoint
- **Component Status**: Database, key management, enclave
- **Statistics**: Device counts and activity

### Logging

- **Application Logs**: FastAPI logging
- **Security Events**: Key operations and device enrollment
- **Error Tracking**: Exception handling and reporting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Port Already in Use**:
   ```bash
   # Change port in .env file
   SERVER_PORT=8001
   ```

2. **Database Errors**:
   ```bash
   # Remove and recreate database
   rm qflare.db
   python run_server.py
   ```

3. **Import Errors**:
   ```bash
   # Install missing dependencies
   pip install -r requirements.txt
   ```

### Getting Help

- Check the API documentation at `/docs`
- Review the health check at `/health`
- Check server logs for error messages
- Ensure all dependencies are installed

## 🎯 Next Steps

1. **Deploy to Production**: Configure for production environment
2. **Add Authentication**: Implement user authentication
3. **Scale Database**: Migrate to production database
4. **Add Monitoring**: Implement comprehensive monitoring
5. **Enhance Security**: Add additional security features

---

**QFLARE** - Quantum-Resistant Federated Learning Environment