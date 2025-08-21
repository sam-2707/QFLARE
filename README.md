# QFLARE: Quantum-Resistant Federated Learning & Authentication for Resilient Edge

QFLARE is a novel framework for performing privacy-preserving machine learning that is secure against both classical and quantum attacks. The system integrates Federated Learning (FL) with a robust, multi-layered security architecture using Post-Quantum Cryptography (PQC) and secure enclaves.

## 🚀 Features

- **Post-Quantum Cryptography**: Uses FrodoKEM-640-AES for key exchange and Dilithium2 for digital signatures
- **Secure Device Enrollment**: One-time token-based enrollment process prevents unauthorized access
- **Trusted Execution Environment**: Secure enclave for model aggregation and poisoning defense
- **Perfect Forward Secrecy**: Challenge-response mechanism for secure session establishment
- **Model Poisoning Defense**: Cosine similarity-based detection of malicious model updates
- **Rate Limiting**: Protection against DoS attacks
- **HTTPS Communication**: All server communication uses secure protocols

## 🏗️ Architecture

```
QFLARE/
├── .env                    # Environment variables (not committed)
├── .env.example           # Environment template
├── docker-compose.yml     # Multi-container orchestration
├── config/
│   └── global_config.yaml # PQC algorithm configuration
├── edge_node/             # Edge device application
├── server/                # Central server application
├── enclaves/              # Secure enclave implementation
├── scripts/               # Administrative tools
└── docs/                  # Documentation
```

## 🔐 Security Features

### Secure Device Enrollment
1. Administrator generates one-time enrollment token
2. Device presents token to `/api/enroll` endpoint
3. Device generates PQC key pairs (KEM + Signature)
4. Server stores public keys, token is revoked
5. Device is now authorized to participate

### Authenticated Session Establishment
1. Device requests challenge from `/api/challenge`
2. Server encrypts session key using device's KEM public key
3. Device decrypts session key using private key
4. All subsequent communication uses session key

### Secure Model Aggregation
1. Device downloads global model securely
2. Local training on private data
3. Model update signed with device's private key
4. Update submitted to server via `/api/submit_model`
5. Server verifies signature and forwards to secure enclave
6. Enclave performs poisoning detection and federated averaging
7. New global model distributed to devices

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Docker and Docker Compose
- liboqs-python (optional, fallback implementations provided)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd QFLARE
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start the services**
   ```bash
   docker-compose up -d
   ```

4. **Generate enrollment token**
   ```bash
   python scripts/generate_token.py --device-id edge_device_001
   ```

5. **Enroll a device**
   ```bash
   python scripts/enroll_device.py --device-id edge_device_001 --enrollment-token <token>
   ```

6. **Start edge node**
   ```bash
   docker-compose up edge_node
   ```

## 📚 Usage

### Server Endpoints

- `GET /` - Main dashboard
- `GET /health` - Health check
- `GET /status` - System status
- `GET /devices` - Device listing
- `POST /api/enroll` - Device enrollment
- `POST /api/challenge` - Session challenge
- `POST /api/submit_model` - Model submission
- `GET /api/global_model` - Global model download
- `GET /api/enclave/status` - Enclave status

### Administrative Scripts

- `scripts/generate_token.py` - Generate enrollment tokens
- `scripts/enroll_device.py` - Enroll devices
- `scripts/federated_start.py` - Orchestrate FL rounds

### Configuration

Edit `config/global_config.yaml` to configure:
- PQC algorithms (FrodoKEM-640-AES, Dilithium2)
- Federated learning parameters
- Server settings

## 🔧 Development

### Local Development Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r server/requirements.txt
   pip install -r edge_node/requirements.txt
   ```

2. **Start server**
   ```bash
   cd server
   python main.py
   ```

3. **Start edge node**
   ```bash
   cd edge_node
   python main.py
   ```

### Testing

```bash
# Run tests
pytest tests/

# Test specific components
pytest tests/test_auth.py
pytest tests/test_fl_training.py
pytest tests/test_key_rotation.py
```

## 🛡️ Security Considerations

### Production Deployment

1. **Use real hardware TEE**: Replace mock enclave with Intel SGX or similar
2. **Secure key storage**: Use hardware security modules (HSM) for key storage
3. **Network security**: Configure firewalls and VPNs
4. **Certificate management**: Use proper SSL certificates
5. **Monitoring**: Implement comprehensive logging and monitoring
6. **Backup**: Regular backups of enrollment tokens and device registry

### Security Best Practices

- Rotate enrollment tokens regularly
- Monitor for suspicious activity
- Keep dependencies updated
- Use strong random number generation
- Implement proper access controls
- Regular security audits

## 📖 API Documentation

See `docs/api_docs.md` for detailed API documentation.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🙏 Acknowledgments

- liboqs for Post-Quantum Cryptography implementations
- FastAPI for the web framework
- Docker for containerization
- The federated learning research community

## 📞 Support

For questions and support, please open an issue on GitHub or contact the development team.
