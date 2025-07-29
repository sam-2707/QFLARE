# QFLARE API Documentation

## Overview

The QFLARE API provides secure endpoints for federated learning with post-quantum cryptography protection. All endpoints use HTTPS and require proper authentication.

## Base URL

```
https://localhost:8000
```

## Authentication

All API endpoints require proper device enrollment and session establishment. See the enrollment process below.

## Endpoints

### Device Enrollment

#### POST /api/enroll

Enroll a new device with the server using a secure one-time token.

**Request Body:**
```json
{
  "device_id": "edge_device_001",
  "enrollment_token": "your_one_time_token_here",
  "kem_public_key": "base64_encoded_kem_public_key",
  "signature_public_key": "base64_encoded_signature_public_key"
}
```

**Response:**
```json
{
  "status": "success",
  "device_id": "edge_device_001",
  "message": "Device enrolled successfully",
  "server_public_key": "server_public_key_placeholder"
}
```

### Session Management

#### POST /api/challenge

Request a session challenge for secure communication.

**Request Body:**
```json
{
  "device_id": "edge_device_001"
}
```

**Response:**
```json
{
  "status": "success",
  "device_id": "edge_device_001",
  "challenge": "base64_encoded_encrypted_session_key",
  "message": "Session challenge generated"
}
```

### Model Management

#### GET /api/global_model

Download the current global model.

**Response:**
```json
{
  "status": "success",
  "model_weights": "base64_encoded_model_weights",
  "model_version": "v1.0",
  "message": "Global model retrieved successfully"
}
```

#### POST /api/submit_model

Submit a model update from an edge device.

**Request Body:**
```json
{
  "device_id": "edge_device_001",
  "model_weights": "base64_encoded_model_weights",
  "signature": "base64_encoded_digital_signature",
  "metadata": {
    "round": 1,
    "data_samples": 1000,
    "training_time": 120.5
  }
}
```

**Response:**
```json
{
  "status": "success",
  "device_id": "edge_device_001",
  "message": "Model update received and stored",
  "aggregation_round": 1
}
```

### Device Management

#### GET /api/devices

List all registered devices.

**Response:**
```json
{
  "devices": [
    {
      "device_id": "edge_device_001",
      "status": "active",
      "last_seen": "2024-01-15T10:30:00Z",
      "public_keys": {
        "kem_public_key": "base64_encoded_key",
        "signature_public_key": "base64_encoded_key"
      }
    }
  ],
  "total_count": 1
}
```

### System Status

#### GET /api/enclave/status

Get the status of the secure enclave.

**Response:**
```json
{
  "enclave_type": "mock_secure_enclave",
  "status": "operational",
  "poison_threshold": 0.8,
  "global_model_hash": "sha256_hash_of_current_model",
  "total_aggregations": 5
}
```

#### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "server": "healthy",
    "enclave": "healthy",
    "registry": "healthy"
  }
}
```

### Web Interface

#### GET /

Main dashboard showing system overview.

#### GET /devices

Web interface for viewing registered devices.

#### GET /status

System status page.

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input data |
| 401 | Unauthorized - Invalid token or signature |
| 404 | Not Found - Device or resource not found |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error - Server error |

## Rate Limiting

- `/api/enroll`: 5 requests per minute
- `/api/challenge`: 30 requests per minute
- `/api/submit_model`: 10 requests per minute
- `/api/global_model`: 60 requests per minute
- `/`: 10 requests per minute
- `/devices`: 30 requests per minute

## Security Features

1. **Post-Quantum Cryptography**: Uses FrodoKEM-640-AES for key exchange and Dilithium2 for signatures
2. **Secure Enrollment**: One-time token-based device enrollment
3. **Perfect Forward Secrecy**: Challenge-response mechanism for session establishment
4. **Digital Signatures**: All model updates are cryptographically signed
5. **Rate Limiting**: Protection against DoS attacks
6. **HTTPS**: All communication uses TLS encryption

## Usage Examples

### Python Client Example

```python
import requests
import base64

# Enroll device
enrollment_data = {
    "device_id": "my_device",
    "enrollment_token": "your_token",
    "kem_public_key": "base64_key",
    "signature_public_key": "base64_key"
}

response = requests.post("https://localhost:8000/api/enroll", json=enrollment_data)
print(response.json())

# Request session challenge
challenge_data = {"device_id": "my_device"}
response = requests.post("https://localhost:8000/api/challenge", json=challenge_data)
challenge = response.json()["challenge"]

# Download global model
response = requests.get("https://localhost:8000/api/global_model")
model_weights = base64.b64decode(response.json()["model_weights"])

# Submit model update
update_data = {
    "device_id": "my_device",
    "model_weights": base64.b64encode(trained_weights).decode(),
    "signature": base64.b64encode(signature).decode(),
    "metadata": {"round": 1}
}

response = requests.post("https://localhost:8000/api/submit_model", json=update_data)
print(response.json())
```

### cURL Examples

```bash
# Enroll device
curl -X POST https://localhost:8000/api/enroll \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "edge_device_001",
    "enrollment_token": "your_token",
    "kem_public_key": "base64_key",
    "signature_public_key": "base64_key"
  }'

# Get global model
curl https://localhost:8000/api/global_model

# Submit model update
curl -X POST https://localhost:8000/api/submit_model \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "edge_device_001",
    "model_weights": "base64_weights",
    "signature": "base64_signature",
    "metadata": {"round": 1}
  }'
```

## Development Notes

- All timestamps are in ISO 8601 format
- Base64 encoding is used for binary data
- Device IDs must contain only alphanumeric characters, underscores, and hyphens
- Enrollment tokens are single-use and expire after 24 hours by default
- Model weights should be serialized as bytes before base64 encoding