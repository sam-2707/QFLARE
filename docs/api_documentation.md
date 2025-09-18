# QFLARE API Documentation

## Overview

QFLARE is a quantum-secure federated learning platform that enables distributed machine learning across edge devices while maintaining data privacy and security through post-quantum cryptography.

## API Base URL

- **Production**: `https://qflare-production.company.com/api/v1`
- **Staging**: `https://qflare-staging.company.com/api/v1`
- **Development**: `http://localhost:8000/api/v1`

## Authentication

All API endpoints (except public health checks) require authentication using JWT tokens.

### OAuth2 Authentication Flow

1. **Authorization Request**
   ```
   GET /auth/authorize?response_type=code&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&scope=read:devices write:devices admin
   ```

2. **Token Exchange**
   ```
   POST /auth/token
   Content-Type: application/x-www-form-urlencoded
   
   grant_type=authorization_code&code={CODE}&client_id={CLIENT_ID}&client_secret={CLIENT_SECRET}&redirect_uri={REDIRECT_URI}
   ```

3. **Using Access Token**
   ```
   Authorization: Bearer {ACCESS_TOKEN}
   ```

### API Key Authentication (for service-to-service)

```
X-API-Key: {API_KEY}
```

## Core Endpoints

### Health & Status

#### GET /health
Check system health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "database": "healthy",
    "redis": "healthy",
    "quantum_keys": "healthy"
  },
  "uptime": 86400
}
```

#### GET /status
Get detailed system status.

**Response:**
```json
{
  "federated_learning": {
    "active_rounds": 2,
    "total_devices": 15,
    "online_devices": 12
  },
  "security": {
    "quantum_keys_rotation": "2024-01-15T08:00:00Z",
    "last_security_scan": "2024-01-15T06:00:00Z"
  },
  "performance": {
    "avg_response_time": 120,
    "memory_usage": 65.2,
    "cpu_usage": 45.1
  }
}
```

### Device Management

#### GET /devices
List all registered devices.

**Query Parameters:**
- `page`: Page number (default: 1)
- `size`: Page size (default: 20)
- `status`: Filter by status (`online`, `offline`, `training`)
- `organization`: Filter by organization ID

**Response:**
```json
{
  "devices": [
    {
      "id": "dev_123456789",
      "name": "Edge Device 001",
      "status": "online",
      "organization_id": "org_abc123",
      "last_seen": "2024-01-15T10:25:00Z",
      "capabilities": {
        "cpu_cores": 4,
        "memory_gb": 8,
        "storage_gb": 256,
        "gpu": false
      },
      "security": {
        "certificate_expiry": "2024-07-15T00:00:00Z",
        "quantum_key_version": "v2.1"
      },
      "training": {
        "current_round": 5,
        "total_rounds_completed": 127,
        "avg_training_time": 45.2
      }
    }
  ],
  "pagination": {
    "page": 1,
    "size": 20,
    "total": 157,
    "pages": 8
  }
}
```

#### POST /devices/register
Register a new device.

**Request:**
```json
{
  "enrollment_token": "tok_abcdef123456",
  "device_info": {
    "name": "Edge Device 002",
    "type": "raspberry_pi",
    "capabilities": {
      "cpu_cores": 4,
      "memory_gb": 4,
      "storage_gb": 64
    }
  },
  "organization_id": "org_abc123"
}
```

**Response:**
```json
{
  "device_id": "dev_987654321",
  "certificate": "-----BEGIN CERTIFICATE-----\n...",
  "quantum_keys": {
    "encryption_key": "base64_encoded_key",
    "signing_key": "base64_encoded_key"
  },
  "config": {
    "server_endpoint": "https://qflare-production.company.com",
    "update_interval": 300,
    "training_config": {
      "batch_size": 32,
      "learning_rate": 0.001
    }
  }
}
```

#### GET /devices/{device_id}
Get specific device details.

#### PUT /devices/{device_id}
Update device configuration.

#### DELETE /devices/{device_id}
Deregister device.

### Federated Learning

#### GET /fl/rounds
List training rounds.

**Response:**
```json
{
  "rounds": [
    {
      "id": "round_789",
      "status": "in_progress",
      "started_at": "2024-01-15T10:00:00Z",
      "participants": 12,
      "target_participants": 15,
      "model_version": "v1.5.2",
      "progress": {
        "completed_devices": 8,
        "avg_accuracy": 0.892,
        "convergence_metric": 0.05
      }
    }
  ]
}
```

#### POST /fl/rounds
Start a new training round.

**Request:**
```json
{
  "model_config": {
    "architecture": "cnn",
    "hyperparameters": {
      "learning_rate": 0.001,
      "batch_size": 32,
      "epochs": 10
    }
  },
  "participant_criteria": {
    "min_participants": 10,
    "max_participants": 50,
    "device_capabilities": {
      "min_memory_gb": 2,
      "min_cpu_cores": 2
    }
  },
  "privacy_settings": {
    "differential_privacy": true,
    "noise_level": 0.1
  }
}
```

#### GET /fl/rounds/{round_id}
Get training round details.

#### POST /fl/rounds/{round_id}/join
Join a training round (device endpoint).

#### POST /fl/rounds/{round_id}/submit
Submit training results.

### Security & Quantum Keys

#### GET /security/keys
List quantum keys status.

**Response:**
```json
{
  "keys": [
    {
      "device_id": "dev_123456789",
      "key_version": "v2.1",
      "created_at": "2024-01-10T00:00:00Z",
      "expires_at": "2024-04-10T00:00:00Z",
      "algorithm": "CRYSTALS-Kyber",
      "status": "active"
    }
  ],
  "rotation_schedule": {
    "next_rotation": "2024-01-20T00:00:00Z",
    "interval_days": 30
  }
}
```

#### POST /security/keys/rotate
Force key rotation.

#### GET /security/audit
Get security audit logs.

### Organizations

#### GET /organizations
List organizations.

#### POST /organizations
Create new organization.

#### GET /organizations/{org_id}
Get organization details.

#### PUT /organizations/{org_id}
Update organization.

### Users & Permissions

#### GET /users
List users.

#### POST /users
Create new user.

#### GET /users/{user_id}
Get user details.

#### PUT /users/{user_id}/permissions
Update user permissions.

### Monitoring & Metrics

#### GET /monitoring/metrics
Get Prometheus metrics.

#### GET /monitoring/alerts
Get active alerts.

#### GET /monitoring/logs
Query system logs.

**Query Parameters:**
- `level`: Log level (`info`, `warning`, `error`)
- `component`: Component name
- `since`: Start time (ISO 8601)
- `until`: End time (ISO 8601)

## WebSocket API

### Real-time Updates

Connect to WebSocket endpoint for real-time updates:

```
WSS /ws/{connection_type}
```

**Connection Types:**
- `dashboard`: General dashboard updates
- `training`: Federated learning progress
- `security`: Security events
- `device/{device_id}`: Device-specific updates

**Message Format:**
```json
{
  "type": "training_progress",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "round_id": "round_789",
    "completed_devices": 9,
    "total_devices": 15,
    "progress_percentage": 60
  }
}
```

## Error Handling

### Standard Error Response

```json
{
  "error": {
    "code": "DEVICE_NOT_FOUND",
    "message": "Device with ID 'dev_invalid' not found",
    "details": {
      "device_id": "dev_invalid",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    "request_id": "req_xyz789"
  }
}
```

### HTTP Status Codes

- `200 OK`: Successful request
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource conflict
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

### Error Codes

| Code | Description |
|------|-------------|
| `INVALID_TOKEN` | JWT token is invalid or expired |
| `INSUFFICIENT_PERMISSIONS` | User lacks required permissions |
| `DEVICE_NOT_FOUND` | Device ID not found |
| `DEVICE_OFFLINE` | Device is not currently online |
| `TRAINING_IN_PROGRESS` | Cannot perform action during training |
| `QUOTA_EXCEEDED` | Organization quota exceeded |
| `INVALID_QUANTUM_KEY` | Quantum key validation failed |
| `ENROLLMENT_TOKEN_EXPIRED` | Device enrollment token expired |

## Rate Limiting

API endpoints are rate-limited based on authentication:

- **Authenticated users**: 1000 requests per hour
- **API keys**: 5000 requests per hour
- **Device endpoints**: 100 requests per minute
- **Public endpoints**: 60 requests per hour per IP

Rate limit headers:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 2024-01-15T11:00:00Z
```

## SDKs and Client Libraries

### Python SDK

```bash
pip install qflare-python-sdk
```

```python
from qflare import QFLAREClient

client = QFLAREClient(
    base_url="https://qflare-production.company.com",
    api_key="your_api_key"
)

# List devices
devices = client.devices.list()

# Start training round
round_id = client.fl.start_round({
    "model_config": {"architecture": "cnn"},
    "participant_criteria": {"min_participants": 10}
})
```

### JavaScript SDK

```bash
npm install @qflare/js-sdk
```

```javascript
import { QFLAREClient } from '@qflare/js-sdk';

const client = new QFLAREClient({
  baseURL: 'https://qflare-production.company.com',
  apiKey: 'your_api_key'
});

// List devices
const devices = await client.devices.list();

// WebSocket connection
const ws = client.connectWebSocket('dashboard');
ws.on('training_progress', (data) => {
  console.log('Training progress:', data);
});
```

## Webhooks

Register webhooks to receive notifications about system events:

### Webhook Events

- `device.registered`: New device registered
- `device.offline`: Device went offline
- `training.round_started`: Training round started
- `training.round_completed`: Training round completed
- `security.key_rotated`: Quantum keys rotated
- `security.alert`: Security alert triggered

### Webhook Payload

```json
{
  "event": "device.registered",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "device_id": "dev_123456789",
    "organization_id": "org_abc123",
    "device_name": "Edge Device 001"
  },
  "webhook_id": "webhook_456",
  "signature": "sha256=abc123..."
}
```

### Webhook Verification

Verify webhook signatures using HMAC-SHA256:

```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    computed = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={computed}", signature)
```

## Environments

### Development Environment

- **URL**: `http://localhost:8000`
- **Database**: SQLite (local file)
- **Redis**: Local Redis instance
- **Authentication**: Disabled for testing
- **Rate Limiting**: Disabled

### Staging Environment

- **URL**: `https://qflare-staging.company.com`
- **Database**: PostgreSQL (shared staging)
- **Redis**: Redis cluster
- **Authentication**: OAuth2 with test credentials
- **Rate Limiting**: Relaxed limits

### Production Environment

- **URL**: `https://qflare-production.company.com`
- **Database**: PostgreSQL (HA cluster)
- **Redis**: Redis cluster with persistence
- **Authentication**: Full OAuth2/OIDC
- **Rate Limiting**: Full enforcement
- **Monitoring**: Full observability stack

## Support

- **Documentation**: https://docs.qflare.company.com
- **API Status**: https://status.qflare.company.com
- **Support Portal**: https://support.qflare.company.com
- **Emergency Contact**: emergency@company.com