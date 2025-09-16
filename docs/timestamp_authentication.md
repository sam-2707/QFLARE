# Timestamp-based Challenge-Response Authentication

## Sequence Diagram Implementation

This document describes the exact implementation of the timestamp-based challenge-response authentication mechanism for QFLARE, following the sequence diagram pattern.

### Overview

The authentication system uses:
- **KEM (Key Encapsulation Mechanism)**: Kyber768 for post-quantum security
- **Digital Signatures**: Dilithium2 for device authentication
- **Timestamp Validation**: 30-second tolerance window
- **Session Management**: 60-minute session duration with automatic cleanup

### Sequence Flow

```
Device                    QFLARE Server              Challenge Manager
  |                            |                           |
  |  1. Generate Keypair       |                           |
  |  (Kyber768 + Dilithium2)   |                           |
  |                            |                           |
  |  2. Register Public Key    |                           |
  |--------------------------->|                           |
  |    POST /api/v2/auth/      |  register_device_key()    |
  |    register-key            |-------------------------->|
  |                            |                           |
  |  3. Challenge Request      |                           |
  |--------------------------->|                           |
  |    POST /api/v2/auth/      |                           |
  |    challenge               |                           |
  |    {                       |                           |
  |      device_id: "edge-001" |                           |
  |      timestamp: 1640995200 |                           |
  |      nonce: "abc123..."    |                           |
  |    }                       |                           |
  |                            |                           |
  |                            |  4. Validate Timestamp    |
  |                            |-------------------------->|
  |                            |     |server_time - device_time| < 30s
  |                            |                           |
  |                            |  5. Generate Session Key  |
  |                            |-------------------------->|
  |                            |     session_key = random(32 bytes)
  |                            |                           |
  |                            |  6. Encrypt with Device   |
  |                            |     Public Key (KEM)      |
  |                            |-------------------------->|
  |                            |     encrypted_key = kyber_encap(session_key)
  |                            |                           |
  |  7. Challenge Response     |                           |
  |<---------------------------|                           |
  |    {                       |                           |
  |      challenge_id: "xyz"   |                           |
  |      encrypted_session_key |                           |
  |      server_timestamp      |                           |
  |      validity_duration: 3600                          |
  |      status: "validated"   |                           |
  |    }                       |                           |
  |                            |                           |
  |  8. Decrypt Session Key    |                           |
  |  (using private key)       |                           |
  |                            |                           |
  |  9. Authenticated Requests |                           |
  |--------------------------->|                           |
  |    Header:                 |                           |
  |    Authorization: Bearer   |                           |
  |    {challenge_id}          |                           |
  |                            |                           |
  |                            | 10. Validate Session      |
  |                            |-------------------------->|
  |                            |    validate_session(challenge_id)
  |                            |                           |
  | 11. API Response           |                           |
  |<---------------------------|                           |
```

### Implementation Details

#### 1. Key Generation and Registration

```python
# Device side: Generate keypair
keypair = challenge_manager.generate_device_keypair(
    device_id="edge-node-001",
    key_type=KeyType.KYBER768
)

# Register public key with server
POST /api/v2/auth/register-key
{
    "device_id": "edge-node-001",
    "public_key": "base64_encoded_public_key",
    "key_type": "kyber768"
}
```

#### 2. Challenge Request Creation

```python
# Device creates challenge request
import time
challenge_request = {
    "device_id": "edge-node-001",
    "timestamp": time.time(),  # Current Unix timestamp
    "nonce": secrets.token_urlsafe(16)
}

POST /api/v2/auth/challenge
```

#### 3. Server Processing

```python
# Server validates timestamp
server_timestamp = time.time()
timestamp_diff = abs(server_timestamp - request['timestamp'])

if timestamp_diff > 30:  # 30-second tolerance
    raise ValueError("Timestamp outside tolerance")

# Generate session key
session_key = secrets.token_bytes(32)  # 256-bit key

# Encrypt using Kyber768 KEM
kem = oqs.KeyEncapsulation("Kyber768")
ciphertext, shared_secret = kem.encap_secret(device_public_key)

# Use shared secret to encrypt session key
encrypted_session_key = aes_encrypt(session_key, shared_secret)
```

#### 4. Session Management

```python
# Create session context
session = SessionContext(
    device_id=device_id,
    session_key=session_key,
    challenge_id=challenge_id,
    created_at=datetime.now(),
    expires_at=datetime.now() + timedelta(hours=1)
)

# Store for validation
active_sessions[challenge_id] = session
```

### API Endpoints

#### Authentication Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v2/auth/register-key` | Register device public key |
| POST | `/api/v2/auth/challenge` | Process challenge request |
| GET | `/api/v2/auth/session/{id}` | Validate session |
| DELETE | `/api/v2/auth/session/{id}` | Revoke session |
| GET | `/api/v2/auth/status` | Get auth system status |

#### Enrollment Token Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v2/enrollment/tokens` | Create enrollment token |
| GET | `/api/v2/enrollment/tokens` | List enrollment tokens |
| POST | `/api/v2/enrollment/generate-token` | Generate token with params |

### Security Features

#### Timestamp Validation
- **Tolerance Window**: 30 seconds (configurable)
- **Protection Against**: Replay attacks, clock skew issues
- **Clock Synchronization**: Devices should sync with NTP

#### Post-Quantum Cryptography
- **KEM Algorithm**: Kyber768 (NIST Level 3 security)
- **Signature Algorithm**: Dilithium2 (NIST Level 2 security)
- **Classical Fallback**: RSA-2048/4096 when PQC unavailable

#### Session Security
- **Session Duration**: 60 minutes (configurable)
- **Key Size**: 256-bit AES session keys
- **Automatic Cleanup**: Expired sessions removed every minute
- **Request Tracking**: Monitor session usage patterns

### Error Handling

#### Timestamp Errors
```json
{
    "success": false,
    "error": "Timestamp outside tolerance: 45s > 30s",
    "server_timestamp": 1640995200,
    "tolerance_seconds": 30
}
```

#### Key Registration Errors
```json
{
    "success": false,
    "error": "Invalid public key format",
    "supported_types": ["kyber768", "dilithium2", "rsa_2048"]
}
```

#### Session Validation Errors
```json
{
    "valid": false,
    "error": "Session not found or expired",
    "challenge_id": "invalid_id"
}
```

### Configuration

```python
# Challenge manager configuration
challenge_manager = TimestampChallengeManager(
    tolerance_seconds=30,        # Timestamp tolerance
    session_duration_minutes=60, # Session lifetime
    max_concurrent_sessions=1000 # Scale limit
)
```

### Monitoring and Metrics

The system provides comprehensive metrics:

```python
status = challenge_manager.get_system_status()
# Returns:
{
    'active_sessions': 45,
    'active_challenges': 12,
    'registered_devices': 156,
    'hsm_available': False,
    'pqc_available': True,
    'tolerance_seconds': 30,
    'session_duration_minutes': 60
}
```

### Production Considerations

1. **Clock Synchronization**: Ensure all devices sync with NTP servers
2. **HSM Integration**: Move private keys to hardware security modules
3. **Load Balancing**: Session state must be shared across server instances
4. **Monitoring**: Track failed authentications and unusual patterns
5. **Rate Limiting**: Implement per-device challenge request limits
6. **Audit Logging**: Log all authentication events for compliance

### Testing

Run the demo to see the complete flow:

```bash
python demo_challenge_response.py
```

This demonstrates:
- Key generation for multiple algorithms
- Complete challenge-response flow
- Session validation and usage
- Timestamp tolerance testing
- System status monitoring