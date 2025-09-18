"""
Pydantic models for request and response data validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import re


class AuthRequest(BaseModel):
    """Request model for device authentication."""
    device_id: str = Field(..., min_length=1, max_length=100, description="Unique device identifier")
    qkey: str = Field(..., min_length=1, description="Simulated quantum key from edge device")
    
    @validator('device_id')
    def validate_device_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Device ID must contain only alphanumeric characters, underscores, and hyphens')
        return v


class AuthResponse(BaseModel):
    """Response model for device authentication."""
    status: str = Field(..., description="Authentication status")
    device_id: str = Field(..., description="Device identifier")
    message: Optional[str] = Field(None, description="Additional message")


class DeviceActionRequest(BaseModel):
    """Request model for device actions."""
    device_id: str = Field(..., min_length=1, max_length=100, description="Device identifier")
    action: str = Field(..., min_length=1, description="Action to perform")
    
    @validator('device_id')
    def validate_device_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Device ID must contain only alphanumeric characters, underscores, and hyphens')
        return v


class EnrollmentRequest(BaseModel):
    """Request model for device enrollment."""
    device_id: str = Field(..., min_length=1, max_length=100, description="Device identifier")
    enrollment_token: str = Field(..., min_length=32, description="One-time enrollment token")
    kem_public_key: str = Field(..., description="KEM public key for key exchange")
    signature_public_key: str = Field(..., description="Digital signature public key")
    
    @validator('device_id')
    def validate_device_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Device ID must contain only alphanumeric characters, underscores, and hyphens')
        return v
    
    @validator('enrollment_token')
    def validate_token(cls, v):
        if not re.match(r'^[A-Za-z0-9_-]+$', v):
            raise ValueError('Enrollment token must contain only alphanumeric characters, underscores, and hyphens')
        return v


class EnrollmentResponse(BaseModel):
    """Response model for device enrollment."""
    status: str = Field(..., description="Enrollment status")
    device_id: str = Field(..., description="Device identifier")
    message: Optional[str] = Field(None, description="Additional message")
    server_public_key: Optional[str] = Field(None, description="Server's public key for verification")


class ChallengeRequest(BaseModel):
    """Request model for session challenge."""
    device_id: str = Field(..., min_length=1, max_length=100, description="Device identifier")
    
    @validator('device_id')
    def validate_device_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Device ID must contain only alphanumeric characters, underscores, and hyphens')
        return v


class ChallengeResponse(BaseModel):
    """Response model for session challenge."""
    status: str = Field(..., description="Challenge status")
    device_id: str = Field(..., description="Device identifier")
    challenge: str = Field(..., description="Encrypted session key")
    message: Optional[str] = Field(None, description="Additional message")


class ModelUpdateRequest(BaseModel):
    """Request model for model update submission."""
    device_id: str = Field(..., min_length=1, max_length=100, description="Device identifier")
    model_weights: str = Field(..., description="Base64 encoded model weights")
    signature: str = Field(..., description="Digital signature of model update")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('device_id')
    def validate_device_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Device ID must contain only alphanumeric characters, underscores, and hyphens')
        return v
    
    @validator('model_weights')
    def validate_model_weights(cls, v):
        # Basic validation for base64 encoding
        try:
            import base64
            base64.b64decode(v)
        except Exception:
            raise ValueError('Model weights must be base64 encoded')
        return v
    
    @validator('signature')
    def validate_signature(cls, v):
        # Basic validation for base64 encoding
        try:
            import base64
            base64.b64decode(v)
        except Exception:
            raise ValueError('Signature must be base64 encoded')
        return v


class ModelUpdateResponse(BaseModel):
    """Response model for model update submission."""
    status: str = Field(..., description="Update status")
    device_id: str = Field(..., description="Device identifier")
    message: Optional[str] = Field(None, description="Additional message")
    aggregation_round: Optional[int] = Field(None, description="Current aggregation round")


class GlobalModelResponse(BaseModel):
    """Response model for global model download."""
    status: str = Field(..., description="Download status")
    model_weights: Optional[str] = Field(None, description="Base64 encoded global model weights")
    model_version: Optional[str] = Field(None, description="Model version identifier")
    message: Optional[str] = Field(None, description="Additional message")


class DeviceInfo(BaseModel):
    """Model for device information."""
    device_id: str = Field(..., description="Device identifier")
    status: str = Field(..., description="Device status")
    last_seen: Optional[datetime] = Field(None, description="Last activity timestamp")
    public_keys: Optional[Dict[str, str]] = Field(None, description="Device public keys")


class DeviceListResponse(BaseModel):
    """Response model for device listing."""
    devices: List[DeviceInfo] = Field(..., description="List of registered devices")
    total_count: int = Field(..., description="Total number of devices")


class EnclaveStatusResponse(BaseModel):
    """Response model for enclave status."""
    enclave_type: str = Field(..., description="Type of secure enclave")
    status: str = Field(..., description="Enclave operational status")
    poison_threshold: float = Field(..., description="Poisoning detection threshold")
    global_model_hash: Optional[str] = Field(None, description="Current global model hash")
    total_aggregations: int = Field(..., description="Total number of aggregations performed")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Component status")


# Authentication Schemas
class LoginRequest(BaseModel):
    """Request model for user login."""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=6, description="Password")
    role: Optional[str] = Field(None, description="User role (admin/user)")
    
    @validator('username')
    def validate_username(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username must contain only alphanumeric characters, underscores, and hyphens')
        return v.lower()


class LoginResponse(BaseModel):
    """Response model for successful login."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    user: Dict[str, Any] = Field(..., description="User information")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class UserInfoResponse(BaseModel):
    """Response model for user information."""
    user: Dict[str, Any] = Field(..., description="User details")
    status: str = Field(..., description="Authentication status")


class TokenValidationResponse(BaseModel):
    """Response model for token validation."""
    valid: bool = Field(..., description="Token validity status")
    user: Optional[Dict[str, Any]] = Field(None, description="User information if token is valid")
    message: str = Field(..., description="Validation message")


class PermissionsResponse(BaseModel):
    """Response model for user permissions."""
    permissions: Dict[str, bool] = Field(..., description="User permission flags")
    role: str = Field(..., description="User role")
    username: str = Field(..., description="Username")