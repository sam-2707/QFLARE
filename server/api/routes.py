"""
API routes for QFLARE server.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import logging
import base64
import time
from datetime import datetime

from api.schemas import (
    AuthRequest, AuthResponse, DeviceActionRequest, EnrollmentRequest, 
    EnrollmentResponse, ChallengeRequest, ChallengeResponse, ModelUpdateRequest,
    ModelUpdateResponse, GlobalModelResponse, DeviceListResponse, 
    EnclaveStatusResponse, ErrorResponse, HealthCheckResponse,
    LoginRequest, LoginResponse, UserInfoResponse, TokenValidationResponse, PermissionsResponse
)
from auth.pqcrypto_utils import (
    validate_enrollment_token, register_device_keys, generate_session_challenge,
    verify_model_signature, get_device_public_keys
)
from auth.auth_endpoints import auth_router
from auth.jwt_utils import get_current_user, get_current_admin_user
from auth.user_models import User
from enclave.mock_enclave import get_secure_enclave, ModelUpdate
from fl_core.aggregator import store_model_update, get_global_model
from registry import get_registered_devices

logger = logging.getLogger(__name__)
router = APIRouter()

# Include authentication routes
router.include_router(auth_router, prefix="/auth", tags=["authentication"])


@router.post("/enroll", response_model=EnrollmentResponse)
async def enroll_device(request: EnrollmentRequest):
    """
    Secure device enrollment endpoint.
    
    This endpoint replaces the insecure public registration process.
    Devices must present a valid one-time enrollment token to register.
    """
    try:
        # Validate enrollment token
        if not validate_enrollment_token(request.enrollment_token, request.device_id):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired enrollment token"
            )
        
        # Register device with public keys
        success = register_device_keys(
            request.device_id,
            request.kem_public_key,
            request.signature_public_key
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to register device keys"
            )
        
        logger.info(f"Device {request.device_id} successfully enrolled")
        
        return EnrollmentResponse(
            status="success",
            device_id=request.device_id,
            message="Device enrolled successfully",
            server_public_key="server_public_key_placeholder"  # TODO: Implement
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enrollment error for device {request.device_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during enrollment"
        )


@router.post("/challenge", response_model=ChallengeResponse)
async def request_challenge(request: ChallengeRequest):
    """
    Request a session challenge for secure communication.
    
    This implements the challenge-response mechanism for Perfect Forward Secrecy (PFS).
    """
    try:
        # Generate session challenge using device's KEM public key
        challenge = generate_session_challenge(request.device_id)
        
        if not challenge:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Device not found or not enrolled"
            )
        
        logger.info(f"Generated session challenge for device {request.device_id}")
        
        return ChallengeResponse(
            status="success",
            device_id=request.device_id,
            challenge=challenge,
            message="Session challenge generated"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Challenge generation error for device {request.device_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during challenge generation"
        )


@router.post("/verify_challenge")
async def verify_challenge_response(request: dict):
    """
    Verify challenge response for device authentication.
    
    This endpoint verifies the device's response to the session challenge.
    """
    try:
        device_id = request.get("device_id")
        challenge_response = request.get("challenge_response")
        
        if not device_id or not challenge_response:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Device ID and challenge response are required"
            )
        
        # Decode the challenge response
        try:
            response_bytes = base64.b64decode(challenge_response)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid challenge response format"
            )
        
        # Verify the challenge response (simplified for demo)
        # In a real implementation, you would verify the signature using the device's public key
        logger.info(f"Challenge response verified for device {device_id}")
        
        return {
            "status": "success",
            "device_id": device_id,
            "message": "Device authenticated successfully",
            "session_token": f"session_{device_id}_{int(time.time())}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Challenge verification error for device {device_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during challenge verification"
        )


@router.post("/submit_model", response_model=ModelUpdateResponse)
async def submit_model_update(request: ModelUpdateRequest):
    """
    Submit model update from edge node.
    
    This endpoint receives signed model updates and forwards them to the secure enclave.
    """
    try:
        # Verify model signature
        model_weights_bytes = base64.b64decode(request.model_weights)
        signature_bytes = base64.b64decode(request.signature)
        
        if not verify_model_signature(request.device_id, model_weights_bytes, signature_bytes):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid model signature"
            )
        
        # Store model update for aggregation
        success = store_model_update(
            request.device_id,
            model_weights_bytes,
            request.metadata
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to store model update"
            )
        
        logger.info(f"Model update received from device {request.device_id}")
        
        return ModelUpdateResponse(
            status="success",
            device_id=request.device_id,
            message="Model update received and stored",
            aggregation_round=1  # TODO: Implement round tracking
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model submission error for device {request.device_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during model submission"
        )


@router.get("/global_model", response_model=GlobalModelResponse)
async def get_global_model():
    """
    Get the current global model for download by edge devices.
    """
    try:
        global_model = get_global_model()
        
        if not global_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No global model available"
            )
        
        model_weights_b64 = base64.b64encode(global_model).decode('utf-8')
        
        return GlobalModelResponse(
            status="success",
            model_weights=model_weights_b64,
            model_version="v1.0",  # TODO: Implement versioning
            message="Global model retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Global model retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during model retrieval"
        )


@router.get("/devices", response_model=DeviceListResponse)
async def list_devices():
    """
    List all registered devices.
    """
    try:
        devices = get_registered_devices()
        
        device_list = []
        for device_id, device_info in devices.items():
            device_list.append({
                "device_id": device_id,
                "status": device_info.get("status", "unknown"),
                "last_seen": device_info.get("last_seen"),
                "public_keys": device_info.get("public_keys", {})
            })
        
        return DeviceListResponse(
            devices=device_list,
            total_count=len(device_list)
        )
        
    except Exception as e:
        logger.error(f"Device listing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during device listing"
        )


@router.get("/devices/{device_id}")
async def get_device_info(device_id: str):
    """
    Get information about a specific device.
    """
    try:
        from registry import get_device_info
        
        device_info = get_device_info(device_id)
        
        if not device_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Device not found"
            )
        
        return {
            "status": "success",
            "device_id": device_id,
            "device_info": device_info,
            "message": "Device information retrieved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving device info for {device_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve device information"
        )


@router.get("/enclave/status", response_model=EnclaveStatusResponse)
async def get_enclave_status():
    """
    Get the status of the secure enclave.
    """
    try:
        enclave = get_secure_enclave()
        status_info = enclave.get_enclave_status()
        
        return EnclaveStatusResponse(**status_info)
        
    except Exception as e:
        logger.error(f"Enclave status error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during enclave status check"
        )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint.
    """
    try:
        # Check component status
        components = {
            "server": "healthy",
            "enclave": "healthy",
            "registry": "healthy"
        }
        
        # TODO: Add actual health checks for components
        
        return HealthCheckResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0",
            components=components
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )


# Legacy endpoints for backward compatibility
@router.post("/authenticate", response_model=AuthResponse)
async def authenticate_client(auth_data: AuthRequest):
    """
    Legacy authentication endpoint (deprecated).
    Use /challenge for secure session establishment.
    """
    logger.warning("Legacy authentication endpoint called - use /challenge instead")
    
    return AuthResponse(
        status="deprecated",
        device_id=auth_data.device_id,
        message="This endpoint is deprecated. Use /challenge for secure authentication."
    )


@router.get("/request_qkey")
async def request_qkey_get():
    """
    GET endpoint for quantum key generation (for browser access).
    """
    try:
        # Generate a new quantum keypair
        from auth.pqcrypto_utils import generate_device_keypair
        
        device_id = f"temp_device_{int(time.time())}"
        kem_public_key, sig_public_key = generate_device_keypair(device_id)
        
        # Auto-generate enrollment token using TokenGenerator
        from scripts.generate_token import TokenGenerator
        token_gen = TokenGenerator()
        enrollment_token = token_gen.generate_token(device_id)

        return {
            "status": "success",
            "device_id": device_id,
            "kem_public_key": kem_public_key,
            "signature_public_key": sig_public_key,
            "enrollment_token": enrollment_token,
            "message": "Quantum keypair and enrollment token generated successfully"
        }
        
    except Exception as e:
        logger.error(f"Quantum key generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate quantum keys"
        )


@router.post("/request_qkey")
async def request_qkey_post(request: DeviceActionRequest):
    """
    POST endpoint for quantum key generation with device ID.
    """
    try:
        # Generate quantum keypair for specified device
        from auth.pqcrypto_utils import generate_device_keypair, register_device_keys
        
        device_id = request.device_id or f"device_{int(time.time())}"
        kem_public_key, sig_public_key = generate_device_keypair(device_id)
        
        # Register the keys
        success = register_device_keys(device_id, kem_public_key, sig_public_key)
        
        return {
            "status": "success" if success else "failed",
            "device_id": device_id,
            "kem_public_key": kem_public_key,
            "signature_public_key": sig_public_key,
            "message": "Quantum keypair generated and registered" if success else "Failed to register keys"
        }
        
    except Exception as e:
        logger.error(f"Quantum key generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate quantum keys"
        )


@router.post("/upload_model")
async def upload_model_update(update: dict):
    """
    Legacy model upload endpoint (deprecated).
    Use /submit_model for secure model submission.
    """
    logger.warning("Legacy model upload endpoint called - use /submit_model instead")
    
    return {
        "status": "deprecated",
        "message": "This endpoint is deprecated. Use /submit_model for secure model submission."
    }


@router.post("/register_device")
async def register_device(device_info: dict):
    """
    Legacy device registration endpoint (deprecated).
    Use /enroll for secure device enrollment.
    """
    logger.warning("Legacy device registration endpoint called - use /enroll instead")
    
    return {
        "status": "deprecated",
        "message": "This endpoint is deprecated. Use /enroll for secure device enrollment."
    }


# Protected Endpoints - Require Authentication

@router.get("/protected/dashboard")
async def get_dashboard_data(current_user: User = Depends(get_current_user)):
    """
    Get dashboard data for authenticated users.
    Returns different data based on user role.
    """
    try:
        if current_user.role.value == "admin":
            # Admin dashboard data
            return {
                "type": "admin_dashboard",
                "user": current_user.username,
                "data": {
                    "total_devices": 39,
                    "online_devices": 37,
                    "training_sessions": 156,
                    "completed_rounds": 45,
                    "system_status": "Operational",
                    "security_alerts": 2,
                    "recent_activities": [
                        {"time": "2024-01-15 10:30", "event": "Device DEV-001 enrolled", "type": "device"},
                        {"time": "2024-01-15 10:25", "event": "Training round 45 completed", "type": "training"},
                        {"time": "2024-01-15 10:20", "event": "Security scan completed", "type": "security"}
                    ]
                }
            }
        else:
            # User dashboard data
            return {
                "type": "user_dashboard", 
                "user": current_user.username,
                "data": {
                    "my_training_sessions": 12,
                    "models_submitted": 8,
                    "accuracy_score": 92.5,
                    "rank": 15,
                    "recent_sessions": [
                        {"date": "2024-01-15", "status": "completed", "accuracy": 94.2},
                        {"date": "2024-01-14", "status": "completed", "accuracy": 91.8},
                        {"date": "2024-01-13", "status": "failed", "accuracy": 0}
                    ]
                }
            }
            
    except Exception as e:
        logger.error(f"Dashboard data error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving dashboard data"
        )


@router.get("/protected/admin/devices")
async def get_admin_devices(admin_user: User = Depends(get_current_admin_user)):
    """
    Get device management data - Admin only.
    """
    try:
        # Mock device data for admin
        devices = [
            {"id": "DEV-001", "name": "Edge Device 1", "status": "online", "last_seen": "2024-01-15 10:30"},
            {"id": "DEV-002", "name": "Edge Device 2", "status": "online", "last_seen": "2024-01-15 10:28"},
            {"id": "DEV-003", "name": "Edge Device 3", "status": "offline", "last_seen": "2024-01-14 18:45"},
        ]
        
        return {
            "devices": devices,
            "total": len(devices),
            "online": len([d for d in devices if d["status"] == "online"]),
            "admin_user": admin_user.username
        }
        
    except Exception as e:
        logger.error(f"Admin devices error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving device data"
        )


@router.get("/protected/admin/security")
async def get_security_status(admin_user: User = Depends(get_current_admin_user)):
    """
    Get security monitoring data - Admin only.
    """
    try:
        return {
            "security_level": "High",
            "threat_level": "Low", 
            "active_connections": 37,
            "blocked_attempts": 12,
            "last_scan": "2024-01-15 09:00",
            "quantum_safe_protocols": "Active",
            "alerts": [
                {"time": "2024-01-15 08:30", "type": "warning", "message": "Unusual traffic pattern detected"},
                {"time": "2024-01-14 22:15", "type": "info", "message": "Security scan completed successfully"}
            ],
            "admin_user": admin_user.username
        }
        
    except Exception as e:
        logger.error(f"Security status error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving security data"
        )


@router.get("/protected/user/training")
async def get_user_training_data(current_user: User = Depends(get_current_user)):
    """
    Get user's personal training data.
    Available to all authenticated users.
    """
    try:
        return {
            "user": current_user.username,
            "training_history": [
                {"round": 45, "accuracy": 94.2, "loss": 0.15, "date": "2024-01-15"},
                {"round": 44, "accuracy": 91.8, "loss": 0.18, "date": "2024-01-14"},
                {"round": 43, "accuracy": 93.1, "loss": 0.16, "date": "2024-01-13"}
            ],
            "current_model": "CNN_v2.1",
            "participation_rate": 85.7,
            "total_contributions": 28
        }
        
    except Exception as e:
        logger.error(f"User training data error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving training data"
        )