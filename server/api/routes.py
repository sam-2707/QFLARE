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

from server.api.schemas import (
    AuthRequest, AuthResponse, DeviceActionRequest, EnrollmentRequest, 
    EnrollmentResponse, ChallengeRequest, ChallengeResponse, ModelUpdateRequest,
    ModelUpdateResponse, GlobalModelResponse, DeviceListResponse, 
    EnclaveStatusResponse, ErrorResponse, HealthCheckResponse
)
from server.auth.pqcrypto_utils import (
    validate_enrollment_token, register_device_keys, generate_session_challenge,
    verify_model_signature, get_device_public_keys
)
from server.enclave.mock_enclave import get_secure_enclave, ModelUpdate
from server.fl_core.aggregator import store_model_update, get_global_model
from server.registry import get_registered_devices

logger = logging.getLogger(__name__)
router = APIRouter()


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


@router.post("/request_qkey")
async def request_qkey(request: DeviceActionRequest):
    """
    Legacy quantum key request endpoint (deprecated).
    """
    logger.warning("Legacy quantum key endpoint called")
    
    return {
        "status": "deprecated",
        "message": "This endpoint is deprecated. Use /challenge for secure key exchange."
    }


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
