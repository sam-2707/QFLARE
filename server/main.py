"""
QFLARE Server - Main Application

This is the main FastAPI application for the QFLARE federated learning server.
"""

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
import json

from api.routes import router as api_router
from database import db_manager
from key_manager import key_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="QFLARE Server",
    description="Quantum-Resistant Federated Learning Server",
    version="1.0.0"
)

# Configure rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Mount static files
from pathlib import Path
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include API routes
app.include_router(api_router, prefix="/api", tags=["api"])


@app.get("/", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def root(request: Request):
    """Main landing page."""
    try:
        devices = db_manager.get_all_devices()
        device_count = len(devices)
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "device_count": device_count}
        )
    except Exception as e:
        logger.error(f"Error rendering root page: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/devices", response_class=HTMLResponse)
@limiter.limit("30/minute")
async def list_devices(request: Request):
    """Display list of registered devices."""
    try:
        devices = db_manager.get_all_devices()
        return templates.TemplateResponse(
            "devices.html", 
            {"request": request, "devices": devices}
        )
    except Exception as e:
        logger.error(f"Error rendering devices page: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Health check endpoint."""
    try:
        # Check component status
        device_stats = db_manager.get_device_statistics()
        server_keys = key_manager.get_server_public_keys()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "server": "healthy",
                "database": "connected",
                "key_management": "active",
                "enclave": "secure"
            },
            "statistics": device_stats,
            "server_keys": {
                "kem_algorithm": server_keys["kem_algorithm"],
                "signature_algorithm": server_keys["signature_algorithm"]
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/status")
@limiter.limit("30/minute")
async def system_status(request: Request):
    """Get system status information."""
    try:
        device_stats = db_manager.get_device_statistics()
        server_keys = key_manager.get_server_public_keys()
        
        return {
            "system_status": "operational",
            "device_statistics": device_stats,
            "security_features": {
                "secure_enrollment": True,
                "post_quantum_crypto": True,
                "secure_enclave": True,
                "poisoning_defense": True
            },
            "server_keys": {
                "kem_algorithm": server_keys["kem_algorithm"],
                "signature_algorithm": server_keys["signature_algorithm"]
            }
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Error getting system status")


# New API endpoints for web interface
@app.post("/api/generate_token")
@limiter.limit("10/minute")
async def generate_enrollment_token(request: Request):
    """Generate enrollment token for device registration."""
    try:
        body = await request.json()
        device_id = body.get("device_id")
        expiration_hours = body.get("expiration_hours", 24)
        
        if not device_id:
            raise HTTPException(status_code=400, detail="Device ID required")
        
        # Generate enrollment token
        token = key_manager.generate_enrollment_token(device_id, expiration_hours)
        
        # Calculate expiration time
        expires_at = datetime.now() + timedelta(hours=expiration_hours)
        
        return {
            "status": "success",
            "token": token,
            "device_id": device_id,
            "expires_at": expires_at.isoformat(),
            "message": "Enrollment token generated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating enrollment token: {e}")
        raise HTTPException(status_code=500, detail="Error generating enrollment token")


@app.post("/api/rotate_keys")
@limiter.limit("5/minute")
async def rotate_server_keys(request: Request):
    """Rotate server keys."""
    try:
        success = key_manager.rotate_server_keys()
        
        if success:
            return {
                "status": "success",
                "message": "Server keys rotated successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to rotate server keys")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rotating server keys: {e}")
        raise HTTPException(status_code=500, detail="Error rotating server keys")


@app.get("/api/devices")
@limiter.limit("30/minute")
async def get_devices_api(request: Request):
    """Get all devices via API."""
    try:
        devices = db_manager.get_all_devices()
        
        # Format devices for API response
        device_list = []
        for device in devices:
            device_list.append({
                "device_id": device["device_id"],
                "status": device["status"],
                "created_at": device["created_at"].isoformat() if device["created_at"] else None,
                "last_seen": device["last_seen"].isoformat() if device["last_seen"] else None,
                "has_kem_key": bool(device["kem_public_key"]),
                "has_signature_key": bool(device["signature_public_key"])
            })
        
        return {
            "devices": device_list,
            "total_count": len(device_list),
            "active_count": len([d for d in device_list if d["status"] == "active"])
        }
        
    except Exception as e:
        logger.error(f"Error getting devices: {e}")
        raise HTTPException(status_code=500, detail="Error getting devices")


@app.put("/api/devices/{device_id}/status")
@limiter.limit("30/minute")
async def update_device_status_api(device_id: str, request: Request):
    """Update device status."""
    try:
        body = await request.json()
        new_status = body.get("status")
        
        if not new_status or new_status not in ["active", "inactive", "suspended"]:
            raise HTTPException(status_code=400, detail="Invalid status")
        
        success = db_manager.update_device_status(device_id, new_status)
        
        if success:
            return {
                "status": "success",
                "device_id": device_id,
                "new_status": new_status,
                "message": f"Device status updated to {new_status}"
            }
        else:
            raise HTTPException(status_code=404, detail="Device not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating device status: {e}")
        raise HTTPException(status_code=500, detail="Error updating device status")


@app.get("/api/server_info")
@limiter.limit("60/minute")
async def get_server_info(request: Request):
    """Get server information."""
    try:
        server_keys = key_manager.get_server_public_keys()
        device_stats = db_manager.get_device_statistics()
        
        return {
            "server_info": {
                "name": "QFLARE Server",
                "version": "1.0.0",
                "host": request.base_url.hostname,
                "port": request.base_url.port or 8000,
                "protocol": request.base_url.scheme,
                "startup_time": datetime.now().isoformat()
            },
            "security": {
                "kem_algorithm": server_keys["kem_algorithm"],
                "signature_algorithm": server_keys["signature_algorithm"],
                "post_quantum_crypto": True
            },
            "statistics": device_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting server info: {e}")
        raise HTTPException(status_code=500, detail="Error getting server info")


# Legacy endpoints for backward compatibility (deprecated)
@app.get("/register", response_class=HTMLResponse)
@limiter.limit("5/minute")
async def register_form(request: Request):
    """
    Legacy registration form (deprecated).
    
    This endpoint is kept for backward compatibility but should not be used.
    Use the secure enrollment process instead.
    """
    logger.warning("Legacy registration form accessed - use secure enrollment instead")
    return templates.TemplateResponse(
        "register.html", 
        {"request": request, "message": "This registration method is deprecated. Use secure enrollment instead."}
    )


@app.post("/register", response_class=HTMLResponse)
@limiter.limit("5/minute")
async def register_device_legacy(request: Request, device_id: str = None):
    """
    Legacy device registration (deprecated).
    
    This endpoint is kept for backward compatibility but should not be used.
    Use the secure enrollment process instead.
    """
    logger.warning("Legacy device registration called - use secure enrollment instead")
    
    if not device_id:
        raise HTTPException(status_code=400, detail="Device ID required")
    
    try:
        # Register device with legacy method
        success = db_manager.register_device(device_id, metadata={"registration_method": "legacy"})
        
        if success:
            return templates.TemplateResponse(
                "register.html", 
                {"request": request, "message": f"Device {device_id} registered (legacy method)"}
            )
        else:
            return templates.TemplateResponse(
                "register.html", 
                {"request": request, "message": f"Failed to register device {device_id}"}
            )
    except Exception as e:
        logger.error(f"Error in legacy device registration: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors."""
    return templates.TemplateResponse(
        "404.html", 
        {"request": request, "message": "Page not found"}, 
        status_code=404
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    """Handle 500 errors."""
    return templates.TemplateResponse(
        "500.html", 
        {"request": request, "message": "Internal server error"}, 
        status_code=500
    )


if __name__ == "__main__":
    import uvicorn
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get configuration
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    
    logger.info(f"Starting QFLARE server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )