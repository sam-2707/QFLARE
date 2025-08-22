"""
QFLARE Server - Main Application

This is the main FastAPI application for the QFLARE federated learning server.
"""

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
import os
from pathlib import Path
from datetime import datetime

from api.routes import router as api_router
from fl_core.client_manager import register_client
from registry import register_device, get_registered_devices

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

# Resolve absolute paths so UI works regardless of CWD
BASE_DIR = Path(__file__).resolve().parent

# Configure templates (absolute path)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Mount static files (absolute path)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Include API routes
app.include_router(api_router, prefix="/api", tags=["api"])


@app.get("/", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def root(request: Request):
    """Main landing page."""
    try:
        device_count = len(get_registered_devices())
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
        devices = get_registered_devices()
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
        from fl_core.aggregator import get_aggregation_status
        from enclave.mock_enclave import get_secure_enclave
        
        # Check component status
        aggregation_status = get_aggregation_status()
        enclave_status = get_secure_enclave().get_enclave_status()
        
        return {
            "status": "healthy",
            "components": {
                "server": "healthy",
                "enclave": enclave_status.get("status", "unknown"),
                "aggregator": "healthy" if aggregation_status else "error"
            },
            "device_count": len(get_registered_devices()),
            "aggregation_status": aggregation_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/status")
@limiter.limit("30/minute")
async def system_status(request: Request):
    """Get system status information."""
    try:
        from registry import get_device_statistics
        from fl_core.aggregator import get_aggregation_status
        
        device_stats = get_device_statistics()
        aggregation_status = get_aggregation_status()
        
        return {
            "system_status": "operational",
            "device_statistics": device_stats,
            "aggregation_status": aggregation_status,
            "security_features": {
                "secure_enrollment": True,
                "post_quantum_crypto": True,
                "secure_enclave": True,
                "poisoning_defense": True
            }
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Error getting system status")


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
async def register_device_legacy(request: Request):
    """
    Legacy device registration (deprecated).
    
    This endpoint is kept for backward compatibility but should not be used.
    Use the secure enrollment process instead.
    """
    logger.warning("Legacy device registration called - use secure enrollment instead")
    
    try:
        # Parse form data
        form_data = await request.form()
        device_id = form_data.get("device_id")
        device_type = form_data.get("device_type", "edge")
        location = form_data.get("location", "")
        description = form_data.get("description", "")
        capabilities = form_data.get("capabilities", "")
        
        if not device_id:
            return templates.TemplateResponse(
                "register.html", 
                {
                    "request": request, 
                    "error": "Device ID is required",
                    "device_id": device_id,
                    "device_type": device_type,
                    "location": location,
                    "description": description,
                    "capabilities": capabilities
                }
            )
        
        # Register device with legacy method
        device_info = {
            "registration_method": "legacy",
            "device_type": device_type,
            "location": location,
            "description": description,
            "capabilities": capabilities,
            "registration_time": datetime.now().isoformat()
        }
        
        success = register_device(device_id, device_info)
        
        if success:
            return templates.TemplateResponse(
                "register.html", 
                {
                    "request": request, 
                    "message": f"Device '{device_id}' registered successfully!",
                    "device_id": "",
                    "device_type": "edge",
                    "location": "",
                    "description": "",
                    "capabilities": ""
                }
            )
        else:
            return templates.TemplateResponse(
                "register.html", 
                {
                    "request": request, 
                    "error": f"Failed to register device '{device_id}'. Device may already exist.",
                    "device_id": device_id,
                    "device_type": device_type,
                    "location": location,
                    "description": description,
                    "capabilities": capabilities
                }
            )
    except Exception as e:
        logger.error(f"Error in legacy device registration: {e}")
        return templates.TemplateResponse(
            "register.html", 
            {
                "request": request, 
                "error": f"Registration failed: {str(e)}",
                "device_id": device_id if 'device_id' in locals() else "",
                "device_type": device_type if 'device_type' in locals() else "edge",
                "location": location if 'location' in locals() else "",
                "description": description if 'description' in locals() else "",
                "capabilities": capabilities if 'capabilities' in locals() else ""
            }
        )


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