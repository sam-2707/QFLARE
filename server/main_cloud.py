#!/usr/bin/env python3
"""
QFLARE Server - Cloud Optimized Version

This version is optimized for cloud deployment platforms like Railway.
"""

import os
import uvicorn
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
from pathlib import Path

# Import your existing modules
try:
    from database import db_manager
    from key_manager import key_manager
    from api.routes import router as api_router
except ImportError:
    # Fallback for cloud deployment
    print("Warning: Some modules not available, using minimal setup")
    db_manager = None
    key_manager = None
    api_router = None

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure templates and static files
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include API routes if available
if api_router:
    app.include_router(api_router, prefix="/api", tags=["api"])

@app.get("/")
async def root():
    """Main landing page."""
    return {"message": "QFLARE Server is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "QFLARE Server is operational",
        "version": "1.0.0"
    }

@app.get("/api/devices")
async def get_devices():
    """Get devices endpoint."""
    if db_manager:
        try:
            devices = db_manager.get_all_devices()
            return {"devices": devices, "total": len(devices)}
        except Exception as e:
            logger.error(f"Error getting devices: {e}")
            return {"devices": [], "total": 0, "error": str(e)}
    else:
        return {"devices": [], "total": 0, "message": "Database not available"}

@app.post("/api/generate_token")
async def generate_token(request: Request):
    """Generate enrollment token."""
    try:
        body = await request.json()
        device_id = body.get("device_id", "unknown")
        
        # Simple token generation for cloud deployment
        import secrets
        token = secrets.token_urlsafe(32)
        
        return {
            "status": "success",
            "token": token,
            "device_id": device_id,
            "message": "Token generated successfully"
        }
    except Exception as e:
        logger.error(f"Error generating token: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Get PORT from environment, with proper fallback
    port_str = os.getenv("PORT", "8000")
    try:
        port = int(port_str)
    except ValueError:
        print(f"Warning: Invalid PORT value '{port_str}', using default 8000")
        port = 8000
    
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting QFLARE Server on {host}:{port}")
    
    uvicorn.run(
        "main_cloud:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    ) 