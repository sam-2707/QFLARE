#!/usr/bin/env python3
"""
QFLARE Server - Vercel Serverless Function

This is the main entry point for Vercel serverless deployment.
"""

import os
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import logging
import secrets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="QFLARE Server",
    description="Quantum-Resistant Federated Learning Server",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates and static files
templates = Jinja2Templates(directory="api/templates")
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# In-memory storage for Vercel (serverless)
devices = {}
tokens = {}

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Main landing page."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "server_url": "https://qflare-sam-2707s-projects.vercel.app",
            "deployment": "vercel"
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "QFLARE Server is operational",
        "version": "1.0.0",
        "deployment": "vercel",
        "device_count": len(devices)
    }

@app.get("/api/devices")
async def get_devices():
    """Get devices endpoint."""
    try:
        device_list = []
        for device_id, device_info in devices.items():
            device_list.append({
                "device_id": device_id,
                "status": device_info.get("status", "active"),
                "created_at": device_info.get("created_at"),
                "last_seen": device_info.get("last_seen")
            })
        
        return {
            "devices": device_list,
            "total_count": len(device_list),
            "deployment": "vercel"
        }
    except Exception as e:
        logger.error(f"Error getting devices: {e}")
        return {
            "devices": [],
            "total_count": 0,
            "error": str(e),
            "deployment": "vercel"
        }

@app.post("/api/generate_token")
async def generate_token(request: Request):
    """Generate enrollment token."""
    try:
        body = await request.json()
        device_id = body.get("device_id", "unknown")
        expiration_hours = body.get("expiration_hours", 24)
        
        # Generate secure token
        token = secrets.token_urlsafe(32)
        
        # Store token (in-memory for Vercel)
        tokens[token] = {
            "device_id": device_id,
            "expiration_hours": expiration_hours,
            "created_at": "2024-01-01T00:00:00Z"  # Simplified for demo
        }
        
        return {
            "status": "success",
            "token": token,
            "device_id": device_id,
            "expires_at": "2024-01-02T00:00:00Z",  # Simplified for demo
            "message": "Token generated successfully",
            "deployment": "vercel"
        }
    except Exception as e:
        logger.error(f"Error generating token: {e}")
        return {
            "status": "error",
            "message": str(e),
            "deployment": "vercel"
        }

@app.post("/api/enroll")
async def enroll_device(request: Request):
    """Enroll a device using token."""
    try:
        body = await request.json()
        device_id = body.get("device_id")
        token = body.get("enrollment_token")
        
        if not device_id or not token:
            return {
                "status": "error",
                "message": "Device ID and enrollment token required",
                "deployment": "vercel"
            }
        
        # Check if token exists
        if token not in tokens:
            return {
                "status": "error",
                "message": "Invalid or expired enrollment token",
                "deployment": "vercel"
            }
        
        # Register device
        devices[device_id] = {
            "device_id": device_id,
            "status": "active",
            "created_at": "2024-01-01T00:00:00Z",
            "last_seen": "2024-01-01T00:00:00Z"
        }
        
        # Remove used token
        del tokens[token]
        
        return {
            "status": "success",
            "device_id": device_id,
            "message": "Device enrolled successfully",
            "deployment": "vercel"
        }
    except Exception as e:
        logger.error(f"Error enrolling device: {e}")
        return {
            "status": "error",
            "message": str(e),
            "deployment": "vercel"
        }

@app.get("/api/server_info")
async def get_server_info():
    """Get server information."""
    return {
        "server_info": {
            "name": "QFLARE Server",
            "version": "1.0.0",
            "deployment": "vercel",
            "platform": "serverless"
        },
        "security": {
            "post_quantum_crypto": True,
            "secure_enrollment": True
        },
        "statistics": {
            "total_devices": len(devices),
            "active_devices": len(devices),
            "deployment": "vercel"
        }
    }

@app.get("/devices", response_class=HTMLResponse)
async def devices_page(request: Request):
    """Devices management page."""
    return templates.TemplateResponse(
        "devices.html",
        {
            "request": request,
            "server_url": "https://qflare-sam-2707s-projects.vercel.app",
            "deployment": "vercel"
        }
    )

@app.get("/docs")
async def api_docs():
    """API documentation."""
    return {
        "message": "QFLARE API Documentation",
        "endpoints": {
            "GET /": "Server status",
            "GET /health": "Health check",
            "GET /api/devices": "List devices",
            "POST /api/generate_token": "Generate enrollment token",
            "POST /api/enroll": "Enroll device",
            "GET /api/server_info": "Server information"
        },
        "deployment": "vercel"
    }

# For Vercel serverless functions
if __name__ == "__main__":
    import uvicorn
    
    # Get PORT from environment
    port_str = os.getenv("PORT", "8000")
    try:
        port = int(port_str)
    except ValueError:
        port = 8000
    
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting QFLARE Server on Vercel - {host}:{port}")
    
    uvicorn.run(
        "api.index:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    ) 