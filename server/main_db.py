"""
QFLARE Server - Main Application (Database-Integrated v2.0)

FastAPI application with unified database backend for quantum-resistant federated learning.
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
import time
import asyncio
from pathlib import Path
from datetime import datetime, UTC

# Import unified database system
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database import (
    init_database, close_database, get_database_health,
    DeviceRepository, KeyExchangeRepository, AuditRepository,
    security_audit_log
)

# Import updated server components
from api.routes import router as api_router
from fl_core.client_manager import register_client
from registry_db import DeviceRegistryDB, initialize_registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title="QFLARE Server",
    description="Quantum-Resistant Federated Learning Server (Database-Integrated)",
    version="2.0.0"
)

# Apply middleware and state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
static_path = Path(__file__).parent / "static"
templates_path = Path(__file__).parent / "templates"

if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

if templates_path.exists():
    templates = Jinja2Templates(directory=str(templates_path))
else:
    templates = None

# Include API routes
app.include_router(api_router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    """Initialize database and server components on startup"""
    try:
        # Initialize unified database
        await init_database()
        logger.info("✅ Unified database initialized successfully")
        
        # Initialize device registry
        await initialize_registry()
        logger.info("✅ Device registry initialized")
        
        # Log server startup
        await security_audit_log(
            event_type="SERVER_STARTUP",
            message="QFLARE server started with database integration",
            threat_level=1
        )
        
        logger.info("🚀 QFLARE Server v2.0 (Database-Integrated) started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize server: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up database connections on shutdown"""
    try:
        # Log server shutdown
        await security_audit_log(
            event_type="SERVER_SHUTDOWN",
            message="QFLARE server shutting down",
            threat_level=1
        )
        
        # Close database connections
        await close_database()
        logger.info("✅ Database connections closed")
        logger.info("🛑 QFLARE Server shutdown completed")
        
    except Exception as e:
        logger.error(f"⚠️ Error during shutdown: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    if templates:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "title": "QFLARE Server",
            "version": "2.0.0",
            "database_integrated": True
        })
    else:
        return HTMLResponse(content="""
        <html>
            <head><title>QFLARE Server v2.0</title></head>
            <body>
                <h1>🔐 QFLARE Server v2.0</h1>
                <h2>Quantum-Resistant Federated Learning (Database-Integrated)</h2>
                <ul>
                    <li><a href="/api/docs">API Documentation</a></li>
                    <li><a href="/health">Health Check</a></li>
                    <li><a href="/api/devices">Device Registry</a></li>
                    <li><a href="/api/sessions">Active Sessions</a></li>
                    <li><a href="/api/audit/recent">Audit Logs</a></li>
                </ul>
                <p><strong>Features:</strong></p>
                <ul>
                    <li>✅ Production Database Backend</li>
                    <li>✅ Quantum-Safe Cryptography</li>
                    <li>✅ Real-time Security Monitoring</li>
                    <li>✅ Comprehensive Audit Logging</li>
                </ul>
            </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Comprehensive health check with database status"""
    try:
        # Get database health
        db_health = await get_database_health()
        
        # Get system statistics
        approved_devices = await DeviceRepository.get_devices_by_status('approved')
        active_sessions = await KeyExchangeRepository.get_active_sessions()
        
        # Get recent audit activity
        recent_logs = await AuditRepository.get_recent_logs(limit=10, hours=1)
        
        health_status = {
            "status": "healthy" if db_health["status"] == "healthy" else "degraded",
            "timestamp": datetime.now(UTC).isoformat(),
            "version": "2.0.0",
            "database": {
                "status": db_health["status"],
                "response_time_ms": db_health.get("response_time_ms", "N/A"),
                "url": db_health.get("database_url", "Unknown")
            },
            "statistics": {
                "active_devices": len(approved_devices),
                "active_sessions": len(active_sessions),
                "recent_audit_events": len(recent_logs)
            },
            "features": {
                "quantum_safe_crypto": True,
                "database_persistence": True,
                "security_monitoring": True,
                "audit_logging": True
            }
        }
        
        # Log health check (low frequency to avoid spam)
        if len(recent_logs) == 0 or (
            recent_logs and 
            "HEALTH_CHECK" not in [log.event_type for log in recent_logs[:5]]
        ):
            await security_audit_log(
                event_type="HEALTH_CHECK",
                message="Server health check performed",
                threat_level=1
            )
        
        return health_status
        
    except Exception as e:
        error_status = {
            "status": "error",
            "timestamp": datetime.now(UTC).isoformat(),
            "version": "2.0.0",
            "error": str(e),
            "database": {"status": "error"}
        }
        return JSONResponse(content=error_status, status_code=503)

@app.get("/api/status")
async def get_server_status():
    """Get detailed server status"""
    try:
        # Get comprehensive statistics
        stats = await DeviceRegistryDB.get_all_devices()
        active_sessions = await KeyExchangeRepository.get_active_sessions()
        session_stats = await KeyExchangeRepository.get_session_statistics(hours=24)
        security_summary = await AuditRepository.get_security_summary(hours=24)
        
        return {
            "server_info": {
                "name": "QFLARE Server",
                "version": "2.0.0",
                "description": "Quantum-Resistant Federated Learning (Database-Integrated)",
                "startup_time": datetime.now(UTC).isoformat()
            },
            "device_statistics": {
                "total_devices": len(stats),
                "active_devices": len([d for d in stats if d.get('status') == 'approved']),
                "device_types": {
                    device_type: len([d for d in stats if d.get('device_type') == device_type])
                    for device_type in set(d.get('device_type', 'unknown') for d in stats)
                }
            },
            "session_statistics": {
                "active_sessions": len(active_sessions),
                "total_sessions_24h": session_stats["total_sessions"],
                "average_duration_ms": session_stats["average_duration_ms"]
            },
            "security_statistics": {
                "total_events_24h": security_summary["total_events"],
                "severity_breakdown": security_summary["severity_counts"],
                "threat_level_counts": security_summary["threat_level_counts"]
            },
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat()
        }

@app.get("/api/devices")
async def get_devices():
    """Get all registered devices from database"""
    try:
        devices = await DeviceRegistryDB.get_all_devices()
        return {
            "devices": devices,
            "count": len(devices),
            "timestamp": datetime.now(UTC).isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "devices": [],
            "count": 0,
            "timestamp": datetime.now(UTC).isoformat()
        }

@app.get("/api/sessions")
async def get_sessions():
    """Get active key exchange sessions"""
    try:
        active_sessions = await KeyExchangeRepository.get_active_sessions()
        
        sessions = []
        for session in active_sessions:
            session_data = {
                "session_id": session.session_id,
                "device_id": session.device_id,
                "algorithm": session.algorithm,
                "status": session.status,
                "security_level": session.security_level,
                "created_at": session.created_at.isoformat(),
                "expires_at": session.expires_at.isoformat(),
                "client_ip": session.client_ip
            }
            sessions.append(session_data)
        
        return {
            "sessions": sessions,
            "count": len(sessions),
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "sessions": [],
            "count": 0,
            "timestamp": datetime.now(UTC).isoformat()
        }

@app.get("/api/audit/recent")
async def get_recent_audit():
    """Get recent audit logs"""
    try:
        recent_logs = await AuditRepository.get_recent_logs(limit=50, hours=24)
        
        logs = []
        for log in recent_logs:
            log_data = {
                "id": log.id,
                "event_type": log.event_type,
                "severity": log.severity,
                "message": log.message,
                "device_id": log.device_id,
                "threat_level": log.threat_level,
                "timestamp": log.event_timestamp.isoformat(),
                "client_ip": log.client_ip
            }
            logs.append(log_data)
        
        return {
            "logs": logs,
            "count": len(logs),
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "logs": [],
            "count": 0,
            "timestamp": datetime.now(UTC).isoformat()
        }

@app.post("/api/device/register")
async def register_device_endpoint(device_data: dict):
    """Register a new device via API"""
    try:
        device_id = device_data.get("device_id")
        if not device_id:
            raise HTTPException(status_code=400, detail="device_id required")
        
        # Register device using new database system
        success = await DeviceRegistryDB.register_device(
            device_id=device_id,
            device_info=device_data,
            organization=device_data.get("organization", "API_CLIENT"),
            location=device_data.get("location", "Remote")
        )
        
        if success:
            return {
                "success": True,
                "message": f"Device {device_id} registered successfully",
                "device_id": device_id,
                "timestamp": datetime.now(UTC).isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Device registration failed")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Rate limited endpoints
@app.get("/api/device/{device_id}")
@limiter.limit("60/minute")
async def get_device_info(request: Request, device_id: str):
    """Get device information by ID"""
    try:
        device = await DeviceRegistryDB.get_device(device_id)
        if device:
            return {
                "device": device,
                "timestamp": datetime.now(UTC).isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Device not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Development server
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting QFLARE Server v2.0 (Database-Integrated)")
    print("📊 Features: Quantum-Safe Crypto + Production Database")
    print("🔗 URL: http://localhost:8001")
    print("📖 API Docs: http://localhost:8001/api/docs")
    
    uvicorn.run(
        "main_db:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )