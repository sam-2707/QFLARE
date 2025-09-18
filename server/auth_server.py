"""
Minimal QFLARE server for testing authentication system.
Focuses on authentication endpoints without quantum crypto dependencies.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime

# Import our authentication modules
from auth.auth_endpoints import auth_router
from auth.jwt_utils import get_current_user, get_current_admin_user
from auth.user_models import User
from fastapi import Depends

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="QFLARE Authentication Server",
    description="Quantum-Resistant Federated Learning Server - Authentication Testing",
    version="1.0.0"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include authentication routes
app.include_router(auth_router, prefix="/api/auth", tags=["authentication"])

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Server health check."""
    return {
        "status": "healthy",
        "service": "qflare-auth-server",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Protected dashboard endpoint
@app.get("/api/protected/dashboard")
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
        return JSONResponse(
            status_code=500,
            content={"error": "Error retrieving dashboard data"}
        )

# Admin-only endpoint
@app.get("/api/protected/admin/devices")
async def get_admin_devices(admin_user: User = Depends(get_current_admin_user)):
    """Get device management data - Admin only."""
    try:
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
        return JSONResponse(
            status_code=500,
            content={"error": "Error retrieving device data"}
        )

# User training data endpoint
@app.get("/api/protected/user/training")
async def get_user_training_data(current_user: User = Depends(get_current_user)):
    """Get user's personal training data."""
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
        return JSONResponse(
            status_code=500,
            content={"error": "Error retrieving training data"}
        )

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting QFLARE Authentication Server...")
    logger.info("Available demo credentials:")
    logger.info("  Admin: admin / admin123")
    logger.info("  User:  user / user123")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )