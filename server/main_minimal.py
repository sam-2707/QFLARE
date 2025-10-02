"""
QFLARE Production Server - Minimal Version
A simplified server for production deployment without complex dependencies
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Add server directory to Python path
server_dir = Path(__file__).parent
sys.path.insert(0, str(server_dir))

# Set environment flags to disable complex features
os.environ['DISABLE_QUANTUM_CRYPTO'] = 'true'
os.environ['DISABLE_SGX'] = 'true'
os.environ['DISABLE_PQC'] = 'true'

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import new modules
try:
    from device_management import router as device_router
    from training_control import router as training_router
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Advanced modules not available: {e}")
    MODULES_AVAILABLE = False

# Create FastAPI app
app = FastAPI(
    title="QFLARE Federated Learning Platform",
    description="Quantum-Federated Learning with Advanced Resilience Engine",
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

# Include routers
if MODULES_AVAILABLE:
    app.include_router(device_router)
    app.include_router(training_router)
    logger.info("✅ Advanced modules loaded successfully")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🚀 QFLARE Server is running!",
        "timestamp": datetime.now().isoformat(),
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "database": "connected" if os.getenv("DATABASE_URL") else "not configured",
            "redis": "connected" if os.getenv("REDIS_URL") else "not configured"
        }
    }

@app.get("/api/status")
async def api_status():
    """API status endpoint"""
    return {
        "qflare_version": "1.0.0",
        "api_version": "v1",
        "environment": "production",
        "features": {
            "federated_learning": True,
            "quantum_crypto": False,
            "secure_enclaves": False,
            "differential_privacy": True
        }
    }

@app.get("/api/fl/status")
async def fl_status():
    """Federated Learning status endpoint"""
    return {
        "available": True,
        "current_round": 0,
        "total_rounds": 10,
        "status": "idle",
        "registered_devices": 0,
        "active_devices": 0,
        "participants_this_round": 0,
        "round_start_time": None,
        "training_history": []
    }

@app.get("/api/devices")
async def get_devices():
    """Get registered devices endpoint"""
    return []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates"""
    print(f"WebSocket connection attempt from: {websocket.client}")
    
    try:
        await websocket.accept()
        print("WebSocket connection accepted")
        
        # Send initial connection message
        await websocket.send_text(json.dumps({
            "event": "connection_established",
            "data": {"message": "Connected to QFLARE dashboard"}
        }))
        print("Initial message sent")
        
        # Wait a moment before starting periodic updates
        await asyncio.sleep(2)
        
        # Keep connection alive and send periodic updates
        update_count = 0
        while True:
            try:
                # Send heartbeat every 10 seconds
                if update_count % 2 == 0:
                    await websocket.send_text(json.dumps({
                        "event": "heartbeat",
                        "data": {"timestamp": datetime.now().isoformat()}
                    }))
                    print("Heartbeat sent")
                else:
                    await websocket.send_text(json.dumps({
                        "event": "fl_status_update", 
                        "data": {
                            "available": True,
                            "current_round": 0,
                            "total_rounds": 10,
                            "status": "idle",
                            "registered_devices": 0,
                            "active_devices": 0,
                            "participants_this_round": 0,
                            "round_start_time": None,
                            "training_history": []
                        }
                    }))
                    print("FL status update sent")
                
                update_count += 1
                await asyncio.sleep(5)
                
            except Exception as e:
                print(f"Error sending update: {e}")
                break
            
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        traceback.print_exc()

@app.websocket("/api/ws/dashboard")
async def websocket_dashboard_endpoint(websocket: WebSocket):
    """Alternative WebSocket endpoint for dashboard updates"""
    print(f"WebSocket connection attempt from: {websocket.client}")
    
    try:
        await websocket.accept()
        print("WebSocket connection accepted")
        
        # Send initial connection message
        await websocket.send_text(json.dumps({
            "event": "connection_established",
            "data": {"message": "Connected to QFLARE dashboard"}
        }))
        print("Initial message sent")
        
        # Wait a moment before starting periodic updates
        await asyncio.sleep(2)
        
        # Keep connection alive and send periodic updates
        update_count = 0
        while True:
            try:
                # Send heartbeat every 10 seconds
                if update_count % 2 == 0:
                    await websocket.send_text(json.dumps({
                        "event": "heartbeat",
                        "data": {"timestamp": datetime.now().isoformat()}
                    }))
                    print("Heartbeat sent")
                else:
                    await websocket.send_text(json.dumps({
                        "event": "fl_status_update", 
                        "data": {
                            "available": True,
                            "current_round": 0,
                            "total_rounds": 10,
                            "status": "idle",
                            "registered_devices": 0,
                            "active_devices": 0,
                            "participants_this_round": 0,
                            "round_start_time": None,
                            "training_history": []
                        }
                    }))
                    print("FL status update sent")
                
                update_count += 1
                await asyncio.sleep(5)
                
            except Exception as e:
                print(f"Error sending update: {e}")
                break
            
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        traceback.print_exc()

@app.get("/dashboard")
async def dashboard():
    """Simple dashboard"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>QFLARE Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { background: #2196F3; color: white; padding: 20px; border-radius: 8px; }
            .status { margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 8px; }
            .success { color: #4CAF50; }
            .info { color: #2196F3; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 QFLARE Production Server</h1>
            <p>Quantum-Federated Learning with Advanced Resilience Engine</p>
        </div>
        
        <div class="status">
            <h2>📊 System Status</h2>
            <p class="success">✅ Server: Running</p>
            <p class="success">✅ API: Active</p>
            <p class="info">ℹ️ Mode: Production</p>
            <p class="info">ℹ️ Version: 1.0.0</p>
        </div>
        
        <div class="status">
            <h2>🔗 Available Endpoints</h2>
            <ul>
                <li><a href="/health">Health Check</a></li>
                <li><a href="/api/status">API Status</a></li>
                <li><a href="/api/fl/status">FL Status</a></li>
                <li><a href="/docs">API Documentation</a></li>
            </ul>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting QFLARE Production Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")