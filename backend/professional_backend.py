#!/usr/bin/env python3
"""
QFLARE Professional Backend Server

This is a comprehensive FastAPI-based backend server for the QFLARE 
Quantum-Resistant Federated Learning Administration Dashboard.

Features:
- RESTful API for federated learning management
- WebSocket support for real-time updates
- Post-quantum cryptography integration
- Performance monitoring and metrics
- Security scanning and compliance
- Client management and coordination
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Authentication Models
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    user: dict

class User(BaseModel):
    username: str
    role: str
    email: str
    name: str
    last_login: Optional[datetime] = None
    is_active: bool = True

# Data Models
class ClientInfo(BaseModel):
    client_id: str
    status: str = "disconnected"
    last_seen: datetime = Field(default_factory=datetime.now)
    rounds_participated: int = 0
    current_accuracy: float = 0.0
    data_samples: int = 0
    connection_quality: str = "unknown"

class TrainingRound(BaseModel):
    round_id: str
    round_number: int
    status: str = "pending"
    participants: List[str] = []
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    global_accuracy: float = 0.0
    aggregation_strategy: str = "fedavg"

class SystemMetrics(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    active_clients: int = 0
    current_round: int = 0
    total_rounds: int = 100
    global_accuracy: float = 0.0
    training_loss: float = 0.0
    convergence_rate: float = 0.0
    byzantine_detections: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_latency: float = 0.0

class SecurityScanResult(BaseModel):
    scan_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    scan_type: str
    status: str
    vulnerabilities_found: int = 0
    critical_issues: int = 0
    recommendations: List[str] = []

class FederatedLearningConfig(BaseModel):
    algorithm: str = "fedavg"
    rounds: int = 100
    min_clients_per_round: int = 2
    max_clients_per_round: int = 10
    client_selection_strategy: str = "random"
    aggregation_strategy: str = "weighted_average"
    byzantine_tolerance: float = 0.3
    differential_privacy: bool = True
    privacy_budget: float = 1.0
    compression_enabled: bool = True
    encryption_algorithm: str = "kyber1024"

# Global state management
class QFLAREState:
    def __init__(self):
        self.clients: Dict[str, ClientInfo] = {}
        self.training_rounds: List[TrainingRound] = []
        self.current_round: Optional[TrainingRound] = None
        self.system_metrics: SystemMetrics = SystemMetrics()
        self.security_scans: List[SecurityScanResult] = []
        self.websocket_connections: List[WebSocket] = []
        self.fl_config: FederatedLearningConfig = FederatedLearningConfig()
        self.training_active: bool = False

# Global state instance
qflare_state = QFLAREState()

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: dict):
        disconnected_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket: {e}")
                disconnected_connections.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected_connections:
            self.disconnect(connection)

manager = ConnectionManager()

# Startup/Shutdown handlers
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting QFLARE Professional Backend Server...")
    
    # Initialize mock data
    await initialize_mock_data()
    
    # Start background tasks
    asyncio.create_task(metrics_update_task())
    asyncio.create_task(training_simulation_task())
    
    logger.info("✅ QFLARE Backend Server started successfully")
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down QFLARE Backend Server...")

# Create FastAPI application
app = FastAPI(
    title="QFLARE Professional Backend",
    description="Quantum-Resistant Federated Learning Administration API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "active_clients": len(qflare_state.clients),
        "training_active": qflare_state.training_active
    }

# Authentication Endpoints
@app.post("/api/auth/login", response_model=LoginResponse)
async def login(credentials: LoginRequest):
    """Authenticate user and return access token"""
    # Demo users (in production, this would check against a database)
    valid_users = {
        "admin": {
            "password": "admin123",
            "role": "admin",
            "name": "System Administrator",
            "email": "admin@qflare.com"
        },
        "user": {
            "password": "user123", 
            "role": "user",
            "name": "Standard User",
            "email": "user@qflare.com"
        }
    }
    
    user_data = valid_users.get(credentials.username.lower())
    if not user_data or user_data["password"] != credentials.password:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )
    
    # Generate mock JWT token (in production, use proper JWT)
    token = f"qflare_token_{credentials.username}_{int(datetime.now().timestamp())}"
    
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        expires_in=3600,
        user={
            "username": credentials.username.lower(),
            "role": user_data["role"],
            "name": user_data["name"],
            "email": user_data["email"],
            "last_login": datetime.now().isoformat(),
            "is_active": True
        }
    )

@app.post("/api/auth/logout")
async def logout():
    """Logout user and invalidate token"""
    return {"message": "Successfully logged out"}

@app.get("/api/auth/validate-token")
async def validate_token():
    """Validate the current token"""
    # In production, this would validate the JWT token
    return {"valid": True, "user": "authenticated"}

# System information
@app.get("/api/system/info")
async def get_system_info():
    """Get system information and current status"""
    return {
        "system_name": "QFLARE",
        "version": "1.0.0",
        "description": "Quantum-Resistant Federated Learning Framework",
        "status": "operational",
        "uptime": "2h 15m",
        "active_clients": len(qflare_state.clients),
        "current_round": qflare_state.system_metrics.current_round,
        "total_rounds": qflare_state.system_metrics.total_rounds,
        "training_active": qflare_state.training_active
    }

# Metrics endpoints
@app.get("/api/metrics")
async def get_metrics():
    """Get current system metrics"""
    return qflare_state.system_metrics.dict()

@app.get("/api/metrics/history")
async def get_metrics_history():
    """Get historical metrics data"""
    # Generate mock historical data
    history = []
    now = datetime.now()
    
    for i in range(50):
        timestamp = now - timedelta(minutes=i)
        history.append({
            "timestamp": timestamp.isoformat(),
            "global_accuracy": min(0.95, 0.1 + (i * 0.017)),
            "training_loss": max(0.1, 2.0 - (i * 0.038)),
            "active_clients": min(10, 2 + (i // 5)),
            "round_number": max(1, 50 - i)
        })
    
    return {"history": reversed(history)}

# Client management endpoints
@app.get("/api/clients")
async def get_clients():
    """Get all connected clients"""
    return {"clients": list(qflare_state.clients.values())}

@app.get("/api/clients/{client_id}")
async def get_client(client_id: str):
    """Get specific client information"""
    if client_id not in qflare_state.clients:
        raise HTTPException(status_code=404, detail="Client not found")
    return qflare_state.clients[client_id].dict()

@app.post("/api/clients/{client_id}/disconnect")
async def disconnect_client(client_id: str):
    """Disconnect a specific client"""
    if client_id not in qflare_state.clients:
        raise HTTPException(status_code=404, detail="Client not found")
    
    qflare_state.clients[client_id].status = "disconnected"
    await manager.broadcast({
        "type": "client_update",
        "data": {
            "client_id": client_id,
            "status": "disconnected"
        }
    })
    
    return {"message": f"Client {client_id} disconnected successfully"}

# Training management endpoints
@app.get("/api/training/status")
async def get_training_status():
    """Get current training status"""
    return {
        "active": qflare_state.training_active,
        "current_round": qflare_state.system_metrics.current_round,
        "total_rounds": qflare_state.system_metrics.total_rounds,
        "progress": (qflare_state.system_metrics.current_round / qflare_state.system_metrics.total_rounds) * 100,
        "global_accuracy": qflare_state.system_metrics.global_accuracy,
        "participants": len([c for c in qflare_state.clients.values() if c.status == "training"])
    }

@app.post("/api/training/start")
async def start_training():
    """Start federated learning training"""
    if qflare_state.training_active:
        raise HTTPException(status_code=400, detail="Training already active")
    
    qflare_state.training_active = True
    qflare_state.system_metrics.current_round = 1
    
    await manager.broadcast({
        "type": "training_started",
        "data": {
            "timestamp": datetime.now().isoformat(),
            "message": "Federated learning training started"
        }
    })
    
    return {"message": "Training started successfully"}

@app.post("/api/training/stop")
async def stop_training():
    """Stop federated learning training"""
    if not qflare_state.training_active:
        raise HTTPException(status_code=400, detail="Training not active")
    
    qflare_state.training_active = False
    
    # Update all clients to idle status
    for client in qflare_state.clients.values():
        if client.status == "training":
            client.status = "connected"
    
    await manager.broadcast({
        "type": "training_stopped",
        "data": {
            "timestamp": datetime.now().isoformat(),
            "message": "Federated learning training stopped"
        }
    })
    
    return {"message": "Training stopped successfully"}

@app.get("/api/training/rounds")
async def get_training_rounds():
    """Get training round history"""
    return {"rounds": [round.dict() for round in qflare_state.training_rounds]}

# Configuration endpoints
@app.get("/api/config")
async def get_config():
    """Get current federated learning configuration"""
    return qflare_state.fl_config.dict()

@app.put("/api/config")
async def update_config(config: FederatedLearningConfig):
    """Update federated learning configuration"""
    qflare_state.fl_config = config
    
    await manager.broadcast({
        "type": "config_updated",
        "data": config.dict()
    })
    
    return {"message": "Configuration updated successfully", "config": config.dict()}

# Security endpoints
@app.get("/api/security/scans")
async def get_security_scans():
    """Get security scan results"""
    return {"scans": [scan.dict() for scan in qflare_state.security_scans]}

@app.post("/api/security/scan")
async def start_security_scan(scan_type: str = "full"):
    """Start a security scan"""
    scan_id = str(uuid.uuid4())
    
    # Simulate security scan
    scan_result = SecurityScanResult(
        scan_id=scan_id,
        scan_type=scan_type,
        status="running"
    )
    
    qflare_state.security_scans.append(scan_result)
    
    # Simulate scan completion after 3 seconds
    async def complete_scan():
        await asyncio.sleep(3)
        scan_result.status = "completed"
        scan_result.vulnerabilities_found = 2
        scan_result.critical_issues = 0
        scan_result.recommendations = [
            "Update dependency: cryptography to version 3.4.8+",
            "Review client authentication mechanisms"
        ]
        
        await manager.broadcast({
            "type": "security_scan_completed",
            "data": scan_result.dict()
        })
    
    asyncio.create_task(complete_scan())
    
    return {"message": f"Security scan {scan_id} started", "scan_id": scan_id}

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and listen for client messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}))
            elif message.get("type") == "client_register":
                # Register new client
                client_id = message.get("client_id", str(uuid.uuid4()))
                client_info = ClientInfo(
                    client_id=client_id,
                    status="connected",
                    data_samples=message.get("data_samples", 1000)
                )
                qflare_state.clients[client_id] = client_info
                
                await manager.broadcast({
                    "type": "client_connected",
                    "data": client_info.dict()
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Static files (for serving documentation if needed)
@app.get("/favicon.ico")
async def favicon():
    """Serve favicon"""
    return JSONResponse(content={"message": "QFLARE Backend"})

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "QFLARE Professional Backend",
        "version": "1.0.0",
        "description": "Quantum-Resistant Federated Learning Administration API",
        "docs": "/api/docs",
        "health": "/health",
        "websocket": "/ws"
    }

# Background tasks
async def metrics_update_task():
    """Background task to update system metrics"""
    while True:
        try:
            # Update metrics
            qflare_state.system_metrics.active_clients = len([c for c in qflare_state.clients.values() if c.status != "disconnected"])
            qflare_state.system_metrics.timestamp = datetime.now()
            
            # Simulate realistic metrics
            if qflare_state.training_active:
                qflare_state.system_metrics.global_accuracy = min(0.95, qflare_state.system_metrics.global_accuracy + 0.001)
                qflare_state.system_metrics.training_loss = max(0.1, qflare_state.system_metrics.training_loss - 0.002)
            
            # Broadcast metrics update
            await manager.broadcast({
                "type": "metrics_update",
                "data": qflare_state.system_metrics.dict()
            })
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            logger.error(f"Error in metrics update task: {e}")
            await asyncio.sleep(10)

async def training_simulation_task():
    """Background task to simulate training rounds"""
    while True:
        try:
            if qflare_state.training_active and qflare_state.system_metrics.current_round < qflare_state.system_metrics.total_rounds:
                # Simulate training round progression
                await asyncio.sleep(15)  # 15 seconds per round
                
                qflare_state.system_metrics.current_round += 1
                
                # Create training round record
                round_record = TrainingRound(
                    round_id=str(uuid.uuid4()),
                    round_number=qflare_state.system_metrics.current_round,
                    status="completed",
                    participants=list(qflare_state.clients.keys()),
                    start_time=datetime.now() - timedelta(seconds=15),
                    end_time=datetime.now(),
                    global_accuracy=qflare_state.system_metrics.global_accuracy
                )
                
                qflare_state.training_rounds.append(round_record)
                
                await manager.broadcast({
                    "type": "training_round_completed",
                    "data": round_record.dict()
                })
                
                # Check if training is complete
                if qflare_state.system_metrics.current_round >= qflare_state.system_metrics.total_rounds:
                    qflare_state.training_active = False
                    await manager.broadcast({
                        "type": "training_completed",
                        "data": {
                            "message": "Federated learning training completed",
                            "final_accuracy": qflare_state.system_metrics.global_accuracy
                        }
                    })
            
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error in training simulation task: {e}")
            await asyncio.sleep(10)

async def initialize_mock_data():
    """Initialize mock data for demonstration"""
    # Add mock clients
    mock_clients = [
        {"id": "client_001", "samples": 1200, "quality": "excellent"},
        {"id": "client_002", "samples": 800, "quality": "good"},
        {"id": "client_003", "samples": 1500, "quality": "excellent"},
        {"id": "client_004", "samples": 600, "quality": "fair"},
    ]
    
    for client_data in mock_clients:
        client_info = ClientInfo(
            client_id=client_data["id"],
            status="connected",
            data_samples=client_data["samples"],
            connection_quality=client_data["quality"],
            rounds_participated=15,
            current_accuracy=0.75 + (hash(client_data["id"]) % 20) / 100
        )
        qflare_state.clients[client_data["id"]] = client_info
    
    # Initialize system metrics
    qflare_state.system_metrics.active_clients = len(qflare_state.clients)
    qflare_state.system_metrics.global_accuracy = 0.78
    qflare_state.system_metrics.training_loss = 0.45
    qflare_state.system_metrics.current_round = 15
    
    # Add mock security scans
    mock_scan = SecurityScanResult(
        scan_id=str(uuid.uuid4()),
        scan_type="dependency_check",
        status="completed",
        vulnerabilities_found=1,
        critical_issues=0,
        recommendations=["Update PyTorch to version 1.13.0+"]
    )
    qflare_state.security_scans.append(mock_scan)
    
    logger.info("✅ Mock data initialized successfully")

if __name__ == "__main__":
    logger.info("🚀 Starting QFLARE Professional Backend Server...")
    uvicorn.run(
        "professional_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
