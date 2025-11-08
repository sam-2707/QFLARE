#!/usr/bin/env python3
"""
QFLARE Simple Backend Server
A lightweight FastAPI backend for QFLARE without monitoring dependencies
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Authentication Models
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    confirm_password: str

class User(BaseModel):
    id: str
    username: str
    email: str
    role: str = "user"
    created_at: datetime
    is_active: bool = True

# Mock users database
USERS_DB = {
    "admin": {
        "id": "admin-123",
        "username": "admin",
        "email": "admin@qflare.com",
        "password": "admin123",  # In production, this should be hashed
        "role": "admin",
        "created_at": datetime.now(),
        "is_active": True
    },
    "user": {
        "id": "user-456", 
        "username": "user",
        "email": "user@qflare.com",
        "password": "user123",
        "role": "user",
        "created_at": datetime.now(),
        "is_active": True
    }
}

# Session storage
SESSIONS = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting QFLARE Simple Backend Server...")
    
    try:
        logger.info("✅ QFLARE Simple Backend Server started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start QFLARE Backend Server: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down QFLARE Simple Backend Server...")
    logger.info("✅ Shutdown completed")

# Create FastAPI app
app = FastAPI(
    title="QFLARE Simple Backend API",
    description="Quantum-Safe Federated Learning Administration REST API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication helper
def get_current_user(request: Request):
    """Extract user from session"""
    token = request.headers.get("Authorization")
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if token.startswith("Bearer "):
        token = token[7:]
    
    session = SESSIONS.get(token)
    if not session or session["expires"] < datetime.now():
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    return session["user"]

# Auth Routes
@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """User login endpoint"""
    user_data = USERS_DB.get(request.username)
    
    if not user_data or user_data["password"] != request.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Create session
    session_token = str(uuid.uuid4())
    SESSIONS[session_token] = {
        "user": {
            "id": user_data["id"],
            "username": user_data["username"],
            "email": user_data["email"],
            "role": user_data["role"]
        },
        "expires": datetime.now() + timedelta(hours=24)
    }
    
    return {
        "token": session_token,
        "user": {
            "id": user_data["id"],
            "username": user_data["username"],
            "email": user_data["email"],
            "role": user_data["role"]
        }
    }

@app.post("/api/auth/register")
async def register(request: RegisterRequest):
    """User registration endpoint"""
    if request.password != request.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    
    if request.username in USERS_DB:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Create new user
    user_id = str(uuid.uuid4())
    USERS_DB[request.username] = {
        "id": user_id,
        "username": request.username,
        "email": request.email,
        "password": request.password,  # In production, hash this
        "role": "user",
        "created_at": datetime.now(),
        "is_active": True
    }
    
    return {"message": "User registered successfully", "user_id": user_id}

@app.post("/api/auth/logout")
async def logout(request: Request):
    """User logout endpoint"""
    token = request.headers.get("Authorization")
    if token and token.startswith("Bearer "):
        token = token[7:]
        SESSIONS.pop(token, None)
    
    return {"message": "Logged out successfully"}

@app.get("/api/auth/me")
async def get_current_user_info(user: dict = Depends(get_current_user)):
    """Get current user information"""
    return {"user": user}

# Dashboard Routes
@app.get("/api/dashboard/stats")
async def get_dashboard_stats(user: dict = Depends(get_current_user)):
    """Get dashboard statistics"""
    return {
        "total_clients": 25,
        "active_training": 3,
        "completed_rounds": 150,
        "model_accuracy": 0.94,
        "system_status": "healthy",
        "last_updated": datetime.now().isoformat()
    }

@app.get("/api/clients")
async def get_clients(user: dict = Depends(get_current_user)):
    """Get federated learning clients"""
    clients = []
    for i in range(1, 26):
        clients.append({
            "id": f"client-{i:03d}",
            "name": f"Client {i}",
            "status": "online" if i % 3 != 0 else "offline",
            "last_seen": datetime.now() - timedelta(minutes=i),
            "data_samples": 1000 + (i * 50),
            "model_version": "v1.2.3",
            "location": f"Node-{i}"
        })
    return {"clients": clients}

@app.get("/api/training/sessions")
async def get_training_sessions(user: dict = Depends(get_current_user)):
    """Get training sessions"""
    return {
        "sessions": [
            {
                "id": "session-001",
                "name": "MNIST Classification",
                "status": "running",
                "progress": 75,
                "accuracy": 0.92,
                "participants": 15,
                "started_at": datetime.now() - timedelta(hours=2),
                "estimated_completion": datetime.now() + timedelta(minutes=30)
            },
            {
                "id": "session-002", 
                "name": "Image Recognition",
                "status": "completed",
                "progress": 100,
                "accuracy": 0.89,
                "participants": 20,
                "started_at": datetime.now() - timedelta(hours=8),
                "completed_at": datetime.now() - timedelta(hours=1)
            }
        ]
    }

@app.get("/api/models")
async def get_models(user: dict = Depends(get_current_user)):
    """Get available models"""
    return {
        "models": [
            {
                "id": "model-001",
                "name": "MNIST CNN",
                "version": "v1.2.3",
                "accuracy": 0.94,
                "size_mb": 2.5,
                "created_at": datetime.now() - timedelta(days=5),
                "status": "active"
            },
            {
                "id": "model-002",
                "name": "ResNet-18",
                "version": "v2.1.0", 
                "accuracy": 0.87,
                "size_mb": 45.2,
                "created_at": datetime.now() - timedelta(days=10),
                "status": "deprecated"
            }
        ]
    }

@app.get("/api/security/status")
async def get_security_status(user: dict = Depends(get_current_user)):
    """Get security status"""
    return {
        "quantum_safe": True,
        "encryption_status": "active",
        "key_rotation": "scheduled",
        "last_security_scan": datetime.now() - timedelta(hours=6),
        "threats_detected": 0,
        "compliance_score": 98
    }

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "uptime": "running"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "QFLARE Simple Backend API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )