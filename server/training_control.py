"""
🏋️ QFLARE Training Control System
Advanced federated learning orchestration and monitoring
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, Query
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from enum import Enum
import uuid
import asyncio
import json
import logging
import numpy as np
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["Training Control"])

class TrainingStatus(str, Enum):
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AggregationMethod(str, Enum):
    FEDAVG = "fedavg"  # Federated Averaging
    FEDPROX = "fedprox"  # FedProx
    SCAFFOLD = "scaffold"  # SCAFFOLD
    FEDNOVA = "fednova"  # FedNova
    CUSTOM = "custom"

class ModelArchitecture(str, Enum):
    CNN = "cnn"
    RESNET = "resnet"
    TRANSFORMER = "transformer"
    LSTM = "lstm"
    CUSTOM = "custom"

class TrainingConfiguration(BaseModel):
    """Configuration for a federated training session"""
    session_name: str = Field(..., min_length=3, max_length=100)
    model_architecture: ModelArchitecture = ModelArchitecture.CNN
    dataset_name: str = Field(..., description="Name of the dataset to use")
    aggregation_method: AggregationMethod = AggregationMethod.FEDAVG
    
    # Training parameters
    global_rounds: int = Field(default=10, ge=1, le=1000)
    local_epochs: int = Field(default=5, ge=1, le=50)
    batch_size: int = Field(default=32, ge=1, le=1024)
    learning_rate: float = Field(default=0.01, gt=0, le=1.0)
    
    # Federated learning parameters
    min_participants: int = Field(default=2, ge=1, le=100)
    max_participants: int = Field(default=10, ge=1, le=100)
    participation_rate: float = Field(default=1.0, gt=0, le=1.0)
    
    # Security and privacy
    differential_privacy: bool = Field(default=False)
    dp_epsilon: Optional[float] = Field(default=None, gt=0)
    secure_aggregation: bool = Field(default=True)
    
    # Advanced settings
    custom_parameters: Dict[str, Any] = Field(default_factory=dict)
    device_requirements: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('max_participants')
    def max_greater_than_min(cls, v, values):
        if 'min_participants' in values and v < values['min_participants']:
            raise ValueError('max_participants must be >= min_participants')
        return v

class TrainingSession(BaseModel):
    """Represents an active or completed training session"""
    session_id: str
    session_name: str
    status: TrainingStatus
    configuration: TrainingConfiguration
    
    # Timing
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    # Progress tracking
    current_round: int = 0
    total_rounds: int
    progress_percentage: float = 0.0
    
    # Participants
    registered_devices: Set[str] = Field(default_factory=set)
    active_devices: Set[str] = Field(default_factory=set)
    completed_devices: Set[str] = Field(default_factory=set)
    failed_devices: Set[str] = Field(default_factory=set)
    
    # Metrics
    global_accuracy: Optional[float] = None
    global_loss: Optional[float] = None
    round_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    device_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Model information
    model_size_bytes: Optional[int] = None
    model_checkpoints: List[str] = Field(default_factory=list)
    
    # Logs and errors
    training_logs: List[str] = Field(default_factory=list)
    error_messages: List[str] = Field(default_factory=list)

class DeviceTrainingUpdate(BaseModel):
    """Update from a device during training"""
    device_id: str
    round_number: int
    local_loss: float
    local_accuracy: Optional[float] = None
    samples_count: int
    training_time: float  # seconds
    model_weights: Optional[Dict[str, Any]] = None  # Serialized weights
    additional_metrics: Dict[str, float] = Field(default_factory=dict)

# In-memory storage (replace with database in production)
training_sessions: Dict[str, TrainingSession] = {}
session_websockets: Dict[str, Set[WebSocket]] = {}

@router.post("/sessions", response_model=Dict[str, str])
async def create_training_session(config: TrainingConfiguration):
    """
    🚀 Create a new federated training session
    """
    session_id = str(uuid.uuid4())
    
    # Check for duplicate session names
    existing_names = [s.session_name for s in training_sessions.values()]
    if config.session_name in existing_names:
        raise HTTPException(
            status_code=400,
            detail=f"Session name '{config.session_name}' already exists"
        )
    
    # Create training session
    session = TrainingSession(
        session_id=session_id,
        session_name=config.session_name,
        status=TrainingStatus.PENDING,
        configuration=config,
        created_at=datetime.now(),
        total_rounds=config.global_rounds
    )
    
    training_sessions[session_id] = session
    session_websockets[session_id] = set()
    
    logger.info(f"🎯 New training session created: {config.session_name} ({session_id})")
    
    return {
        "session_id": session_id,
        "message": f"Training session '{config.session_name}' created successfully!",
        "status": "created"
    }

@router.get("/sessions", response_model=List[TrainingSession])
async def list_training_sessions(
    status: Optional[TrainingStatus] = None,
    limit: int = Query(default=20, le=100)
):
    """
    📋 List all training sessions with optional filtering
    """
    sessions = list(training_sessions.values())
    
    if status:
        sessions = [s for s in sessions if s.status == status]
    
    # Sort by creation time (most recent first)
    sessions.sort(key=lambda x: x.created_at, reverse=True)
    
    return sessions[:limit]

@router.get("/sessions/{session_id}", response_model=TrainingSession)
async def get_training_session(session_id: str):
    """
    🔍 Get detailed information about a training session
    """
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    return training_sessions[session_id]

@router.put("/sessions/{session_id}/start")
async def start_training_session(session_id: str, background_tasks: BackgroundTasks):
    """
    ▶️ Start a training session
    """
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    session = training_sessions[session_id]
    
    if session.status != TrainingStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot start session with status: {session.status}"
        )
    
    # Check minimum participants
    if len(session.registered_devices) < session.configuration.min_participants:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {session.configuration.min_participants} devices, but only {len(session.registered_devices)} registered"
        )
    
    # Update session status
    session.status = TrainingStatus.INITIALIZING
    session.started_at = datetime.now()
    
    # Estimate completion time
    estimated_duration = timedelta(
        minutes=session.configuration.global_rounds * session.configuration.local_epochs * 2
    )
    session.estimated_completion = session.started_at + estimated_duration
    
    # Start training in background
    background_tasks.add_task(run_federated_training, session_id)
    
    logger.info(f"🚀 Training session started: {session.session_name} ({session_id})")
    
    return {
        "session_id": session_id,
        "status": "starting",
        "message": "Training session is starting..."
    }

@router.put("/sessions/{session_id}/pause")
async def pause_training_session(session_id: str):
    """
    ⏸️ Pause a running training session
    """
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    session = training_sessions[session_id]
    
    if session.status != TrainingStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot pause session with status: {session.status}"
        )
    
    session.status = TrainingStatus.PAUSED
    session.training_logs.append(f"Session paused at {datetime.now()}")
    
    await broadcast_session_update(session_id, {"type": "status_update", "status": "paused"})
    
    return {"message": "Training session paused"}

@router.put("/sessions/{session_id}/resume")
async def resume_training_session(session_id: str, background_tasks: BackgroundTasks):
    """
    ▶️ Resume a paused training session
    """
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    session = training_sessions[session_id]
    
    if session.status != TrainingStatus.PAUSED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume session with status: {session.status}"
        )
    
    session.status = TrainingStatus.RUNNING
    session.training_logs.append(f"Session resumed at {datetime.now()}")
    
    # Continue training in background
    background_tasks.add_task(continue_federated_training, session_id)
    
    await broadcast_session_update(session_id, {"type": "status_update", "status": "running"})
    
    return {"message": "Training session resumed"}

@router.delete("/sessions/{session_id}")
async def cancel_training_session(session_id: str):
    """
    🛑 Cancel and delete a training session
    """
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    session = training_sessions[session_id]
    session.status = TrainingStatus.CANCELLED
    session.completed_at = datetime.now()
    
    await broadcast_session_update(session_id, {"type": "status_update", "status": "cancelled"})
    
    # Clean up
    del training_sessions[session_id]
    if session_id in session_websockets:
        del session_websockets[session_id]
    
    logger.info(f"🛑 Training session cancelled: {session.session_name} ({session_id})")
    
    return {"message": "Training session cancelled and deleted"}

@router.post("/sessions/{session_id}/devices/{device_id}/register")
async def register_device_for_training(session_id: str, device_id: str):
    """
    📝 Register a device to participate in training
    """
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    session = training_sessions[session_id]
    
    if session.status not in [TrainingStatus.PENDING, TrainingStatus.RUNNING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot register for session with status: {session.status}"
        )
    
    if len(session.registered_devices) >= session.configuration.max_participants:
        raise HTTPException(
            status_code=400,
            detail="Training session is full"
        )
    
    session.registered_devices.add(device_id)
    session.training_logs.append(f"Device {device_id} registered at {datetime.now()}")
    
    await broadcast_session_update(session_id, {
        "type": "device_registered",
        "device_id": device_id,
        "total_registered": len(session.registered_devices)
    })
    
    return {
        "message": "Device registered for training",
        "registered_devices": len(session.registered_devices),
        "max_participants": session.configuration.max_participants
    }

@router.post("/sessions/{session_id}/devices/{device_id}/update")
async def submit_training_update(session_id: str, device_id: str, update: DeviceTrainingUpdate):
    """
    📊 Submit training update from a device
    """
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    session = training_sessions[session_id]
    
    if device_id not in session.registered_devices:
        raise HTTPException(status_code=400, detail="Device not registered for this session")
    
    # Store device metrics
    session.device_metrics[device_id] = {
        "round": update.round_number,
        "loss": update.local_loss,
        "accuracy": update.local_accuracy,
        "samples": update.samples_count,
        "training_time": update.training_time,
        "timestamp": datetime.now(),
        **update.additional_metrics
    }
    
    # Add to completed devices for this round
    session.completed_devices.add(device_id)
    
    # Log the update
    session.training_logs.append(
        f"Update from {device_id}: Round {update.round_number}, Loss: {update.local_loss:.4f}"
    )
    
    await broadcast_session_update(session_id, {
        "type": "device_update",
        "device_id": device_id,
        "round": update.round_number,
        "metrics": session.device_metrics[device_id]
    })
    
    return {"message": "Training update received", "status": "acknowledged"}

@router.websocket("/sessions/{session_id}/ws")
async def training_session_websocket(websocket: WebSocket, session_id: str):
    """
    🔌 WebSocket endpoint for real-time training session updates
    """
    await websocket.accept()
    
    if session_id not in session_websockets:
        session_websockets[session_id] = set()
    
    session_websockets[session_id].add(websocket)
    
    try:
        # Send initial session state
        if session_id in training_sessions:
            session = training_sessions[session_id]
            await websocket.send_json({
                "type": "session_state",
                "session": session.dict()
            })
        
        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle client messages (like device status updates)
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        if session_id in session_websockets:
            session_websockets[session_id].discard(websocket)

@router.get("/sessions/{session_id}/metrics")
async def get_session_metrics(session_id: str):
    """
    📈 Get comprehensive metrics for a training session
    """
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    session = training_sessions[session_id]
    
    # Calculate aggregated metrics
    device_count = len(session.registered_devices)
    active_count = len(session.active_devices)
    completed_count = len(session.completed_devices)
    failed_count = len(session.failed_devices)
    
    # Calculate average metrics
    avg_loss = 0.0
    avg_accuracy = 0.0
    total_samples = 0
    
    if session.device_metrics:
        losses = [m.get("loss", 0) for m in session.device_metrics.values()]
        accuracies = [m.get("accuracy", 0) for m in session.device_metrics.values() if m.get("accuracy")]
        samples = [m.get("samples", 0) for m in session.device_metrics.values()]
        
        avg_loss = sum(losses) / len(losses) if losses else 0
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        total_samples = sum(samples)
    
    # Calculate training efficiency
    elapsed_time = 0
    if session.started_at:
        elapsed_time = (datetime.now() - session.started_at).total_seconds() / 3600  # hours
    
    return {
        "session_id": session_id,
        "session_name": session.session_name,
        "status": session.status,
        "progress": {
            "current_round": session.current_round,
            "total_rounds": session.total_rounds,
            "percentage": session.progress_percentage
        },
        "participants": {
            "registered": device_count,
            "active": active_count,
            "completed": completed_count,
            "failed": failed_count
        },
        "performance": {
            "global_accuracy": session.global_accuracy,
            "global_loss": session.global_loss,
            "average_local_loss": avg_loss,
            "average_local_accuracy": avg_accuracy,
            "total_samples": total_samples
        },
        "timing": {
            "created_at": session.created_at,
            "started_at": session.started_at,
            "elapsed_hours": elapsed_time,
            "estimated_completion": session.estimated_completion
        },
        "model_info": {
            "architecture": session.configuration.model_architecture,
            "size_bytes": session.model_size_bytes,
            "checkpoints": len(session.model_checkpoints)
        }
    }

# Helper functions

async def broadcast_session_update(session_id: str, message: Dict[str, Any]):
    """Broadcast update to all WebSocket connections for a session"""
    if session_id not in session_websockets:
        return
    
    disconnected = set()
    for websocket in session_websockets[session_id]:
        try:
            await websocket.send_json(message)
        except:
            disconnected.add(websocket)
    
    # Clean up disconnected websockets
    for ws in disconnected:
        session_websockets[session_id].discard(ws)

async def run_federated_training(session_id: str):
    """Main federated training loop"""
    if session_id not in training_sessions:
        return
    
    session = training_sessions[session_id]
    
    try:
        session.status = TrainingStatus.RUNNING
        await broadcast_session_update(session_id, {"type": "status_update", "status": "running"})
        
        for round_num in range(1, session.configuration.global_rounds + 1):
            if session.status != TrainingStatus.RUNNING:
                break  # Training was paused or cancelled
            
            session.current_round = round_num
            session.progress_percentage = (round_num / session.configuration.global_rounds) * 100
            
            # Simulate training round
            session.training_logs.append(f"Starting round {round_num}/{session.configuration.global_rounds}")
            
            # Broadcast round start
            await broadcast_session_update(session_id, {
                "type": "round_start",
                "round": round_num,
                "progress": session.progress_percentage
            })
            
            # Simulate federated learning round
            await asyncio.sleep(10)  # Simulate training time
            
            # Update global metrics (simulate improvement)
            base_accuracy = 0.7 + (round_num / session.configuration.global_rounds) * 0.2
            base_loss = 0.5 - (round_num / session.configuration.global_rounds) * 0.3
            
            session.global_accuracy = base_accuracy + np.random.normal(0, 0.02)
            session.global_loss = max(0.1, base_loss + np.random.normal(0, 0.05))
            
            # Store round metrics
            round_metrics = {
                "round": round_num,
                "global_accuracy": session.global_accuracy,
                "global_loss": session.global_loss,
                "timestamp": datetime.now(),
                "participants": len(session.active_devices)
            }
            session.round_metrics.append(round_metrics)
            
            # Broadcast round completion
            await broadcast_session_update(session_id, {
                "type": "round_complete",
                "round": round_num,
                "metrics": round_metrics
            })
        
        # Training completed
        session.status = TrainingStatus.COMPLETED
        session.completed_at = datetime.now()
        session.progress_percentage = 100.0
        
        await broadcast_session_update(session_id, {
            "type": "training_complete",
            "final_accuracy": session.global_accuracy,
            "final_loss": session.global_loss
        })
        
        logger.info(f"✅ Training session completed: {session.session_name} ({session_id})")
    
    except Exception as e:
        session.status = TrainingStatus.FAILED
        session.completed_at = datetime.now()
        session.error_messages.append(f"Training failed: {str(e)}")
        
        await broadcast_session_update(session_id, {
            "type": "training_failed",
            "error": str(e)
        })
        
        logger.error(f"❌ Training session failed: {session.session_name} ({session_id}): {e}")

async def continue_federated_training(session_id: str):
    """Continue training after pause"""
    # This would continue from where training was paused
    await run_federated_training(session_id)