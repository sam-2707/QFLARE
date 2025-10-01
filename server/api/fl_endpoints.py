"""
QFLARE Federated Learning API Endpoints

This module implements the core FL endpoints for model submission, aggregation,
and global model distribution.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, Any, List, Optional
import logging
import time
import json
import pickle
import io
from datetime import datetime
import numpy as np
import torch

from ..fl_core.fl_controller import FLController
from ..fl_core.model_aggregator import FederatedAveraging
from ..fl_core.security import ModelValidator
from ..registry import get_registered_devices

logger = logging.getLogger(__name__)

# Initialize FL controller
fl_controller = FLController()
router = APIRouter()

# Global FL state
fl_state = {
    "current_round": 0,
    "total_rounds": 10,
    "status": "idle",  # idle, training, aggregating, completed
    "participants": {},
    "global_model": None,
    "round_start_time": None,
    "training_history": []
}

@router.get("/fl/status")
async def get_fl_status():
    """Get current federated learning status."""
    registered_devices = get_registered_devices()
    active_devices = [d for d in registered_devices.values() if d.get("status") == "enrolled"]
    
    return JSONResponse({
        "success": True,
        "fl_status": {
            "current_round": fl_state["current_round"],
            "total_rounds": fl_state["total_rounds"],
            "status": fl_state["status"],
            "registered_devices": len(registered_devices),
            "active_devices": len(active_devices),
            "participants_this_round": len(fl_state["participants"]),
            "round_start_time": fl_state["round_start_time"],
            "training_history": fl_state["training_history"][-5:]  # Last 5 rounds
        }
    })

@router.post("/fl/start_round")
async def start_training_round(
    target_participants: int = Form(default=3),
    local_epochs: int = Form(default=5),
    learning_rate: float = Form(default=0.01)
):
    """Start a new federated learning training round."""
    try:
        if fl_state["status"] != "idle":
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot start new round. Current status: {fl_state['status']}"
            )
        
        # Get active devices
        registered_devices = get_registered_devices()
        active_devices = [d for d in registered_devices.values() if d.get("status") == "enrolled"]
        
        if len(active_devices) < target_participants:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough active devices. Need {target_participants}, have {len(active_devices)}"
            )
        
        # Select participants (for now, select first N active devices)
        selected_devices = active_devices[:target_participants]
        
        # Initialize round
        fl_state["current_round"] += 1
        fl_state["status"] = "training"
        fl_state["round_start_time"] = datetime.now().isoformat()
        fl_state["participants"] = {
            device["device_id"]: {
                "device_id": device["device_id"],
                "status": "selected",
                "model_submitted": False,
                "submission_time": None,
                "model_size": 0,
                "training_loss": None
            }
            for device in selected_devices
        }
        
        logger.info(f"Started FL round {fl_state['current_round']} with {len(selected_devices)} participants")
        
        return JSONResponse({
            "success": True,
            "message": f"Training round {fl_state['current_round']} started",
            "round_info": {
                "round_number": fl_state["current_round"],
                "participants": list(fl_state["participants"].keys()),
                "target_participants": target_participants,
                "local_epochs": local_epochs,
                "learning_rate": learning_rate,
                "deadline": "30 minutes from now"  # TODO: Implement proper deadline
            }
        })
        
    except Exception as e:
        logger.error(f"Error starting FL round: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fl/submit_model")
async def submit_model_update(
    device_id: str = Form(...),
    model_file: UploadFile = File(...),
    training_loss: float = Form(...),
    local_epochs: int = Form(...),
    num_samples: int = Form(...)
):
    """Submit a local model update from an edge device."""
    try:
        # Validate device is registered and participating
        if device_id not in fl_state["participants"]:
            raise HTTPException(
                status_code=403,
                detail="Device not selected for current training round"
            )
        
        if fl_state["status"] != "training":
            raise HTTPException(
                status_code=400,
                detail=f"Not accepting submissions. Current status: {fl_state['status']}"
            )
        
        # Validate model file
        if not model_file.filename.endswith(('.pt', '.pth', '.pkl')):
            raise HTTPException(
                status_code=400,
                detail="Model file must be a PyTorch (.pt, .pth) or pickle (.pkl) file"
            )
        
        # Read model data
        model_data = await model_file.read()
        
        # Basic security validation
        validator = ModelValidator()
        if not validator.validate_model_update(model_data, device_id):
            raise HTTPException(
                status_code=400,
                detail="Model update failed security validation"
            )
        
        # Store model update
        participant = fl_state["participants"][device_id]
        participant.update({
            "status": "submitted",
            "model_submitted": True,
            "submission_time": datetime.now().isoformat(),
            "model_size": len(model_data),
            "training_loss": training_loss,
            "local_epochs": local_epochs,
            "num_samples": num_samples,
            "model_data": model_data
        })
        
        logger.info(f"Received model update from {device_id}: {len(model_data)} bytes, loss: {training_loss}")
        
        # Check if all participants have submitted
        submitted_count = sum(1 for p in fl_state["participants"].values() if p["model_submitted"])
        total_participants = len(fl_state["participants"])
        
        response_data = {
            "success": True,
            "message": "Model update received successfully",
            "submission_info": {
                "device_id": device_id,
                "model_size": len(model_data),
                "training_loss": training_loss,
                "submitted_count": submitted_count,
                "total_participants": total_participants,
                "round_complete": submitted_count == total_participants
            }
        }
        
        # Trigger aggregation if all models received
        if submitted_count == total_participants:
            await trigger_model_aggregation()
            response_data["message"] += ". All models received, starting aggregation."
        
        return JSONResponse(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting model update: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fl/global_model")
async def get_global_model(device_id: str):
    """Download the latest global model."""
    try:
        # Validate device is registered
        registered_devices = get_registered_devices()
        device_ids = [d["device_id"] for d in registered_devices.values()]
        
        if device_id not in device_ids:
            raise HTTPException(
                status_code=403,
                detail="Device not registered"
            )
        
        # Check if global model is available
        if fl_state["global_model"] is None:
            # Return initial model (random initialization)
            initial_model = create_initial_model()
            model_data = serialize_model(initial_model)
        else:
            model_data = fl_state["global_model"]
        
        # Create streaming response
        model_stream = io.BytesIO(model_data)
        
        return StreamingResponse(
            model_stream,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=global_model_round_{fl_state['current_round']}.pt",
                "X-Model-Round": str(fl_state["current_round"]),
                "X-Model-Size": str(len(model_data))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving global model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fl/training_history")
async def get_training_history():
    """Get federated learning training history."""
    return JSONResponse({
        "success": True,
        "training_history": fl_state["training_history"],
        "current_round": fl_state["current_round"],
        "total_rounds": fl_state["total_rounds"]
    })

@router.post("/fl/reset")
async def reset_fl_system():
    """Reset the federated learning system (admin only)."""
    try:
        fl_state.update({
            "current_round": 0,
            "status": "idle",
            "participants": {},
            "global_model": None,
            "round_start_time": None,
            "training_history": []
        })
        
        logger.info("FL system reset successfully")
        
        return JSONResponse({
            "success": True,
            "message": "Federated learning system reset successfully"
        })
        
    except Exception as e:
        logger.error(f"Error resetting FL system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions

async def trigger_model_aggregation():
    """Trigger model aggregation after all participants submit."""
    try:
        fl_state["status"] = "aggregating"
        
        # Collect all model updates
        model_updates = []
        weights = []
        
        for participant in fl_state["participants"].values():
            if participant["model_submitted"]:
                model_data = participant["model_data"]
                model_updates.append(model_data)
                weights.append(participant["num_samples"])  # Weight by number of samples
        
        # Perform federated averaging
        aggregator = FederatedAveraging()
        aggregated_model = aggregator.aggregate(model_updates, weights)
        
        # Store global model
        fl_state["global_model"] = aggregated_model
        fl_state["status"] = "idle"
        
        # Record training history
        round_history = {
            "round": fl_state["current_round"],
            "timestamp": datetime.now().isoformat(),
            "participants": len(fl_state["participants"]),
            "avg_loss": np.mean([p["training_loss"] for p in fl_state["participants"].values()]),
            "total_samples": sum([p["num_samples"] for p in fl_state["participants"].values()])
        }
        fl_state["training_history"].append(round_history)
        
        logger.info(f"Model aggregation completed for round {fl_state['current_round']}")
        
    except Exception as e:
        logger.error(f"Error in model aggregation: {e}")
        fl_state["status"] = "error"

def create_initial_model():
    """Create initial model for the first round."""
    # Simple CNN model for MNIST
    import torch.nn as nn
    
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 64 * 7 * 7)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    return SimpleCNN()

def serialize_model(model):
    """Serialize PyTorch model to bytes."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()