"""
QFLARE Federated Learning Controller

This module orchestrates federated learning training rounds, manages client selection,
and coordinates model aggregation.
"""

import logging
import time
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)

class FLController:
    """
    Orchestrates federated learning training rounds, manages client selection, and coordinates aggregation.
    """
    
    def __init__(self, 
                 min_participants: int = 2,
                 max_participants: int = 10,
                 round_timeout: int = 1800):  # 30 minutes
        """
        Initialize FL Controller.
        
        Args:
            min_participants: Minimum number of participants required
            max_participants: Maximum number of participants per round
            round_timeout: Timeout for training rounds in seconds
        """
        self.min_participants = min_participants
        self.max_participants = max_participants
        self.round_timeout = round_timeout
        
        # Training state
        self.current_round = 0
        self.is_training = False
        self.round_start_time = None
        self.selected_participants = {}
        
        logger.info(f"FL Controller initialized with {min_participants}-{max_participants} participants")
    
    def can_start_round(self, available_devices: List[Dict]) -> bool:
        """Check if we can start a new training round."""
        if self.is_training:
            return False
        
        active_devices = [d for d in available_devices if d.get("status") == "enrolled"]
        return len(active_devices) >= self.min_participants
    
    def select_participants(self, available_devices: List[Dict], target_count: Optional[int] = None) -> List[Dict]:
        """
        Select participants for the next training round.
        
        Args:
            available_devices: List of available devices
            target_count: Target number of participants (optional)
            
        Returns:
            List of selected devices
        """
        # Filter to active devices only
        active_devices = [d for d in available_devices if d.get("status") == "enrolled"]
        
        if len(active_devices) < self.min_participants:
            raise ValueError(f"Not enough active devices: {len(active_devices)} < {self.min_participants}")
        
        # Determine selection count
        if target_count is None:
            target_count = min(len(active_devices), self.max_participants)
        else:
            target_count = min(target_count, len(active_devices), self.max_participants)
        
        # Selection strategy: random selection for now
        # TODO: Implement more sophisticated selection (availability, performance, etc.)
        selected = random.sample(active_devices, target_count)
        
        logger.info(f"Selected {len(selected)} participants from {len(active_devices)} active devices")
        return selected
    
    def start_training_round(self, 
                           selected_devices: List[Dict],
                           training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start a new federated learning round.
        
        Args:
            selected_devices: List of selected participant devices
            training_config: Training configuration
            
        Returns:
            Round information dictionary
        """
        if self.is_training:
            raise ValueError("Training round already in progress")
        
        # Initialize round
        self.current_round += 1
        self.is_training = True
        self.round_start_time = datetime.now()
        
        # Initialize participant tracking
        self.selected_participants = {
            device["device_id"]: {
                "device_info": device,
                "status": "selected",
                "model_submitted": False,
                "submission_time": None,
                "training_metrics": {}
            }
            for device in selected_devices
        }
        
        round_info = {
            "round_number": self.current_round,
            "start_time": self.round_start_time.isoformat(),
            "deadline": (self.round_start_time + timedelta(seconds=self.round_timeout)).isoformat(),
            "participants": list(self.selected_participants.keys()),
            "training_config": training_config,
            "status": "active"
        }
        
        logger.info(f"Started FL round {self.current_round} with {len(selected_devices)} participants")
        return round_info
    
    def submit_model_update(self, 
                          device_id: str, 
                          model_data: bytes,
                          training_metrics: Dict[str, Any]) -> bool:
        """
        Record model submission from a participant.
        
        Args:
            device_id: ID of submitting device
            model_data: Serialized model data
            training_metrics: Training metrics from local training
            
        Returns:
            True if submission successful, False otherwise
        """
        if not self.is_training:
            logger.warning(f"Model submission from {device_id} rejected - no active round")
            return False
        
        if device_id not in self.selected_participants:
            logger.warning(f"Model submission from {device_id} rejected - not selected participant")
            return False
        
        participant = self.selected_participants[device_id]
        
        if participant["model_submitted"]:
            logger.warning(f"Model submission from {device_id} rejected - already submitted")
            return False
        
        # Record submission
        participant.update({
            "status": "submitted",
            "model_submitted": True,
            "submission_time": datetime.now().isoformat(),
            "model_data": model_data,
            "training_metrics": training_metrics
        })
        
        logger.info(f"Model update received from {device_id}")
        return True
    
    def check_round_completion(self) -> bool:
        """Check if the current training round is complete."""
        if not self.is_training:
            return False
        
        submitted_count = sum(1 for p in self.selected_participants.values() if p["model_submitted"])
        total_participants = len(self.selected_participants)
        
        # Round complete if all participants submitted or timeout reached
        all_submitted = submitted_count == total_participants
        timeout_reached = datetime.now() > (self.round_start_time + timedelta(seconds=self.round_timeout))
        
        if all_submitted:
            logger.info(f"Round {self.current_round} complete - all {total_participants} participants submitted")
        elif timeout_reached:
            logger.info(f"Round {self.current_round} timeout - {submitted_count}/{total_participants} submitted")
        
        return all_submitted or timeout_reached
    
    def get_submitted_models(self) -> List[Dict[str, Any]]:
        """Get all submitted model updates for aggregation."""
        submitted_models = []
        
        for device_id, participant in self.selected_participants.items():
            if participant["model_submitted"]:
                submitted_models.append({
                    "device_id": device_id,
                    "model_data": participant["model_data"],
                    "training_metrics": participant["training_metrics"],
                    "submission_time": participant["submission_time"]
                })
        
        return submitted_models
    
    def end_training_round(self) -> Dict[str, Any]:
        """End the current training round and return summary."""
        if not self.is_training:
            raise ValueError("No active training round to end")
        
        submitted_count = sum(1 for p in self.selected_participants.values() if p["model_submitted"])
        total_participants = len(self.selected_participants)
        
        round_summary = {
            "round_number": self.current_round,
            "start_time": self.round_start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_participants": total_participants,
            "submitted_models": submitted_count,
            "participation_rate": submitted_count / total_participants if total_participants > 0 else 0,
            "duration_seconds": (datetime.now() - self.round_start_time).total_seconds()
        }
        
        # Reset state
        self.is_training = False
        self.round_start_time = None
        self.selected_participants = {}
        
        logger.info(f"Ended FL round {self.current_round} - {submitted_count}/{total_participants} participated")
        return round_summary
    
    def get_round_status(self) -> Dict[str, Any]:
        """Get current round status information."""
        if not self.is_training:
            return {
                "active": False,
                "round_number": self.current_round,
                "status": "idle"
            }
        
        submitted_count = sum(1 for p in self.selected_participants.values() if p["model_submitted"])
        total_participants = len(self.selected_participants)
        
        time_elapsed = (datetime.now() - self.round_start_time).total_seconds()
        time_remaining = max(0, self.round_timeout - time_elapsed)
        
        return {
            "active": True,
            "round_number": self.current_round,
            "status": "training",
            "start_time": self.round_start_time.isoformat(),
            "participants": total_participants,
            "submitted": submitted_count,
            "participation_rate": submitted_count / total_participants if total_participants > 0 else 0,
            "time_elapsed": time_elapsed,
            "time_remaining": time_remaining,
            "deadline": (self.round_start_time + timedelta(seconds=self.round_timeout)).isoformat()
        } 