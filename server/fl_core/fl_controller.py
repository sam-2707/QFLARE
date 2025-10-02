"""
QFLARE Federated Learning Controller

This module orchestrates federated learning training rounds, manages client selection,
and coordinates model aggregation with real ML training.
"""

import logging
import time
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import json

from ..ml.training import FederatedTrainer, create_real_fl_trainer
from ..security.mock_enclave import mock_secure_compute

# Import WebSocket functions for real-time updates
try:
    from ..websocket.manager import (
        broadcast_fl_status_update,
        broadcast_training_progress,
        broadcast_model_aggregation,
        broadcast_device_status
    )
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

logger = logging.getLogger(__name__)

class FLController:
    """
    Orchestrates federated learning training rounds, manages client selection, and coordinates aggregation.
    """
    
    def __init__(self, 
                 min_participants: int = 2,
                 max_participants: int = 10,
                 round_timeout: int = 1800,  # 30 minutes
                 training_config: Optional[Dict[str, Any]] = None):
        """
        Initialize FL Controller.
        
        Args:
            min_participants: Minimum number of participants required
            max_participants: Maximum number of participants per round
            round_timeout: Timeout for training rounds in seconds
            training_config: Configuration for ML training
        """
        self.min_participants = min_participants
        self.max_participants = max_participants
        self.round_timeout = round_timeout
        
        # Training state
        self.current_round = 0
        self.is_training = False
        self.round_start_time = None
        self.selected_participants = {}
        
        # Initialize real ML trainer
        default_config = {
            "dataset": "mnist",
            "model": "mnist", 
            "data_dir": "../data",
            "device": "auto"
        }
        config = {**default_config, **(training_config or {})}
        self.ml_trainer = create_real_fl_trainer(config)
        
        logger.info(f"FL Controller initialized with {min_participants}-{max_participants} participants")
        logger.info(f"Using dataset: {config['dataset']}, model: {config['model']}")
    
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
    
    async def run_real_training_round(self, 
                                    available_devices: List[Dict],
                                    training_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run a complete federated learning round with real ML training.
        
        Args:
            available_devices: List of available devices
            training_config: Training configuration override
            
        Returns:
            Round results with training metrics
        """
        if self.is_training:
            raise ValueError("Training round already in progress")
        
        # Use default training config if not provided
        default_config = {
            "epochs": 5,
            "learning_rate": 0.01,
            "batch_size": 32,
            "participation_rate": 0.3
        }
        config = {**default_config, **(training_config or {})}
        
        # Select participants
        selected_devices = self.select_participants(available_devices)
        num_clients = len(selected_devices)
        
        if num_clients < self.min_participants:
            raise ValueError(f"Not enough participants: {num_clients} < {self.min_participants}")
        
        # Start round tracking
        round_info = self.start_training_round(selected_devices, config)
        
        # Broadcast training start
        if WEBSOCKET_AVAILABLE:
            await broadcast_training_progress(self.current_round, {
                "status": "started",
                "participants": num_clients,
                "config": config,
                "start_time": self.round_start_time.isoformat()
            })
        
        try:
            # Run real ML training round
            training_results = self.ml_trainer.run_federated_round(
                num_clients=num_clients,
                participation_rate=1.0,  # All selected clients participate
                epochs=config["epochs"],
                learning_rate=config["learning_rate"],
                batch_size=config["batch_size"]
            )
            
            # Broadcast training progress
            if WEBSOCKET_AVAILABLE:
                await broadcast_training_progress(self.current_round, {
                    "status": "aggregating",
                    "global_accuracy": training_results["global_metrics"]["test_accuracy"],
                    "global_loss": training_results["global_metrics"]["test_loss"],
                    "client_count": len(training_results["client_metrics"])
                })
            
            # Simulate model submissions for selected devices
            for i, device in enumerate(selected_devices):
                device_id = device["device_id"]
                
                # Get client metrics if available
                client_metrics = {}
                if i < len(training_results["client_metrics"]):
                    client_metrics = training_results["client_metrics"][i]
                
                # Create mock model data (in real implementation, this would come from edge nodes)
                model_data = self.ml_trainer.get_model_weights()
                
                # Submit model update
                self.submit_model_update(device_id, model_data, client_metrics)
            
            # End round and get summary
            round_summary = self.end_training_round()
            
            # Store training results in database
            await self._store_training_results(round_summary, training_results)
            
            # Combine results
            complete_results = {
                **round_summary,
                "ml_results": training_results,
                "global_accuracy": training_results["global_metrics"]["test_accuracy"],
                "global_loss": training_results["global_metrics"]["test_loss"],
                "aggregated_samples": training_results["aggregated_samples"],
                "training_config": config
            }
            
            # Broadcast completion
            if WEBSOCKET_AVAILABLE:
                await broadcast_training_progress(self.current_round, {
                    "status": "completed",
                    "global_accuracy": training_results["global_metrics"]["test_accuracy"],
                    "global_loss": training_results["global_metrics"]["test_loss"],
                    "duration": round_summary["duration_seconds"],
                    "participants": round_summary["total_participants"]
                })
                
                await broadcast_model_aggregation({
                    "round_number": self.current_round,
                    "aggregated_samples": training_results["aggregated_samples"],
                    "client_metrics": training_results["client_metrics"],
                    "global_metrics": training_results["global_metrics"]
                })
            
            logger.info(f"Real FL round {self.current_round} completed successfully")
            logger.info(f"Global accuracy: {training_results['global_metrics']['test_accuracy']:.2f}%")
            
            return complete_results
            
        except Exception as e:
            # End round on error
            if self.is_training:
                self.end_training_round()
            logger.error(f"Training round failed: {str(e)}")
            raise
    
    async def _store_training_results(self, round_summary: Dict[str, Any], training_results: Dict[str, Any]):
        """Store training results in database."""
        try:
            from pathlib import Path
            import sqlite3
            
            db_path = Path(__file__).parent.parent.parent / "data" / "qflare_core.db"
            db_path.parent.mkdir(exist_ok=True)
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Store round summary
            cursor.execute("""
                INSERT INTO training_rounds 
                (round_number, start_time, end_time, participants, submitted_models, 
                 global_accuracy, global_loss, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                round_summary["round_number"],
                round_summary["start_time"],
                round_summary["end_time"],
                round_summary["total_participants"],
                round_summary["submitted_models"],
                training_results["global_metrics"]["test_accuracy"],
                training_results["global_metrics"]["test_loss"],
                round_summary["duration_seconds"]
            ))
            
            # Store client metrics
            for client_metric in training_results.get("client_metrics", []):
                cursor.execute("""
                    INSERT INTO client_training_metrics
                    (round_number, client_id, accuracy, loss, training_time, samples)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    round_summary["round_number"],
                    client_metric.get("client_id", 0),
                    client_metric.get("accuracy", 0),
                    client_metric.get("loss", 0),
                    client_metric.get("training_time", 0),
                    client_metric.get("samples", 0)
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store training results: {str(e)}")
            # Don't raise - training was successful even if storage failed
    
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
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive FL system status."""
        return {
            "available": True,
            "current_round": self.current_round,
            "total_rounds": getattr(self, 'total_rounds', 10),
            "status": "training" if self.is_training else "idle",
            "registered_devices": len(getattr(self, 'registered_devices', {})),
            "active_devices": len([d for d in getattr(self, 'registered_devices', {}).values() if d.get('status') == 'active']),
            "participants_this_round": len(self.selected_participants) if self.is_training else 0,
            "round_start_time": self.round_start_time.isoformat() if self.round_start_time else None,
            "training_history": getattr(self, 'training_history', [])
        }
    
    def register_device(self, device_id: str, capabilities: Dict[str, Any]) -> bool:
        """Register a device for federated learning."""
        if not hasattr(self, 'registered_devices'):
            self.registered_devices = {}
        
        self.registered_devices[device_id] = {
            "device_id": device_id,
            "capabilities": capabilities,
            "status": "active",
            "registered_at": datetime.now().isoformat(),
            "submissions": 0
        }
        logger.info(f"Registered device {device_id} for FL")
        return True
    
    def submit_model(self, data: Dict[str, Any]) -> bool:
        """Submit a model update from a device."""
        device_id = data.get("device_id")
        round_number = data.get("round_number")
        model_weights = data.get("model_weights")
        metrics = data.get("metrics", {})
        samples = data.get("samples", 0)
        
        if not self.is_training:
            logger.warning(f"Model submission from {device_id} rejected - no active training")
            return False
        
        if device_id not in self.selected_participants:
            logger.warning(f"Model submission from {device_id} rejected - not a participant")
            return False
        
        # Record submission
        self.selected_participants[device_id].update({
            "model_submitted": True,
            "submission_time": datetime.now().isoformat(),
            "model_weights": model_weights,
            "metrics": metrics,
            "samples": samples
        })
        
        # Update device stats
        if hasattr(self, 'registered_devices') and device_id in self.registered_devices:
            self.registered_devices[device_id]["submissions"] += 1
        
        logger.info(f"Model submitted from {device_id} for round {round_number}")
        return True
    
    def get_global_model(self) -> Optional[Dict[str, Any]]:
        """Get the current global model."""
        if not hasattr(self, 'global_model') or not self.global_model:
            return None
        
        return self.global_model
    
    def start_training(self, rounds: int = 10, min_participants: int = 2) -> bool:
        """Start FL training."""
        if self.is_training:
            logger.warning("Training already in progress")
            return False
        
        if not hasattr(self, 'registered_devices'):
            self.registered_devices = {}
        
        active_devices = [d for d in self.registered_devices.values() if d.get('status') == 'active']
        
        if len(active_devices) < min_participants:
            logger.warning(f"Not enough devices to start training: {len(active_devices)} < {min_participants}")
            return False
        
        # Start training
        self.is_training = True
        self.current_round = 1
        self.round_start_time = datetime.now()
        self.total_rounds = rounds
        
        # Select participants (all active devices for now)
        self.selected_participants = {
            d['device_id']: {
                "device_info": d,
                "status": "selected",
                "model_submitted": False
            }
            for d in active_devices
        }
        
        logger.info(f"Started FL training for {rounds} rounds with {len(active_devices)} devices")
        return True
    
    def stop_training(self):
        """Stop FL training."""
        if self.is_training:
            logger.info(f"Stopping FL training at round {self.current_round}")
            self.is_training = False
            self.round_start_time = None
            self.selected_participants = {}
    
    def list_devices(self) -> List[Dict[str, Any]]:
        """List all registered devices."""
        if not hasattr(self, 'registered_devices'):
            return []
        
        return list(self.registered_devices.values())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        if not hasattr(self, 'training_history'):
            self.training_history = []
        
        return {
            "total_rounds": self.current_round,
            "active_training": self.is_training,
            "history": self.training_history,
            "current_participants": len(self.selected_participants) if self.is_training else 0
        }
