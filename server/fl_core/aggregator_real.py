"""
Federated Learning Aggregator

This module handles real model aggregation using PyTorch and database storage.
Replaces mock aggregation with actual federated averaging algorithms.
"""

import logging
import time
import sqlite3
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Try to import torch, use mock if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch not available, using mock aggregation")

from ..security.mock_enclave import mock_secure_compute
from ..ml.models import serialize_model_weights, deserialize_model_weights, create_model

logger = logging.getLogger(__name__)


def get_database_connection():
    """Get database connection for aggregator operations."""
    db_path = Path(__file__).parent.parent.parent / "data" / "qflare_core.db"
    db_path.parent.mkdir(exist_ok=True)
    return sqlite3.connect(str(db_path))


class RealModelAggregator:
    """
    Real model aggregator using PyTorch federated averaging.
    Implements FedAvg algorithm with weighted averaging based on client data sizes.
    """
    
    def __init__(self, model_type: str = "mnist"):
        self.model_type = model_type
        self.aggregation_history = []
        if TORCH_AVAILABLE:
            self.global_model = create_model(model_type)
        
    def store_model_update(self, device_id: str, model_weights: bytes, 
                          training_metrics: Dict[str, Any] = None) -> bool:
        """
        Store a model update from a federated client.
        
        Args:
            device_id: Device identifier
            model_weights: Serialized PyTorch model weights
            training_metrics: Training metrics from client
            
        Returns:
            True if update was stored successfully
        """
        try:
            if training_metrics is None:
                training_metrics = {}
            
            conn = get_database_connection()
            cursor = conn.cursor()
            
            # Ensure table exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    model_weights BLOB NOT NULL,
                    training_metrics TEXT,
                    timestamp TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    samples INTEGER DEFAULT 0
                )
            """)
            
            # Store model update
            samples = training_metrics.get('samples', 0)
            cursor.execute("""
                INSERT INTO model_updates 
                (device_id, model_weights, training_metrics, timestamp, status, samples)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                device_id,
                model_weights,
                json.dumps(training_metrics),
                datetime.now().isoformat(),
                "pending",
                samples
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Model update stored for device {device_id} ({samples} samples)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store model update for device {device_id}: {str(e)}")
            return False
    
    def get_pending_updates(self) -> List[Dict[str, Any]]:
        """Get all pending model updates from database."""
        try:
            conn = get_database_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, device_id, model_weights, training_metrics, timestamp, samples
                FROM model_updates 
                WHERE status = 'pending'
                ORDER BY timestamp ASC
            """)
            
            updates = []
            for row in cursor.fetchall():
                updates.append({
                    'id': row[0],
                    'device_id': row[1],
                    'model_weights': row[2],
                    'training_metrics': json.loads(row[3]) if row[3] else {},
                    'timestamp': row[4],
                    'samples': row[5]
                })
            
            conn.close()
            return updates
            
        except Exception as e:
            logger.error(f"Failed to get pending updates: {str(e)}")
            return []
    
    def federated_averaging(self, model_updates: List[Dict[str, Any]]) -> Tuple[bytes, Dict[str, Any]]:
        """
        Perform FedAvg aggregation with weighted averaging.
        
        Args:
            model_updates: List of model updates with weights and metadata
            
        Returns:
            Tuple of (aggregated_weights, aggregation_metrics)
        """
        if not model_updates:
            raise ValueError("No model updates provided for aggregation")
        
        logger.info(f"Starting FedAvg aggregation of {len(model_updates)} models")
        
        if not TORCH_AVAILABLE:
            # Mock aggregation without PyTorch
            logger.warning("Using mock aggregation - PyTorch not available")
            return model_updates[0]['model_weights'], {
                'num_clients': len(model_updates),
                'total_samples': sum(u.get('samples', 1) for u in model_updates),
                'algorithm': 'mock_fedavg'
            }
        
        # Extract client weights and sample counts
        client_weights = []
        client_samples = []
        
        for update in model_updates:
            # Deserialize model weights
            temp_model = create_model(self.model_type)
            temp_model = deserialize_model_weights(temp_model, update['model_weights'])
            client_weights.append(temp_model.state_dict())
            client_samples.append(max(1, update.get('samples', 1)))  # Ensure non-zero
        
        # Calculate weighted average
        total_samples = sum(client_samples)
        aggregated_state = {}
        
        # Initialize aggregated state dict with zeros
        for name, param in client_weights[0].items():
            aggregated_state[name] = torch.zeros_like(param)
        
        # Weighted averaging
        for i, (state_dict, num_samples) in enumerate(zip(client_weights, client_samples)):
            weight = num_samples / total_samples
            
            for name, param in state_dict.items():
                aggregated_state[name] += weight * param
        
        # Create aggregated model and serialize
        aggregated_model = create_model(self.model_type)
        aggregated_model.load_state_dict(aggregated_state)
        aggregated_weights = serialize_model_weights(aggregated_model)
        
        # Calculate aggregation metrics
        metrics = {
            'num_clients': len(model_updates),
            'total_samples': total_samples,
            'client_samples': client_samples,
            'aggregation_time': time.time(),
            'algorithm': 'fedavg'
        }
        
        logger.info(f"FedAvg completed: {len(model_updates)} clients, {total_samples} total samples")
        return aggregated_weights, metrics
    
    def aggregate_pending_models(self, min_updates: int = 2) -> Optional[Dict[str, Any]]:
        """
        Aggregate all pending model updates using FedAvg.
        
        Args:
            min_updates: Minimum number of updates required
            
        Returns:
            Aggregation results or None if insufficient updates
        """
        try:
            # Get pending updates
            pending_updates = self.get_pending_updates()
            
            if len(pending_updates) < min_updates:
                logger.info(f"Not enough updates for aggregation: {len(pending_updates)}/{min_updates}")
                return None
            
            # Perform aggregation
            start_time = time.time()
            aggregated_weights, metrics = self.federated_averaging(pending_updates)
            aggregation_duration = time.time() - start_time
            
            # Store aggregated model
            self._store_global_model(aggregated_weights, metrics)
            
            # Mark updates as processed
            self._mark_updates_processed([u['id'] for u in pending_updates])
            
            # Record in history
            aggregation_record = {
                'timestamp': datetime.now().isoformat(),
                'num_clients': len(pending_updates),
                'total_samples': metrics['total_samples'],
                'duration_seconds': aggregation_duration,
                'client_devices': [u['device_id'] for u in pending_updates]
            }
            self.aggregation_history.append(aggregation_record)
            
            logger.info(f"Model aggregation completed in {aggregation_duration:.2f}s")
            
            return {
                'aggregated_weights': aggregated_weights,
                'aggregation_metrics': metrics,
                'aggregation_record': aggregation_record,
                'num_updates_processed': len(pending_updates)
            }
            
        except Exception as e:
            logger.error(f"Model aggregation failed: {str(e)}")
            return None
    
    def _store_global_model(self, weights: bytes, metrics: Dict[str, Any]):
        """Store the aggregated global model."""
        try:
            conn = get_database_connection()
            cursor = conn.cursor()
            
            # Ensure table exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS global_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_weights BLOB NOT NULL,
                    metrics TEXT,
                    timestamp TEXT NOT NULL,
                    round_number INTEGER
                )
            """)
            
            # Store global model
            cursor.execute("""
                INSERT INTO global_models 
                (model_weights, metrics, timestamp, round_number)
                VALUES (?, ?, ?, ?)
            """, (
                weights,
                json.dumps(metrics),
                datetime.now().isoformat(),
                len(self.aggregation_history) + 1
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store global model: {str(e)}")
    
    def _mark_updates_processed(self, update_ids: List[int]):
        """Mark model updates as processed."""
        try:
            conn = get_database_connection()
            cursor = conn.cursor()
            
            for update_id in update_ids:
                cursor.execute("""
                    UPDATE model_updates 
                    SET status = 'processed', processed_at = ?
                    WHERE id = ?
                """, (datetime.now().isoformat(), update_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to mark updates as processed: {str(e)}")
    
    def get_latest_global_model(self) -> Optional[bytes]:
        """Get the latest global model weights."""
        try:
            conn = get_database_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT model_weights FROM global_models 
                ORDER BY timestamp DESC LIMIT 1
            """)
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Failed to get latest global model: {str(e)}")
            return None
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        return {
            'total_aggregations': len(self.aggregation_history),
            'aggregation_history': self.aggregation_history[-10:],  # Last 10
            'model_type': self.model_type,
            'torch_available': TORCH_AVAILABLE
        }


# Global aggregator instance
_global_aggregator = RealModelAggregator()


def store_model_update(device_id: str, model_weights: bytes, 
                      training_metrics: Dict[str, Any] = None) -> bool:
    """Store a model update using the global aggregator."""
    return _global_aggregator.store_model_update(device_id, model_weights, training_metrics)


def aggregate_models(min_updates: int = 2) -> Optional[Dict[str, Any]]:
    """Aggregate pending models using the global aggregator."""
    return _global_aggregator.aggregate_pending_models(min_updates)


def get_latest_global_model() -> Optional[bytes]:
    """Get the latest global model."""
    return _global_aggregator.get_latest_global_model()


def get_aggregation_stats() -> Dict[str, Any]:
    """Get aggregation statistics."""
    return _global_aggregator.get_aggregation_stats()


if __name__ == "__main__":
    # Test the real aggregator
    print("Testing Real Model Aggregator...")
    
    aggregator = RealModelAggregator()
    
    if TORCH_AVAILABLE:
        # Test model creation and serialization
        test_model = create_model("mnist")
        test_weights = serialize_model_weights(test_model)
        
        # Test storing updates
        success = aggregator.store_model_update("test_device_1", test_weights, {"samples": 100})
        print(f"Store update test: {'PASS' if success else 'FAIL'}")
        
        # Test getting pending updates
        pending = aggregator.get_pending_updates()
        print(f"Pending updates test: {'PASS' if len(pending) > 0 else 'FAIL'}")
    else:
        print("PyTorch not available - skipping ML tests")
    
    print("Real aggregator tests completed!")