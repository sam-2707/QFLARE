"""
Federated Learning Aggregator

This module handles model aggregation by delegating to the secure enclave.
"""

import logging
import time
import base64
from typing import List, Dict, Any, Optional
from datetime import datetime

from server.enclave.mock_enclave import get_secure_enclave, ModelUpdate

logger = logging.getLogger(__name__)

# In-memory storage for model updates (use database in production)
pending_updates = {}
global_model_weights = None
aggregation_round = 0


def store_model_update(device_id: str, model_weights: bytes, metadata: Dict[str, Any] = None) -> bool:
    """
    Store a model update from a device for later aggregation.
    
    Args:
        device_id: Device identifier
        model_weights: Model weights as bytes
        metadata: Additional metadata about the update
        
    Returns:
        True if update was stored successfully, False otherwise
    """
    try:
        if metadata is None:
            metadata = {}
        
        # Create model update object
        model_update = ModelUpdate(
            device_id=device_id,
            model_weights=model_weights,
            signature=b"",  # Signature is verified before calling this function
            timestamp=time.time(),
            metadata=metadata
        )
        
        # Store update
        pending_updates[device_id] = model_update
        
        logger.info(f"Stored model update from device {device_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error storing model update from device {device_id}: {e}")
        return False


def get_global_model() -> Optional[bytes]:
    """
    Get the current global model weights.
    
    Returns:
        Global model weights as bytes, or None if no model available
    """
    return global_model_weights


def aggregate_models() -> Optional[bytes]:
    
    global global_model_weights, aggregation_round
    """
    Aggregate all pending model updates using the secure enclave.
    
    Returns:
        Aggregated model weights as bytes, or None if aggregation failed
    """
    try:
        if not pending_updates:
            logger.warning("No pending model updates to aggregate")
            return None
        
        # Get secure enclave
        enclave = get_secure_enclave()
        
        # Convert pending updates to list
        model_updates = list(pending_updates.values())
        
        # Perform secure aggregation
        aggregated_weights, metadata = enclave.aggregate_models(
            model_updates, 
            global_model_weights
        )
        
        # Update global model
       
        global_model_weights = aggregated_weights
        aggregation_round += 1
        
        # Clear pending updates
        pending_updates.clear()
        
        logger.info(f"Successfully aggregated {len(model_updates)} model updates")
        logger.info(f"Aggregation metadata: {metadata}")
        
        return aggregated_weights
        
    except Exception as e:
        logger.error(f"Error during model aggregation: {e}")
        return None


def get_aggregation_status() -> Dict[str, Any]:
    """
    Get the current status of model aggregation.
    
    Returns:
        Dictionary with aggregation status information
    """
    try:
        enclave = get_secure_enclave()
        enclave_status = enclave.get_enclave_status()
        
        return {
            "pending_updates": len(pending_updates),
            "aggregation_round": aggregation_round,
            "global_model_available": global_model_weights is not None,
            "enclave_status": enclave_status,
            "last_aggregation": time.time() if aggregation_round > 0 else None
        }
        
    except Exception as e:
        logger.error(f"Error getting aggregation status: {e}")
        return {
            "error": str(e),
            "pending_updates": 0,
            "aggregation_round": 0,
            "global_model_available": False
        }


def get_pending_updates() -> List[Dict[str, Any]]:
    """
    Get list of pending model updates.
    
    Returns:
        List of pending update information
    """
    updates = []
    for device_id, update in pending_updates.items():
        updates.append({
            "device_id": device_id,
            "timestamp": update.timestamp,
            "metadata": update.metadata
        })
    return updates


def clear_pending_updates() -> bool:
    """
    Clear all pending model updates.
    
    Returns:
        True if updates were cleared, False otherwise
    """
    try:
        pending_updates.clear()
        logger.info("Cleared all pending model updates")
        return True
    except Exception as e:
        logger.error(f"Error clearing pending updates: {e}")
        return False


def set_global_model(model_weights: bytes) -> bool:
    """
    Set the global model weights (for initialization or manual updates).
    
    Args:
        model_weights: Model weights as bytes
        
    Returns:
        True if model was set successfully, False otherwise
    """
    try:
        global global_model_weights
        global_model_weights = model_weights
        logger.info("Global model weights updated")
        return True
    except Exception as e:
        logger.error(f"Error setting global model: {e}")
        return False


# Legacy functions for backward compatibility
def fed_avg(client_weights: List[bytes]) -> bytes:
    """
    Legacy federated averaging function (deprecated).
    
    This function is kept for backward compatibility but should not be used
    in new code. Use the secure enclave aggregation instead.
    """
    logger.warning("Legacy federated averaging called - use secure enclave aggregation")
    
    if not client_weights:
        raise ValueError("No client weights provided")
    
    # Simple averaging for backward compatibility
    import numpy as np
    weight_arrays = [np.frombuffer(weights, dtype=np.float32) for weights in client_weights]
    min_length = min(len(arr) for arr in weight_arrays)
    normalized_arrays = [arr[:min_length] for arr in weight_arrays]
    averaged = np.mean(normalized_arrays, axis=0)
    return averaged.tobytes()


def weighted_avg(client_weights: List[bytes], weights: List[float]) -> bytes:
    """
    Legacy weighted averaging function (deprecated).
    
    This function is kept for backward compatibility but should not be used
    in new code. Use the secure enclave aggregation instead.
    """
    logger.warning("Legacy weighted averaging called - use secure enclave aggregation")
    
    if not client_weights or len(client_weights) != len(weights):
        raise ValueError("Invalid weights provided")
    
    # Simple weighted averaging for backward compatibility
    import numpy as np
    weight_arrays = [np.frombuffer(weights, dtype=np.float32) for weights in client_weights]
    min_length = min(len(arr) for arr in weight_arrays)
    normalized_arrays = [arr[:min_length] for arr in weight_arrays]
    
    # Apply weights
    weighted_arrays = [arr * w for arr, w in zip(normalized_arrays, weights)]
    averaged = np.mean(weighted_arrays, axis=0)
    return averaged.tobytes()


def save_aggregated_model(model_bytes: bytes, path: str = "models/global_model.pkl") -> bool:
    """
    Save aggregated model to file (legacy function).
    
    Args:
        model_bytes: Model weights as bytes
        path: File path to save model
        
    Returns:
        True if model was saved successfully, False otherwise
    """
    try:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            f.write(model_bytes)
        
        logger.info(f"Saved aggregated model to {path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving aggregated model: {e}")
        return False


def aggregate_model(device_id: str, model_update: Dict[str, Any]) -> bool:
    """
    Legacy model aggregation function (deprecated).
    
    This function is kept for backward compatibility but should not be used
    in new code. Use store_model_update() and aggregate_models() instead.
    """
    logger.warning("Legacy model aggregation called - use secure aggregation instead")
    
    try:
        # Extract model weights from update
        model_weights = model_update.get("weights", b"")
        if isinstance(model_weights, str):
            model_weights = base64.b64decode(model_weights)
        
        # Store for later aggregation
        return store_model_update(device_id, model_weights, model_update)
        
    except Exception as e:
        logger.error(f"Error in legacy model aggregation: {e}")
        return False
