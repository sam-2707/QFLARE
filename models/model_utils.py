"""
Model utility functions for federated learning.
Handles model serialization, deserialization, and aggregation.
"""

import torch
import torch.nn as nn
import io
import pickle
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import base64
import hashlib

logger = logging.getLogger(__name__)

class ModelSerializer:
    """
    Handles serialization and deserialization of PyTorch models for federated learning.
    """
    
    @staticmethod
    def serialize_model(model: nn.Module) -> bytes:
        """
        Serialize a PyTorch model to bytes.
        
        Args:
            model: PyTorch model to serialize
            
        Returns:
            Serialized model as bytes
        """
        try:
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error serializing model: {e}")
            raise
    
    @staticmethod
    def deserialize_model(model_bytes: bytes, model_class: nn.Module) -> nn.Module:
        """
        Deserialize bytes to a PyTorch model.
        
        Args:
            model_bytes: Serialized model bytes
            model_class: Model class/instance to load weights into
            
        Returns:
            PyTorch model with loaded weights
        """
        try:
            buffer = io.BytesIO(model_bytes)
            state_dict = torch.load(buffer, map_location='cpu')
            model_class.load_state_dict(state_dict)
            return model_class
        except Exception as e:
            logger.error(f"Error deserializing model: {e}")
            raise
    
    @staticmethod
    def serialize_weights(model: nn.Module) -> bytes:
        """
        Serialize only model weights (state_dict) to bytes.
        
        Args:
            model: PyTorch model
            
        Returns:
            Serialized weights as bytes
        """
        try:
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error serializing weights: {e}")
            raise
    
    @staticmethod
    def deserialize_weights(weights_bytes: bytes, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        Deserialize bytes to model weights (state_dict).
        
        Args:
            weights_bytes: Serialized weights bytes
            device: Device to load weights on
            
        Returns:
            Model state_dict
        """
        try:
            buffer = io.BytesIO(weights_bytes)
            state_dict = torch.load(buffer, map_location=device)
            return state_dict
        except Exception as e:
            logger.error(f"Error deserializing weights: {e}")
            raise
    
    @staticmethod
    def get_model_size(model: nn.Module) -> int:
        """
        Get the size of a model in bytes.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model size in bytes
        """
        try:
            serialized = ModelSerializer.serialize_model(model)
            return len(serialized)
        except Exception as e:
            logger.error(f"Error getting model size: {e}")
            return 0
    
    @staticmethod
    def compute_model_hash(model: nn.Module) -> str:
        """
        Compute SHA-256 hash of model weights.
        
        Args:
            model: PyTorch model
            
        Returns:
            Hex string of model hash
        """
        try:
            serialized = ModelSerializer.serialize_weights(model)
            return hashlib.sha256(serialized).hexdigest()
        except Exception as e:
            logger.error(f"Error computing model hash: {e}")
            return ""

class FederatedAggregator:
    """
    Handles aggregation of model updates in federated learning.
    """
    
    @staticmethod
    def federated_averaging(model_updates: List[Dict[str, torch.Tensor]], 
                          weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """
        Perform FedAvg (Federated Averaging) on model updates.
        
        Args:
            model_updates: List of model state_dicts
            weights: Optional weights for each model update (default: equal weights)
            
        Returns:
            Aggregated model state_dict
        """
        try:
            if not model_updates:
                raise ValueError("No model updates provided")
            
            # Default to equal weights
            if weights is None:
                weights = [1.0 / len(model_updates)] * len(model_updates)
            
            if len(weights) != len(model_updates):
                raise ValueError("Number of weights must match number of model updates")
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Initialize aggregated model with zeros
            aggregated_model = {}
            
            # Get parameter names from first model
            param_names = model_updates[0].keys()
            
            # Aggregate each parameter
            for param_name in param_names:
                # Start with zeros
                aggregated_param = torch.zeros_like(model_updates[0][param_name])
                
                # Weighted sum
                for i, update in enumerate(model_updates):
                    if param_name in update:
                        aggregated_param += weights[i] * update[param_name]
                
                aggregated_model[param_name] = aggregated_param
            
            logger.info(f"Aggregated {len(model_updates)} model updates using FedAvg")
            return aggregated_model
            
        except Exception as e:
            logger.error(f"Error in federated averaging: {e}")
            raise
    
    @staticmethod
    def weighted_aggregation(model_updates: List[Dict[str, torch.Tensor]], 
                           num_samples: List[int]) -> Dict[str, torch.Tensor]:
        """
        Perform weighted aggregation based on number of training samples.
        
        Args:
            model_updates: List of model state_dicts
            num_samples: Number of training samples for each model update
            
        Returns:
            Aggregated model state_dict
        """
        try:
            if len(model_updates) != len(num_samples):
                raise ValueError("Number of model updates must match number of sample counts")
            
            # Calculate weights based on sample counts
            total_samples = sum(num_samples)
            weights = [n / total_samples for n in num_samples]
            
            logger.info(f"Using sample-weighted aggregation with weights: {weights}")
            return FederatedAggregator.federated_averaging(model_updates, weights)
            
        except Exception as e:
            logger.error(f"Error in weighted aggregation: {e}")
            raise

class ModelCompatibility:
    """
    Handles model compatibility and versioning for federated learning.
    """
    
    @staticmethod
    def check_compatibility(model1: Dict[str, torch.Tensor], 
                          model2: Dict[str, torch.Tensor]) -> bool:
        """
        Check if two models are compatible for aggregation.
        
        Args:
            model1: First model state_dict
            model2: Second model state_dict
            
        Returns:
            True if models are compatible, False otherwise
        """
        try:
            # Check if parameter names match
            if set(model1.keys()) != set(model2.keys()):
                logger.warning("Model parameter names don't match")
                return False
            
            # Check if parameter shapes match
            for param_name in model1.keys():
                if model1[param_name].shape != model2[param_name].shape:
                    logger.warning(f"Parameter {param_name} shape mismatch: "
                                 f"{model1[param_name].shape} vs {model2[param_name].shape}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking model compatibility: {e}")
            return False
    
    @staticmethod
    def validate_model_update(model_update: Dict[str, torch.Tensor],
                            reference_model: Dict[str, torch.Tensor],
                            max_norm: float = 10.0) -> bool:
        """
        Validate a model update for potential poisoning attacks.
        
        Args:
            model_update: Model update to validate
            reference_model: Reference model (e.g., previous global model)
            max_norm: Maximum allowed L2 norm of update
            
        Returns:
            True if update is valid, False otherwise
        """
        try:
            if not ModelCompatibility.check_compatibility(model_update, reference_model):
                return False
            
            # Calculate L2 norm of update
            update_norm = 0.0
            for param_name in model_update.keys():
                if param_name in reference_model:
                    diff = model_update[param_name] - reference_model[param_name]
                    update_norm += torch.norm(diff).item() ** 2
            
            update_norm = update_norm ** 0.5
            
            if update_norm > max_norm:
                logger.warning(f"Model update norm {update_norm:.4f} exceeds maximum {max_norm}")
                return False
            
            logger.info(f"Model update validation passed (norm: {update_norm:.4f})")
            return True
            
        except Exception as e:
            logger.error(f"Error validating model update: {e}")
            return False

# Backward compatibility functions
def serialize_model(model: nn.Module) -> bytes:
    """Backward compatibility function."""
    return ModelSerializer.serialize_model(model)

def deserialize_model(model_bytes: bytes, model_class: nn.Module) -> nn.Module:
    """Backward compatibility function."""
    return ModelSerializer.deserialize_model(model_bytes, model_class)

def aggregate_models(model_updates: List[Dict[str, torch.Tensor]], 
                    weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
    """Backward compatibility function."""
    return FederatedAggregator.federated_averaging(model_updates, weights)