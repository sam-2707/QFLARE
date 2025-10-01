"""
QFLARE Model Aggregation for Federated Learning

This module implements federated averaging and other aggregation algorithms
for combining local model updates into a global model.
"""

import logging
import pickle
import io
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict

logger = logging.getLogger(__name__)

class FederatedAveraging:
    """
    Implements the FedAvg algorithm for model aggregation.
    Combines local model updates using weighted averaging.
    """
    
    def __init__(self, 
                 use_sample_weights: bool = True,
                 min_participation_rate: float = 0.5):
        """
        Initialize federated averaging aggregator.
        
        Args:
            use_sample_weights: Whether to weight by number of training samples
            min_participation_rate: Minimum participation rate to proceed with aggregation
        """
        self.use_sample_weights = use_sample_weights
        self.min_participation_rate = min_participation_rate
        
        logger.info(f"FederatedAveraging initialized (sample_weights={use_sample_weights})")
    
    def aggregate(self, 
                  model_updates: List[bytes], 
                  weights: Optional[List[float]] = None,
                  metadata: Optional[List[Dict[str, Any]]] = None) -> bytes:
        """
        Aggregate model updates using federated averaging.
        
        Args:
            model_updates: List of serialized model state dictionaries
            weights: Optional weights for each model (e.g., number of samples)
            metadata: Optional metadata for each model update
            
        Returns:
            Serialized aggregated model state dictionary
        """
        if not model_updates:
            raise ValueError("No model updates provided for aggregation")
        
        logger.info(f"Aggregating {len(model_updates)} model updates")
        
        try:
            # Deserialize model updates
            state_dicts = []
            for i, model_data in enumerate(model_updates):
                try:
                    state_dict = self._deserialize_model(model_data)
                    state_dicts.append(state_dict)
                except Exception as e:
                    logger.warning(f"Failed to deserialize model update {i}: {e}")
                    continue
            
            if not state_dicts:
                raise ValueError("No valid model updates could be deserialized")
            
            # Prepare weights
            if weights is None:
                weights = [1.0] * len(state_dicts)
            elif len(weights) != len(state_dicts):
                logger.warning(f"Weight count mismatch: {len(weights)} weights for {len(state_dicts)} models")
                weights = weights[:len(state_dicts)] + [1.0] * max(0, len(state_dicts) - len(weights))
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight <= 0:
                raise ValueError("Total weight must be positive")
            
            normalized_weights = [w / total_weight for w in weights]
            
            # Perform weighted averaging
            aggregated_state_dict = self._weighted_average_state_dicts(state_dicts, normalized_weights)
            
            # Serialize aggregated model
            aggregated_model_data = self._serialize_model(aggregated_state_dict)
            
            logger.info(f"Model aggregation completed - {len(aggregated_model_data)} bytes")
            return aggregated_model_data
            
        except Exception as e:
            logger.error(f"Error during model aggregation: {e}")
            raise
    
    def _deserialize_model(self, model_data: bytes) -> Dict[str, Any]:
        """Deserialize model data to state dictionary."""
        try:
            # Try different deserialization methods
            buffer = io.BytesIO(model_data)
            
            # First try pickle
            try:
                buffer.seek(0)
                state_dict = pickle.load(buffer)
                if isinstance(state_dict, dict):
                    return state_dict
            except:
                pass
            
            # Try PyTorch if available
            try:
                import torch
                buffer.seek(0)
                state_dict = torch.load(buffer, map_location='cpu')
                if isinstance(state_dict, dict):
                    return state_dict
            except ImportError:
                pass
            except:
                pass
            
            # If all else fails, try direct evaluation (unsafe but for demo)
            try:
                buffer.seek(0)
                data_str = buffer.read().decode('utf-8')
                state_dict = eval(data_str)  # UNSAFE - only for demo
                if isinstance(state_dict, dict):
                    return state_dict
            except:
                pass
            
            raise ValueError("Could not deserialize model data")
            
        except Exception as e:
            raise ValueError(f"Model deserialization failed: {e}")
    
    def _serialize_model(self, state_dict: Dict[str, Any]) -> bytes:
        """Serialize state dictionary to bytes."""
        try:
            # Try PyTorch serialization if available
            try:
                import torch
                buffer = io.BytesIO()
                torch.save(state_dict, buffer)
                return buffer.getvalue()
            except ImportError:
                pass
            
            # Fallback to pickle
            buffer = io.BytesIO()
            pickle.dump(state_dict, buffer)
            return buffer.getvalue()
            
        except Exception as e:
            raise ValueError(f"Model serialization failed: {e}")
    
    def _weighted_average_state_dicts(self, 
                                    state_dicts: List[Dict[str, Any]], 
                                    weights: List[float]) -> Dict[str, Any]:
        """
        Perform weighted averaging of model state dictionaries.
        
        Args:
            state_dicts: List of model state dictionaries
            weights: Normalized weights for each model
            
        Returns:
            Averaged state dictionary
        """
        if not state_dicts:
            raise ValueError("No state dictionaries provided")
        
        # Initialize aggregated state dict with first model's structure
        aggregated_state_dict = OrderedDict()
        reference_keys = set(state_dicts[0].keys())
        
        # Check all models have the same structure
        for i, state_dict in enumerate(state_dicts[1:], 1):
            model_keys = set(state_dict.keys())
            if model_keys != reference_keys:
                logger.warning(f"Model {i} has different keys than reference model")
                # Use intersection of keys
                reference_keys = reference_keys.intersection(model_keys)
        
        # Average each parameter
        for key in reference_keys:
            try:
                # Extract parameter values from all models
                param_values = []
                valid_weights = []
                
                for state_dict, weight in zip(state_dicts, weights):
                    if key in state_dict:
                        param_values.append(self._to_numpy(state_dict[key]))
                        valid_weights.append(weight)
                
                if not param_values:
                    logger.warning(f"No valid values found for parameter {key}")
                    continue
                
                # Renormalize weights for this parameter
                total_valid_weight = sum(valid_weights)
                if total_valid_weight > 0:
                    valid_weights = [w / total_valid_weight for w in valid_weights]
                else:
                    valid_weights = [1.0 / len(valid_weights)] * len(valid_weights)
                
                # Compute weighted average
                averaged_param = np.zeros_like(param_values[0])
                for param_value, weight in zip(param_values, valid_weights):
                    averaged_param += weight * param_value
                
                # Convert back to original type
                aggregated_state_dict[key] = self._from_numpy(averaged_param, state_dicts[0][key])
                
            except Exception as e:
                logger.warning(f"Failed to average parameter {key}: {e}")
                # Use first model's parameter as fallback
                aggregated_state_dict[key] = state_dicts[0][key]
        
        logger.debug(f"Averaged {len(aggregated_state_dict)} parameters")
        return aggregated_state_dict
    
    def _to_numpy(self, tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        if isinstance(tensor, np.ndarray):
            return tensor.copy()
        
        try:
            # Try PyTorch tensor
            if hasattr(tensor, 'detach'):
                return tensor.detach().cpu().numpy()
            if hasattr(tensor, 'numpy'):
                return tensor.numpy()
        except:
            pass
        
        # Try direct conversion
        try:
            return np.array(tensor)
        except:
            raise ValueError(f"Cannot convert to numpy array: {type(tensor)}")
    
    def _from_numpy(self, arr: np.ndarray, original_tensor):
        """Convert numpy array back to original tensor type."""
        try:
            # Try PyTorch tensor
            if hasattr(original_tensor, 'detach'):
                import torch
                return torch.from_numpy(arr).to(original_tensor.dtype).to(original_tensor.device)
        except:
            pass
        
        # Return as numpy array if conversion fails
        return arr


class SecureAggregation:
    """
    Placeholder for secure aggregation techniques.
    TODO: Implement secure multi-party computation for privacy-preserving aggregation.
    """
    
    def __init__(self):
        self.fedavg = FederatedAveraging()
        logger.info("SecureAggregation initialized (using FedAvg for now)")
    
    def aggregate(self, model_updates: List[bytes], weights: Optional[List[float]] = None) -> bytes:
        """
        Placeholder for secure aggregation.
        Currently delegates to FederatedAveraging.
        """
        logger.warning("Using FederatedAveraging instead of secure aggregation (not implemented)")
        return self.fedavg.aggregate(model_updates, weights)


class ModelAggregationValidator:
    """
    Validates model updates before aggregation to detect potential attacks.
    """
    
    def __init__(self, 
                 cosine_similarity_threshold: float = 0.5,
                 magnitude_threshold: float = 2.0):
        """
        Initialize aggregation validator.
        
        Args:
            cosine_similarity_threshold: Threshold for cosine similarity-based detection
            magnitude_threshold: Threshold for parameter magnitude detection
        """
        self.cosine_similarity_threshold = cosine_similarity_threshold
        self.magnitude_threshold = magnitude_threshold
        
        logger.info(f"ModelAggregationValidator initialized (similarity_threshold={cosine_similarity_threshold})")
    
    def validate_updates(self, 
                        model_updates: List[bytes],
                        metadata: Optional[List[Dict[str, Any]]] = None) -> List[bool]:
        """
        Validate model updates for potential poisoning attacks.
        
        Args:
            model_updates: List of serialized model updates
            metadata: Optional metadata for each update
            
        Returns:
            List of boolean values indicating which updates are valid
        """
        if len(model_updates) < 2:
            # Can't validate with less than 2 updates
            return [True] * len(model_updates)
        
        try:
            # Deserialize updates for analysis
            aggregator = FederatedAveraging()
            state_dicts = []
            
            for model_data in model_updates:
                try:
                    state_dict = aggregator._deserialize_model(model_data)
                    state_dicts.append(state_dict)
                except:
                    state_dicts.append(None)
            
            # Validate each update
            valid_flags = []
            for i, state_dict in enumerate(state_dicts):
                if state_dict is None:
                    valid_flags.append(False)
                    continue
                
                is_valid = self._validate_single_update(state_dict, state_dicts, i)
                valid_flags.append(is_valid)
            
            logger.info(f"Validation complete: {sum(valid_flags)}/{len(valid_flags)} updates valid")
            return valid_flags
            
        except Exception as e:
            logger.error(f"Error during update validation: {e}")
            # Return all valid if validation fails
            return [True] * len(model_updates)
    
    def _validate_single_update(self, 
                               update_state_dict: Dict[str, Any],
                               all_state_dicts: List[Dict[str, Any]],
                               update_index: int) -> bool:
        """Validate a single model update against others."""
        try:
            # Skip None entries and self
            other_state_dicts = [
                state_dict for i, state_dict in enumerate(all_state_dicts)
                if i != update_index and state_dict is not None
            ]
            
            if not other_state_dicts:
                return True
            
            # Compute cosine similarity with other updates
            similarities = []
            for other_state_dict in other_state_dicts:
                similarity = self._compute_cosine_similarity(update_state_dict, other_state_dict)
                if similarity is not None:
                    similarities.append(similarity)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                if avg_similarity < self.cosine_similarity_threshold:
                    logger.warning(f"Update {update_index} rejected: low similarity ({avg_similarity:.3f})")
                    return False
            
            # Check parameter magnitudes
            if not self._check_parameter_magnitudes(update_state_dict):
                logger.warning(f"Update {update_index} rejected: abnormal parameter magnitudes")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating update {update_index}: {e}")
            return True  # Default to valid if validation fails
    
    def _compute_cosine_similarity(self, 
                                  state_dict1: Dict[str, Any],
                                  state_dict2: Dict[str, Any]) -> Optional[float]:
        """Compute cosine similarity between two state dictionaries."""
        try:
            common_keys = set(state_dict1.keys()).intersection(set(state_dict2.keys()))
            if not common_keys:
                return None
            
            # Flatten parameters
            vec1 = []
            vec2 = []
            
            aggregator = FederatedAveraging()
            
            for key in common_keys:
                param1 = aggregator._to_numpy(state_dict1[key]).flatten()
                param2 = aggregator._to_numpy(state_dict2[key]).flatten()
                
                vec1.extend(param1)
                vec2.extend(param2)
            
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.warning(f"Error computing cosine similarity: {e}")
            return None
    
    def _check_parameter_magnitudes(self, state_dict: Dict[str, Any]) -> bool:
        """Check if parameter magnitudes are within reasonable bounds."""
        try:
            aggregator = FederatedAveraging()
            
            for key, param in state_dict.items():
                param_array = aggregator._to_numpy(param)
                
                # Check for extreme values
                if np.any(np.isnan(param_array)) or np.any(np.isinf(param_array)):
                    return False
                
                # Check magnitude
                max_magnitude = np.max(np.abs(param_array))
                if max_magnitude > self.magnitude_threshold:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking parameter magnitudes: {e}")
            return True  # Default to valid if check fails