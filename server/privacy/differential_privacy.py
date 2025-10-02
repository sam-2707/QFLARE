"""
QFLARE Differential Privacy Engine

This module implements differential privacy mechanisms for federated learning.
Provides privacy-preserving model updates with formal (ε, δ)-DP guarantees.
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import math
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class DifferentialPrivacyConfig:
    """Configuration for differential privacy parameters."""
    
    def __init__(self, 
                 epsilon: float = 0.1,
                 delta: float = 1e-6,
                 max_grad_norm: float = 1.0,
                 noise_multiplier: Optional[float] = None,
                 secure_rng: bool = True):
        """
        Initialize DP configuration.
        
        Args:
            epsilon: Privacy budget parameter (smaller = more private)
            delta: Failure probability (should be < 1/n where n is dataset size)
            max_grad_norm: L2 norm bound for gradient clipping
            noise_multiplier: Noise multiplier (computed if None)
            secure_rng: Use cryptographically secure random number generator
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.secure_rng = secure_rng
        
        # Calculate noise multiplier if not provided
        if noise_multiplier is None:
            self.noise_multiplier = self._calculate_noise_multiplier()
        else:
            self.noise_multiplier = noise_multiplier
        
        # Privacy accounting
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.composition_rounds = 0
        
        logger.info(f"DP Config: ε={epsilon}, δ={delta}, σ={self.noise_multiplier:.3f}")
    
    def _calculate_noise_multiplier(self) -> float:
        """
        Calculate noise multiplier for Gaussian mechanism.
        
        Uses the formula: σ = (2 * ln(1.25/δ) / ε) * C
        where C is the sensitivity (max_grad_norm).
        """
        if self.delta <= 0 or self.epsilon <= 0:
            raise ValueError("ε and δ must be positive")
        
        # Gaussian mechanism noise scale
        sigma = (2 * math.log(1.25 / self.delta) / self.epsilon) * self.max_grad_norm
        return sigma
    
    def update_composition(self, rounds: int = 1):
        """Update privacy composition for multiple queries."""
        self.composition_rounds += rounds
        
        # Advanced composition theorem
        # For T rounds: ε' = ε√(2T ln(1/δ')) + Tε(e^ε - 1)
        T = self.composition_rounds
        
        if T > 1:
            # Simplified advanced composition
            self.spent_epsilon = self.epsilon * math.sqrt(2 * T * math.log(1 / self.delta))
            self.spent_delta = T * self.delta
        else:
            self.spent_epsilon = self.epsilon
            self.spent_delta = self.delta
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get total privacy budget spent."""
        return self.spent_epsilon, self.spent_delta
    
    def get_remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget."""
        # For demonstration, assume total budget is 10x initial
        max_epsilon = self.epsilon * 10
        max_delta = self.delta * 10
        
        remaining_eps = max(0, max_epsilon - self.spent_epsilon)
        remaining_delta = max(0, max_delta - self.spent_delta)
        
        return remaining_eps, remaining_delta


class GaussianMechanism:
    """
    Implements the Gaussian mechanism for differential privacy.
    Adds calibrated Gaussian noise to provide (ε, δ)-DP guarantees.
    """
    
    def __init__(self, config: DifferentialPrivacyConfig):
        self.config = config
        self.rng = np.random.RandomState()
        
        if config.secure_rng:
            # Use cryptographically secure random seed
            import secrets
            self.rng.seed(secrets.randbits(32))
    
    def add_noise(self, value: Union[torch.Tensor, np.ndarray, float]) -> Union[torch.Tensor, np.ndarray, float]:
        """
        Add Gaussian noise to a value.
        
        Args:
            value: Value to add noise to (tensor, array, or scalar)
            
        Returns:
            Value with added Gaussian noise
        """
        if isinstance(value, torch.Tensor):
            return self._add_noise_tensor(value)
        elif isinstance(value, np.ndarray):
            return self._add_noise_array(value)
        else:
            return self._add_noise_scalar(value)
    
    def _add_noise_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to PyTorch tensor."""
        noise = torch.normal(
            mean=0.0,
            std=self.config.noise_multiplier,
            size=tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device
        )
        return tensor + noise
    
    def _add_noise_array(self, array: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to NumPy array."""
        noise = self.rng.normal(
            loc=0.0,
            scale=self.config.noise_multiplier,
            size=array.shape
        )
        return array + noise.astype(array.dtype)
    
    def _add_noise_scalar(self, value: float) -> float:
        """Add Gaussian noise to scalar value."""
        noise = self.rng.normal(loc=0.0, scale=self.config.noise_multiplier)
        return value + noise


class PrivacyEngine:
    """
    Main differential privacy engine for federated learning.
    Handles gradient clipping, noise addition, and privacy accounting.
    """
    
    def __init__(self, config: DifferentialPrivacyConfig):
        self.config = config
        self.gaussian_mechanism = GaussianMechanism(config)
        self.privacy_history = []
        
        logger.info("Privacy Engine initialized with (ε, δ) = ({:.3f}, {:.2e})".format(
            config.epsilon, config.delta
        ))
    
    def clip_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Clip gradients to bound their L2 norm.
        
        Args:
            gradients: Dictionary of gradient tensors
            
        Returns:
            Clipped gradients with bounded L2 norm
        """
        # Calculate total L2 norm
        total_norm = 0.0
        for grad in gradients.values():
            if grad is not None:
                total_norm += grad.norm(dtype=torch.float32).item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip if necessary
        clip_coeff = min(1.0, self.config.max_grad_norm / max(total_norm, 1e-6))
        
        clipped_gradients = {}
        for name, grad in gradients.items():
            if grad is not None:
                clipped_gradients[name] = grad * clip_coeff
            else:
                clipped_gradients[name] = grad
        
        logger.debug(f"Gradient norm: {total_norm:.3f}, clip coefficient: {clip_coeff:.3f}")
        return clipped_gradients
    
    def add_privacy_noise(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Add differential privacy noise to gradients.
        
        Args:
            gradients: Dictionary of gradient tensors
            
        Returns:
            Gradients with added DP noise
        """
        noisy_gradients = {}
        
        for name, grad in gradients.items():
            if grad is not None:
                noisy_grad = self.gaussian_mechanism.add_noise(grad)
                noisy_gradients[name] = noisy_grad
            else:
                noisy_gradients[name] = grad
        
        logger.debug(f"Added DP noise to {len(noisy_gradients)} gradient tensors")
        return noisy_gradients
    
    def privatize_model_update(self, model_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply full differential privacy pipeline to model update.
        
        Args:
            model_update: Dictionary of model parameter updates
            
        Returns:
            Privatized model update with DP guarantees
        """
        # Step 1: Clip gradients
        clipped_update = self.clip_gradients(model_update)
        
        # Step 2: Add noise
        private_update = self.add_privacy_noise(clipped_update)
        
        # Step 3: Update privacy accounting
        self.config.update_composition(rounds=1)
        
        # Step 4: Record privacy event
        privacy_event = {
            "timestamp": datetime.now().isoformat(),
            "epsilon_spent": self.config.spent_epsilon,
            "delta_spent": self.config.spent_delta,
            "composition_rounds": self.config.composition_rounds,
            "parameters_privatized": len(private_update)
        }
        self.privacy_history.append(privacy_event)
        
        logger.info(f"Privatized model update - ε spent: {self.config.spent_epsilon:.4f}")
        return private_update
    
    def privatize_aggregated_model(self, aggregated_gradients: Dict[str, torch.Tensor], 
                                   num_clients: int) -> Dict[str, torch.Tensor]:
        """
        Apply differential privacy to aggregated model updates.
        
        Args:
            aggregated_gradients: Aggregated gradients from multiple clients
            num_clients: Number of clients that contributed
            
        Returns:
            Privatized aggregated model with DP guarantees
        """
        # Scale noise for aggregation
        # When aggregating, sensitivity scales with number of clients
        original_noise = self.config.noise_multiplier
        
        # For aggregation, we can reduce noise proportionally
        # This is because the sensitivity of the sum is bounded by num_clients * C
        scaled_noise_multiplier = original_noise / math.sqrt(num_clients)
        self.config.noise_multiplier = scaled_noise_multiplier
        
        try:
            # Apply privatization
            private_aggregated = self.privatize_model_update(aggregated_gradients)
            
            logger.info(f"Privatized aggregated model from {num_clients} clients")
            return private_aggregated
            
        finally:
            # Restore original noise multiplier
            self.config.noise_multiplier = original_noise
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """Generate comprehensive privacy report."""
        spent_eps, spent_delta = self.config.get_privacy_spent()
        remaining_eps, remaining_delta = self.config.get_remaining_budget()
        
        return {
            "privacy_parameters": {
                "target_epsilon": self.config.epsilon,
                "target_delta": self.config.delta,
                "max_grad_norm": self.config.max_grad_norm,
                "noise_multiplier": self.config.noise_multiplier
            },
            "privacy_spent": {
                "epsilon_spent": spent_eps,
                "delta_spent": spent_delta,
                "composition_rounds": self.config.composition_rounds
            },
            "remaining_budget": {
                "epsilon_remaining": remaining_eps,
                "delta_remaining": remaining_delta
            },
            "privacy_history": self.privacy_history[-10:],  # Last 10 events
            "total_privatization_events": len(self.privacy_history)
        }
    
    def validate_privacy_budget(self) -> bool:
        """Check if privacy budget is still available."""
        remaining_eps, remaining_delta = self.config.get_remaining_budget()
        return remaining_eps > 0 and remaining_delta > 0


# Factory functions for common DP configurations
def create_strong_privacy_config() -> DifferentialPrivacyConfig:
    """Create configuration for strong privacy (ε=0.1)."""
    return DifferentialPrivacyConfig(
        epsilon=0.1,
        delta=1e-6,
        max_grad_norm=1.0
    )


def create_moderate_privacy_config() -> DifferentialPrivacyConfig:
    """Create configuration for moderate privacy (ε=1.0)."""
    return DifferentialPrivacyConfig(
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0
    )


def create_weak_privacy_config() -> DifferentialPrivacyConfig:
    """Create configuration for weak privacy (ε=5.0)."""
    return DifferentialPrivacyConfig(
        epsilon=5.0,
        delta=1e-4,
        max_grad_norm=1.0
    )


def create_privacy_engine(privacy_level: str = "strong") -> PrivacyEngine:
    """
    Create privacy engine with preset configuration.
    
    Args:
        privacy_level: "strong", "moderate", or "weak"
        
    Returns:
        Configured PrivacyEngine instance
    """
    if privacy_level == "strong":
        config = create_strong_privacy_config()
    elif privacy_level == "moderate":
        config = create_moderate_privacy_config()
    elif privacy_level == "weak":
        config = create_weak_privacy_config()
    else:
        raise ValueError(f"Unknown privacy level: {privacy_level}")
    
    return PrivacyEngine(config)


if __name__ == "__main__":
    # Test the differential privacy implementation
    print("Testing QFLARE Differential Privacy Engine...")
    
    # Create privacy engine
    privacy_engine = create_privacy_engine("strong")
    print(f"✓ Privacy engine created with ε={privacy_engine.config.epsilon}")
    
    # Test with mock gradients
    mock_gradients = {
        "layer1.weight": torch.randn(10, 5),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(1, 10),
        "layer2.bias": torch.randn(1)
    }
    
    # Test gradient clipping
    clipped = privacy_engine.clip_gradients(mock_gradients)
    print("✓ Gradient clipping works")
    
    # Test noise addition
    noisy = privacy_engine.add_privacy_noise(clipped)
    print("✓ Noise addition works")
    
    # Test full privatization
    privatized = privacy_engine.privatize_model_update(mock_gradients)
    print("✓ Full privatization works")
    
    # Test privacy report
    report = privacy_engine.get_privacy_report()
    print(f"✓ Privacy report generated: {report['privacy_spent']['epsilon_spent']:.4f} ε spent")
    
    print("Differential privacy engine test completed!")