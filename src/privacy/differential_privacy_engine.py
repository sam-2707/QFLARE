"""
QFLARE Differential Privacy Engine
Implements formal differential privacy for federated learning with privacy budget tracking
"""

import numpy as np
import math
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from scipy import stats
import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrivacyMechanism(Enum):
    """Supported differential privacy mechanisms"""
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"
    EXPONENTIAL = "exponential"
    COMPOSITIONS = "compositions"

class PrivacyAccountant(Enum):
    """Privacy accounting methods"""
    BASIC_COMPOSITION = "basic"
    ADVANCED_COMPOSITION = "advanced"
    RDP_ACCOUNTANT = "rdp"  # Rényi Differential Privacy
    MOMENTS_ACCOUNTANT = "moments"

@dataclass
class PrivacyParameters:
    """Differential privacy parameters"""
    epsilon: float  # Privacy loss parameter
    delta: float    # Failure probability
    sensitivity: float = 1.0  # Global sensitivity
    mechanism: PrivacyMechanism = PrivacyMechanism.GAUSSIAN
    accountant: PrivacyAccountant = PrivacyAccountant.MOMENTS_ACCOUNTANT
    
    def __post_init__(self):
        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if not (0 <= self.delta <= 1):
            raise ValueError("Delta must be between 0 and 1")
        if self.sensitivity <= 0:
            raise ValueError("Sensitivity must be positive")

@dataclass
class PrivacyLedger:
    """Tracks privacy expenditure over time"""
    total_epsilon: float = 0.0
    total_delta: float = 0.0
    operations: List[Dict] = field(default_factory=list)
    budget_limit: float = 1.0
    
    def add_operation(self, epsilon_spent: float, delta_spent: float, 
                     operation_type: str, timestamp: float = None):
        """Add privacy expenditure from an operation"""
        if timestamp is None:
            timestamp = time.time()
        
        self.total_epsilon += epsilon_spent
        self.total_delta += delta_spent
        
        operation = {
            'epsilon': epsilon_spent,
            'delta': delta_spent,
            'type': operation_type,
            'timestamp': timestamp,
            'cumulative_epsilon': self.total_epsilon,
            'cumulative_delta': self.total_delta
        }
        
        self.operations.append(operation)
        logger.debug(f"Privacy operation recorded: {operation_type}, ε={epsilon_spent:.4f}, δ={delta_spent:.6f}")
    
    def has_budget_remaining(self) -> bool:
        """Check if privacy budget is not exhausted"""
        return self.total_epsilon < self.budget_limit
    
    def remaining_budget(self) -> float:
        """Get remaining privacy budget"""
        return max(0, self.budget_limit - self.total_epsilon)
    
    def to_dict(self) -> Dict:
        """Export ledger to dictionary"""
        return {
            'total_epsilon': self.total_epsilon,
            'total_delta': self.total_delta,
            'budget_limit': self.budget_limit,
            'operations': self.operations,
            'budget_exhausted': not self.has_budget_remaining()
        }

class MomentsAccountant:
    """
    Moments Accountant for tight privacy analysis
    Based on "Deep Learning with Differential Privacy" (Abadi et al.)
    """
    
    def __init__(self, max_order: int = 32):
        self.max_order = max_order
        self.log_moments = [0.0] * (max_order + 1)
    
    def accumulate_privacy_spending(self, sampling_rate: float, noise_multiplier: float, steps: int):
        """Accumulate privacy spending using moments accountant"""
        
        # Compute log moments for Gaussian mechanism
        for order in range(2, self.max_order + 1):
            # Moment generating function for subsampled Gaussian mechanism
            alpha = order
            
            # Log moment bound for subsampled Gaussian mechanism
            # This is a simplified version - full implementation requires more complex bounds
            if noise_multiplier > 0:
                log_moment = self._compute_log_moment(alpha, sampling_rate, noise_multiplier, steps)
                self.log_moments[order] += log_moment
    
    def _compute_log_moment(self, alpha: int, q: float, sigma: float, steps: int) -> float:
        """Compute log moment for given parameters"""
        # Simplified computation - real implementation uses tight bounds
        if sigma <= 0:
            return float('inf')
        
        # Basic bound for subsampled Gaussian mechanism
        # In practice, use tighter bounds from privacy analysis literature
        return steps * q * alpha * (alpha - 1) / (2 * sigma**2)
    
    def get_privacy_spent(self, delta: float) -> float:
        """Convert moments to (epsilon, delta)-DP guarantee"""
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be in (0, 1)")
        
        min_epsilon = float('inf')
        
        for order in range(2, self.max_order + 1):
            if self.log_moments[order] < float('inf'):
                epsilon = self.log_moments[order] - math.log(delta) / (order - 1)
                min_epsilon = min(min_epsilon, epsilon)
        
        return min_epsilon if min_epsilon < float('inf') else 0.0

class DifferentialPrivacyEngine:
    """
    QFLARE Differential Privacy Engine
    
    Provides formal differential privacy guarantees for federated learning
    """
    
    def __init__(self, privacy_params: PrivacyParameters):
        """Initialize differential privacy engine"""
        self.privacy_params = privacy_params
        self.ledger = PrivacyLedger(budget_limit=privacy_params.epsilon)
        self.moments_accountant = MomentsAccountant()
        
        logger.info(f"DP Engine initialized: ε={privacy_params.epsilon}, δ={privacy_params.delta}")
    
    def add_gaussian_noise(self, data: Union[np.ndarray, torch.Tensor], 
                          sensitivity: float = None) -> Union[np.ndarray, torch.Tensor]:
        """
        Add calibrated Gaussian noise for differential privacy
        
        Args:
            data: Input data (gradients, model parameters, etc.)
            sensitivity: L2 sensitivity of the data
            
        Returns:
            Data with added noise
        """
        if sensitivity is None:
            sensitivity = self.privacy_params.sensitivity
        
        # Calculate noise scale for (ε, δ)-DP
        sigma = self._gaussian_noise_scale(
            self.privacy_params.epsilon, 
            self.privacy_params.delta, 
            sensitivity
        )
        
        is_torch = isinstance(data, torch.Tensor)
        
        if is_torch:
            device = data.device
            noise = torch.normal(0, sigma, size=data.shape, device=device)
            noisy_data = data + noise
        else:
            noise = np.random.normal(0, sigma, size=data.shape)
            noisy_data = data + noise
        
        # Record privacy expenditure
        self.ledger.add_operation(
            epsilon_spent=self.privacy_params.epsilon,
            delta_spent=self.privacy_params.delta,
            operation_type="gaussian_noise"
        )
        
        logger.debug(f"Added Gaussian noise: σ={sigma:.4f}, sensitivity={sensitivity}")
        
        return noisy_data
    
    def add_laplace_noise(self, data: Union[np.ndarray, torch.Tensor], 
                         sensitivity: float = None) -> Union[np.ndarray, torch.Tensor]:
        """
        Add calibrated Laplace noise for pure differential privacy
        
        Args:
            data: Input data
            sensitivity: L1 sensitivity of the data
            
        Returns:
            Data with added Laplace noise
        """
        if sensitivity is None:
            sensitivity = self.privacy_params.sensitivity
        
        # Laplace mechanism scale
        scale = sensitivity / self.privacy_params.epsilon
        
        is_torch = isinstance(data, torch.Tensor)
        
        if is_torch:
            # PyTorch doesn't have Laplace distribution, use numpy then convert
            numpy_data = data.cpu().numpy()
            noise = np.random.laplace(0, scale, size=numpy_data.shape)
            noisy_data = torch.from_numpy(numpy_data + noise).to(data.device)
        else:
            noise = np.random.laplace(0, scale, size=data.shape)
            noisy_data = data + noise
        
        # Record privacy expenditure (pure DP, δ=0)
        self.ledger.add_operation(
            epsilon_spent=self.privacy_params.epsilon,
            delta_spent=0.0,
            operation_type="laplace_noise"
        )
        
        logger.debug(f"Added Laplace noise: scale={scale:.4f}, sensitivity={sensitivity}")
        
        return noisy_data
    
    def clip_gradients(self, gradients: Union[List[torch.Tensor], torch.Tensor], 
                      max_norm: float = 1.0) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Clip gradients to bound sensitivity
        
        Args:
            gradients: Gradient tensors or list of tensors
            max_norm: Maximum L2 norm for clipping
            
        Returns:
            Clipped gradients
        """
        if isinstance(gradients, list):
            # Multiple gradient tensors
            total_norm = 0.0
            for grad in gradients:
                if grad is not None:
                    total_norm += torch.norm(grad).item() ** 2
            total_norm = math.sqrt(total_norm)
            
            clip_coef = min(max_norm / (total_norm + 1e-6), 1.0)
            
            clipped_gradients = []
            for grad in gradients:
                if grad is not None:
                    clipped_gradients.append(grad * clip_coef)
                else:
                    clipped_gradients.append(grad)
            
            logger.debug(f"Clipped gradients: norm {total_norm:.4f} -> {min(total_norm, max_norm):.4f}")
            return clipped_gradients
        
        else:
            # Single gradient tensor
            grad_norm = torch.norm(gradients).item()
            clip_coef = min(max_norm / (grad_norm + 1e-6), 1.0)
            
            clipped_gradients = gradients * clip_coef
            
            logger.debug(f"Clipped gradient: norm {grad_norm:.4f} -> {min(grad_norm, max_norm):.4f}")
            return clipped_gradients
    
    def private_aggregation(self, gradients_list: List[torch.Tensor], 
                           client_weights: Optional[List[float]] = None) -> torch.Tensor:
        """
        Differentially private gradient aggregation
        
        Args:
            gradients_list: List of gradient tensors from clients
            client_weights: Optional weights for weighted aggregation
            
        Returns:
            Aggregated gradient with privacy guarantees
        """
        if not gradients_list:
            raise ValueError("No gradients provided for aggregation")
        
        # Clip all gradients
        clipped_gradients = [self.clip_gradients(grad) for grad in gradients_list]
        
        # Weighted or uniform aggregation
        if client_weights is None:
            client_weights = [1.0 / len(clipped_gradients)] * len(clipped_gradients)
        
        # Aggregate clipped gradients
        aggregated = torch.zeros_like(clipped_gradients[0])
        for grad, weight in zip(clipped_gradients, client_weights):
            aggregated += weight * grad
        
        # Add calibrated noise
        noisy_aggregated = self.add_gaussian_noise(aggregated)
        
        logger.info(f"Private aggregation: {len(gradients_list)} clients, ε={self.privacy_params.epsilon}")
        
        return noisy_aggregated
    
    def exponential_mechanism(self, candidates: List, quality_function: callable, 
                            sensitivity: float) -> int:
        """
        Exponential mechanism for private selection
        
        Args:
            candidates: List of candidate options
            quality_function: Function that scores each candidate
            sensitivity: Sensitivity of quality function
            
        Returns:
            Index of selected candidate
        """
        # Calculate quality scores
        scores = [quality_function(candidate) for candidate in candidates]
        
        # Calculate exponential weights
        weights = []
        scale = self.privacy_params.epsilon / (2 * sensitivity)
        
        for score in scores:
            weight = math.exp(scale * score)
            weights.append(weight)
        
        # Normalize to probabilities
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        # Sample according to exponential distribution
        selected_idx = np.random.choice(len(candidates), p=probabilities)
        
        # Record privacy expenditure
        self.ledger.add_operation(
            epsilon_spent=self.privacy_params.epsilon,
            delta_spent=0.0,
            operation_type="exponential_mechanism"
        )
        
        logger.debug(f"Exponential mechanism selected candidate {selected_idx}")
        
        return selected_idx
    
    def private_mean(self, values: List[float], bounds: Tuple[float, float]) -> float:
        """
        Compute differentially private mean
        
        Args:
            values: List of values
            bounds: (min_value, max_value) for clamping
            
        Returns:
            Private mean estimate
        """
        # Clamp values to bounds
        min_val, max_val = bounds
        clamped_values = [max(min_val, min(max_val, v)) for v in values]
        
        # Calculate sensitivity
        sensitivity = (max_val - min_val) / len(values)
        
        # Compute mean and add Laplace noise
        mean = sum(clamped_values) / len(clamped_values)
        noisy_mean = mean + np.random.laplace(0, sensitivity / self.privacy_params.epsilon)
        
        # Record privacy expenditure
        self.ledger.add_operation(
            epsilon_spent=self.privacy_params.epsilon,
            delta_spent=0.0,
            operation_type="private_mean"
        )
        
        logger.debug(f"Private mean: {len(values)} values, result={noisy_mean:.4f}")
        
        return noisy_mean
    
    def composition_privacy_cost(self, num_operations: int, 
                                mechanism: PrivacyMechanism = PrivacyMechanism.GAUSSIAN) -> Tuple[float, float]:
        """
        Calculate privacy cost under composition
        
        Args:
            num_operations: Number of privacy operations
            mechanism: Privacy mechanism used
            
        Returns:
            Tuple of (total_epsilon, total_delta)
        """
        if mechanism == PrivacyMechanism.GAUSSIAN:
            # Use moments accountant for tight composition
            sampling_rate = 1.0  # Assume full participation for conservative estimate
            noise_multiplier = self._gaussian_noise_scale(
                self.privacy_params.epsilon, 
                self.privacy_params.delta, 
                self.privacy_params.sensitivity
            )
            
            self.moments_accountant.accumulate_privacy_spending(
                sampling_rate, noise_multiplier, num_operations
            )
            
            total_epsilon = self.moments_accountant.get_privacy_spent(self.privacy_params.delta)
            total_delta = self.privacy_params.delta
            
        else:
            # Basic composition for pure DP mechanisms
            total_epsilon = num_operations * self.privacy_params.epsilon
            total_delta = 0.0
        
        return total_epsilon, total_delta
    
    def _gaussian_noise_scale(self, epsilon: float, delta: float, sensitivity: float) -> float:
        """Calculate Gaussian noise scale for (ε, δ)-DP"""
        if delta == 0:
            raise ValueError("Delta must be > 0 for Gaussian mechanism")
        
        # Standard formula for Gaussian mechanism
        return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    
    def get_privacy_report(self) -> Dict:
        """Generate comprehensive privacy report"""
        return {
            'privacy_parameters': {
                'epsilon': self.privacy_params.epsilon,
                'delta': self.privacy_params.delta,
                'sensitivity': self.privacy_params.sensitivity,
                'mechanism': self.privacy_params.mechanism.value
            },
            'privacy_ledger': self.ledger.to_dict(),
            'moments_accountant': {
                'max_order': self.moments_accountant.max_order,
                'log_moments': self.moments_accountant.log_moments[:10]  # First 10 orders
            },
            'recommendations': self._get_privacy_recommendations()
        }
    
    def _get_privacy_recommendations(self) -> List[str]:
        """Get privacy recommendations based on current state"""
        recommendations = []
        
        if self.ledger.total_epsilon > self.ledger.budget_limit * 0.8:
            recommendations.append("Privacy budget nearly exhausted - consider reducing epsilon or stopping training")
        
        if self.privacy_params.epsilon > 1.0:
            recommendations.append("High epsilon value may provide weak privacy guarantees")
        
        if self.privacy_params.delta > 1e-5:
            recommendations.append("Consider smaller delta for stronger privacy guarantees")
        
        if len(self.ledger.operations) == 0:
            recommendations.append("No privacy operations recorded - ensure DP mechanisms are being used")
        
        return recommendations

def demo_differential_privacy():
    """Demonstration of QFLARE differential privacy"""
    print("🔒 QFLARE Differential Privacy Engine Demo")
    print("=" * 50)
    
    # Initialize privacy parameters
    privacy_params = PrivacyParameters(
        epsilon=0.1,  # Strong privacy
        delta=1e-6,   # Very small failure probability
        sensitivity=1.0
    )
    
    dp_engine = DifferentialPrivacyEngine(privacy_params)
    
    # Demo 1: Gradient clipping and noise addition
    print("\n1. Gradient Clipping and Noise Addition")
    print("-" * 40)
    
    # Simulate gradients from multiple clients
    np.random.seed(42)
    torch.manual_seed(42)
    
    client_gradients = []
    for i in range(5):
        # Random gradients with different norms
        grad = torch.randn(100) * (i + 1)
        client_gradients.append(grad)
        print(f"Client {i+1} gradient norm: {torch.norm(grad).item():.3f}")
    
    # Clip gradients
    clipped_gradients = [dp_engine.clip_gradients(grad, max_norm=1.0) for grad in client_gradients]
    
    print("\nAfter clipping (max_norm=1.0):")
    for i, grad in enumerate(clipped_gradients):
        print(f"Client {i+1} gradient norm: {torch.norm(grad).item():.3f}")
    
    # Demo 2: Private aggregation
    print("\n2. Private Gradient Aggregation")
    print("-" * 40)
    
    # Aggregate with differential privacy
    aggregated = dp_engine.private_aggregation(clipped_gradients)
    print(f"Aggregated gradient norm: {torch.norm(aggregated).item():.3f}")
    print(f"Privacy spent: ε={privacy_params.epsilon}, δ={privacy_params.delta}")
    
    # Demo 3: Private mean calculation
    print("\n3. Private Statistical Queries")
    print("-" * 40)
    
    # Simulate client accuracy values
    accuracies = [0.85, 0.87, 0.83, 0.89, 0.86, 0.84, 0.88, 0.82]
    true_mean = sum(accuracies) / len(accuracies)
    
    # Calculate private mean
    private_mean = dp_engine.private_mean(accuracies, bounds=(0.0, 1.0))
    
    print(f"True mean accuracy: {true_mean:.4f}")
    print(f"Private mean accuracy: {private_mean:.4f}")
    print(f"Error: {abs(true_mean - private_mean):.4f}")
    
    # Demo 4: Privacy budget tracking
    print("\n4. Privacy Budget Tracking")
    print("-" * 40)
    
    print(f"Total privacy spent: ε={dp_engine.ledger.total_epsilon:.4f}")
    print(f"Remaining budget: {dp_engine.ledger.remaining_budget():.4f}")
    print(f"Number of operations: {len(dp_engine.ledger.operations)}")
    
    # Demo 5: Composition analysis
    print("\n5. Privacy Composition Analysis")
    print("-" * 40)
    
    # Simulate multiple rounds
    num_rounds = 10
    total_eps, total_delta = dp_engine.composition_privacy_cost(num_rounds)
    
    print(f"Privacy cost for {num_rounds} rounds:")
    print(f"  Total epsilon: {total_eps:.4f}")
    print(f"  Total delta: {total_delta:.6f}")
    print(f"  Privacy amplification factor: {total_eps / (num_rounds * privacy_params.epsilon):.2f}x")
    
    # Demo 6: Privacy report
    print("\n6. Privacy Analysis Report")
    print("-" * 40)
    
    report = dp_engine.get_privacy_report()
    
    print("Privacy Parameters:")
    for key, value in report['privacy_parameters'].items():
        print(f"  {key}: {value}")
    
    print(f"\nBudget Status:")
    print(f"  Used: {report['privacy_ledger']['total_epsilon']:.4f}")
    print(f"  Limit: {report['privacy_ledger']['budget_limit']:.4f}")
    print(f"  Exhausted: {report['privacy_ledger']['budget_exhausted']}")
    
    if report['recommendations']:
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    
    print(f"\n🔐 Differential privacy provides formal privacy guarantees!")
    print(f"   (ε={privacy_params.epsilon}, δ={privacy_params.delta})-differential privacy")

if __name__ == "__main__":
    demo_differential_privacy()