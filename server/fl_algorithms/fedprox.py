"""
FedProx Algorithm Implementation for QFLARE

FedProx addresses the challenge of statistical heterogeneity (non-IID data) in federated learning
by adding a proximal term to the local objective function. This helps maintain consistency
between local and global models while allowing for some degree of personalization.

Reference: "Federated Optimization in Heterogeneous Networks" (Li et al., 2018)
https://arxiv.org/abs/1812.06127

Key Features:
- Proximal term μ/2 * ||w - w_global||² added to local loss
- Handles non-IID data distribution across devices
- Configurable proximal parameter μ for different heterogeneity levels
- Support for partial local updates (variable local epochs)
- Advanced convergence monitoring and adaptive μ tuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import copy
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class FedProxConfig:
    """Configuration for FedProx algorithm."""
    mu: float = 0.01  # Proximal term coefficient
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    adaptive_mu: bool = True  # Adaptively tune μ based on heterogeneity
    min_mu: float = 0.001
    max_mu: float = 1.0
    mu_adjustment_factor: float = 1.2
    convergence_threshold: float = 1e-6
    max_iterations: int = 1000
    device_sampling_fraction: float = 1.0  # Fraction of devices to sample each round
    gradient_clipping: bool = True
    gradient_clip_norm: float = 1.0
    momentum: float = 0.9
    weight_decay: float = 1e-4


@dataclass
class FedProxClientState:
    """State information for a FedProx client."""
    device_id: str
    model_state: Dict[str, torch.Tensor]
    optimizer_state: Optional[Dict] = None
    local_loss_history: List[float] = field(default_factory=list)
    gradient_norm_history: List[float] = field(default_factory=list)
    data_size: int = 0
    heterogeneity_score: float = 0.0
    last_participation_round: int = -1
    total_local_iterations: int = 0


@dataclass
class FedProxRoundResult:
    """Results from a FedProx training round."""
    round_id: int
    participating_clients: List[str]
    global_loss: float
    local_losses: Dict[str, float]
    convergence_metrics: Dict[str, float]
    mu_values: Dict[str, float]
    aggregation_weights: Dict[str, float]
    round_duration: float
    gradient_diversity: float
    heterogeneity_estimate: float


class FedProxLoss(nn.Module):
    """Custom loss function for FedProx with proximal term."""
    
    def __init__(self, base_loss_fn: nn.Module, mu: float, global_model_params: Dict[str, torch.Tensor]):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.mu = mu
        self.global_model_params = global_model_params
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, 
                local_model_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute FedProx loss with proximal term."""
        # Base loss (e.g., cross-entropy)
        base_loss = self.base_loss_fn(outputs, targets)
        
        # Proximal term: μ/2 * ||w - w_global||²
        proximal_term = 0.0
        for name, local_param in local_model_params.items():
            if name in self.global_model_params:
                global_param = self.global_model_params[name]
                proximal_term += torch.norm(local_param - global_param, p=2) ** 2
        
        proximal_term = (self.mu / 2.0) * proximal_term
        
        total_loss = base_loss + proximal_term
        return total_loss


class FedProxOptimizer:
    """Custom optimizer for FedProx algorithm."""
    
    def __init__(self, model: nn.Module, config: FedProxConfig, 
                 global_model_params: Dict[str, torch.Tensor]):
        self.model = model
        self.config = config
        self.global_model_params = global_model_params
        
        # Create base optimizer
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        # Track gradients for diversity computation
        self.gradient_history = []
    
    def step(self, loss: torch.Tensor):
        """Perform one optimization step with gradient clipping."""
        # Compute gradients
        loss.backward()
        
        # Apply gradient clipping if enabled
        if self.config.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip_norm
            )
        
        # Store gradient norms for analysis
        total_grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        self.gradient_history.append(total_grad_norm)
        
        # Perform optimization step
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def get_gradient_diversity(self) -> float:
        """Compute gradient diversity metric."""
        if len(self.gradient_history) < 2:
            return 0.0
        
        grad_norms = np.array(self.gradient_history[-10:])  # Last 10 gradients
        return np.std(grad_norms) / (np.mean(grad_norms) + 1e-8)


class FedProxAggregator:
    """Advanced aggregation strategy for FedProx."""
    
    def __init__(self, config: FedProxConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def aggregate_models(self, client_models: Dict[str, Dict[str, torch.Tensor]], 
                        client_data_sizes: Dict[str, int],
                        client_losses: Dict[str, float]) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Aggregate client models using advanced weighting strategies.
        
        Returns:
            Tuple of (aggregated_model_params, aggregation_weights)
        """
        if not client_models:
            raise ValueError("No client models to aggregate")
        
        # Compute aggregation weights
        weights = self._compute_aggregation_weights(client_data_sizes, client_losses)
        
        # Aggregate model parameters
        aggregated_params = {}
        
        # Get parameter names from first client
        first_client = next(iter(client_models.values()))
        
        for param_name in first_client.keys():
            weighted_param = None
            total_weight = 0.0
            
            for client_id, model_params in client_models.items():
                if client_id in weights:
                    weight = weights[client_id]
                    param_tensor = model_params[param_name]
                    
                    if weighted_param is None:
                        weighted_param = weight * param_tensor
                    else:
                        weighted_param += weight * param_tensor
                    
                    total_weight += weight
            
            # Normalize by total weight
            if total_weight > 0:
                aggregated_params[param_name] = weighted_param / total_weight
            else:
                aggregated_params[param_name] = weighted_param
        
        self.logger.info(f"Aggregated {len(client_models)} client models with weights: {weights}")
        
        return aggregated_params, weights
    
    def _compute_aggregation_weights(self, client_data_sizes: Dict[str, int], 
                                   client_losses: Dict[str, float]) -> Dict[str, float]:
        """Compute advanced aggregation weights considering data size and loss."""
        weights = {}
        
        # Normalize data sizes
        total_data = sum(client_data_sizes.values())
        if total_data == 0:
            # Equal weights if no data size info
            num_clients = len(client_data_sizes)
            return {client_id: 1.0/num_clients for client_id in client_data_sizes.keys()}
        
        # Compute loss-based adjustment
        losses = list(client_losses.values())
        if len(losses) > 1:
            loss_std = np.std(losses)
            loss_mean = np.mean(losses)
        else:
            loss_std = 0.0
            loss_mean = losses[0] if losses else 1.0
        
        for client_id in client_data_sizes.keys():
            # Base weight from data size
            data_weight = client_data_sizes[client_id] / total_data
            
            # Adjustment based on loss (lower loss gets higher weight)
            if client_id in client_losses and loss_std > 0:
                loss_adjustment = 1.0 - (client_losses[client_id] - loss_mean) / (3 * loss_std)
                loss_adjustment = max(0.1, min(2.0, loss_adjustment))  # Clamp adjustment
            else:
                loss_adjustment = 1.0
            
            weights[client_id] = data_weight * loss_adjustment
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights


class FedProxHeterogeneityEstimator:
    """Estimates data heterogeneity across clients for adaptive μ tuning."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gradient_history = {}
        self.loss_history = {}
    
    def update_client_info(self, client_id: str, gradients: Dict[str, torch.Tensor], 
                          loss: float):
        """Update client gradient and loss information."""
        if client_id not in self.gradient_history:
            self.gradient_history[client_id] = []
            self.loss_history[client_id] = []
        
        # Store flattened gradient norm
        grad_norm = 0.0
        for grad in gradients.values():
            grad_norm += torch.norm(grad).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        self.gradient_history[client_id].append(grad_norm)
        self.loss_history[client_id].append(loss)
        
        # Keep only recent history
        max_history = 10
        if len(self.gradient_history[client_id]) > max_history:
            self.gradient_history[client_id] = self.gradient_history[client_id][-max_history:]
            self.loss_history[client_id] = self.loss_history[client_id][-max_history:]
    
    def estimate_heterogeneity(self) -> float:
        """Estimate overall heterogeneity score [0, 1]."""
        if len(self.gradient_history) < 2:
            return 0.5  # Default moderate heterogeneity
        
        # Compute gradient diversity
        all_grad_norms = []
        for client_grads in self.gradient_history.values():
            if client_grads:
                all_grad_norms.extend(client_grads[-3:])  # Last 3 gradients per client
        
        if len(all_grad_norms) < 2:
            return 0.5
        
        grad_diversity = np.std(all_grad_norms) / (np.mean(all_grad_norms) + 1e-8)
        
        # Compute loss diversity
        all_losses = []
        for client_losses in self.loss_history.values():
            if client_losses:
                all_losses.extend(client_losses[-3:])
        
        if len(all_losses) < 2:
            loss_diversity = 0.0
        else:
            loss_diversity = np.std(all_losses) / (np.mean(all_losses) + 1e-8)
        
        # Combine diversity measures
        heterogeneity = (grad_diversity + loss_diversity) / 2.0
        heterogeneity = min(1.0, max(0.0, heterogeneity))  # Clamp to [0, 1]
        
        self.logger.debug(f"Heterogeneity estimate: {heterogeneity:.4f} "
                         f"(grad_div: {grad_diversity:.4f}, loss_div: {loss_diversity:.4f})")
        
        return heterogeneity


class FedProxAlgorithm:
    """Main FedProx algorithm implementation."""
    
    def __init__(self, config: FedProxConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Algorithm components
        self.aggregator = FedProxAggregator(config)
        self.heterogeneity_estimator = FedProxHeterogeneityEstimator()
        
        # State tracking
        self.global_model_params: Optional[Dict[str, torch.Tensor]] = None
        self.client_states: Dict[str, FedProxClientState] = {}
        self.round_history: List[FedProxRoundResult] = []
        self.current_round = 0
        
        # Adaptive μ tracking
        self.current_mu = config.mu
        self.mu_history = []
    
    def initialize_global_model(self, model_template: nn.Module):
        """Initialize the global model parameters."""
        self.global_model_params = {
            name: param.clone().detach()
            for name, param in model_template.named_parameters()
        }
        self.logger.info("Global model initialized for FedProx")
    
    def register_client(self, client_id: str, data_size: int):
        """Register a new client with the FedProx algorithm."""
        self.client_states[client_id] = FedProxClientState(
            device_id=client_id,
            model_state={},
            data_size=data_size
        )
        self.logger.info(f"Registered client {client_id} with data size {data_size}")
    
    def client_update(self, client_id: str, model: nn.Module, 
                     train_loader: torch.utils.data.DataLoader,
                     loss_fn: nn.Module) -> Tuple[Dict[str, torch.Tensor], float, Dict[str, Any]]:
        """
        Perform local training for a client using FedProx.
        
        Returns:
            Tuple of (updated_model_params, local_loss, metrics)
        """
        if client_id not in self.client_states:
            raise ValueError(f"Client {client_id} not registered")
        
        if self.global_model_params is None:
            raise ValueError("Global model not initialized")
        
        client_state = self.client_states[client_id]
        
        # Set model to training mode
        model.train()
        
        # Load global model parameters
        for name, param in model.named_parameters():
            if name in self.global_model_params:
                param.data.copy_(self.global_model_params[name])
        
        # Create FedProx loss function
        fedprox_loss = FedProxLoss(loss_fn, self.current_mu, self.global_model_params)
        
        # Create FedProx optimizer
        optimizer = FedProxOptimizer(model, self.config, self.global_model_params)
        
        # Training metrics
        epoch_losses = []
        total_samples = 0
        
        start_time = time.time()
        
        # Local training loop
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                # Forward pass
                outputs = model(data)
                
                # Get current model parameters for proximal term
                current_params = {name: param for name, param in model.named_parameters()}
                
                # Compute FedProx loss
                loss = fedprox_loss(outputs, targets, current_params)
                
                # Optimization step
                optimizer.step(loss)
                
                # Track metrics
                epoch_loss += loss.item() * len(data)
                epoch_samples += len(data)
                total_samples += len(data)
                client_state.total_local_iterations += 1
            
            avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
            epoch_losses.append(avg_epoch_loss)
            
            self.logger.debug(f"Client {client_id} epoch {epoch+1}/{self.config.local_epochs}: "
                            f"loss = {avg_epoch_loss:.6f}")
        
        training_time = time.time() - start_time
        
        # Compute final metrics
        final_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        gradient_diversity = optimizer.get_gradient_diversity()
        
        # Update client state
        client_state.model_state = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
        client_state.local_loss_history.append(final_loss)
        client_state.gradient_norm_history.extend(optimizer.gradient_history)
        client_state.last_participation_round = self.current_round
        
        # Prepare metrics
        metrics = {
            'training_time': training_time,
            'total_samples': total_samples,
            'gradient_diversity': gradient_diversity,
            'mu_value': self.current_mu,
            'local_epochs': self.config.local_epochs,
            'epoch_losses': epoch_losses
        }
        
        self.logger.info(f"Client {client_id} training completed: "
                        f"loss = {final_loss:.6f}, time = {training_time:.2f}s")
        
        return client_state.model_state, final_loss, metrics
    
    def federated_averaging(self, participating_clients: List[str]) -> FedProxRoundResult:
        """
        Perform federated averaging for the current round.
        
        Args:
            participating_clients: List of client IDs that participated in this round
            
        Returns:
            FedProxRoundResult with round statistics
        """
        start_time = time.time()
        
        # Collect client models and metadata
        client_models = {}
        client_data_sizes = {}
        client_losses = {}
        
        for client_id in participating_clients:
            if client_id in self.client_states:
                client_state = self.client_states[client_id]
                client_models[client_id] = client_state.model_state
                client_data_sizes[client_id] = client_state.data_size
                
                if client_state.local_loss_history:
                    client_losses[client_id] = client_state.local_loss_history[-1]
                else:
                    client_losses[client_id] = float('inf')
        
        if not client_models:
            raise ValueError("No client models available for aggregation")
        
        # Perform aggregation
        aggregated_params, aggregation_weights = self.aggregator.aggregate_models(
            client_models, client_data_sizes, client_losses
        )
        
        # Update global model
        self.global_model_params = aggregated_params
        
        # Compute round metrics
        global_loss = np.mean(list(client_losses.values()))
        gradient_diversity = self._compute_gradient_diversity(participating_clients)
        heterogeneity_estimate = self.heterogeneity_estimator.estimate_heterogeneity()
        
        # Adaptive μ tuning
        if self.config.adaptive_mu:
            self._update_mu(heterogeneity_estimate)
        
        # Convergence check
        convergence_metrics = self._compute_convergence_metrics()
        
        round_duration = time.time() - start_time
        
        # Create round result
        round_result = FedProxRoundResult(
            round_id=self.current_round,
            participating_clients=participating_clients,
            global_loss=global_loss,
            local_losses=client_losses,
            convergence_metrics=convergence_metrics,
            mu_values={client_id: self.current_mu for client_id in participating_clients},
            aggregation_weights=aggregation_weights,
            round_duration=round_duration,
            gradient_diversity=gradient_diversity,
            heterogeneity_estimate=heterogeneity_estimate
        )
        
        self.round_history.append(round_result)
        self.current_round += 1
        
        self.logger.info(f"Round {round_result.round_id} completed: "
                        f"global_loss = {global_loss:.6f}, "
                        f"participants = {len(participating_clients)}, "
                        f"μ = {self.current_mu:.6f}, "
                        f"heterogeneity = {heterogeneity_estimate:.4f}")
        
        return round_result
    
    def _compute_gradient_diversity(self, participating_clients: List[str]) -> float:
        """Compute gradient diversity across participating clients."""
        all_grad_norms = []
        
        for client_id in participating_clients:
            if client_id in self.client_states:
                client_state = self.client_states[client_id]
                if client_state.gradient_norm_history:
                    # Use recent gradient norms
                    recent_norms = client_state.gradient_norm_history[-3:]
                    all_grad_norms.extend(recent_norms)
        
        if len(all_grad_norms) < 2:
            return 0.0
        
        return np.std(all_grad_norms) / (np.mean(all_grad_norms) + 1e-8)
    
    def _update_mu(self, heterogeneity_estimate: float):
        """Adaptively update the proximal parameter μ based on heterogeneity."""
        # Increase μ for higher heterogeneity, decrease for lower
        target_mu = self.config.mu * (1.0 + heterogeneity_estimate)
        
        # Gradual adjustment
        adjustment_factor = self.config.mu_adjustment_factor
        if target_mu > self.current_mu:
            self.current_mu = min(target_mu, self.current_mu * adjustment_factor)
        else:
            self.current_mu = max(target_mu, self.current_mu / adjustment_factor)
        
        # Clamp to valid range
        self.current_mu = max(self.config.min_mu, min(self.config.max_mu, self.current_mu))
        
        self.mu_history.append(self.current_mu)
        
        self.logger.debug(f"Updated μ: {self.current_mu:.6f} (heterogeneity: {heterogeneity_estimate:.4f})")
    
    def _compute_convergence_metrics(self) -> Dict[str, float]:
        """Compute convergence metrics for the algorithm."""
        metrics = {}
        
        if len(self.round_history) < 2:
            return {'loss_improvement': 0.0, 'convergence_rate': 0.0}
        
        # Loss improvement
        current_loss = self.round_history[-1].global_loss
        previous_loss = self.round_history[-2].global_loss
        loss_improvement = previous_loss - current_loss
        
        # Convergence rate (exponential moving average of loss improvements)
        if len(self.round_history) >= 5:
            recent_losses = [r.global_loss for r in self.round_history[-5:]]
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            convergence_rate = -loss_trend  # Negative slope means convergence
        else:
            convergence_rate = loss_improvement
        
        metrics['loss_improvement'] = loss_improvement
        metrics['convergence_rate'] = convergence_rate
        
        return metrics
    
    def get_global_model_params(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get the current global model parameters."""
        return self.global_model_params
    
    def get_client_state(self, client_id: str) -> Optional[FedProxClientState]:
        """Get the state of a specific client."""
        return self.client_states.get(client_id)
    
    def get_algorithm_stats(self) -> Dict[str, Any]:
        """Get comprehensive algorithm statistics."""
        stats = {
            'current_round': self.current_round,
            'total_clients': len(self.client_states),
            'current_mu': self.current_mu,
            'mu_history': self.mu_history.copy(),
            'round_count': len(self.round_history)
        }
        
        if self.round_history:
            latest_round = self.round_history[-1]
            stats.update({
                'latest_global_loss': latest_round.global_loss,
                'latest_heterogeneity': latest_round.heterogeneity_estimate,
                'latest_gradient_diversity': latest_round.gradient_diversity,
                'avg_round_duration': np.mean([r.round_duration for r in self.round_history])
            })
        
        return stats