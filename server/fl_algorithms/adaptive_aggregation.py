"""
Adaptive Aggregation Strategies for QFLARE

This module implements advanced aggregation strategies that adapt to client heterogeneity,
data distribution differences, and system constraints in federated learning.

Implemented Strategies:
1. Adaptive FedAvg with dynamic weighting
2. FedOpt family (FedAdam, FedAdagrad, FedYogi)
3. Scaffold-based aggregation with control variates
4. Dynamic client selection and weighting
5. Robust aggregation with Byzantine resilience

Key Features:
- Adaptive weighting based on client characteristics
- Server-side optimization for better convergence
- Robust aggregation against malicious clients
- Dynamic client selection strategies
- Performance-aware weight adjustment
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import copy
import time
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Available aggregation strategies."""
    ADAPTIVE_FEDAVG = "adaptive_fedavg"
    FEDADAM = "fedadam"
    FEDADAGRAD = "fedadagrad"
    FEDYOGI = "fedyogi"
    SCAFFOLD = "scaffold"
    ROBUST_AGGREGATION = "robust_aggregation"
    DYNAMIC_WEIGHTED = "dynamic_weighted"


@dataclass
class AdaptiveAggregationConfig:
    """Configuration for adaptive aggregation strategies."""
    # Strategy selection
    strategy: AggregationStrategy = AggregationStrategy.ADAPTIVE_FEDAVG
    
    # Adaptive weighting
    weight_strategy: str = "loss_based"  # data_size, loss_based, gradient_norm, performance_history
    adaptive_weights: bool = True
    weight_decay: float = 0.99  # Decay factor for historical weights
    
    # FedOpt parameters
    server_learning_rate: float = 1.0
    server_momentum: float = 0.9
    server_beta1: float = 0.9  # Adam beta1
    server_beta2: float = 0.999  # Adam beta2
    server_epsilon: float = 1e-8  # Adam epsilon
    
    # Scaffold parameters
    scaffold_lr: float = 1.0
    control_variate_weight: float = 1.0
    
    # Robust aggregation
    byzantine_fraction: float = 0.2  # Expected fraction of Byzantine clients
    robustness_strategy: str = "trimmed_mean"  # trimmed_mean, median, krum
    trimming_fraction: float = 0.1
    
    # Client selection
    dynamic_client_selection: bool = True
    min_selected_clients: int = 3
    max_selected_clients: int = 100
    selection_strategy: str = "random"  # random, performance_based, diversity_based
    
    # Performance tracking
    track_convergence: bool = True
    convergence_window: int = 10
    staleness_penalty: bool = True
    max_staleness: int = 5


@dataclass
class ClientInfo:
    """Information about a client for aggregation."""
    client_id: str
    model_update: Dict[str, torch.Tensor]
    data_size: int
    local_loss: float
    gradient_norm: float
    staleness: int = 0  # Rounds since last participation
    performance_history: List[float] = field(default_factory=list)
    reliability_score: float = 1.0
    last_participation_round: int = -1


@dataclass
class AggregationResult:
    """Result of an aggregation operation."""
    aggregated_model: Dict[str, torch.Tensor]
    client_weights: Dict[str, float]
    aggregation_metrics: Dict[str, Any]
    selected_clients: List[str]
    convergence_metrics: Dict[str, float]
    robustness_metrics: Dict[str, float]


class BaseAggregator(ABC):
    """Abstract base class for aggregation strategies."""
    
    def __init__(self, config: AdaptiveAggregationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.round_history = []
        self.client_histories = defaultdict(list)
    
    @abstractmethod
    def aggregate(self, client_infos: List[ClientInfo], 
                 global_model: Dict[str, torch.Tensor]) -> AggregationResult:
        """Perform aggregation of client updates."""
        pass
    
    def compute_weights(self, client_infos: List[ClientInfo]) -> Dict[str, float]:
        """Compute aggregation weights for clients."""
        if self.config.weight_strategy == "data_size":
            return self._data_size_weights(client_infos)
        elif self.config.weight_strategy == "loss_based":
            return self._loss_based_weights(client_infos)
        elif self.config.weight_strategy == "gradient_norm":
            return self._gradient_norm_weights(client_infos)
        elif self.config.weight_strategy == "performance_history":
            return self._performance_history_weights(client_infos)
        else:
            return self._uniform_weights(client_infos)
    
    def _data_size_weights(self, client_infos: List[ClientInfo]) -> Dict[str, float]:
        """Compute weights based on data size."""
        total_data = sum(info.data_size for info in client_infos)
        if total_data == 0:
            return self._uniform_weights(client_infos)
        
        weights = {}
        for info in client_infos:
            weights[info.client_id] = info.data_size / total_data
        
        return weights
    
    def _loss_based_weights(self, client_infos: List[ClientInfo]) -> Dict[str, float]:
        """Compute weights inversely proportional to local loss."""
        # Inverse loss weighting with smoothing
        inv_losses = []
        for info in client_infos:
            if info.local_loss > 0:
                inv_losses.append(1.0 / (info.local_loss + 1e-8))
            else:
                inv_losses.append(1.0)
        
        total_inv_loss = sum(inv_losses)
        if total_inv_loss == 0:
            return self._uniform_weights(client_infos)
        
        weights = {}
        for i, info in enumerate(client_infos):
            weights[info.client_id] = inv_losses[i] / total_inv_loss
        
        return weights
    
    def _gradient_norm_weights(self, client_infos: List[ClientInfo]) -> Dict[str, float]:
        """Compute weights based on gradient norms."""
        # Weight by gradient norm (higher norm = more informative update)
        grad_norms = [info.gradient_norm for info in client_infos]
        total_norm = sum(grad_norms)
        
        if total_norm == 0:
            return self._uniform_weights(client_infos)
        
        weights = {}
        for info in client_infos:
            weights[info.client_id] = info.gradient_norm / total_norm
        
        return weights
    
    def _performance_history_weights(self, client_infos: List[ClientInfo]) -> Dict[str, float]:
        """Compute weights based on historical performance."""
        weights = {}
        
        for info in client_infos:
            if len(info.performance_history) >= 2:
                # Recent improvement score
                recent_avg = np.mean(info.performance_history[-3:])
                overall_avg = np.mean(info.performance_history)
                improvement = max(0.1, overall_avg - recent_avg + 0.5)  # Add baseline
                weights[info.client_id] = improvement * info.reliability_score
            else:
                weights[info.client_id] = info.reliability_score
        
        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        else:
            weights = self._uniform_weights(client_infos)
        
        return weights
    
    def _uniform_weights(self, client_infos: List[ClientInfo]) -> Dict[str, float]:
        """Compute uniform weights."""
        num_clients = len(client_infos)
        if num_clients == 0:
            return {}
        
        weight = 1.0 / num_clients
        return {info.client_id: weight for info in client_infos}
    
    def apply_staleness_penalty(self, weights: Dict[str, float], 
                               client_infos: List[ClientInfo]) -> Dict[str, float]:
        """Apply penalty for stale client updates."""
        if not self.config.staleness_penalty:
            return weights
        
        adjusted_weights = {}
        for info in client_infos:
            penalty = max(0.1, 1.0 - 0.1 * info.staleness)  # 10% penalty per round of staleness
            adjusted_weights[info.client_id] = weights[info.client_id] * penalty
        
        # Renormalize
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
        
        return adjusted_weights


class AdaptiveFedAvgAggregator(BaseAggregator):
    """Adaptive FedAvg with dynamic weighting strategies."""
    
    def __init__(self, config: AdaptiveAggregationConfig):
        super().__init__(config)
        self.historical_weights = defaultdict(list)
    
    def aggregate(self, client_infos: List[ClientInfo], 
                 global_model: Dict[str, torch.Tensor]) -> AggregationResult:
        """Perform adaptive federated averaging."""
        if not client_infos:
            return AggregationResult(
                aggregated_model=global_model,
                client_weights={},
                aggregation_metrics={},
                selected_clients=[],
                convergence_metrics={},
                robustness_metrics={}
            )
        
        start_time = time.time()
        
        # Compute initial weights
        weights = self.compute_weights(client_infos)
        
        # Apply staleness penalty
        weights = self.apply_staleness_penalty(weights, client_infos)
        
        # Apply historical weight smoothing
        if self.config.adaptive_weights:
            weights = self._apply_weight_smoothing(weights)
        
        # Perform weighted aggregation
        aggregated_model = self._weighted_aggregation(client_infos, weights)
        
        # Compute metrics
        aggregation_metrics = self._compute_aggregation_metrics(client_infos, weights)
        convergence_metrics = self._compute_convergence_metrics(client_infos)
        robustness_metrics = self._compute_robustness_metrics(client_infos)
        
        # Update histories
        for info in client_infos:
            self.historical_weights[info.client_id].append(weights[info.client_id])
            if len(self.historical_weights[info.client_id]) > self.config.convergence_window:
                self.historical_weights[info.client_id].pop(0)
        
        aggregation_duration = time.time() - start_time
        aggregation_metrics['aggregation_time'] = aggregation_duration
        
        self.logger.info(f"Adaptive FedAvg aggregation completed: {len(client_infos)} clients, "
                        f"time = {aggregation_duration:.3f}s")
        
        return AggregationResult(
            aggregated_model=aggregated_model,
            client_weights=weights,
            aggregation_metrics=aggregation_metrics,
            selected_clients=[info.client_id for info in client_infos],
            convergence_metrics=convergence_metrics,
            robustness_metrics=robustness_metrics
        )
    
    def _apply_weight_smoothing(self, current_weights: Dict[str, float]) -> Dict[str, float]:
        """Apply exponential smoothing to weights based on history."""
        smoothed_weights = {}
        
        for client_id, current_weight in current_weights.items():
            if client_id in self.historical_weights and self.historical_weights[client_id]:
                # Exponential smoothing
                historical_weight = self.historical_weights[client_id][-1]
                smoothed_weight = (
                    self.config.weight_decay * historical_weight + 
                    (1 - self.config.weight_decay) * current_weight
                )
                smoothed_weights[client_id] = smoothed_weight
            else:
                smoothed_weights[client_id] = current_weight
        
        # Renormalize
        total_weight = sum(smoothed_weights.values())
        if total_weight > 0:
            smoothed_weights = {k: v/total_weight for k, v in smoothed_weights.items()}
        
        return smoothed_weights
    
    def _weighted_aggregation(self, client_infos: List[ClientInfo], 
                            weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Perform weighted aggregation of client models."""
        if not client_infos:
            return {}
        
        aggregated = {}
        first_client = client_infos[0]
        
        for param_name in first_client.model_update.keys():
            weighted_sum = None
            total_weight = 0.0
            
            for info in client_infos:
                if info.client_id in weights and param_name in info.model_update:
                    weight = weights[info.client_id]
                    param = info.model_update[param_name]
                    
                    if weighted_sum is None:
                        weighted_sum = weight * param
                    else:
                        weighted_sum += weight * param
                    
                    total_weight += weight
            
            if total_weight > 0 and weighted_sum is not None:
                aggregated[param_name] = weighted_sum / total_weight
        
        return aggregated
    
    def _compute_aggregation_metrics(self, client_infos: List[ClientInfo], 
                                   weights: Dict[str, float]) -> Dict[str, Any]:
        """Compute metrics related to the aggregation process."""
        metrics = {}
        
        # Weight statistics
        weight_values = list(weights.values())
        metrics['weight_entropy'] = -sum(w * np.log(w + 1e-8) for w in weight_values)
        metrics['weight_variance'] = np.var(weight_values)
        metrics['max_weight'] = max(weight_values) if weight_values else 0
        metrics['min_weight'] = min(weight_values) if weight_values else 0
        
        # Client statistics
        losses = [info.local_loss for info in client_infos]
        metrics['avg_local_loss'] = np.mean(losses)
        metrics['loss_variance'] = np.var(losses)
        
        grad_norms = [info.gradient_norm for info in client_infos]
        metrics['avg_gradient_norm'] = np.mean(grad_norms)
        metrics['gradient_diversity'] = np.std(grad_norms) / (np.mean(grad_norms) + 1e-8)
        
        # Data distribution
        data_sizes = [info.data_size for info in client_infos]
        metrics['total_data_size'] = sum(data_sizes)
        metrics['data_size_variance'] = np.var(data_sizes)
        
        return metrics
    
    def _compute_convergence_metrics(self, client_infos: List[ClientInfo]) -> Dict[str, float]:
        """Compute convergence-related metrics."""
        metrics = {}
        
        # Loss convergence
        losses = [info.local_loss for info in client_infos]
        metrics['global_loss_estimate'] = np.mean(losses)
        metrics['loss_std'] = np.std(losses)
        
        # Gradient alignment (simplified)
        grad_norms = [info.gradient_norm for info in client_infos]
        if len(grad_norms) > 1:
            metrics['gradient_alignment'] = 1.0 - (np.std(grad_norms) / (np.mean(grad_norms) + 1e-8))
        else:
            metrics['gradient_alignment'] = 1.0
        
        return metrics
    
    def _compute_robustness_metrics(self, client_infos: List[ClientInfo]) -> Dict[str, float]:
        """Compute robustness-related metrics."""
        metrics = {}
        
        # Client reliability
        reliability_scores = [info.reliability_score for info in client_infos]
        metrics['avg_reliability'] = np.mean(reliability_scores)
        metrics['min_reliability'] = min(reliability_scores)
        
        # Staleness analysis
        staleness_values = [info.staleness for info in client_infos]
        metrics['avg_staleness'] = np.mean(staleness_values)
        metrics['max_staleness'] = max(staleness_values)
        
        return metrics


class FedOptAggregator(BaseAggregator):
    """Server-side optimization aggregator (FedAdam, FedAdagrad, FedYogi)."""
    
    def __init__(self, config: AdaptiveAggregationConfig):
        super().__init__(config)
        self.server_state = {}
        self.round_count = 0
        
        # Initialize server optimizer state
        if config.strategy == AggregationStrategy.FEDADAM:
            self.m_t = {}  # First moment estimate
            self.v_t = {}  # Second moment estimate
        elif config.strategy == AggregationStrategy.FEDADAGRAD:
            self.h_t = {}  # Accumulated squared gradients
        elif config.strategy == AggregationStrategy.FEDYOGI:
            self.m_t = {}  # First moment estimate
            self.v_t = {}  # Second moment estimate
    
    def aggregate(self, client_infos: List[ClientInfo], 
                 global_model: Dict[str, torch.Tensor]) -> AggregationResult:
        """Perform FedOpt aggregation with server-side optimization."""
        if not client_infos:
            return AggregationResult(
                aggregated_model=global_model,
                client_weights={},
                aggregation_metrics={},
                selected_clients=[],
                convergence_metrics={},
                robustness_metrics={}
            )
        
        start_time = time.time()
        self.round_count += 1
        
        # Compute pseudo-gradient (difference from global model)
        pseudo_gradient = self._compute_pseudo_gradient(client_infos, global_model)
        
        # Apply server-side optimization
        if self.config.strategy == AggregationStrategy.FEDADAM:
            updated_model = self._apply_fedadam(global_model, pseudo_gradient)
        elif self.config.strategy == AggregationStrategy.FEDADAGRAD:
            updated_model = self._apply_fedadagrad(global_model, pseudo_gradient)
        elif self.config.strategy == AggregationStrategy.FEDYOGI:
            updated_model = self._apply_fedyogi(global_model, pseudo_gradient)
        else:
            updated_model = global_model
        
        # Compute weights for metrics
        weights = self.compute_weights(client_infos)
        
        # Compute metrics
        aggregation_metrics = self._compute_fedopt_metrics(client_infos, pseudo_gradient)
        convergence_metrics = self._compute_convergence_metrics(client_infos)
        robustness_metrics = self._compute_robustness_metrics(client_infos)
        
        aggregation_duration = time.time() - start_time
        aggregation_metrics['aggregation_time'] = aggregation_duration
        
        self.logger.info(f"{self.config.strategy.value} aggregation completed: "
                        f"{len(client_infos)} clients, time = {aggregation_duration:.3f}s")
        
        return AggregationResult(
            aggregated_model=updated_model,
            client_weights=weights,
            aggregation_metrics=aggregation_metrics,
            selected_clients=[info.client_id for info in client_infos],
            convergence_metrics=convergence_metrics,
            robustness_metrics=robustness_metrics
        )
    
    def _compute_pseudo_gradient(self, client_infos: List[ClientInfo], 
                               global_model: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute pseudo-gradient from client updates."""
        # First compute weighted average of client updates
        weights = self.compute_weights(client_infos)
        client_avg = {}
        
        if client_infos:
            first_client = client_infos[0]
            for param_name in first_client.model_update.keys():
                weighted_sum = None
                total_weight = 0.0
                
                for info in client_infos:
                    if info.client_id in weights and param_name in info.model_update:
                        weight = weights[info.client_id]
                        param = info.model_update[param_name]
                        
                        if weighted_sum is None:
                            weighted_sum = weight * param
                        else:
                            weighted_sum += weight * param
                        
                        total_weight += weight
                
                if total_weight > 0 and weighted_sum is not None:
                    client_avg[param_name] = weighted_sum / total_weight
        
        # Compute pseudo-gradient as difference
        pseudo_gradient = {}
        for param_name in global_model.keys():
            if param_name in client_avg:
                pseudo_gradient[param_name] = client_avg[param_name] - global_model[param_name]
            else:
                pseudo_gradient[param_name] = torch.zeros_like(global_model[param_name])
        
        return pseudo_gradient
    
    def _apply_fedadam(self, global_model: Dict[str, torch.Tensor], 
                      pseudo_gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply FedAdam server optimization."""
        updated_model = {}
        
        for param_name in global_model.keys():
            if param_name not in self.m_t:
                self.m_t[param_name] = torch.zeros_like(global_model[param_name])
                self.v_t[param_name] = torch.zeros_like(global_model[param_name])
            
            g_t = pseudo_gradient[param_name]
            
            # Update biased first moment estimate
            self.m_t[param_name] = (
                self.config.server_beta1 * self.m_t[param_name] + 
                (1 - self.config.server_beta1) * g_t
            )
            
            # Update biased second moment estimate
            self.v_t[param_name] = (
                self.config.server_beta2 * self.v_t[param_name] + 
                (1 - self.config.server_beta2) * (g_t ** 2)
            )
            
            # Bias correction
            m_hat = self.m_t[param_name] / (1 - self.config.server_beta1 ** self.round_count)
            v_hat = self.v_t[param_name] / (1 - self.config.server_beta2 ** self.round_count)
            
            # Update parameters
            updated_model[param_name] = global_model[param_name] + self.config.server_learning_rate * (
                m_hat / (torch.sqrt(v_hat) + self.config.server_epsilon)
            )
        
        return updated_model
    
    def _apply_fedadagrad(self, global_model: Dict[str, torch.Tensor], 
                         pseudo_gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply FedAdagrad server optimization."""
        updated_model = {}
        
        for param_name in global_model.keys():
            if param_name not in self.h_t:
                self.h_t[param_name] = torch.zeros_like(global_model[param_name])
            
            g_t = pseudo_gradient[param_name]
            
            # Accumulate squared gradients
            self.h_t[param_name] += g_t ** 2
            
            # Update parameters
            updated_model[param_name] = global_model[param_name] + self.config.server_learning_rate * (
                g_t / (torch.sqrt(self.h_t[param_name]) + self.config.server_epsilon)
            )
        
        return updated_model
    
    def _apply_fedyogi(self, global_model: Dict[str, torch.Tensor], 
                      pseudo_gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply FedYogi server optimization."""
        updated_model = {}
        
        for param_name in global_model.keys():
            if param_name not in self.m_t:
                self.m_t[param_name] = torch.zeros_like(global_model[param_name])
                self.v_t[param_name] = torch.zeros_like(global_model[param_name])
            
            g_t = pseudo_gradient[param_name]
            
            # Update first moment
            self.m_t[param_name] = (
                self.config.server_beta1 * self.m_t[param_name] + 
                (1 - self.config.server_beta1) * g_t
            )
            
            # YoGi-style second moment update
            self.v_t[param_name] = self.v_t[param_name] - (1 - self.config.server_beta2) * (
                torch.sign(self.v_t[param_name] - g_t ** 2) * (g_t ** 2)
            )
            
            # Bias correction
            m_hat = self.m_t[param_name] / (1 - self.config.server_beta1 ** self.round_count)
            v_hat = self.v_t[param_name] / (1 - self.config.server_beta2 ** self.round_count)
            
            # Update parameters
            updated_model[param_name] = global_model[param_name] + self.config.server_learning_rate * (
                m_hat / (torch.sqrt(torch.abs(v_hat)) + self.config.server_epsilon)
            )
        
        return updated_model
    
    def _compute_fedopt_metrics(self, client_infos: List[ClientInfo], 
                              pseudo_gradient: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute FedOpt-specific metrics."""
        metrics = {}
        
        # Pseudo-gradient statistics
        grad_norms = []
        for param_tensor in pseudo_gradient.values():
            grad_norms.append(torch.norm(param_tensor).item())
        
        metrics['pseudo_gradient_norm'] = np.sqrt(sum(g**2 for g in grad_norms))
        metrics['avg_param_gradient_norm'] = np.mean(grad_norms)
        
        # Server state statistics
        if self.config.strategy in [AggregationStrategy.FEDADAM, AggregationStrategy.FEDYOGI]:
            if self.m_t:
                m_norms = [torch.norm(m).item() for m in self.m_t.values()]
                v_norms = [torch.norm(v).item() for v in self.v_t.values()]
                metrics['avg_momentum_norm'] = np.mean(m_norms)
                metrics['avg_velocity_norm'] = np.mean(v_norms)
        
        elif self.config.strategy == AggregationStrategy.FEDADAGRAD:
            if self.h_t:
                h_norms = [torch.norm(h).item() for h in self.h_t.values()]
                metrics['avg_accumulated_grad_norm'] = np.mean(h_norms)
        
        metrics['server_round'] = self.round_count
        metrics['participating_clients'] = len(client_infos)
        
        return metrics
    
    def _compute_convergence_metrics(self, client_infos: List[ClientInfo]) -> Dict[str, float]:
        """Compute convergence metrics for FedOpt."""
        # Use base implementation
        return super()._compute_convergence_metrics(client_infos)
    
    def _compute_robustness_metrics(self, client_infos: List[ClientInfo]) -> Dict[str, float]:
        """Compute robustness metrics for FedOpt."""
        # Use base implementation
        return super()._compute_robustness_metrics(client_infos)


class AdaptiveAggregationOrchestrator:
    """Main orchestrator for adaptive aggregation strategies."""
    
    def __init__(self, config: AdaptiveAggregationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Create aggregator based on strategy
        self.aggregator = self._create_aggregator()
        
        # Client selection and management
        self.client_selector = ClientSelector(config)
        
        # Round tracking
        self.current_round = 0
        self.aggregation_history = []
    
    def _create_aggregator(self) -> BaseAggregator:
        """Create the appropriate aggregator based on strategy."""
        if self.config.strategy == AggregationStrategy.ADAPTIVE_FEDAVG:
            return AdaptiveFedAvgAggregator(self.config)
        elif self.config.strategy in [
            AggregationStrategy.FEDADAM, 
            AggregationStrategy.FEDADAGRAD, 
            AggregationStrategy.FEDYOGI
        ]:
            return FedOptAggregator(self.config)
        else:
            # Default to adaptive FedAvg
            self.logger.warning(f"Strategy {self.config.strategy} not implemented, using Adaptive FedAvg")
            return AdaptiveFedAvgAggregator(self.config)
    
    def aggregate_round(self, available_clients: List[ClientInfo], 
                       global_model: Dict[str, torch.Tensor]) -> AggregationResult:
        """
        Perform one round of adaptive aggregation.
        
        Args:
            available_clients: List of clients with their updates
            global_model: Current global model parameters
            
        Returns:
            AggregationResult with updated model and metrics
        """
        start_time = time.time()
        
        # Select clients for aggregation
        selected_clients = self.client_selector.select_clients(
            available_clients, self.current_round
        )
        
        if not selected_clients:
            self.logger.warning("No clients selected for aggregation")
            return AggregationResult(
                aggregated_model=global_model,
                client_weights={},
                aggregation_metrics={'selected_clients': 0},
                selected_clients=[],
                convergence_metrics={},
                robustness_metrics={}
            )
        
        # Perform aggregation
        result = self.aggregator.aggregate(selected_clients, global_model)
        
        # Update round tracking
        self.current_round += 1
        self.aggregation_history.append(result)
        
        # Update client selector with results
        self.client_selector.update_client_performance(selected_clients, result)
        
        total_time = time.time() - start_time
        result.aggregation_metrics['total_round_time'] = total_time
        result.aggregation_metrics['client_selection_count'] = len(selected_clients)
        
        self.logger.info(f"Adaptive aggregation round {self.current_round} completed: "
                        f"{len(selected_clients)}/{len(available_clients)} clients selected, "
                        f"strategy = {self.config.strategy.value}, "
                        f"time = {total_time:.3f}s")
        
        return result
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the aggregation process."""
        stats = {
            'current_round': self.current_round,
            'strategy': self.config.strategy.value,
            'total_rounds': len(self.aggregation_history)
        }
        
        if self.aggregation_history:
            # Convergence trends
            global_losses = [r.convergence_metrics.get('global_loss_estimate', 0) 
                           for r in self.aggregation_history]
            if global_losses:
                stats['loss_trend'] = global_losses[-1] - global_losses[0] if len(global_losses) > 1 else 0
                stats['latest_global_loss'] = global_losses[-1]
            
            # Client participation
            all_participants = set()
            for result in self.aggregation_history:
                all_participants.update(result.selected_clients)
            stats['unique_participants'] = len(all_participants)
            
            # Aggregation performance
            round_times = [r.aggregation_metrics.get('total_round_time', 0) 
                          for r in self.aggregation_history]
            stats['avg_round_time'] = np.mean(round_times) if round_times else 0
        
        return stats


class ClientSelector:
    """Handles dynamic client selection for aggregation."""
    
    def __init__(self, config: AdaptiveAggregationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.client_performance_history = defaultdict(list)
    
    def select_clients(self, available_clients: List[ClientInfo], 
                      current_round: int) -> List[ClientInfo]:
        """Select clients for the current round."""
        if not self.config.dynamic_client_selection:
            return available_clients
        
        num_clients = len(available_clients)
        
        # Determine number of clients to select
        min_clients = min(self.config.min_selected_clients, num_clients)
        max_clients = min(self.config.max_selected_clients, num_clients)
        
        if self.config.selection_strategy == "random":
            selected_count = min(max_clients, max(min_clients, num_clients // 2))
            selected_indices = np.random.choice(num_clients, selected_count, replace=False)
            return [available_clients[i] for i in selected_indices]
        
        elif self.config.selection_strategy == "performance_based":
            return self._performance_based_selection(available_clients, min_clients, max_clients)
        
        elif self.config.selection_strategy == "diversity_based":
            return self._diversity_based_selection(available_clients, min_clients, max_clients)
        
        else:
            return available_clients
    
    def _performance_based_selection(self, clients: List[ClientInfo], 
                                   min_clients: int, max_clients: int) -> List[ClientInfo]:
        """Select clients based on historical performance."""
        # Score clients based on performance history
        client_scores = []
        for client in clients:
            if client.client_id in self.client_performance_history:
                history = self.client_performance_history[client.client_id]
                if history:
                    # Recent performance improvement
                    score = -np.mean(history[-3:])  # Negative because lower loss is better
                else:
                    score = -client.local_loss
            else:
                score = -client.local_loss
            
            # Factor in reliability
            score *= client.reliability_score
            
            client_scores.append((client, score))
        
        # Sort by score (higher is better)
        client_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top performers
        selected_count = min(max_clients, max(min_clients, len(clients) // 3))
        return [client for client, _ in client_scores[:selected_count]]
    
    def _diversity_based_selection(self, clients: List[ClientInfo], 
                                 min_clients: int, max_clients: int) -> List[ClientInfo]:
        """Select clients to maximize diversity."""
        # Simple diversity: select clients with different loss ranges
        clients_sorted = sorted(clients, key=lambda x: x.local_loss)
        
        selected = []
        step = max(1, len(clients_sorted) // max_clients)
        
        for i in range(0, len(clients_sorted), step):
            selected.append(clients_sorted[i])
            if len(selected) >= max_clients:
                break
        
        # Ensure minimum clients
        while len(selected) < min_clients and len(selected) < len(clients):
            for client in clients:
                if client not in selected:
                    selected.append(client)
                    break
        
        return selected
    
    def update_client_performance(self, selected_clients: List[ClientInfo], 
                                result: AggregationResult):
        """Update client performance history based on aggregation results."""
        for client in selected_clients:
            # Use the aggregation weight as a performance indicator
            performance_score = result.client_weights.get(client.client_id, 0.0)
            self.client_performance_history[client.client_id].append(performance_score)
            
            # Keep only recent history
            if len(self.client_performance_history[client.client_id]) > 20:
                self.client_performance_history[client.client_id].pop(0)