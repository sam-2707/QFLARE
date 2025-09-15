"""
FedBN (Federated Batch Normalization) Algorithm Implementation for QFLARE

FedBN addresses the challenge of batch normalization in federated learning where
different clients have different data distributions. It keeps local batch normalization
statistics while sharing other parameters globally.

Reference: "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization"
(Li et al., 2021) https://arxiv.org/abs/2102.07623

Key Features:
- Separate handling of batch normalization layers
- Local BN statistics maintenance across rounds
- Global aggregation of non-BN parameters only
- Support for different normalization layers (BatchNorm1d, BatchNorm2d, etc.)
- Advanced BN statistics analysis and monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
import copy
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class FedBNConfig:
    """Configuration for FedBN algorithm."""
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    momentum: float = 0.9
    weight_decay: float = 1e-4
    
    # BN-specific settings
    bn_track_running_stats: bool = True
    bn_momentum: float = 0.1  # BN momentum for running statistics
    bn_eps: float = 1e-5
    
    # Advanced features
    gradient_clipping: bool = True
    gradient_clip_norm: float = 1.0
    device_sampling_fraction: float = 1.0
    
    # BN analysis settings
    analyze_bn_diversity: bool = True
    bn_stats_history_size: int = 10
    
    # Convergence settings
    convergence_threshold: float = 1e-6
    max_iterations: int = 1000


@dataclass
class BNStatistics:
    """Batch normalization statistics for a single layer."""
    running_mean: torch.Tensor
    running_var: torch.Tensor
    num_batches_tracked: torch.Tensor
    layer_name: str
    layer_type: str  # 'BatchNorm1d', 'BatchNorm2d', etc.
    
    def clone(self) -> 'BNStatistics':
        """Create a deep copy of BN statistics."""
        return BNStatistics(
            running_mean=self.running_mean.clone(),
            running_var=self.running_var.clone(),
            num_batches_tracked=self.num_batches_tracked.clone(),
            layer_name=self.layer_name,
            layer_type=self.layer_type
        )
    
    def compute_distance(self, other: 'BNStatistics') -> float:
        """Compute statistical distance between two BN statistics."""
        if self.layer_name != other.layer_name:
            return float('inf')
        
        mean_dist = torch.norm(self.running_mean - other.running_mean).item()
        var_dist = torch.norm(self.running_var - other.running_var).item()
        
        return mean_dist + var_dist


@dataclass
class FedBNClientState:
    """State information for a FedBN client."""
    device_id: str
    model_state: Dict[str, torch.Tensor]
    bn_statistics: Dict[str, BNStatistics] = field(default_factory=dict)
    bn_stats_history: List[Dict[str, BNStatistics]] = field(default_factory=list)
    optimizer_state: Optional[Dict] = None
    local_loss_history: List[float] = field(default_factory=list)
    data_size: int = 0
    last_participation_round: int = -1
    total_local_iterations: int = 0
    bn_diversity_scores: List[float] = field(default_factory=list)


@dataclass
class FedBNRoundResult:
    """Results from a FedBN training round."""
    round_id: int
    participating_clients: List[str]
    global_loss: float
    local_losses: Dict[str, float]
    convergence_metrics: Dict[str, float]
    aggregation_weights: Dict[str, float]
    round_duration: float
    
    # BN-specific metrics
    bn_diversity_score: float
    bn_layer_count: int
    bn_stats_analysis: Dict[str, Any]
    non_bn_params_count: int
    total_params_count: int


class FedBNParameterSeparator:
    """Separates batch normalization parameters from other model parameters."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.bn_layer_types = {
            'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d',
            'SyncBatchNorm', 'GroupNorm', 'LayerNorm',
            'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d'
        }
    
    def separate_parameters(self, model: nn.Module) -> Tuple[Dict[str, torch.Tensor], 
                                                            Dict[str, torch.Tensor],
                                                            Dict[str, BNStatistics]]:
        """
        Separate model parameters into BN and non-BN parameters.
        
        Returns:
            Tuple of (bn_params, non_bn_params, bn_statistics)
        """
        bn_params = {}
        non_bn_params = {}
        bn_statistics = {}
        
        for name, module in model.named_modules():
            module_type = type(module).__name__
            
            if module_type in self.bn_layer_types:
                # Handle BN parameters
                for param_name, param in module.named_parameters():
                    full_param_name = f"{name}.{param_name}"
                    bn_params[full_param_name] = param.clone().detach()
                
                # Extract BN statistics
                if hasattr(module, 'running_mean') and hasattr(module, 'running_var'):
                    bn_stats = BNStatistics(
                        running_mean=module.running_mean.clone(),
                        running_var=module.running_var.clone(),
                        num_batches_tracked=module.num_batches_tracked.clone() if hasattr(module, 'num_batches_tracked') else torch.tensor(0),
                        layer_name=name,
                        layer_type=module_type
                    )
                    bn_statistics[name] = bn_stats
        
        # Get all non-BN parameters
        for name, param in model.named_parameters():
            if not any(name.startswith(bn_name) for bn_name in bn_params.keys()):
                non_bn_params[name] = param.clone().detach()
        
        self.logger.debug(f"Separated parameters: {len(bn_params)} BN params, "
                         f"{len(non_bn_params)} non-BN params, "
                         f"{len(bn_statistics)} BN layers")
        
        return bn_params, non_bn_params, bn_statistics
    
    def is_bn_parameter(self, param_name: str, model: nn.Module) -> bool:
        """Check if a parameter belongs to a batch normalization layer."""
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if module_type in self.bn_layer_types and param_name.startswith(name):
                return True
        return False


class FedBNAggregator:
    """Aggregator for FedBN that only aggregates non-BN parameters."""
    
    def __init__(self, config: FedBNConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.parameter_separator = FedBNParameterSeparator()
    
    def aggregate_models(self, client_models: Dict[str, Dict[str, torch.Tensor]], 
                        client_data_sizes: Dict[str, int],
                        client_losses: Dict[str, float]) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Aggregate only non-BN parameters from client models.
        
        Returns:
            Tuple of (aggregated_non_bn_params, aggregation_weights)
        """
        if not client_models:
            raise ValueError("No client models to aggregate")
        
        # Separate BN and non-BN parameters for each client
        client_non_bn_params = {}
        
        for client_id, model_params in client_models.items():
            non_bn_params = {}
            for param_name, param_tensor in model_params.items():
                # Simple heuristic: if parameter name contains batch norm keywords
                if not self._is_bn_param_name(param_name):
                    non_bn_params[param_name] = param_tensor
            client_non_bn_params[client_id] = non_bn_params
        
        # Compute aggregation weights
        weights = self._compute_aggregation_weights(client_data_sizes, client_losses)
        
        # Aggregate non-BN parameters only
        aggregated_params = {}
        
        if client_non_bn_params:
            first_client = next(iter(client_non_bn_params.values()))
            
            for param_name in first_client.keys():
                weighted_param = None
                total_weight = 0.0
                
                for client_id, non_bn_params in client_non_bn_params.items():
                    if client_id in weights and param_name in non_bn_params:
                        weight = weights[client_id]
                        param_tensor = non_bn_params[param_name]
                        
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
        
        self.logger.info(f"Aggregated {len(client_models)} client models "
                        f"({len(aggregated_params)} non-BN params) with weights: {weights}")
        
        return aggregated_params, weights
    
    def _is_bn_param_name(self, param_name: str) -> bool:
        """Check if parameter name suggests it belongs to a BN layer."""
        bn_keywords = ['batchnorm', 'batch_norm', 'bn', 'norm']
        param_lower = param_name.lower()
        return any(keyword in param_lower for keyword in bn_keywords)
    
    def _compute_aggregation_weights(self, client_data_sizes: Dict[str, int], 
                                   client_losses: Dict[str, float]) -> Dict[str, float]:
        """Compute aggregation weights based on data size and loss."""
        weights = {}
        
        # Normalize data sizes
        total_data = sum(client_data_sizes.values())
        if total_data == 0:
            # Equal weights if no data size info
            num_clients = len(client_data_sizes)
            return {client_id: 1.0/num_clients for client_id in client_data_sizes.keys()}
        
        for client_id in client_data_sizes.keys():
            # Base weight from data size
            data_weight = client_data_sizes[client_id] / total_data
            
            # Simple adjustment based on loss (could be made more sophisticated)
            if client_id in client_losses:
                # Lower loss gets slightly higher weight
                loss_adjustment = 1.0 / (1.0 + client_losses[client_id])
                loss_adjustment = max(0.5, min(1.5, loss_adjustment))
            else:
                loss_adjustment = 1.0
            
            weights[client_id] = data_weight * loss_adjustment
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights


class FedBNStatisticsAnalyzer:
    """Analyzes batch normalization statistics across clients."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_bn_diversity(self, client_bn_stats: Dict[str, Dict[str, BNStatistics]]) -> Dict[str, Any]:
        """
        Analyze diversity of BN statistics across clients.
        
        Args:
            client_bn_stats: Dict mapping client_id to their BN statistics
            
        Returns:
            Dictionary with diversity analysis results
        """
        analysis = {
            'layer_diversity': {},
            'overall_diversity': 0.0,
            'layer_count': 0,
            'client_count': len(client_bn_stats),
            'mean_distances': {},
            'variance_ratios': {}
        }
        
        if not client_bn_stats:
            return analysis
        
        # Get all layer names
        all_layers = set()
        for client_stats in client_bn_stats.values():
            all_layers.update(client_stats.keys())
        
        analysis['layer_count'] = len(all_layers)
        total_diversity = 0.0
        
        for layer_name in all_layers:
            # Collect statistics for this layer across clients
            layer_stats = []
            for client_id, client_stats in client_bn_stats.items():
                if layer_name in client_stats:
                    layer_stats.append(client_stats[layer_name])
            
            if len(layer_stats) < 2:
                continue
            
            # Compute pairwise distances
            distances = []
            for i in range(len(layer_stats)):
                for j in range(i + 1, len(layer_stats)):
                    dist = layer_stats[i].compute_distance(layer_stats[j])
                    if dist != float('inf'):
                        distances.append(dist)
            
            if distances:
                layer_diversity = np.mean(distances)
                analysis['layer_diversity'][layer_name] = layer_diversity
                analysis['mean_distances'][layer_name] = np.mean(distances)
                total_diversity += layer_diversity
                
                # Compute variance ratios
                running_means = [stats.running_mean for stats in layer_stats]
                running_vars = [stats.running_var for stats in layer_stats]
                
                if running_means:
                    mean_var = torch.var(torch.stack(running_means), dim=0).mean().item()
                    var_var = torch.var(torch.stack(running_vars), dim=0).mean().item()
                    analysis['variance_ratios'][layer_name] = {
                        'mean_variance': mean_var,
                        'var_variance': var_var
                    }
        
        # Overall diversity score
        if analysis['layer_diversity']:
            analysis['overall_diversity'] = total_diversity / len(analysis['layer_diversity'])
        
        self.logger.debug(f"BN diversity analysis: {len(all_layers)} layers, "
                         f"overall diversity: {analysis['overall_diversity']:.6f}")
        
        return analysis
    
    def compute_bn_drift(self, current_stats: Dict[str, BNStatistics], 
                        previous_stats: Dict[str, BNStatistics]) -> Dict[str, float]:
        """Compute drift in BN statistics between rounds."""
        drift_scores = {}
        
        for layer_name in current_stats.keys():
            if layer_name in previous_stats:
                current = current_stats[layer_name]
                previous = previous_stats[layer_name]
                drift = current.compute_distance(previous)
                drift_scores[layer_name] = drift
        
        return drift_scores


class FedBNAlgorithm:
    """Main FedBN algorithm implementation."""
    
    def __init__(self, config: FedBNConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Algorithm components
        self.parameter_separator = FedBNParameterSeparator()
        self.aggregator = FedBNAggregator(config)
        self.bn_analyzer = FedBNStatisticsAnalyzer()
        
        # State tracking
        self.global_non_bn_params: Optional[Dict[str, torch.Tensor]] = None
        self.client_states: Dict[str, FedBNClientState] = {}
        self.round_history: List[FedBNRoundResult] = []
        self.current_round = 0
        
        # BN statistics tracking
        self.global_bn_layer_names: Set[str] = set()
    
    def initialize_global_model(self, model_template: nn.Module):
        """Initialize the global model parameters (non-BN only)."""
        bn_params, non_bn_params, bn_statistics = self.parameter_separator.separate_parameters(model_template)
        
        self.global_non_bn_params = non_bn_params
        self.global_bn_layer_names = set(bn_statistics.keys())
        
        self.logger.info(f"Global model initialized for FedBN: "
                        f"{len(non_bn_params)} non-BN params, "
                        f"{len(bn_statistics)} BN layers")
    
    def register_client(self, client_id: str, data_size: int):
        """Register a new client with the FedBN algorithm."""
        self.client_states[client_id] = FedBNClientState(
            device_id=client_id,
            model_state={},
            data_size=data_size
        )
        self.logger.info(f"Registered client {client_id} with data size {data_size}")
    
    def client_update(self, client_id: str, model: nn.Module, 
                     train_loader: torch.utils.data.DataLoader,
                     loss_fn: nn.Module) -> Tuple[Dict[str, torch.Tensor], float, Dict[str, Any]]:
        """
        Perform local training for a client using FedBN.
        
        Returns:
            Tuple of (updated_model_params, local_loss, metrics)
        """
        if client_id not in self.client_states:
            raise ValueError(f"Client {client_id} not registered")
        
        if self.global_non_bn_params is None:
            raise ValueError("Global model not initialized")
        
        client_state = self.client_states[client_id]
        
        # Set model to training mode
        model.train()
        
        # Load global non-BN parameters, keep local BN parameters
        self._load_global_non_bn_params(model)
        
        # If client has previous BN statistics, restore them
        if client_state.bn_statistics:
            self._restore_client_bn_stats(model, client_state.bn_statistics)
        
        # Create optimizer
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        # Training metrics
        epoch_losses = []
        total_samples = 0
        
        start_time = time.time()
        
        # Local training loop
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(data)
                loss = loss_fn(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Apply gradient clipping if enabled
                if self.config.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.config.gradient_clip_norm
                    )
                
                optimizer.step()
                
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
        
        # Extract updated model state and BN statistics
        bn_params, non_bn_params, bn_statistics = self.parameter_separator.separate_parameters(model)
        
        # Update client state
        client_state.model_state = {**non_bn_params, **bn_params}  # Store both for completeness
        client_state.bn_statistics = bn_statistics
        
        # Update BN statistics history
        if self.config.analyze_bn_diversity:
            client_state.bn_stats_history.append(copy.deepcopy(bn_statistics))
            if len(client_state.bn_stats_history) > self.config.bn_stats_history_size:
                client_state.bn_stats_history.pop(0)
        
        # Compute final metrics
        final_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        client_state.local_loss_history.append(final_loss)
        client_state.last_participation_round = self.current_round
        
        # Prepare metrics
        metrics = {
            'training_time': training_time,
            'total_samples': total_samples,
            'local_epochs': self.config.local_epochs,
            'epoch_losses': epoch_losses,
            'bn_layer_count': len(bn_statistics),
            'non_bn_param_count': len(non_bn_params)
        }
        
        self.logger.info(f"Client {client_id} training completed: "
                        f"loss = {final_loss:.6f}, time = {training_time:.2f}s, "
                        f"BN layers = {len(bn_statistics)}")
        
        return client_state.model_state, final_loss, metrics
    
    def federated_averaging(self, participating_clients: List[str]) -> FedBNRoundResult:
        """
        Perform federated averaging for the current round (non-BN params only).
        
        Args:
            participating_clients: List of client IDs that participated in this round
            
        Returns:
            FedBNRoundResult with round statistics
        """
        start_time = time.time()
        
        # Collect client models and metadata
        client_models = {}
        client_data_sizes = {}
        client_losses = {}
        client_bn_stats = {}
        
        for client_id in participating_clients:
            if client_id in self.client_states:
                client_state = self.client_states[client_id]
                client_models[client_id] = client_state.model_state
                client_data_sizes[client_id] = client_state.data_size
                client_bn_stats[client_id] = client_state.bn_statistics
                
                if client_state.local_loss_history:
                    client_losses[client_id] = client_state.local_loss_history[-1]
                else:
                    client_losses[client_id] = float('inf')
        
        if not client_models:
            raise ValueError("No client models available for aggregation")
        
        # Perform aggregation (non-BN parameters only)
        aggregated_non_bn_params, aggregation_weights = self.aggregator.aggregate_models(
            client_models, client_data_sizes, client_losses
        )
        
        # Update global non-BN parameters
        self.global_non_bn_params = aggregated_non_bn_params
        
        # Analyze BN statistics
        if self.config.analyze_bn_diversity:
            bn_analysis = self.bn_analyzer.analyze_bn_diversity(client_bn_stats)
        else:
            bn_analysis = {'overall_diversity': 0.0, 'layer_count': 0}
        
        # Compute round metrics
        global_loss = np.mean(list(client_losses.values()))
        
        # Count parameters
        non_bn_count = len(aggregated_non_bn_params)
        total_count = sum(len(model_state) for model_state in client_models.values())
        total_count = total_count // len(client_models) if client_models else 0
        
        # Convergence metrics
        convergence_metrics = self._compute_convergence_metrics()
        
        round_duration = time.time() - start_time
        
        # Create round result
        round_result = FedBNRoundResult(
            round_id=self.current_round,
            participating_clients=participating_clients,
            global_loss=global_loss,
            local_losses=client_losses,
            convergence_metrics=convergence_metrics,
            aggregation_weights=aggregation_weights,
            round_duration=round_duration,
            bn_diversity_score=bn_analysis['overall_diversity'],
            bn_layer_count=bn_analysis['layer_count'],
            bn_stats_analysis=bn_analysis,
            non_bn_params_count=non_bn_count,
            total_params_count=total_count
        )
        
        self.round_history.append(round_result)
        self.current_round += 1
        
        self.logger.info(f"Round {round_result.round_id} completed: "
                        f"global_loss = {global_loss:.6f}, "
                        f"participants = {len(participating_clients)}, "
                        f"BN diversity = {bn_analysis['overall_diversity']:.6f}, "
                        f"BN layers = {bn_analysis['layer_count']}")
        
        return round_result
    
    def _load_global_non_bn_params(self, model: nn.Module):
        """Load global non-BN parameters into the model."""
        if self.global_non_bn_params is None:
            return
        
        model_state = model.state_dict()
        for param_name, param_value in self.global_non_bn_params.items():
            if param_name in model_state:
                model_state[param_name].copy_(param_value)
    
    def _restore_client_bn_stats(self, model: nn.Module, bn_statistics: Dict[str, BNStatistics]):
        """Restore client's BN statistics to the model."""
        for name, module in model.named_modules():
            if name in bn_statistics:
                stats = bn_statistics[name]
                if hasattr(module, 'running_mean'):
                    module.running_mean.copy_(stats.running_mean)
                if hasattr(module, 'running_var'):
                    module.running_var.copy_(stats.running_var)
                if hasattr(module, 'num_batches_tracked'):
                    module.num_batches_tracked.copy_(stats.num_batches_tracked)
    
    def _compute_convergence_metrics(self) -> Dict[str, float]:
        """Compute convergence metrics for the algorithm."""
        metrics = {}
        
        if len(self.round_history) < 2:
            return {'loss_improvement': 0.0, 'convergence_rate': 0.0}
        
        # Loss improvement
        current_loss = self.round_history[-1].global_loss
        previous_loss = self.round_history[-2].global_loss
        loss_improvement = previous_loss - current_loss
        
        # Convergence rate
        if len(self.round_history) >= 5:
            recent_losses = [r.global_loss for r in self.round_history[-5:]]
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            convergence_rate = -loss_trend
        else:
            convergence_rate = loss_improvement
        
        metrics['loss_improvement'] = loss_improvement
        metrics['convergence_rate'] = convergence_rate
        
        return metrics
    
    def get_global_model_params(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get the current global non-BN model parameters."""
        return self.global_non_bn_params
    
    def get_client_state(self, client_id: str) -> Optional[FedBNClientState]:
        """Get the state of a specific client."""
        return self.client_states.get(client_id)
    
    def get_algorithm_stats(self) -> Dict[str, Any]:
        """Get comprehensive algorithm statistics."""
        stats = {
            'current_round': self.current_round,
            'total_clients': len(self.client_states),
            'bn_layer_count': len(self.global_bn_layer_names),
            'round_count': len(self.round_history)
        }
        
        if self.round_history:
            latest_round = self.round_history[-1]
            stats.update({
                'latest_global_loss': latest_round.global_loss,
                'latest_bn_diversity': latest_round.bn_diversity_score,
                'avg_round_duration': np.mean([r.round_duration for r in self.round_history]),
                'total_params': latest_round.total_params_count,
                'non_bn_params': latest_round.non_bn_params_count
            })
            
            # BN diversity trend
            bn_diversities = [r.bn_diversity_score for r in self.round_history]
            if len(bn_diversities) >= 2:
                stats['bn_diversity_trend'] = bn_diversities[-1] - bn_diversities[-2]
        
        return stats