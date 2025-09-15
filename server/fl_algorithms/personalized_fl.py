"""
Personalized Federated Learning Algorithms for QFLARE

This module implements various personalized federated learning approaches that adapt
to individual client characteristics while maintaining privacy and collaboration benefits.

Implemented Algorithms:
1. Per-FedAvg: Personalized FedAvg with meta-learning
2. FedPer: Personalized layers + shared layers
3. PFNM: Personalized Federated Neural Matching
4. FedRep: Federated Representation Learning

Key Features:
- Client-specific model personalization
- Adaptive personalization strategies
- Privacy-preserving personalization
- Performance monitoring and adaptation
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

logger = logging.getLogger(__name__)


@dataclass
class PersonalizationConfig:
    """Configuration for personalized federated learning."""
    # General settings
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    momentum: float = 0.9
    weight_decay: float = 1e-4
    
    # Personalization strategy
    strategy: str = "per_fedavg"  # per_fedavg, fedper, pfnm, fedrep
    
    # Per-FedAvg specific
    meta_learning_rate: float = 0.1
    meta_steps: int = 5
    
    # FedPer specific
    personalized_layers: List[str] = field(default_factory=lambda: ["classifier", "fc"])
    
    # PFNM specific
    matching_threshold: float = 0.8
    neuron_matching_method: str = "hungarian"  # hungarian, greedy
    
    # FedRep specific
    representation_epochs: int = 3
    head_epochs: int = 2
    
    # Adaptive personalization
    adaptive_personalization: bool = True
    personalization_strength: float = 0.5  # [0, 1]
    min_personalization: float = 0.1
    max_personalization: float = 0.9
    
    # Privacy and security
    differential_privacy: bool = False
    dp_noise_scale: float = 0.1
    dp_clip_norm: float = 1.0
    
    # Performance monitoring
    track_personalization_metrics: bool = True
    performance_window: int = 10


@dataclass
class PersonalizedClientState:
    """State for a personalized FL client."""
    device_id: str
    personalized_model: Dict[str, torch.Tensor]
    global_model_cache: Dict[str, torch.Tensor] = field(default_factory=dict)
    personalization_history: List[float] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)
    local_loss_history: List[float] = field(default_factory=list)
    data_distribution: Optional[Dict[str, float]] = None
    personalization_strength: float = 0.5
    last_participation_round: int = -1
    total_local_iterations: int = 0
    meta_optimizer_state: Optional[Dict] = None


@dataclass
class PersonalizedRoundResult:
    """Results from a personalized FL round."""
    round_id: int
    participating_clients: List[str]
    global_loss: float
    personalized_losses: Dict[str, float]
    personalization_metrics: Dict[str, Any]
    convergence_metrics: Dict[str, float]
    round_duration: float
    avg_personalization_strength: float
    performance_improvements: Dict[str, float]


class PersonalizationStrategy(ABC):
    """Abstract base class for personalization strategies."""
    
    @abstractmethod
    def personalize_model(self, client_id: str, global_model: Dict[str, torch.Tensor],
                         local_data: torch.utils.data.DataLoader,
                         client_state: PersonalizedClientState) -> Dict[str, torch.Tensor]:
        """Personalize the global model for a specific client."""
        pass
    
    @abstractmethod
    def aggregate_for_global(self, client_updates: Dict[str, Dict[str, torch.Tensor]],
                           client_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Aggregate client updates for global model update."""
        pass


class PerFedAvgStrategy(PersonalizationStrategy):
    """Per-FedAvg: Personalized FedAvg with meta-learning approach."""
    
    def __init__(self, config: PersonalizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def personalize_model(self, client_id: str, global_model: Dict[str, torch.Tensor],
                         local_data: torch.utils.data.DataLoader,
                         client_state: PersonalizedClientState) -> Dict[str, torch.Tensor]:
        """
        Personalize using meta-learning approach.
        
        1. Start from global model
        2. Perform few gradient steps on local data
        3. Use meta-learning to adapt the model
        """
        # Initialize with global model
        personalized_model = copy.deepcopy(global_model)
        
        # Create temporary model for meta-learning
        temp_model = self._dict_to_model(personalized_model)
        temp_model.train()
        
        # Meta-learning optimizer
        meta_optimizer = optim.SGD(
            temp_model.parameters(),
            lr=self.config.meta_learning_rate
        )
        
        # Perform meta-learning steps
        for meta_step in range(self.config.meta_steps):
            meta_optimizer.zero_grad()
            
            # Sample a batch for meta-learning
            try:
                data_batch = next(iter(local_data))
                if len(data_batch) == 2:
                    inputs, targets = data_batch
                else:
                    continue
                
                # Forward pass
                outputs = temp_model(inputs)
                
                # Compute loss
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(outputs, targets)
                
                # Backward pass
                loss.backward()
                meta_optimizer.step()
                
            except (StopIteration, RuntimeError) as e:
                self.logger.warning(f"Meta-learning step {meta_step} failed for client {client_id}: {e}")
                break
        
        # Extract personalized parameters
        personalized_model = {
            name: param.data.clone()
            for name, param in temp_model.named_parameters()
        }
        
        self.logger.debug(f"Per-FedAvg personalization completed for client {client_id}")
        return personalized_model
    
    def aggregate_for_global(self, client_updates: Dict[str, Dict[str, torch.Tensor]],
                           client_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Standard FedAvg aggregation."""
        if not client_updates:
            return {}
        
        aggregated = {}
        first_client = next(iter(client_updates.values()))
        
        for param_name in first_client.keys():
            weighted_sum = None
            total_weight = 0.0
            
            for client_id, update in client_updates.items():
                if client_id in client_weights and param_name in update:
                    weight = client_weights[client_id]
                    param = update[param_name]
                    
                    if weighted_sum is None:
                        weighted_sum = weight * param
                    else:
                        weighted_sum += weight * param
                    
                    total_weight += weight
            
            if total_weight > 0:
                aggregated[param_name] = weighted_sum / total_weight
        
        return aggregated
    
    def _dict_to_model(self, param_dict: Dict[str, torch.Tensor]) -> nn.Module:
        """Convert parameter dictionary to a simple model (placeholder)."""
        # This is a simplified implementation
        # In practice, you'd need the actual model architecture
        class SimpleModel(nn.Module):
            def __init__(self, param_dict):
                super().__init__()
                for name, param in param_dict.items():
                    self.register_parameter(name, nn.Parameter(param.clone()))
            
            def forward(self, x):
                # Simplified forward pass - would need actual model architecture
                return x
        
        return SimpleModel(param_dict)


class FedPerStrategy(PersonalizationStrategy):
    """FedPer: Separate personalized layers from shared layers."""
    
    def __init__(self, config: PersonalizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.personalized_layer_names = set(config.personalized_layers)
    
    def personalize_model(self, client_id: str, global_model: Dict[str, torch.Tensor],
                         local_data: torch.utils.data.DataLoader,
                         client_state: PersonalizedClientState) -> Dict[str, torch.Tensor]:
        """
        Personalize by keeping some layers local and updating shared layers from global.
        """
        personalized_model = {}
        
        # Use global parameters for shared layers
        for param_name, param_value in global_model.items():
            if not self._is_personalized_layer(param_name):
                personalized_model[param_name] = param_value.clone()
        
        # Use local parameters for personalized layers
        if client_state.personalized_model:
            for param_name, param_value in client_state.personalized_model.items():
                if self._is_personalized_layer(param_name):
                    personalized_model[param_name] = param_value.clone()
        else:
            # Initialize personalized layers from global if no local version exists
            for param_name, param_value in global_model.items():
                if self._is_personalized_layer(param_name):
                    personalized_model[param_name] = param_value.clone()
        
        self.logger.debug(f"FedPer personalization completed for client {client_id}")
        return personalized_model
    
    def aggregate_for_global(self, client_updates: Dict[str, Dict[str, torch.Tensor]],
                           client_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Aggregate only shared layers."""
        if not client_updates:
            return {}
        
        aggregated = {}
        first_client = next(iter(client_updates.values()))
        
        for param_name in first_client.keys():
            # Only aggregate shared layers
            if not self._is_personalized_layer(param_name):
                weighted_sum = None
                total_weight = 0.0
                
                for client_id, update in client_updates.items():
                    if client_id in client_weights and param_name in update:
                        weight = client_weights[client_id]
                        param = update[param_name]
                        
                        if weighted_sum is None:
                            weighted_sum = weight * param
                        else:
                            weighted_sum += weight * param
                        
                        total_weight += weight
                
                if total_weight > 0:
                    aggregated[param_name] = weighted_sum / total_weight
        
        return aggregated
    
    def _is_personalized_layer(self, param_name: str) -> bool:
        """Check if a parameter belongs to a personalized layer."""
        param_lower = param_name.lower()
        return any(layer_name.lower() in param_lower for layer_name in self.personalized_layer_names)


class PFNMStrategy(PersonalizationStrategy):
    """PFNM: Personalized Federated Neural Matching."""
    
    def __init__(self, config: PersonalizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def personalize_model(self, client_id: str, global_model: Dict[str, torch.Tensor],
                         local_data: torch.utils.data.DataLoader,
                         client_state: PersonalizedClientState) -> Dict[str, torch.Tensor]:
        """
        Personalize using neural matching to align global and local features.
        """
        # This is a simplified implementation of PFNM
        # The full implementation would require sophisticated neuron matching algorithms
        
        personalized_model = copy.deepcopy(global_model)
        
        # If client has previous personalized model, perform matching
        if client_state.personalized_model:
            for param_name in personalized_model.keys():
                if param_name in client_state.personalized_model:
                    global_param = global_model[param_name]
                    local_param = client_state.personalized_model[param_name]
                    
                    # Simplified matching: weighted combination based on similarity
                    similarity = self._compute_similarity(global_param, local_param)
                    
                    if similarity > self.config.matching_threshold:
                        # High similarity: use more global
                        weight = 0.7
                    else:
                        # Low similarity: use more local
                        weight = 0.3
                    
                    personalized_model[param_name] = (
                        weight * global_param + (1 - weight) * local_param
                    )
        
        self.logger.debug(f"PFNM personalization completed for client {client_id}")
        return personalized_model
    
    def aggregate_for_global(self, client_updates: Dict[str, Dict[str, torch.Tensor]],
                           client_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Aggregate with neural matching considerations."""
        # Simplified aggregation - full PFNM would include matching-aware aggregation
        return self._standard_aggregation(client_updates, client_weights)
    
    def _compute_similarity(self, param1: torch.Tensor, param2: torch.Tensor) -> float:
        """Compute similarity between two parameter tensors."""
        if param1.shape != param2.shape:
            return 0.0
        
        # Cosine similarity
        param1_flat = param1.flatten()
        param2_flat = param2.flatten()
        
        cos_sim = torch.cosine_similarity(param1_flat, param2_flat, dim=0)
        return cos_sim.item()
    
    def _standard_aggregation(self, client_updates: Dict[str, Dict[str, torch.Tensor]],
                            client_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Standard weighted aggregation."""
        if not client_updates:
            return {}
        
        aggregated = {}
        first_client = next(iter(client_updates.values()))
        
        for param_name in first_client.keys():
            weighted_sum = None
            total_weight = 0.0
            
            for client_id, update in client_updates.items():
                if client_id in client_weights and param_name in update:
                    weight = client_weights[client_id]
                    param = update[param_name]
                    
                    if weighted_sum is None:
                        weighted_sum = weight * param
                    else:
                        weighted_sum += weight * param
                    
                    total_weight += weight
            
            if total_weight > 0:
                aggregated[param_name] = weighted_sum / total_weight
        
        return aggregated


class FedRepStrategy(PersonalizationStrategy):
    """FedRep: Federated Representation Learning."""
    
    def __init__(self, config: PersonalizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def personalize_model(self, client_id: str, global_model: Dict[str, torch.Tensor],
                         local_data: torch.utils.data.DataLoader,
                         client_state: PersonalizedClientState) -> Dict[str, torch.Tensor]:
        """
        Personalize by separating representation learning and head training.
        """
        personalized_model = copy.deepcopy(global_model)
        
        # In FedRep, we would:
        # 1. Update representation layers using global knowledge
        # 2. Train head layers locally for personalization
        
        # Simplified implementation: identify head vs representation layers
        head_params = {}
        repr_params = {}
        
        for param_name, param_value in personalized_model.items():
            if self._is_head_layer(param_name):
                head_params[param_name] = param_value
            else:
                repr_params[param_name] = param_value
        
        # Use client's personalized head if available
        if client_state.personalized_model:
            for param_name in head_params.keys():
                if param_name in client_state.personalized_model:
                    personalized_model[param_name] = client_state.personalized_model[param_name]
        
        self.logger.debug(f"FedRep personalization completed for client {client_id}")
        return personalized_model
    
    def aggregate_for_global(self, client_updates: Dict[str, Dict[str, torch.Tensor]],
                           client_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Aggregate only representation layers."""
        if not client_updates:
            return {}
        
        aggregated = {}
        first_client = next(iter(client_updates.values()))
        
        for param_name in first_client.keys():
            # Only aggregate representation layers
            if not self._is_head_layer(param_name):
                weighted_sum = None
                total_weight = 0.0
                
                for client_id, update in client_updates.items():
                    if client_id in client_weights and param_name in update:
                        weight = client_weights[client_id]
                        param = update[param_name]
                        
                        if weighted_sum is None:
                            weighted_sum = weight * param
                        else:
                            weighted_sum += weight * param
                        
                        total_weight += weight
                
                if total_weight > 0:
                    aggregated[param_name] = weighted_sum / total_weight
        
        return aggregated
    
    def _is_head_layer(self, param_name: str) -> bool:
        """Check if a parameter belongs to the head (classifier) layers."""
        head_keywords = ['classifier', 'fc', 'head', 'linear', 'output']
        param_lower = param_name.lower()
        return any(keyword in param_lower for keyword in head_keywords)


class PersonalizedFederatedLearning:
    """Main personalized federated learning algorithm."""
    
    def __init__(self, config: PersonalizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize personalization strategy
        self.strategy = self._create_strategy()
        
        # State tracking
        self.global_model_params: Optional[Dict[str, torch.Tensor]] = None
        self.client_states: Dict[str, PersonalizedClientState] = {}
        self.round_history: List[PersonalizedRoundResult] = []
        self.current_round = 0
        
        # Performance tracking
        self.global_performance_history = []
        self.personalization_effectiveness = {}
    
    def _create_strategy(self) -> PersonalizationStrategy:
        """Create the appropriate personalization strategy."""
        if self.config.strategy == "per_fedavg":
            return PerFedAvgStrategy(self.config)
        elif self.config.strategy == "fedper":
            return FedPerStrategy(self.config)
        elif self.config.strategy == "pfnm":
            return PFNMStrategy(self.config)
        elif self.config.strategy == "fedrep":
            return FedRepStrategy(self.config)
        else:
            raise ValueError(f"Unknown personalization strategy: {self.config.strategy}")
    
    def initialize_global_model(self, model_template: nn.Module):
        """Initialize the global model parameters."""
        self.global_model_params = {
            name: param.clone().detach()
            for name, param in model_template.named_parameters()
        }
        self.logger.info(f"Global model initialized for personalized FL ({self.config.strategy})")
    
    def register_client(self, client_id: str, data_size: int, 
                       data_distribution: Optional[Dict[str, float]] = None):
        """Register a new client with personalized FL."""
        self.client_states[client_id] = PersonalizedClientState(
            device_id=client_id,
            personalized_model={},
            data_distribution=data_distribution,
            personalization_strength=self.config.personalization_strength
        )
        self.logger.info(f"Registered personalized client {client_id} with data size {data_size}")
    
    def client_update(self, client_id: str, model: nn.Module, 
                     train_loader: torch.utils.data.DataLoader,
                     loss_fn: nn.Module) -> Tuple[Dict[str, torch.Tensor], float, Dict[str, Any]]:
        """
        Perform personalized local training for a client.
        
        Returns:
            Tuple of (updated_model_params, local_loss, metrics)
        """
        if client_id not in self.client_states:
            raise ValueError(f"Client {client_id} not registered")
        
        if self.global_model_params is None:
            raise ValueError("Global model not initialized")
        
        client_state = self.client_states[client_id]
        
        # Personalize the global model for this client
        personalized_params = self.strategy.personalize_model(
            client_id, self.global_model_params, train_loader, client_state
        )
        
        # Load personalized parameters into model
        self._load_params_to_model(model, personalized_params)
        
        # Set model to training mode
        model.train()
        
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
                
                # Add differential privacy noise if enabled
                if self.config.differential_privacy:
                    loss = self._add_dp_noise(loss)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.dp_clip_norm)
                
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item() * len(data)
                epoch_samples += len(data)
                total_samples += len(data)
                client_state.total_local_iterations += 1
            
            avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
            epoch_losses.append(avg_epoch_loss)
        
        training_time = time.time() - start_time
        
        # Extract updated model parameters
        updated_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        
        # Update client state
        client_state.personalized_model = updated_params
        final_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        client_state.local_loss_history.append(final_loss)
        client_state.last_participation_round = self.current_round
        
        # Update performance tracking
        if self.config.track_personalization_metrics:
            self._update_personalization_metrics(client_id, final_loss)
        
        # Adaptive personalization strength
        if self.config.adaptive_personalization:
            self._adapt_personalization_strength(client_id)
        
        # Prepare metrics
        metrics = {
            'training_time': training_time,
            'total_samples': total_samples,
            'personalization_strength': client_state.personalization_strength,
            'strategy': self.config.strategy,
            'epoch_losses': epoch_losses
        }
        
        self.logger.info(f"Personalized client {client_id} training completed: "
                        f"loss = {final_loss:.6f}, time = {training_time:.2f}s")
        
        return updated_params, final_loss, metrics
    
    def federated_averaging(self, participating_clients: List[str]) -> PersonalizedRoundResult:
        """
        Perform personalized federated averaging.
        
        Args:
            participating_clients: List of client IDs that participated in this round
            
        Returns:
            PersonalizedRoundResult with round statistics
        """
        start_time = time.time()
        
        # Collect client updates and metadata
        client_updates = {}
        client_weights = {}
        client_losses = {}
        
        total_data = 0
        for client_id in participating_clients:
            if client_id in self.client_states:
                client_state = self.client_states[client_id]
                client_updates[client_id] = client_state.personalized_model
                
                # Use data size for weighting (could be made more sophisticated)
                data_size = getattr(client_state, 'data_size', 1)
                total_data += data_size
                client_weights[client_id] = data_size
                
                if client_state.local_loss_history:
                    client_losses[client_id] = client_state.local_loss_history[-1]
                else:
                    client_losses[client_id] = float('inf')
        
        # Normalize weights
        if total_data > 0:
            client_weights = {k: v/total_data for k, v in client_weights.items()}
        else:
            num_clients = len(participating_clients)
            client_weights = {k: 1.0/num_clients for k in participating_clients}
        
        # Perform strategy-specific aggregation
        aggregated_params = self.strategy.aggregate_for_global(client_updates, client_weights)
        
        # Update global model (only parameters that were aggregated)
        if aggregated_params:
            for param_name, param_value in aggregated_params.items():
                if param_name in self.global_model_params:
                    self.global_model_params[param_name] = param_value
        
        # Compute round metrics
        global_loss = np.mean(list(client_losses.values()))
        personalization_metrics = self._compute_personalization_metrics(participating_clients)
        convergence_metrics = self._compute_convergence_metrics()
        
        # Compute performance improvements
        performance_improvements = {}
        for client_id in participating_clients:
            if client_id in self.client_states:
                improvements = self._compute_client_improvement(client_id)
                performance_improvements[client_id] = improvements
        
        round_duration = time.time() - start_time
        
        # Create round result
        round_result = PersonalizedRoundResult(
            round_id=self.current_round,
            participating_clients=participating_clients,
            global_loss=global_loss,
            personalized_losses=client_losses,
            personalization_metrics=personalization_metrics,
            convergence_metrics=convergence_metrics,
            round_duration=round_duration,
            avg_personalization_strength=np.mean([
                self.client_states[cid].personalization_strength 
                for cid in participating_clients if cid in self.client_states
            ]),
            performance_improvements=performance_improvements
        )
        
        self.round_history.append(round_result)
        self.current_round += 1
        
        self.logger.info(f"Personalized round {round_result.round_id} completed: "
                        f"global_loss = {global_loss:.6f}, "
                        f"participants = {len(participating_clients)}, "
                        f"avg_personalization = {round_result.avg_personalization_strength:.4f}")
        
        return round_result
    
    def _load_params_to_model(self, model: nn.Module, params: Dict[str, torch.Tensor]):
        """Load parameters into model."""
        model_state = model.state_dict()
        for param_name, param_value in params.items():
            if param_name in model_state:
                model_state[param_name].copy_(param_value)
    
    def _add_dp_noise(self, loss: torch.Tensor) -> torch.Tensor:
        """Add differential privacy noise to loss."""
        noise = torch.normal(0, self.config.dp_noise_scale, size=loss.shape)
        return loss + noise
    
    def _update_personalization_metrics(self, client_id: str, current_loss: float):
        """Update personalization effectiveness metrics."""
        client_state = self.client_states[client_id]
        client_state.performance_history.append(current_loss)
        
        # Keep only recent history
        if len(client_state.performance_history) > self.config.performance_window:
            client_state.performance_history.pop(0)
    
    def _adapt_personalization_strength(self, client_id: str):
        """Adaptively adjust personalization strength based on performance."""
        client_state = self.client_states[client_id]
        
        if len(client_state.performance_history) < 3:
            return
        
        # Check if performance is improving
        recent_losses = client_state.performance_history[-3:]
        if len(recent_losses) >= 2:
            loss_trend = recent_losses[-1] - recent_losses[0]
            
            if loss_trend < 0:  # Improving (loss decreasing)
                # Slightly increase personalization
                client_state.personalization_strength = min(
                    self.config.max_personalization,
                    client_state.personalization_strength * 1.05
                )
            else:  # Not improving
                # Slightly decrease personalization
                client_state.personalization_strength = max(
                    self.config.min_personalization,
                    client_state.personalization_strength * 0.95
                )
    
    def _compute_personalization_metrics(self, participating_clients: List[str]) -> Dict[str, Any]:
        """Compute metrics related to personalization effectiveness."""
        metrics = {}
        
        personalization_strengths = []
        performance_variances = []
        
        for client_id in participating_clients:
            if client_id in self.client_states:
                client_state = self.client_states[client_id]
                personalization_strengths.append(client_state.personalization_strength)
                
                if len(client_state.performance_history) >= 2:
                    performance_variances.append(np.var(client_state.performance_history))
        
        if personalization_strengths:
            metrics['avg_personalization_strength'] = np.mean(personalization_strengths)
            metrics['personalization_diversity'] = np.std(personalization_strengths)
        
        if performance_variances:
            metrics['avg_performance_variance'] = np.mean(performance_variances)
        
        metrics['strategy'] = self.config.strategy
        metrics['participants_count'] = len(participating_clients)
        
        return metrics
    
    def _compute_convergence_metrics(self) -> Dict[str, float]:
        """Compute convergence metrics."""
        metrics = {}
        
        if len(self.round_history) < 2:
            return {'loss_improvement': 0.0, 'convergence_rate': 0.0}
        
        # Global loss improvement
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
    
    def _compute_client_improvement(self, client_id: str) -> float:
        """Compute performance improvement for a specific client."""
        if client_id not in self.client_states:
            return 0.0
        
        client_state = self.client_states[client_id]
        
        if len(client_state.performance_history) < 2:
            return 0.0
        
        # Compare recent performance to earlier performance
        if len(client_state.performance_history) >= 5:
            early_avg = np.mean(client_state.performance_history[:2])
            recent_avg = np.mean(client_state.performance_history[-2:])
            return early_avg - recent_avg  # Positive means improvement
        else:
            return client_state.performance_history[0] - client_state.performance_history[-1]
    
    def get_global_model_params(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get the current global model parameters."""
        return self.global_model_params
    
    def get_client_state(self, client_id: str) -> Optional[PersonalizedClientState]:
        """Get the state of a specific client."""
        return self.client_states.get(client_id)
    
    def get_algorithm_stats(self) -> Dict[str, Any]:
        """Get comprehensive algorithm statistics."""
        stats = {
            'current_round': self.current_round,
            'total_clients': len(self.client_states),
            'strategy': self.config.strategy,
            'round_count': len(self.round_history)
        }
        
        if self.round_history:
            latest_round = self.round_history[-1]
            stats.update({
                'latest_global_loss': latest_round.global_loss,
                'avg_personalization_strength': latest_round.avg_personalization_strength,
                'avg_round_duration': np.mean([r.round_duration for r in self.round_history])
            })
        
        # Client-specific stats
        if self.client_states:
            all_strengths = [cs.personalization_strength for cs in self.client_states.values()]
            stats['personalization_strength_stats'] = {
                'mean': np.mean(all_strengths),
                'std': np.std(all_strengths),
                'min': np.min(all_strengths),
                'max': np.max(all_strengths)
            }
        
        return stats