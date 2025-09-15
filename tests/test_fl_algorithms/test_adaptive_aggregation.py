"""
Test Suite for Adaptive Aggregation Algorithms

This module contains comprehensive tests for adaptive aggregation strategies including
FedOpt family (FedAdam, FedAdagrad, FedYogi), robust aggregation, and dynamic client selection.

Test Coverage:
- Adaptive aggregation configuration validation
- Server-side optimization algorithms (FedAdam, FedAdagrad, FedYogi)
- Robust aggregation methods (median, trimmed mean, Krum)
- Dynamic client selection strategies
- Byzantine fault tolerance
- Convergence analysis and adaptive parameters
- Integration with different model architectures
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Tuple

# Import the modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'server'))

from fl_algorithms.adaptive_aggregation import (
    AdaptiveAggregationOrchestrator, AdaptiveAggregationConfig,
    FedOptConfig, FedAdamOptimizer, FedAdagradOptimizer, FedYogiOptimizer,
    RobustAggregationConfig, MedianAggregator, TrimmedMeanAggregator, KrumAggregator,
    DynamicClientSelectionConfig, DynamicClientSelector,
    ConvergenceAnalyzer, AdaptiveAggregationResult
)


class SimpleModel(nn.Module):
    """Simple neural network for testing."""
    
    def __init__(self, input_size=10, hidden_size=32, output_size=5):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def create_mock_client_updates(num_clients=5, param_size=10, add_noise=True, byzantine_count=0):
    """Create mock client updates for testing."""
    updates = {}
    
    # Create normal clients
    for i in range(num_clients - byzantine_count):
        client_id = f'client{i+1}'
        
        # Create parameters with some variation
        base_param = torch.randn(param_size)
        if add_noise:
            noise = torch.randn(param_size) * 0.1
            param = base_param + noise
        else:
            param = base_param
        
        updates[client_id] = {
            'linear1.weight': param.view(1, -1),
            'linear1.bias': torch.randn(1),
            'linear2.weight': torch.randn(5, 1),
            'linear2.bias': torch.randn(5)
        }
    
    # Create Byzantine clients (if any)
    for i in range(byzantine_count):
        client_id = f'byzantine{i+1}'
        
        # Byzantine clients send malicious updates
        updates[client_id] = {
            'linear1.weight': torch.ones(1, param_size) * 100,  # Extremely large values
            'linear1.bias': torch.ones(1) * 100,
            'linear2.weight': torch.ones(5, 1) * 100,
            'linear2.bias': torch.ones(5) * 100
        }
    
    return updates


def create_mock_client_weights(client_ids, equal_weights=True):
    """Create mock client weights for testing."""
    if equal_weights:
        weight = 1.0 / len(client_ids)
        return {client_id: weight for client_id in client_ids}
    else:
        # Create unequal weights
        weights = torch.softmax(torch.randn(len(client_ids)), dim=0)
        return {client_id: weight.item() for client_id, weight in zip(client_ids, weights)}


@pytest.fixture
def basic_config():
    """Create a basic adaptive aggregation configuration."""
    return AdaptiveAggregationConfig(
        method="fedavg",
        server_learning_rate=1.0,
        momentum=0.0,
        adaptive_lr=True,
        convergence_window=5
    )


@pytest.fixture
def fedopt_config():
    """Create a FedOpt configuration."""
    return AdaptiveAggregationConfig(
        method="fedopt",
        fedopt=FedOptConfig(
            optimizer="fedadam",
            server_learning_rate=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8
        ),
        adaptive_lr=True
    )


@pytest.fixture
def robust_config():
    """Create a robust aggregation configuration."""
    return AdaptiveAggregationConfig(
        method="robust",
        robust_aggregation=RobustAggregationConfig(
            method="median",
            byzantine_tolerance=0.3,
            outlier_detection=True
        )
    )


@pytest.fixture
def dynamic_selection_config():
    """Create a dynamic client selection configuration."""
    return AdaptiveAggregationConfig(
        method="fedavg",
        dynamic_client_selection=DynamicClientSelectionConfig(
            enabled=True,
            selection_strategy="contribution_based",
            max_clients_per_round=10,
            contribution_threshold=0.1
        )
    )


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def orchestrator(basic_config):
    """Create an adaptive aggregation orchestrator."""
    return AdaptiveAggregationOrchestrator(basic_config)


class TestAdaptiveAggregationConfig:
    """Test adaptive aggregation configuration validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AdaptiveAggregationConfig()
        assert config.method == "fedavg"
        assert config.server_learning_rate == 1.0
        assert config.momentum == 0.0
        assert config.adaptive_lr == True
        assert config.convergence_window == 10
    
    def test_fedopt_config(self):
        """Test FedOpt specific configuration."""
        fedopt_config = FedOptConfig(
            optimizer="fedadam",
            server_learning_rate=0.01,
            beta1=0.9,
            beta2=0.999
        )
        
        config = AdaptiveAggregationConfig(
            method="fedopt",
            fedopt=fedopt_config
        )
        
        assert config.method == "fedopt"
        assert config.fedopt.optimizer == "fedadam"
        assert config.fedopt.server_learning_rate == 0.01
        assert config.fedopt.beta1 == 0.9
        assert config.fedopt.beta2 == 0.999
    
    def test_robust_aggregation_config(self):
        """Test robust aggregation configuration."""
        robust_config = RobustAggregationConfig(
            method="krum",
            byzantine_tolerance=0.2,
            outlier_detection=True
        )
        
        config = AdaptiveAggregationConfig(
            method="robust",
            robust_aggregation=robust_config
        )
        
        assert config.method == "robust"
        assert config.robust_aggregation.method == "krum"
        assert config.robust_aggregation.byzantine_tolerance == 0.2
        assert config.robust_aggregation.outlier_detection == True
    
    def test_dynamic_client_selection_config(self):
        """Test dynamic client selection configuration."""
        selection_config = DynamicClientSelectionConfig(
            enabled=True,
            selection_strategy="performance_based",
            max_clients_per_round=20,
            contribution_threshold=0.05
        )
        
        config = AdaptiveAggregationConfig(
            method="fedavg",
            dynamic_client_selection=selection_config
        )
        
        assert config.dynamic_client_selection.enabled == True
        assert config.dynamic_client_selection.selection_strategy == "performance_based"
        assert config.dynamic_client_selection.max_clients_per_round == 20
        assert config.dynamic_client_selection.contribution_threshold == 0.05


class TestFedOptOptimizers:
    """Test FedOpt family optimizers."""
    
    def test_fedadam_optimizer(self):
        """Test FedAdam optimizer."""
        config = FedOptConfig(
            optimizer="fedadam",
            server_learning_rate=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8
        )
        
        optimizer = FedAdamOptimizer(config)
        assert optimizer.config == config
        assert len(optimizer.momentum_buffer) == 0
        assert len(optimizer.velocity_buffer) == 0
    
    def test_fedadam_step(self):
        """Test FedAdam optimization step."""
        config = FedOptConfig(optimizer="fedadam", server_learning_rate=0.1)
        optimizer = FedAdamOptimizer(config)
        
        # Mock current and new parameters
        current_params = {
            'layer1.weight': torch.zeros(3, 3),
            'layer1.bias': torch.zeros(3)
        }
        
        new_params = {
            'layer1.weight': torch.ones(3, 3) * 0.1,
            'layer1.bias': torch.ones(3) * 0.1
        }
        
        # Perform optimization step
        updated_params = optimizer.step(current_params, new_params)
        
        # Should have same parameter names
        assert set(updated_params.keys()) == set(current_params.keys())
        
        # Should be different from both current and new (due to momentum)
        assert not torch.equal(updated_params['layer1.weight'], current_params['layer1.weight'])
        assert not torch.equal(updated_params['layer1.weight'], new_params['layer1.weight'])
        
        # Should have momentum and velocity buffers
        assert len(optimizer.momentum_buffer) == 2
        assert len(optimizer.velocity_buffer) == 2
    
    def test_fedadagrad_optimizer(self):
        """Test FedAdagrad optimizer."""
        config = FedOptConfig(
            optimizer="fedadagrad",
            server_learning_rate=0.1,
            epsilon=1e-8
        )
        
        optimizer = FedAdagradOptimizer(config)
        assert optimizer.config == config
        assert len(optimizer.accumulator) == 0
    
    def test_fedadagrad_step(self):
        """Test FedAdagrad optimization step."""
        config = FedOptConfig(optimizer="fedadagrad", server_learning_rate=0.1)
        optimizer = FedAdagradOptimizer(config)
        
        current_params = {
            'param1': torch.zeros(2, 2),
            'param2': torch.zeros(2)
        }
        
        new_params = {
            'param1': torch.ones(2, 2) * 0.2,
            'param2': torch.ones(2) * 0.2
        }
        
        # Perform optimization step
        updated_params = optimizer.step(current_params, new_params)
        
        # Should have accumulator
        assert len(optimizer.accumulator) == 2
        assert 'param1' in optimizer.accumulator
        assert 'param2' in optimizer.accumulator
        
        # Second step should use accumulated gradients
        new_params2 = {
            'param1': updated_params['param1'] + 0.1,
            'param2': updated_params['param2'] + 0.1
        }
        
        updated_params2 = optimizer.step(updated_params, new_params2)
        assert not torch.equal(updated_params2['param1'], updated_params['param1'])
    
    def test_fedyogi_optimizer(self):
        """Test FedYogi optimizer."""
        config = FedOptConfig(
            optimizer="fedyogi",
            server_learning_rate=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8
        )
        
        optimizer = FedYogiOptimizer(config)
        assert optimizer.config == config
        assert len(optimizer.momentum_buffer) == 0
        assert len(optimizer.velocity_buffer) == 0
    
    def test_fedyogi_step(self):
        """Test FedYogi optimization step."""
        config = FedOptConfig(optimizer="fedyogi", server_learning_rate=0.1)
        optimizer = FedYogiOptimizer(config)
        
        current_params = {
            'weight': torch.zeros(3, 3),
            'bias': torch.zeros(3)
        }
        
        new_params = {
            'weight': torch.ones(3, 3) * 0.15,
            'bias': torch.ones(3) * 0.15
        }
        
        # Multiple steps to test momentum accumulation
        updated_params = current_params
        for _ in range(3):
            updated_params = optimizer.step(updated_params, new_params)
        
        # Should have momentum and velocity buffers
        assert len(optimizer.momentum_buffer) == 2
        assert len(optimizer.velocity_buffer) == 2
        
        # Buffers should have proper shapes
        assert optimizer.momentum_buffer['weight'].shape == (3, 3)
        assert optimizer.velocity_buffer['bias'].shape == (3,)


class TestRobustAggregators:
    """Test robust aggregation methods."""
    
    def test_median_aggregator(self):
        """Test median aggregation."""
        config = RobustAggregationConfig(method="median")
        aggregator = MedianAggregator(config)
        
        # Create client updates with outliers
        client_updates = {
            'client1': {'param': torch.tensor([1.0, 2.0, 3.0])},
            'client2': {'param': torch.tensor([1.1, 2.1, 3.1])},
            'client3': {'param': torch.tensor([0.9, 1.9, 2.9])},
            'outlier': {'param': torch.tensor([10.0, 20.0, 30.0])}  # Outlier
        }
        
        client_weights = {cid: 0.25 for cid in client_updates.keys()}
        
        aggregated = aggregator.aggregate(client_updates, client_weights)
        
        # Should compute median (approximately [1.0, 2.0, 3.0])
        assert 'param' in aggregated
        median_values = aggregated['param']
        
        # Check that outlier was excluded
        assert median_values[0] < 5.0  # Should not be influenced by outlier
        assert median_values[1] < 10.0
        assert median_values[2] < 15.0
    
    def test_trimmed_mean_aggregator(self):
        """Test trimmed mean aggregation."""
        config = RobustAggregationConfig(
            method="trimmed_mean",
            trim_fraction=0.25  # Trim 25% from each end
        )
        aggregator = TrimmedMeanAggregator(config)
        
        # Create client updates with outliers
        client_updates = {
            'client1': {'param': torch.tensor([1.0])},
            'client2': {'param': torch.tensor([2.0])},
            'client3': {'param': torch.tensor([3.0])},
            'client4': {'param': torch.tensor([4.0])},
            'outlier1': {'param': torch.tensor([100.0])},  # High outlier
            'outlier2': {'param': torch.tensor([-100.0])}  # Low outlier
        }
        
        client_weights = {cid: 1.0/6 for cid in client_updates.keys()}
        
        aggregated = aggregator.aggregate(client_updates, client_weights)
        
        # Should trim outliers and compute mean of middle values
        trimmed_mean = aggregated['param']
        
        # Should be approximately mean of [1, 2, 3, 4] = 2.5
        assert 1.0 < trimmed_mean.item() < 5.0
        assert abs(trimmed_mean.item() - 2.5) < 1.0
    
    def test_krum_aggregator(self):
        """Test Krum aggregation."""
        config = RobustAggregationConfig(
            method="krum",
            byzantine_tolerance=0.3  # Tolerate up to 30% Byzantine clients
        )
        aggregator = KrumAggregator(config)
        
        # Create normal and Byzantine updates
        client_updates = create_mock_client_updates(
            num_clients=5, 
            param_size=3, 
            add_noise=False, 
            byzantine_count=1
        )
        
        client_weights = create_mock_client_weights(list(client_updates.keys()))
        
        aggregated = aggregator.aggregate(client_updates, client_weights)
        
        # Should exclude Byzantine updates
        assert 'linear1.weight' in aggregated
        assert 'linear1.bias' in aggregated
        
        # Should not be influenced by Byzantine values (which are 100)
        weight_values = aggregated['linear1.weight'].flatten()
        bias_values = aggregated['linear1.bias']
        
        assert torch.all(torch.abs(weight_values) < 50)  # Much less than Byzantine 100
        assert torch.all(torch.abs(bias_values) < 50)
    
    def test_outlier_detection(self):
        """Test outlier detection in robust aggregation."""
        config = RobustAggregationConfig(
            method="median",
            outlier_detection=True,
            outlier_threshold=2.0
        )
        aggregator = MedianAggregator(config)
        
        # Create updates with clear outliers
        client_updates = {
            'normal1': {'param': torch.tensor([1.0, 1.0])},
            'normal2': {'param': torch.tensor([1.1, 1.1])},
            'normal3': {'param': torch.tensor([0.9, 0.9])},
            'outlier': {'param': torch.tensor([5.0, 5.0])}  # Clear outlier
        }
        
        client_weights = {cid: 0.25 for cid in client_updates.keys()}
        
        aggregated = aggregator.aggregate(client_updates, client_weights)
        
        # Should detect and handle outlier
        result_param = aggregated['param']
        
        # Result should be close to normal values, not outlier
        assert torch.all(torch.abs(result_param - 1.0) < 1.0)


class TestDynamicClientSelection:
    """Test dynamic client selection strategies."""
    
    def test_contribution_based_selection(self):
        """Test contribution-based client selection."""
        config = DynamicClientSelectionConfig(
            selection_strategy="contribution_based",
            max_clients_per_round=3,
            contribution_threshold=0.1
        )
        
        selector = DynamicClientSelector(config)
        
        # Mock client performance history
        client_performances = {
            'client1': [0.8, 0.85, 0.9],    # Good performance
            'client2': [0.5, 0.55, 0.6],    # Moderate performance
            'client3': [0.3, 0.35, 0.4],    # Poor performance
            'client4': [0.85, 0.88, 0.92],  # Best performance
            'client5': [0.1, 0.15, 0.2]     # Very poor performance
        }
        
        # Mock client resources
        client_resources = {cid: 1.0 for cid in client_performances.keys()}
        
        # Select clients
        selected_clients = selector.select_clients(
            list(client_performances.keys()),
            client_performances,
            client_resources
        )
        
        # Should select top performing clients
        assert len(selected_clients) <= 3
        assert 'client4' in selected_clients  # Best performance
        assert 'client1' in selected_clients  # Good performance
        assert 'client5' not in selected_clients  # Very poor performance
    
    def test_performance_based_selection(self):
        """Test performance-based client selection."""
        config = DynamicClientSelectionConfig(
            selection_strategy="performance_based",
            max_clients_per_round=2,
            contribution_threshold=0.05
        )
        
        selector = DynamicClientSelector(config)
        
        client_performances = {
            'fast_client': [0.9, 0.95, 0.98],     # Fast improvement
            'slow_client': [0.8, 0.81, 0.82],     # Slow improvement
            'declining_client': [0.7, 0.65, 0.6], # Declining performance
            'stable_client': [0.85, 0.85, 0.85]   # Stable performance
        }
        
        client_resources = {cid: 1.0 for cid in client_performances.keys()}
        
        selected_clients = selector.select_clients(
            list(client_performances.keys()),
            client_performances,
            client_resources
        )
        
        # Should prefer clients with good performance trends
        assert len(selected_clients) <= 2
        assert 'fast_client' in selected_clients  # Fast improvement
        assert 'declining_client' not in selected_clients  # Declining
    
    def test_resource_based_selection(self):
        """Test resource-based client selection."""
        config = DynamicClientSelectionConfig(
            selection_strategy="resource_based",
            max_clients_per_round=3
        )
        
        selector = DynamicClientSelector(config)
        
        client_performances = {cid: [0.8] for cid in ['client1', 'client2', 'client3', 'client4']}
        
        # Mock different resource levels
        client_resources = {
            'client1': 1.0,   # High resources
            'client2': 0.5,   # Medium resources
            'client3': 0.2,   # Low resources
            'client4': 0.8    # Good resources
        }
        
        selected_clients = selector.select_clients(
            list(client_performances.keys()),
            client_performances,
            client_resources
        )
        
        # Should prefer clients with more resources
        assert 'client1' in selected_clients  # Highest resources
        assert 'client4' in selected_clients  # Good resources
        assert len(selected_clients) <= 3


class TestConvergenceAnalyzer:
    """Test convergence analysis functionality."""
    
    def test_convergence_detection(self):
        """Test convergence detection."""
        analyzer = ConvergenceAnalyzer(window_size=5, threshold=0.01)
        
        # Simulate converging loss values
        converging_losses = [1.0, 0.5, 0.25, 0.15, 0.12, 0.11, 0.105, 0.104, 0.103]
        
        for loss in converging_losses:
            analyzer.update(loss)
        
        # Should detect convergence
        assert analyzer.is_converged() == True
        assert analyzer.get_convergence_rate() > 0
    
    def test_non_convergence(self):
        """Test non-convergence detection."""
        analyzer = ConvergenceAnalyzer(window_size=3, threshold=0.1)
        
        # Simulate oscillating loss values
        oscillating_losses = [1.0, 0.5, 1.2, 0.3, 1.5, 0.2, 1.8]
        
        for loss in oscillating_losses:
            analyzer.update(loss)
        
        # Should not detect convergence
        assert analyzer.is_converged() == False
    
    def test_adaptive_learning_rate(self):
        """Test adaptive learning rate based on convergence."""
        analyzer = ConvergenceAnalyzer(window_size=3)
        
        # Test learning rate adaptation
        initial_lr = 0.1
        
        # Good convergence should maintain or slightly increase LR
        good_losses = [1.0, 0.8, 0.6, 0.5]
        for loss in good_losses:
            analyzer.update(loss)
        
        adapted_lr = analyzer.adapt_learning_rate(initial_lr)
        assert adapted_lr >= initial_lr * 0.9  # Should not decrease much
        
        # Poor convergence should decrease LR
        analyzer = ConvergenceAnalyzer(window_size=3)
        poor_losses = [1.0, 1.1, 1.2, 1.3]
        for loss in poor_losses:
            analyzer.update(loss)
        
        adapted_lr = analyzer.adapt_learning_rate(initial_lr)
        assert adapted_lr < initial_lr  # Should decrease


class TestAdaptiveAggregationOrchestrator:
    """Test main adaptive aggregation orchestrator."""
    
    def test_orchestrator_creation(self, basic_config):
        """Test orchestrator creation."""
        orchestrator = AdaptiveAggregationOrchestrator(basic_config)
        
        assert orchestrator.config == basic_config
        assert orchestrator.convergence_analyzer is not None
        assert orchestrator.round_count == 0
    
    def test_fedavg_aggregation(self, orchestrator, simple_model):
        """Test basic FedAvg aggregation."""
        # Initialize with model
        orchestrator.initialize_model(simple_model)
        
        # Create mock client updates
        client_updates = create_mock_client_updates(num_clients=3, param_size=10, add_noise=True)
        client_weights = create_mock_client_weights(list(client_updates.keys()))
        
        # Perform aggregation
        result = orchestrator.aggregate(client_updates, client_weights)
        
        # Verify result
        assert isinstance(result, AdaptiveAggregationResult)
        assert result.round_id == 0
        assert result.aggregation_method == "fedavg"
        assert result.global_model_params is not None
        assert len(result.participating_clients) == 3
        assert result.convergence_metrics is not None
    
    def test_fedopt_aggregation(self, fedopt_config, simple_model):
        """Test FedOpt aggregation."""
        orchestrator = AdaptiveAggregationOrchestrator(fedopt_config)
        orchestrator.initialize_model(simple_model)
        
        client_updates = create_mock_client_updates(num_clients=4, param_size=10)
        client_weights = create_mock_client_weights(list(client_updates.keys()))
        
        # Perform multiple rounds to test server-side optimization
        results = []
        for round_num in range(3):
            result = orchestrator.aggregate(client_updates, client_weights)
            results.append(result)
            
            # Verify FedOpt specific features
            assert result.aggregation_method == "fedopt"
            assert 'server_optimizer_state' in result.aggregation_info
            assert result.adapted_learning_rate > 0
        
        # Should have optimizer state history
        assert orchestrator.round_count == 3
    
    def test_robust_aggregation(self, robust_config, simple_model):
        """Test robust aggregation."""
        orchestrator = AdaptiveAggregationOrchestrator(robust_config)
        orchestrator.initialize_model(simple_model)
        
        # Include Byzantine clients
        client_updates = create_mock_client_updates(
            num_clients=6, 
            param_size=10, 
            byzantine_count=2
        )
        client_weights = create_mock_client_weights(list(client_updates.keys()))
        
        result = orchestrator.aggregate(client_updates, client_weights)
        
        # Should handle Byzantine clients
        assert result.aggregation_method == "robust"
        assert 'byzantine_detection' in result.aggregation_info
        assert result.global_model_params is not None
        
        # Parameters should not be corrupted by Byzantine clients
        for param_name, param_value in result.global_model_params.items():
            assert torch.all(torch.abs(param_value) < 50)  # Should exclude Byzantine values
    
    def test_dynamic_client_selection(self, dynamic_selection_config, simple_model):
        """Test dynamic client selection."""
        orchestrator = AdaptiveAggregationOrchestrator(dynamic_selection_config)
        orchestrator.initialize_model(simple_model)
        
        # Create many clients
        client_updates = create_mock_client_updates(num_clients=15, param_size=10)
        client_weights = create_mock_client_weights(list(client_updates.keys()))
        
        # Mock client performance history
        client_performances = {
            client_id: [np.random.uniform(0.3, 0.9) for _ in range(3)]
            for client_id in client_updates.keys()
        }
        
        result = orchestrator.aggregate(
            client_updates, 
            client_weights, 
            client_performances=client_performances
        )
        
        # Should select subset of clients
        assert len(result.participating_clients) <= 10  # max_clients_per_round
        assert 'client_selection_info' in result.aggregation_info
    
    def test_adaptive_learning_rate(self, orchestrator, simple_model):
        """Test adaptive learning rate adjustment."""
        orchestrator.initialize_model(simple_model)
        
        # Create updates that simulate convergence
        client_updates = create_mock_client_updates(num_clients=3, param_size=10, add_noise=False)
        client_weights = create_mock_client_weights(list(client_updates.keys()))
        
        # Perform multiple rounds with decreasing loss
        initial_lr = orchestrator.config.server_learning_rate
        losses = [1.0, 0.8, 0.6, 0.5, 0.45, 0.42]
        
        for i, target_loss in enumerate(losses):
            # Mock the loss for this round
            result = orchestrator.aggregate(client_updates, client_weights)
            orchestrator.convergence_analyzer.update(target_loss)
            
            if i > 2:  # After some rounds
                # Learning rate should adapt based on convergence
                current_lr = result.adapted_learning_rate
                assert current_lr > 0
    
    def test_convergence_detection(self, orchestrator, simple_model):
        """Test convergence detection."""
        orchestrator.initialize_model(simple_model)
        
        client_updates = create_mock_client_updates(num_clients=3, param_size=5, add_noise=False)
        client_weights = create_mock_client_weights(list(client_updates.keys()))
        
        # Simulate convergence with small losses
        converged_losses = [0.05, 0.04, 0.035, 0.032, 0.031]
        
        for loss in converged_losses:
            result = orchestrator.aggregate(client_updates, client_weights)
            orchestrator.convergence_analyzer.update(loss)
        
        # Should detect convergence
        assert orchestrator.convergence_analyzer.is_converged() == True
        
        # Final result should indicate convergence
        final_result = orchestrator.aggregate(client_updates, client_weights)
        assert final_result.convergence_metrics['is_converged'] == True
    
    def test_algorithm_stats(self, orchestrator, simple_model):
        """Test algorithm statistics."""
        orchestrator.initialize_model(simple_model)
        
        # Perform some rounds
        client_updates = create_mock_client_updates(num_clients=3, param_size=5)
        client_weights = create_mock_client_weights(list(client_updates.keys()))
        
        for _ in range(3):
            orchestrator.aggregate(client_updates, client_weights)
        
        stats = orchestrator.get_algorithm_stats()
        
        assert 'round_count' in stats
        assert 'aggregation_method' in stats
        assert 'convergence_status' in stats
        assert stats['round_count'] == 3
        assert stats['aggregation_method'] == orchestrator.config.method


class TestAdaptiveAggregationIntegration:
    """Integration tests for adaptive aggregation."""
    
    def test_full_adaptive_round(self):
        """Test a complete adaptive aggregation round."""
        config = AdaptiveAggregationConfig(
            method="fedopt",
            fedopt=FedOptConfig(optimizer="fedadam"),
            robust_aggregation=RobustAggregationConfig(
                method="median",
                byzantine_tolerance=0.2
            ),
            dynamic_client_selection=DynamicClientSelectionConfig(
                enabled=True,
                max_clients_per_round=5
            ),
            adaptive_lr=True
        )
        
        orchestrator = AdaptiveAggregationOrchestrator(config)
        model = SimpleModel()
        orchestrator.initialize_model(model)
        
        # Create diverse client updates
        client_updates = create_mock_client_updates(
            num_clients=8, 
            param_size=10, 
            byzantine_count=1
        )
        client_weights = create_mock_client_weights(list(client_updates.keys()))
        
        # Mock performance history
        client_performances = {
            client_id: [np.random.uniform(0.4, 0.8) for _ in range(5)]
            for client_id in client_updates.keys()
            if not client_id.startswith('byzantine')
        }
        
        # Byzantine clients have poor performance
        for client_id in client_updates.keys():
            if client_id.startswith('byzantine'):
                client_performances[client_id] = [0.1, 0.1, 0.1, 0.1, 0.1]
        
        # Perform aggregation
        result = orchestrator.aggregate(
            client_updates, 
            client_weights,
            client_performances=client_performances
        )
        
        # Verify comprehensive result
        assert isinstance(result, AdaptiveAggregationResult)
        assert result.global_model_params is not None
        assert len(result.participating_clients) <= 5  # Dynamic selection
        assert 'client_selection_info' in result.aggregation_info
        assert 'byzantine_detection' in result.aggregation_info
        assert result.adapted_learning_rate > 0
        assert result.convergence_metrics is not None
    
    def test_multiple_rounds_adaptation(self):
        """Test adaptation over multiple rounds."""
        config = AdaptiveAggregationConfig(
            method="fedopt",
            fedopt=FedOptConfig(optimizer="fedyogi", server_learning_rate=0.01),
            adaptive_lr=True,
            convergence_window=3
        )
        
        orchestrator = AdaptiveAggregationOrchestrator(config)
        model = SimpleModel()
        orchestrator.initialize_model(model)
        
        # Simulate training over multiple rounds
        client_updates = create_mock_client_updates(num_clients=4, param_size=5)
        client_weights = create_mock_client_weights(list(client_updates.keys()))
        
        # Track adaptation over rounds
        learning_rates = []
        convergence_rates = []
        
        # Simulate decreasing loss over time (convergence)
        simulated_losses = [1.0, 0.7, 0.5, 0.4, 0.35, 0.32, 0.31, 0.305]
        
        for i, loss in enumerate(simulated_losses):
            result = orchestrator.aggregate(client_updates, client_weights)
            orchestrator.convergence_analyzer.update(loss)
            
            learning_rates.append(result.adapted_learning_rate)
            convergence_rates.append(
                result.convergence_metrics.get('convergence_rate', 0.0)
            )
        
        # Verify adaptation occurred
        assert len(learning_rates) == 8
        assert len(convergence_rates) == 8
        
        # Should eventually detect convergence
        final_result = orchestrator.aggregate(client_updates, client_weights)
        assert final_result.convergence_metrics.get('is_converged', False) == True
    
    @pytest.mark.parametrize("method", ["fedavg", "fedopt", "robust"])
    def test_different_aggregation_methods(self, method):
        """Test different aggregation methods."""
        if method == "fedopt":
            config = AdaptiveAggregationConfig(
                method=method,
                fedopt=FedOptConfig(optimizer="fedadam")
            )
        elif method == "robust":
            config = AdaptiveAggregationConfig(
                method=method,
                robust_aggregation=RobustAggregationConfig(method="krum")
            )
        else:  # fedavg
            config = AdaptiveAggregationConfig(method=method)
        
        orchestrator = AdaptiveAggregationOrchestrator(config)
        model = SimpleModel()
        orchestrator.initialize_model(model)
        
        client_updates = create_mock_client_updates(
            num_clients=5,
            param_size=8,
            byzantine_count=1 if method == "robust" else 0
        )
        client_weights = create_mock_client_weights(list(client_updates.keys()))
        
        try:
            result = orchestrator.aggregate(client_updates, client_weights)
            
            # All methods should complete successfully
            assert isinstance(result, AdaptiveAggregationResult)
            assert result.aggregation_method == method
            assert result.global_model_params is not None
            
        except Exception as e:
            pytest.fail(f"Method {method} failed: {e}")


class TestAdaptiveAggregationPerformance:
    """Performance tests for adaptive aggregation."""
    
    def test_large_scale_aggregation(self):
        """Test aggregation with many clients."""
        config = AdaptiveAggregationConfig(
            method="fedavg",
            dynamic_client_selection=DynamicClientSelectionConfig(
                enabled=True,
                max_clients_per_round=20
            )
        )
        
        orchestrator = AdaptiveAggregationOrchestrator(config)
        model = SimpleModel(input_size=100, hidden_size=256, output_size=50)
        orchestrator.initialize_model(model)
        
        # Create many client updates
        num_clients = 100
        client_updates = {}
        
        for i in range(num_clients):
            client_id = f'client{i:03d}'
            # Simplified updates for performance
            client_updates[client_id] = {
                'linear1.weight': torch.randn(256, 100) * 0.1,
                'linear1.bias': torch.randn(256) * 0.1,
                'linear2.weight': torch.randn(50, 256) * 0.1,
                'linear2.bias': torch.randn(50) * 0.1
            }
        
        client_weights = create_mock_client_weights(list(client_updates.keys()))
        
        # Should handle large scale efficiently
        result = orchestrator.aggregate(client_updates, client_weights)
        
        assert result.global_model_params is not None
        assert len(result.participating_clients) <= 20  # Dynamic selection
        assert result.aggregation_method == "fedavg"
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large models."""
        config = AdaptiveAggregationConfig(method="fedavg")
        orchestrator = AdaptiveAggregationOrchestrator(config)
        
        # Create larger model
        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(1000, 500),
                    nn.ReLU(),
                    nn.Linear(500, 250),
                    nn.ReLU(),
                    nn.Linear(250, 100),
                    nn.ReLU(),
                    nn.Linear(100, 10)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        large_model = LargeModel()
        orchestrator.initialize_model(large_model)
        
        # Create client updates
        client_updates = {}
        for i in range(5):
            client_id = f'client{i+1}'
            client_updates[client_id] = {
                name: param.clone() + torch.randn_like(param) * 0.01
                for name, param in large_model.named_parameters()
            }
        
        client_weights = create_mock_client_weights(list(client_updates.keys()))
        
        # Should handle large model efficiently
        result = orchestrator.aggregate(client_updates, client_weights)
        
        assert result.global_model_params is not None
        assert len(result.global_model_params) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])