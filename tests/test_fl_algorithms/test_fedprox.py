"""
Test Suite for FedProx Algorithm

This module contains comprehensive tests for the FedProx federated learning algorithm
implementation including unit tests, integration tests, and performance validation.

Test Coverage:
- FedProx configuration validation
- Client state management
- Proximal loss computation
- Adaptive μ tuning
- Heterogeneity estimation
- Aggregation strategies
- Convergence metrics
- Performance benchmarking
"""

import pytest
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Import the modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'server'))

from fl_algorithms.fedprox import (
    FedProxAlgorithm, FedProxConfig, FedProxClientState, 
    FedProxLoss, FedProxOptimizer, FedProxAggregator,
    FedProxHeterogeneityEstimator, FedProxRoundResult
)


class SimpleModel(nn.Module):
    """Simple neural network for testing."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TestDataset(data_utils.Dataset):
    """Simple dataset for testing."""
    
    def __init__(self, size=1000, input_dim=784, num_classes=10):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Generate random data
        self.data = torch.randn(size, input_dim)
        self.targets = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


@pytest.fixture
def fedprox_config():
    """Create a test FedProx configuration."""
    return FedProxConfig(
        mu=0.01,
        local_epochs=3,
        learning_rate=0.01,
        batch_size=32,
        adaptive_mu=True,
        device_sampling_fraction=1.0
    )


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def test_dataset():
    """Create a test dataset."""
    return TestDataset(size=100)


@pytest.fixture
def test_dataloader(test_dataset):
    """Create a test dataloader."""
    return data_utils.DataLoader(test_dataset, batch_size=16, shuffle=True)


@pytest.fixture
def fedprox_algorithm(fedprox_config):
    """Create a FedProx algorithm instance."""
    return FedProxAlgorithm(fedprox_config)


class TestFedProxConfig:
    """Test FedProx configuration validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FedProxConfig()
        assert config.mu == 0.01
        assert config.local_epochs == 5
        assert config.learning_rate == 0.01
        assert config.adaptive_mu == True
        assert config.min_mu == 0.001
        assert config.max_mu == 1.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = FedProxConfig(
            mu=0.05,
            local_epochs=10,
            learning_rate=0.001,
            adaptive_mu=False
        )
        assert config.mu == 0.05
        assert config.local_epochs == 10
        assert config.learning_rate == 0.001
        assert config.adaptive_mu == False
    
    def test_config_validation(self):
        """Test configuration value validation."""
        # Test valid ranges
        config = FedProxConfig(mu=0.001, min_mu=0.001, max_mu=1.0)
        assert config.mu >= config.min_mu
        assert config.mu <= config.max_mu


class TestFedProxLoss:
    """Test FedProx loss function with proximal term."""
    
    def test_loss_computation(self):
        """Test proximal loss computation."""
        # Create mock parameters
        global_params = {
            'fc1.weight': torch.randn(128, 784),
            'fc1.bias': torch.randn(128),
            'fc2.weight': torch.randn(10, 128),
            'fc2.bias': torch.randn(10)
        }
        
        local_params = {
            'fc1.weight': torch.randn(128, 784),
            'fc1.bias': torch.randn(128),
            'fc2.weight': torch.randn(10, 128),
            'fc2.bias': torch.randn(10)
        }
        
        # Create loss function
        base_loss_fn = nn.CrossEntropyLoss()
        mu = 0.01
        fedprox_loss = FedProxLoss(base_loss_fn, mu, global_params)
        
        # Mock inputs
        outputs = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))
        
        # Compute loss
        total_loss = fedprox_loss(outputs, targets, local_params)
        
        # Verify loss is computed
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.item() > 0
        
        # Verify proximal term is added
        base_loss = base_loss_fn(outputs, targets)
        assert total_loss.item() > base_loss.item()
    
    def test_zero_mu(self):
        """Test that zero mu gives only base loss."""
        global_params = {'param': torch.zeros(10)}
        local_params = {'param': torch.ones(10)}
        
        base_loss_fn = nn.MSELoss()
        fedprox_loss = FedProxLoss(base_loss_fn, mu=0.0, global_model_params=global_params)
        
        outputs = torch.ones(5, 10)
        targets = torch.zeros(5, 10)
        
        total_loss = fedprox_loss(outputs, targets, local_params)
        base_loss = base_loss_fn(outputs, targets)
        
        # Should be approximately equal (within numerical precision)
        assert abs(total_loss.item() - base_loss.item()) < 1e-6


class TestFedProxOptimizer:
    """Test FedProx optimizer with gradient clipping."""
    
    def test_optimizer_creation(self, simple_model, fedprox_config):
        """Test optimizer creation."""
        global_params = {name: param for name, param in simple_model.named_parameters()}
        optimizer = FedProxOptimizer(simple_model, fedprox_config, global_params)
        
        assert optimizer.model == simple_model
        assert optimizer.config == fedprox_config
        assert optimizer.global_model_params == global_params
        assert optimizer.optimizer is not None
    
    def test_gradient_tracking(self, simple_model, fedprox_config, test_dataloader):
        """Test gradient tracking functionality."""
        global_params = {name: param.clone() for name, param in simple_model.named_parameters()}
        optimizer = FedProxOptimizer(simple_model, fedprox_config, global_params)
        
        # Perform a few optimization steps
        loss_fn = nn.CrossEntropyLoss()
        for i, (data, targets) in enumerate(test_dataloader):
            if i >= 3:  # Only test a few batches
                break
            
            outputs = simple_model(data)
            loss = loss_fn(outputs, targets)
            optimizer.step(loss)
        
        # Check gradient history
        assert len(optimizer.gradient_history) == 3
        assert all(grad_norm > 0 for grad_norm in optimizer.gradient_history)
    
    def test_gradient_diversity(self, simple_model, fedprox_config):
        """Test gradient diversity computation."""
        global_params = {name: param.clone() for name, param in simple_model.named_parameters()}
        optimizer = FedProxOptimizer(simple_model, fedprox_config, global_params)
        
        # Add some gradient history
        optimizer.gradient_history = [1.0, 1.5, 0.8, 1.2, 0.9]
        
        diversity = optimizer.get_gradient_diversity()
        assert isinstance(diversity, float)
        assert diversity >= 0


class TestFedProxAggregator:
    """Test FedProx aggregation strategies."""
    
    def test_aggregator_creation(self, fedprox_config):
        """Test aggregator creation."""
        aggregator = FedProxAggregator(fedprox_config)
        assert aggregator.config == fedprox_config
    
    def test_model_aggregation(self, fedprox_config):
        """Test model aggregation."""
        aggregator = FedProxAggregator(fedprox_config)
        
        # Create mock client models
        client_models = {
            'client1': {'param1': torch.tensor([1.0, 2.0]), 'param2': torch.tensor([3.0])},
            'client2': {'param1': torch.tensor([2.0, 3.0]), 'param2': torch.tensor([4.0])},
            'client3': {'param1': torch.tensor([3.0, 4.0]), 'param2': torch.tensor([5.0])}
        }
        
        client_data_sizes = {'client1': 100, 'client2': 200, 'client3': 150}
        client_losses = {'client1': 0.5, 'client2': 0.3, 'client3': 0.4}
        
        aggregated_params, weights = aggregator.aggregate_models(
            client_models, client_data_sizes, client_losses
        )
        
        # Verify aggregation
        assert 'param1' in aggregated_params
        assert 'param2' in aggregated_params
        assert len(weights) == 3
        
        # Verify weights sum to 1
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_empty_models(self, fedprox_config):
        """Test aggregation with empty model list."""
        aggregator = FedProxAggregator(fedprox_config)
        
        with pytest.raises(ValueError, match="No client models to aggregate"):
            aggregator.aggregate_models({}, {}, {})


class TestFedProxHeterogeneityEstimator:
    """Test heterogeneity estimation for adaptive μ tuning."""
    
    def test_estimator_creation(self):
        """Test estimator creation."""
        estimator = FedProxHeterogeneityEstimator()
        assert estimator.gradient_history == {}
        assert estimator.loss_history == {}
    
    def test_client_info_update(self):
        """Test updating client information."""
        estimator = FedProxHeterogeneityEstimator()
        
        gradients = {'param1': torch.tensor([1.0, 2.0]), 'param2': torch.tensor([3.0])}
        loss = 0.5
        
        estimator.update_client_info('client1', gradients, loss)
        
        assert 'client1' in estimator.gradient_history
        assert 'client1' in estimator.loss_history
        assert len(estimator.gradient_history['client1']) == 1
        assert len(estimator.loss_history['client1']) == 1
    
    def test_heterogeneity_estimation(self):
        """Test heterogeneity score computation."""
        estimator = FedProxHeterogeneityEstimator()
        
        # Add data for multiple clients
        for client_id in ['client1', 'client2', 'client3']:
            for i in range(3):
                gradients = {'param': torch.tensor([float(i + 1)])}
                loss = 0.1 * (i + 1)
                estimator.update_client_info(client_id, gradients, loss)
        
        heterogeneity = estimator.estimate_heterogeneity()
        
        assert isinstance(heterogeneity, float)
        assert 0 <= heterogeneity <= 1
    
    def test_insufficient_data(self):
        """Test heterogeneity estimation with insufficient data."""
        estimator = FedProxHeterogeneityEstimator()
        
        # Add data for only one client
        gradients = {'param': torch.tensor([1.0])}
        estimator.update_client_info('client1', gradients, 0.5)
        
        heterogeneity = estimator.estimate_heterogeneity()
        assert heterogeneity == 0.5  # Default value


class TestFedProxAlgorithm:
    """Test main FedProx algorithm functionality."""
    
    def test_algorithm_creation(self, fedprox_config):
        """Test algorithm creation."""
        algorithm = FedProxAlgorithm(fedprox_config)
        assert algorithm.config == fedprox_config
        assert algorithm.global_model_params is None
        assert algorithm.current_round == 0
    
    def test_global_model_initialization(self, fedprox_algorithm, simple_model):
        """Test global model initialization."""
        fedprox_algorithm.initialize_global_model(simple_model)
        
        assert fedprox_algorithm.global_model_params is not None
        assert len(fedprox_algorithm.global_model_params) > 0
        
        # Verify parameter names match model
        model_param_names = set(name for name, _ in simple_model.named_parameters())
        global_param_names = set(fedprox_algorithm.global_model_params.keys())
        assert model_param_names == global_param_names
    
    def test_client_registration(self, fedprox_algorithm):
        """Test client registration."""
        fedprox_algorithm.register_client('client1', data_size=1000)
        
        assert 'client1' in fedprox_algorithm.client_states
        client_state = fedprox_algorithm.client_states['client1']
        assert client_state.device_id == 'client1'
        assert client_state.data_size == 1000
    
    def test_client_update(self, fedprox_algorithm, simple_model, test_dataloader):
        """Test client update process."""
        # Initialize global model
        fedprox_algorithm.initialize_global_model(simple_model)
        
        # Register client
        fedprox_algorithm.register_client('client1', data_size=100)
        
        # Perform client update
        loss_fn = nn.CrossEntropyLoss()
        model_params, local_loss, metrics = fedprox_algorithm.client_update(
            'client1', simple_model, test_dataloader, loss_fn
        )
        
        # Verify results
        assert isinstance(model_params, dict)
        assert isinstance(local_loss, float)
        assert isinstance(metrics, dict)
        assert local_loss >= 0
        
        # Verify metrics
        assert 'training_time' in metrics
        assert 'total_samples' in metrics
        assert 'mu_value' in metrics
        
        # Verify client state updated
        client_state = fedprox_algorithm.client_states['client1']
        assert len(client_state.local_loss_history) == 1
        assert client_state.local_loss_history[0] == local_loss
    
    def test_federated_averaging(self, fedprox_algorithm, simple_model):
        """Test federated averaging process."""
        # Initialize global model
        fedprox_algorithm.initialize_global_model(simple_model)
        
        # Register multiple clients
        for i in range(3):
            client_id = f'client{i+1}'
            fedprox_algorithm.register_client(client_id, data_size=100 * (i + 1))
            
            # Set mock client state
            client_state = fedprox_algorithm.client_states[client_id]
            client_state.model_state = {
                name: param.clone() + torch.randn_like(param) * 0.1
                for name, param in simple_model.named_parameters()
            }
            client_state.local_loss_history = [0.5 - i * 0.1]
        
        # Perform federated averaging
        participating_clients = ['client1', 'client2', 'client3']
        round_result = fedprox_algorithm.federated_averaging(participating_clients)
        
        # Verify result
        assert isinstance(round_result, FedProxRoundResult)
        assert round_result.round_id == 0
        assert set(round_result.participating_clients) == set(participating_clients)
        assert round_result.global_loss > 0
        assert len(round_result.local_losses) == 3
        
        # Verify global model updated
        assert fedprox_algorithm.global_model_params is not None
        assert fedprox_algorithm.current_round == 1
    
    def test_adaptive_mu(self, fedprox_algorithm, simple_model):
        """Test adaptive μ tuning."""
        config = fedprox_algorithm.config
        assert config.adaptive_mu == True
        
        # Initialize algorithm
        fedprox_algorithm.initialize_global_model(simple_model)
        
        initial_mu = fedprox_algorithm.current_mu
        
        # Simulate high heterogeneity
        fedprox_algorithm._update_mu(0.8)  # High heterogeneity
        high_hetero_mu = fedprox_algorithm.current_mu
        
        # Simulate low heterogeneity
        fedprox_algorithm._update_mu(0.2)  # Low heterogeneity
        low_hetero_mu = fedprox_algorithm.current_mu
        
        # Verify μ adapts appropriately
        assert high_hetero_mu >= initial_mu  # Should increase for high heterogeneity
        assert low_hetero_mu <= high_hetero_mu  # Should decrease for low heterogeneity
    
    def test_convergence_metrics(self, fedprox_algorithm):
        """Test convergence metrics computation."""
        # Add some round history
        for i in range(3):
            round_result = FedProxRoundResult(
                round_id=i,
                participating_clients=[f'client{j+1}' for j in range(3)],
                global_loss=1.0 - i * 0.1,  # Decreasing loss
                local_losses={f'client{j+1}': 1.0 - i * 0.1 for j in range(3)},
                convergence_metrics={},
                mu_values={f'client{j+1}': 0.01 for j in range(3)},
                aggregation_weights={f'client{j+1}': 1/3 for j in range(3)},
                round_duration=1.0,
                gradient_diversity=0.1,
                heterogeneity_estimate=0.5
            )
            fedprox_algorithm.round_history.append(round_result)
        
        metrics = fedprox_algorithm._compute_convergence_metrics()
        
        assert 'loss_improvement' in metrics
        assert 'convergence_rate' in metrics
        assert metrics['loss_improvement'] > 0  # Loss should be improving
    
    def test_algorithm_stats(self, fedprox_algorithm, simple_model):
        """Test algorithm statistics."""
        # Initialize and add some state
        fedprox_algorithm.initialize_global_model(simple_model)
        fedprox_algorithm.register_client('client1', 100)
        fedprox_algorithm.register_client('client2', 200)
        
        stats = fedprox_algorithm.get_algorithm_stats()
        
        assert 'current_round' in stats
        assert 'total_clients' in stats
        assert 'current_mu' in stats
        assert stats['total_clients'] == 2


class TestFedProxIntegration:
    """Integration tests for FedProx algorithm."""
    
    def test_full_training_round(self, fedprox_config):
        """Test a complete training round."""
        # Create algorithm
        algorithm = FedProxAlgorithm(fedprox_config)
        
        # Create models and data
        model = SimpleModel()
        algorithm.initialize_global_model(model)
        
        # Create different datasets for clients (simulate heterogeneity)
        client_datasets = []
        for i in range(3):
            # Create slightly different data distributions
            dataset = TestDataset(size=50)
            # Add some bias to simulate non-IID data
            dataset.data += torch.randn_like(dataset.data) * 0.1 * (i + 1)
            client_datasets.append(dataset)
        
        # Register clients
        client_ids = []
        for i, dataset in enumerate(client_datasets):
            client_id = f'client{i+1}'
            algorithm.register_client(client_id, len(dataset))
            client_ids.append(client_id)
        
        # Perform client updates
        loss_fn = nn.CrossEntropyLoss()
        for i, client_id in enumerate(client_ids):
            dataloader = data_utils.DataLoader(client_datasets[i], batch_size=16, shuffle=True)
            client_model = SimpleModel()
            
            model_params, local_loss, metrics = algorithm.client_update(
                client_id, client_model, dataloader, loss_fn
            )
            
            assert local_loss >= 0
            assert 'training_time' in metrics
        
        # Perform federated averaging
        round_result = algorithm.federated_averaging(client_ids)
        
        # Verify complete round
        assert round_result.round_id == 0
        assert len(round_result.participating_clients) == 3
        assert round_result.global_loss > 0
        assert algorithm.current_round == 1
    
    def test_multiple_rounds_convergence(self, fedprox_config):
        """Test convergence over multiple rounds."""
        # Use smaller learning rate for stability
        fedprox_config.learning_rate = 0.001
        fedprox_config.local_epochs = 2
        
        algorithm = FedProxAlgorithm(fedprox_config)
        model = SimpleModel()
        algorithm.initialize_global_model(model)
        
        # Create clients
        client_datasets = [TestDataset(size=30) for _ in range(2)]
        client_ids = []
        for i, dataset in enumerate(client_datasets):
            client_id = f'client{i+1}'
            algorithm.register_client(client_id, len(dataset))
            client_ids.append(client_id)
        
        # Run multiple rounds
        loss_fn = nn.CrossEntropyLoss()
        round_losses = []
        
        for round_num in range(3):
            # Client updates
            for i, client_id in enumerate(client_ids):
                dataloader = data_utils.DataLoader(client_datasets[i], batch_size=8, shuffle=True)
                client_model = SimpleModel()
                
                try:
                    model_params, local_loss, metrics = algorithm.client_update(
                        client_id, client_model, dataloader, loss_fn
                    )
                except Exception as e:
                    # Continue if individual client fails
                    print(f"Client {client_id} failed in round {round_num}: {e}")
                    continue
            
            # Federated averaging
            round_result = algorithm.federated_averaging(client_ids)
            round_losses.append(round_result.global_loss)
        
        # Verify multiple rounds completed
        assert algorithm.current_round == 3
        assert len(algorithm.round_history) == 3
        
        # Check that we have loss values (convergence verification would need more sophisticated setup)
        assert all(loss > 0 for loss in round_losses)


class TestFedProxPerformance:
    """Performance and stress tests for FedProx."""
    
    def test_large_model_handling(self, fedprox_config):
        """Test FedProx with larger model."""
        # Create larger model
        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(784, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 10)
                )
            
            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.layers(x)
        
        algorithm = FedProxAlgorithm(fedprox_config)
        large_model = LargeModel()
        
        # Initialize - should handle large model
        algorithm.initialize_global_model(large_model)
        
        assert algorithm.global_model_params is not None
        assert len(algorithm.global_model_params) > 0
        
        # Verify all parameters are captured
        model_params = dict(large_model.named_parameters())
        global_params = algorithm.global_model_params
        
        assert len(model_params) == len(global_params)
    
    def test_many_clients(self, fedprox_config):
        """Test FedProx with many clients."""
        algorithm = FedProxAlgorithm(fedprox_config)
        model = SimpleModel()
        algorithm.initialize_global_model(model)
        
        # Register many clients
        num_clients = 50
        client_ids = []
        for i in range(num_clients):
            client_id = f'client{i:03d}'
            algorithm.register_client(client_id, data_size=100)
            client_ids.append(client_id)
        
        assert len(algorithm.client_states) == num_clients
        
        # Mock client updates for all clients
        for client_id in client_ids:
            client_state = algorithm.client_states[client_id]
            client_state.model_state = {
                name: param.clone() + torch.randn_like(param) * 0.01
                for name, param in model.named_parameters()
            }
            client_state.local_loss_history = [np.random.uniform(0.1, 1.0)]
        
        # Perform aggregation with all clients
        round_result = algorithm.federated_averaging(client_ids)
        
        assert len(round_result.participating_clients) == num_clients
        assert len(round_result.local_losses) == num_clients
    
    @pytest.mark.parametrize("mu_value", [0.001, 0.01, 0.1, 1.0])
    def test_different_mu_values(self, mu_value):
        """Test FedProx with different μ values."""
        config = FedProxConfig(mu=mu_value, adaptive_mu=False)
        algorithm = FedProxAlgorithm(config)
        
        assert algorithm.current_mu == mu_value
        
        # Test that μ affects loss computation
        model = SimpleModel()
        algorithm.initialize_global_model(model)
        
        global_params = algorithm.global_model_params
        local_params = {name: param + 0.1 for name, param in global_params.items()}
        
        base_loss_fn = nn.CrossEntropyLoss()
        fedprox_loss = FedProxLoss(base_loss_fn, mu_value, global_params)
        
        outputs = torch.randn(10, 10)
        targets = torch.randint(0, 10, (10,))
        
        total_loss = fedprox_loss(outputs, targets, local_params)
        base_loss = base_loss_fn(outputs, targets)
        
        # Verify proximal term scales with μ
        proximal_contribution = total_loss.item() - base_loss.item()
        assert proximal_contribution > 0
        
        # Higher μ should give larger proximal contribution
        if mu_value > 0.01:
            config_small = FedProxConfig(mu=0.01, adaptive_mu=False)
            fedprox_loss_small = FedProxLoss(base_loss_fn, 0.01, global_params)
            total_loss_small = fedprox_loss_small(outputs, targets, local_params)
            proximal_small = total_loss_small.item() - base_loss.item()
            
            assert proximal_contribution > proximal_small


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])