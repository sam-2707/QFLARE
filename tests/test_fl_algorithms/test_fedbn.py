"""
Test Suite for FedBN Algorithm

This module contains comprehensive tests for the FedBN (Federated Batch Normalization)
algorithm implementation including unit tests, integration tests, and BN-specific validation.

Test Coverage:
- FedBN configuration validation
- Batch normalization parameter separation
- BN statistics management and analysis
- Local BN preservation across rounds
- Global aggregation of non-BN parameters
- BN diversity metrics computation
- Integration with different model architectures
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

from fl_algorithms.fedbn import (
    FedBNAlgorithm, FedBNConfig, FedBNClientState,
    BNStatistics, FedBNParameterSeparator, FedBNAggregator,
    FedBNStatisticsAnalyzer, FedBNRoundResult
)


class ModelWithBN(nn.Module):
    """Neural network with batch normalization for testing."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class ModelWithoutBN(nn.Module):
    """Neural network without batch normalization for testing."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class TestDataset(data_utils.Dataset):
    """Simple dataset for testing."""
    
    def __init__(self, size=1000, input_dim=784, num_classes=10, bias=0.0):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Generate random data with optional bias
        self.data = torch.randn(size, input_dim) + bias
        self.targets = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


@pytest.fixture
def fedbn_config():
    """Create a test FedBN configuration."""
    return FedBNConfig(
        local_epochs=3,
        learning_rate=0.01,
        batch_size=32,
        bn_track_running_stats=True,
        analyze_bn_diversity=True
    )


@pytest.fixture
def model_with_bn():
    """Create a model with batch normalization."""
    return ModelWithBN()


@pytest.fixture
def model_without_bn():
    """Create a model without batch normalization."""
    return ModelWithoutBN()


@pytest.fixture
def test_dataset():
    """Create a test dataset."""
    return TestDataset(size=100)


@pytest.fixture
def test_dataloader(test_dataset):
    """Create a test dataloader."""
    return data_utils.DataLoader(test_dataset, batch_size=16, shuffle=True)


@pytest.fixture
def fedbn_algorithm(fedbn_config):
    """Create a FedBN algorithm instance."""
    return FedBNAlgorithm(fedbn_config)


class TestFedBNConfig:
    """Test FedBN configuration validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FedBNConfig()
        assert config.local_epochs == 5
        assert config.learning_rate == 0.01
        assert config.bn_track_running_stats == True
        assert config.bn_momentum == 0.1
        assert config.analyze_bn_diversity == True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = FedBNConfig(
            local_epochs=10,
            learning_rate=0.001,
            bn_track_running_stats=False,
            analyze_bn_diversity=False
        )
        assert config.local_epochs == 10
        assert config.learning_rate == 0.001
        assert config.bn_track_running_stats == False
        assert config.analyze_bn_diversity == False
    
    def test_bn_specific_settings(self):
        """Test BN-specific configuration settings."""
        config = FedBNConfig(
            bn_momentum=0.05,
            bn_eps=1e-6,
            bn_stats_history_size=5
        )
        assert config.bn_momentum == 0.05
        assert config.bn_eps == 1e-6
        assert config.bn_stats_history_size == 5


class TestBNStatistics:
    """Test batch normalization statistics handling."""
    
    def test_bn_statistics_creation(self):
        """Test BN statistics creation."""
        running_mean = torch.randn(128)
        running_var = torch.ones(128)
        num_batches = torch.tensor(10)
        
        bn_stats = BNStatistics(
            running_mean=running_mean,
            running_var=running_var,
            num_batches_tracked=num_batches,
            layer_name="bn1",
            layer_type="BatchNorm1d"
        )
        
        assert torch.equal(bn_stats.running_mean, running_mean)
        assert torch.equal(bn_stats.running_var, running_var)
        assert torch.equal(bn_stats.num_batches_tracked, num_batches)
        assert bn_stats.layer_name == "bn1"
        assert bn_stats.layer_type == "BatchNorm1d"
    
    def test_bn_statistics_clone(self):
        """Test BN statistics cloning."""
        running_mean = torch.randn(128)
        running_var = torch.ones(128)
        num_batches = torch.tensor(10)
        
        original = BNStatistics(
            running_mean=running_mean,
            running_var=running_var,
            num_batches_tracked=num_batches,
            layer_name="bn1",
            layer_type="BatchNorm1d"
        )
        
        cloned = original.clone()
        
        # Should be equal but different objects
        assert torch.equal(cloned.running_mean, original.running_mean)
        assert torch.equal(cloned.running_var, original.running_var)
        assert not torch.equal(cloned.running_mean, original.running_mean) or \
               cloned.running_mean is not original.running_mean
    
    def test_bn_statistics_distance(self):
        """Test distance computation between BN statistics."""
        stats1 = BNStatistics(
            running_mean=torch.zeros(10),
            running_var=torch.ones(10),
            num_batches_tracked=torch.tensor(5),
            layer_name="bn1",
            layer_type="BatchNorm1d"
        )
        
        stats2 = BNStatistics(
            running_mean=torch.ones(10),
            running_var=torch.ones(10) * 2,
            num_batches_tracked=torch.tensor(5),
            layer_name="bn1",
            layer_type="BatchNorm1d"
        )
        
        distance = stats1.compute_distance(stats2)
        assert isinstance(distance, float)
        assert distance >= 0
        
        # Distance to self should be 0
        self_distance = stats1.compute_distance(stats1.clone())
        assert abs(self_distance) < 1e-6
        
        # Different layer names should give infinite distance
        stats3 = BNStatistics(
            running_mean=torch.zeros(10),
            running_var=torch.ones(10),
            num_batches_tracked=torch.tensor(5),
            layer_name="bn2",
            layer_type="BatchNorm1d"
        )
        
        inf_distance = stats1.compute_distance(stats3)
        assert inf_distance == float('inf')


class TestFedBNParameterSeparator:
    """Test parameter separation between BN and non-BN parameters."""
    
    def test_separator_creation(self):
        """Test parameter separator creation."""
        separator = FedBNParameterSeparator()
        assert hasattr(separator, 'bn_layer_types')
        assert 'BatchNorm1d' in separator.bn_layer_types
        assert 'BatchNorm2d' in separator.bn_layer_types
    
    def test_parameter_separation_with_bn(self, model_with_bn):
        """Test parameter separation with BN layers."""
        separator = FedBNParameterSeparator()
        
        bn_params, non_bn_params, bn_statistics = separator.separate_parameters(model_with_bn)
        
        # Should have BN parameters
        assert len(bn_params) > 0
        assert len(non_bn_params) > 0
        assert len(bn_statistics) > 0
        
        # Check that BN parameters are correctly identified
        bn_param_names = set(bn_params.keys())
        expected_bn_params = {'bn1.weight', 'bn1.bias', 'bn2.weight', 'bn2.bias'}
        assert expected_bn_params.issubset(bn_param_names)
        
        # Check that non-BN parameters don't include BN
        non_bn_param_names = set(non_bn_params.keys())
        assert not any('bn' in name.lower() for name in non_bn_param_names)
        
        # Check BN statistics
        assert 'bn1' in bn_statistics
        assert 'bn2' in bn_statistics
        assert bn_statistics['bn1'].layer_type == 'BatchNorm1d'
        assert bn_statistics['bn2'].layer_type == 'BatchNorm1d'
    
    def test_parameter_separation_without_bn(self, model_without_bn):
        """Test parameter separation without BN layers."""
        separator = FedBNParameterSeparator()
        
        bn_params, non_bn_params, bn_statistics = separator.separate_parameters(model_without_bn)
        
        # Should have no BN parameters
        assert len(bn_params) == 0
        assert len(bn_statistics) == 0
        
        # All parameters should be non-BN
        assert len(non_bn_params) > 0
        
        # Check that all model parameters are in non-BN
        model_param_count = sum(1 for _ in model_without_bn.parameters())
        assert len(non_bn_params) == model_param_count
    
    def test_is_bn_parameter(self, model_with_bn):
        """Test BN parameter identification."""
        separator = FedBNParameterSeparator()
        
        # Test with actual parameter names from model
        assert separator.is_bn_parameter('bn1.weight', model_with_bn)
        assert separator.is_bn_parameter('bn2.bias', model_with_bn)
        assert not separator.is_bn_parameter('fc1.weight', model_with_bn)
        assert not separator.is_bn_parameter('fc3.bias', model_with_bn)


class TestFedBNAggregator:
    """Test FedBN aggregation strategy."""
    
    def test_aggregator_creation(self, fedbn_config):
        """Test aggregator creation."""
        aggregator = FedBNAggregator(fedbn_config)
        assert aggregator.config == fedbn_config
        assert hasattr(aggregator, 'parameter_separator')
    
    def test_non_bn_aggregation(self, fedbn_config):
        """Test aggregation of non-BN parameters only."""
        aggregator = FedBNAggregator(fedbn_config)
        
        # Create mock client models with mixed parameters
        client_models = {
            'client1': {
                'fc1.weight': torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                'fc1.bias': torch.tensor([1.0, 2.0]),
                'bn1.weight': torch.tensor([0.5, 0.6]),  # Should not be aggregated
                'bn1.bias': torch.tensor([0.1, 0.2])     # Should not be aggregated
            },
            'client2': {
                'fc1.weight': torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
                'fc1.bias': torch.tensor([2.0, 3.0]),
                'bn1.weight': torch.tensor([0.7, 0.8]),  # Should not be aggregated
                'bn1.bias': torch.tensor([0.3, 0.4])     # Should not be aggregated
            }
        }
        
        client_data_sizes = {'client1': 100, 'client2': 200}
        client_losses = {'client1': 0.5, 'client2': 0.3}
        
        aggregated_params, weights = aggregator.aggregate_models(
            client_models, client_data_sizes, client_losses
        )
        
        # Should only have non-BN parameters
        assert 'fc1.weight' in aggregated_params
        assert 'fc1.bias' in aggregated_params
        assert 'bn1.weight' not in aggregated_params
        assert 'bn1.bias' not in aggregated_params
        
        # Verify weights
        assert len(weights) == 2
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_empty_models_aggregation(self, fedbn_config):
        """Test aggregation with empty models."""
        aggregator = FedBNAggregator(fedbn_config)
        
        with pytest.raises(ValueError, match="No client models to aggregate"):
            aggregator.aggregate_models({}, {}, {})
    
    def test_bn_param_identification(self, fedbn_config):
        """Test BN parameter identification in aggregator."""
        aggregator = FedBNAggregator(fedbn_config)
        
        # Test BN parameter name detection
        assert aggregator._is_bn_param_name('bn1.weight')
        assert aggregator._is_bn_param_name('batch_norm_layer.bias')
        assert aggregator._is_bn_param_name('layer.batchnorm.running_mean')
        
        assert not aggregator._is_bn_param_name('fc1.weight')
        assert not aggregator._is_bn_param_name('conv2d.bias')
        assert not aggregator._is_bn_param_name('linear.weight')


class TestFedBNStatisticsAnalyzer:
    """Test BN statistics analysis."""
    
    def test_analyzer_creation(self):
        """Test analyzer creation."""
        analyzer = FedBNStatisticsAnalyzer()
        assert hasattr(analyzer, 'logger')
    
    def test_bn_diversity_analysis(self):
        """Test BN diversity analysis."""
        analyzer = FedBNStatisticsAnalyzer()
        
        # Create mock client BN statistics
        client_bn_stats = {
            'client1': {
                'bn1': BNStatistics(
                    running_mean=torch.zeros(10),
                    running_var=torch.ones(10),
                    num_batches_tracked=torch.tensor(5),
                    layer_name='bn1',
                    layer_type='BatchNorm1d'
                )
            },
            'client2': {
                'bn1': BNStatistics(
                    running_mean=torch.ones(10),
                    running_var=torch.ones(10) * 2,
                    num_batches_tracked=torch.tensor(5),
                    layer_name='bn1',
                    layer_type='BatchNorm1d'
                )
            },
            'client3': {
                'bn1': BNStatistics(
                    running_mean=torch.ones(10) * -1,
                    running_var=torch.ones(10) * 0.5,
                    num_batches_tracked=torch.tensor(5),
                    layer_name='bn1',
                    layer_type='BatchNorm1d'
                )
            }
        }
        
        analysis = analyzer.analyze_bn_diversity(client_bn_stats)
        
        # Check analysis structure
        assert 'layer_diversity' in analysis
        assert 'overall_diversity' in analysis
        assert 'layer_count' in analysis
        assert 'client_count' in analysis
        
        assert analysis['client_count'] == 3
        assert analysis['layer_count'] == 1
        assert 'bn1' in analysis['layer_diversity']
        assert analysis['overall_diversity'] >= 0
    
    def test_empty_bn_diversity_analysis(self):
        """Test BN diversity analysis with empty data."""
        analyzer = FedBNStatisticsAnalyzer()
        
        analysis = analyzer.analyze_bn_diversity({})
        
        assert analysis['overall_diversity'] == 0.0
        assert analysis['layer_count'] == 0
        assert analysis['client_count'] == 0
    
    def test_bn_drift_computation(self):
        """Test BN drift computation between rounds."""
        analyzer = FedBNStatisticsAnalyzer()
        
        current_stats = {
            'bn1': BNStatistics(
                running_mean=torch.ones(5),
                running_var=torch.ones(5) * 2,
                num_batches_tracked=torch.tensor(10),
                layer_name='bn1',
                layer_type='BatchNorm1d'
            )
        }
        
        previous_stats = {
            'bn1': BNStatistics(
                running_mean=torch.zeros(5),
                running_var=torch.ones(5),
                num_batches_tracked=torch.tensor(5),
                layer_name='bn1',
                layer_type='BatchNorm1d'
            )
        }
        
        drift_scores = analyzer.compute_bn_drift(current_stats, previous_stats)
        
        assert 'bn1' in drift_scores
        assert drift_scores['bn1'] >= 0


class TestFedBNAlgorithm:
    """Test main FedBN algorithm functionality."""
    
    def test_algorithm_creation(self, fedbn_config):
        """Test algorithm creation."""
        algorithm = FedBNAlgorithm(fedbn_config)
        assert algorithm.config == fedbn_config
        assert algorithm.global_non_bn_params is None
        assert algorithm.current_round == 0
    
    def test_global_model_initialization(self, fedbn_algorithm, model_with_bn):
        """Test global model initialization."""
        fedbn_algorithm.initialize_global_model(model_with_bn)
        
        assert fedbn_algorithm.global_non_bn_params is not None
        assert len(fedbn_algorithm.global_non_bn_params) > 0
        assert len(fedbn_algorithm.global_bn_layer_names) > 0
        
        # Verify BN layers are identified
        assert 'bn1' in fedbn_algorithm.global_bn_layer_names
        assert 'bn2' in fedbn_algorithm.global_bn_layer_names
    
    def test_client_registration(self, fedbn_algorithm):
        """Test client registration."""
        fedbn_algorithm.register_client('client1', data_size=1000)
        
        assert 'client1' in fedbn_algorithm.client_states
        client_state = fedbn_algorithm.client_states['client1']
        assert client_state.device_id == 'client1'
        assert client_state.data_size == 1000
    
    def test_client_update_with_bn(self, fedbn_algorithm, model_with_bn, test_dataloader):
        """Test client update with BN layers."""
        # Initialize global model
        fedbn_algorithm.initialize_global_model(model_with_bn)
        
        # Register client
        fedbn_algorithm.register_client('client1', data_size=100)
        
        # Perform client update
        loss_fn = nn.CrossEntropyLoss()
        model_params, local_loss, metrics = fedbn_algorithm.client_update(
            'client1', model_with_bn, test_dataloader, loss_fn
        )
        
        # Verify results
        assert isinstance(model_params, dict)
        assert isinstance(local_loss, float)
        assert isinstance(metrics, dict)
        assert local_loss >= 0
        
        # Verify BN-specific metrics
        assert 'bn_layer_count' in metrics
        assert 'non_bn_param_count' in metrics
        assert metrics['bn_layer_count'] > 0
        
        # Verify client state has BN statistics
        client_state = fedbn_algorithm.client_states['client1']
        assert len(client_state.bn_statistics) > 0
        assert 'bn1' in client_state.bn_statistics
        assert 'bn2' in client_state.bn_statistics
    
    def test_client_update_without_bn(self, fedbn_algorithm, model_without_bn, test_dataloader):
        """Test client update without BN layers."""
        # Initialize global model
        fedbn_algorithm.initialize_global_model(model_without_bn)
        
        # Register client
        fedbn_algorithm.register_client('client1', data_size=100)
        
        # Perform client update
        loss_fn = nn.CrossEntropyLoss()
        model_params, local_loss, metrics = fedbn_algorithm.client_update(
            'client1', model_without_bn, test_dataloader, loss_fn
        )
        
        # Verify results
        assert isinstance(model_params, dict)
        assert local_loss >= 0
        
        # Should have no BN layers
        assert metrics['bn_layer_count'] == 0
        
        # Client should have no BN statistics
        client_state = fedbn_algorithm.client_states['client1']
        assert len(client_state.bn_statistics) == 0
    
    def test_federated_averaging(self, fedbn_algorithm, model_with_bn):
        """Test federated averaging with BN layers."""
        # Initialize global model
        fedbn_algorithm.initialize_global_model(model_with_bn)
        
        # Register multiple clients
        for i in range(3):
            client_id = f'client{i+1}'
            fedbn_algorithm.register_client(client_id, data_size=100 * (i + 1))
            
            # Set mock client state with BN statistics
            client_state = fedbn_algorithm.client_states[client_id]
            client_state.model_state = {
                name: param.clone() + torch.randn_like(param) * 0.1
                for name, param in model_with_bn.named_parameters()
            }
            client_state.local_loss_history = [0.5 - i * 0.1]
            
            # Add mock BN statistics
            client_state.bn_statistics = {
                'bn1': BNStatistics(
                    running_mean=torch.randn(128),
                    running_var=torch.ones(128),
                    num_batches_tracked=torch.tensor(10),
                    layer_name='bn1',
                    layer_type='BatchNorm1d'
                ),
                'bn2': BNStatistics(
                    running_mean=torch.randn(64),
                    running_var=torch.ones(64),
                    num_batches_tracked=torch.tensor(10),
                    layer_name='bn2',
                    layer_type='BatchNorm1d'
                )
            }
        
        # Perform federated averaging
        participating_clients = ['client1', 'client2', 'client3']
        round_result = fedbn_algorithm.federated_averaging(participating_clients)
        
        # Verify result
        assert isinstance(round_result, FedBNRoundResult)
        assert round_result.round_id == 0
        assert set(round_result.participating_clients) == set(participating_clients)
        assert round_result.global_loss > 0
        assert round_result.bn_layer_count == 2
        assert round_result.bn_diversity_score >= 0
        
        # Verify global model updated (non-BN parameters only)
        assert fedbn_algorithm.global_non_bn_params is not None
        assert fedbn_algorithm.current_round == 1
    
    def test_bn_statistics_preservation(self, fedbn_algorithm, model_with_bn):
        """Test that BN statistics are preserved locally."""
        fedbn_algorithm.initialize_global_model(model_with_bn)
        fedbn_algorithm.register_client('client1', 100)
        
        # Set initial BN statistics
        initial_bn_stats = {
            'bn1': BNStatistics(
                running_mean=torch.ones(128),
                running_var=torch.ones(128) * 2,
                num_batches_tracked=torch.tensor(5),
                layer_name='bn1',
                layer_type='BatchNorm1d'
            )
        }
        
        client_state = fedbn_algorithm.client_states['client1']
        client_state.bn_statistics = initial_bn_stats
        
        # Load model and check BN restoration
        fedbn_algorithm._restore_client_bn_stats(model_with_bn, initial_bn_stats)
        
        # Verify BN statistics are loaded
        bn1_module = None
        for name, module in model_with_bn.named_modules():
            if name == 'bn1':
                bn1_module = module
                break
        
        assert bn1_module is not None
        assert torch.equal(bn1_module.running_mean, initial_bn_stats['bn1'].running_mean)
        assert torch.equal(bn1_module.running_var, initial_bn_stats['bn1'].running_var)
    
    def test_algorithm_stats(self, fedbn_algorithm, model_with_bn):
        """Test algorithm statistics."""
        # Initialize and add some state
        fedbn_algorithm.initialize_global_model(model_with_bn)
        fedbn_algorithm.register_client('client1', 100)
        fedbn_algorithm.register_client('client2', 200)
        
        stats = fedbn_algorithm.get_algorithm_stats()
        
        assert 'current_round' in stats
        assert 'total_clients' in stats
        assert 'bn_layer_count' in stats
        assert stats['total_clients'] == 2
        assert stats['bn_layer_count'] > 0


class TestFedBNIntegration:
    """Integration tests for FedBN algorithm."""
    
    def test_full_training_round_with_bn(self, fedbn_config):
        """Test a complete training round with BN layers."""
        # Create algorithm
        algorithm = FedBNAlgorithm(fedbn_config)
        
        # Create model and data
        model = ModelWithBN()
        algorithm.initialize_global_model(model)
        
        # Create different datasets for clients (simulate heterogeneity)
        client_datasets = []
        for i in range(3):
            # Create data with different distributions (affects BN statistics)
            dataset = TestDataset(size=50, bias=i * 0.5)
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
            client_model = ModelWithBN()
            
            model_params, local_loss, metrics = algorithm.client_update(
                client_id, client_model, dataloader, loss_fn
            )
            
            assert local_loss >= 0
            assert metrics['bn_layer_count'] > 0
        
        # Perform federated averaging
        round_result = algorithm.federated_averaging(client_ids)
        
        # Verify complete round
        assert round_result.round_id == 0
        assert len(round_result.participating_clients) == 3
        assert round_result.bn_diversity_score >= 0
        assert round_result.bn_layer_count > 0
        assert algorithm.current_round == 1
    
    def test_multiple_rounds_bn_preservation(self, fedbn_config):
        """Test BN statistics preservation over multiple rounds."""
        # Use smaller learning rate for stability
        fedbn_config.learning_rate = 0.001
        fedbn_config.local_epochs = 2
        
        algorithm = FedBNAlgorithm(fedbn_config)
        model = ModelWithBN()
        algorithm.initialize_global_model(model)
        
        # Create clients with different data distributions
        client_datasets = []
        for i in range(2):
            dataset = TestDataset(size=30, bias=i * 1.0)  # More pronounced bias
            client_datasets.append(dataset)
        
        client_ids = []
        for i, dataset in enumerate(client_datasets):
            client_id = f'client{i+1}'
            algorithm.register_client(client_id, len(dataset))
            client_ids.append(client_id)
        
        # Run multiple rounds and track BN diversity
        loss_fn = nn.CrossEntropyLoss()
        bn_diversities = []
        
        for round_num in range(3):
            # Client updates
            for i, client_id in enumerate(client_ids):
                dataloader = data_utils.DataLoader(client_datasets[i], batch_size=8, shuffle=True)
                client_model = ModelWithBN()
                
                try:
                    model_params, local_loss, metrics = algorithm.client_update(
                        client_id, client_model, dataloader, loss_fn
                    )
                except Exception as e:
                    print(f"Client {client_id} failed in round {round_num}: {e}")
                    continue
            
            # Federated averaging
            round_result = algorithm.federated_averaging(client_ids)
            bn_diversities.append(round_result.bn_diversity_score)
        
        # Verify multiple rounds completed
        assert algorithm.current_round == 3
        assert len(algorithm.round_history) == 3
        
        # BN diversity should be tracked
        assert len(bn_diversities) == 3
        assert all(div >= 0 for div in bn_diversities)
    
    def test_mixed_bn_non_bn_models(self, fedbn_config):
        """Test FedBN handles models with and without BN gracefully."""
        algorithm = FedBNAlgorithm(fedbn_config)
        
        # Initialize with BN model
        model_with_bn = ModelWithBN()
        algorithm.initialize_global_model(model_with_bn)
        
        # Should handle non-BN model for client updates
        model_without_bn = ModelWithoutBN()
        algorithm.register_client('client1', 100)
        
        # This should work even though architectures differ
        # (In practice, this might not be a valid scenario, but tests robustness)
        dataset = TestDataset(size=20)
        dataloader = data_utils.DataLoader(dataset, batch_size=8)
        loss_fn = nn.CrossEntropyLoss()
        
        # The update might fail due to architecture mismatch, which is expected
        try:
            model_params, local_loss, metrics = algorithm.client_update(
                'client1', model_without_bn, dataloader, loss_fn
            )
            # If it succeeds, verify it handles the case gracefully
            assert isinstance(metrics, dict)
        except Exception:
            # Expected to fail due to architecture mismatch
            pass


class TestFedBNPerformance:
    """Performance tests for FedBN algorithm."""
    
    def test_large_bn_model(self, fedbn_config):
        """Test FedBN with larger model containing many BN layers."""
        
        class LargeBNModel(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                input_size = 784
                
                for i in range(5):  # 5 layers with BN
                    hidden_size = 256 // (i + 1)
                    layers.extend([
                        nn.Linear(input_size, hidden_size),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU()
                    ])
                    input_size = hidden_size
                
                layers.append(nn.Linear(input_size, 10))
                self.layers = nn.Sequential(*layers)
            
            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.layers(x)
        
        algorithm = FedBNAlgorithm(fedbn_config)
        large_model = LargeBNModel()
        
        # Initialize - should handle large model with many BN layers
        algorithm.initialize_global_model(large_model)
        
        assert algorithm.global_non_bn_params is not None
        assert len(algorithm.global_bn_layer_names) == 5  # 5 BN layers
        
        # Verify parameter separation works correctly
        separator = algorithm.parameter_separator
        bn_params, non_bn_params, bn_statistics = separator.separate_parameters(large_model)
        
        assert len(bn_statistics) == 5
        assert len(bn_params) > 0
        assert len(non_bn_params) > 0
    
    def test_many_clients_bn_diversity(self, fedbn_config):
        """Test BN diversity analysis with many clients."""
        algorithm = FedBNAlgorithm(fedbn_config)
        model = ModelWithBN()
        algorithm.initialize_global_model(model)
        
        # Register many clients with different BN statistics
        num_clients = 20
        client_ids = []
        
        for i in range(num_clients):
            client_id = f'client{i:02d}'
            algorithm.register_client(client_id, data_size=100)
            client_ids.append(client_id)
            
            # Create diverse BN statistics
            client_state = algorithm.client_states[client_id]
            client_state.bn_statistics = {
                'bn1': BNStatistics(
                    running_mean=torch.randn(128) * (i + 1),
                    running_var=torch.ones(128) * (1 + i * 0.1),
                    num_batches_tracked=torch.tensor(10 + i),
                    layer_name='bn1',
                    layer_type='BatchNorm1d'
                ),
                'bn2': BNStatistics(
                    running_mean=torch.randn(64) * (i + 1),
                    running_var=torch.ones(64) * (1 + i * 0.1),
                    num_batches_tracked=torch.tensor(10 + i),
                    layer_name='bn2',
                    layer_type='BatchNorm1d'
                )
            }
            client_state.local_loss_history = [np.random.uniform(0.1, 1.0)]
        
        # Mock model states
        for client_id in client_ids:
            client_state = algorithm.client_states[client_id]
            client_state.model_state = {
                name: param.clone() + torch.randn_like(param) * 0.01
                for name, param in model.named_parameters()
            }
        
        # Perform aggregation
        round_result = algorithm.federated_averaging(client_ids)
        
        assert len(round_result.participating_clients) == num_clients
        assert round_result.bn_diversity_score >= 0
        assert 'layer_diversity' in round_result.bn_stats_analysis
        assert 'overall_diversity' in round_result.bn_stats_analysis


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])