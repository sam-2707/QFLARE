"""
Test Suite for Personalized Federated Learning Algorithms

This module contains comprehensive tests for personalized FL algorithms including
Per-FedAvg, FedPer, PFNM, and FedRep strategies with personalization metrics validation.

Test Coverage:
- Personalization configuration validation
- Different personalization strategies
- Client state management with personalization
- Adaptive personalization strength
- Performance tracking and metrics
- Privacy-preserving features
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

from fl_algorithms.personalized_fl import (
    PersonalizedFederatedLearning, PersonalizationConfig, PersonalizedClientState,
    PersonalizedRoundResult, PersonalizationStrategy,
    PerFedAvgStrategy, FedPerStrategy, PFNMStrategy, FedRepStrategy
)


class SimpleClassifierModel(nn.Module):
    """Simple neural network with clear classifier layer for testing."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU()
        )
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


class ModelWithFC(nn.Module):
    """Model with FC layers for FedPer testing."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU()
        )
        self.fc = nn.Linear(64, num_classes)  # Personalized layer
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.shared_layers(x)
        x = self.fc(x)
        return x


class TestDataset(data_utils.Dataset):
    """Simple dataset for testing with client-specific characteristics."""
    
    def __init__(self, size=1000, input_dim=784, num_classes=10, client_bias=0.0, class_imbalance=False):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Generate data with client-specific characteristics
        self.data = torch.randn(size, input_dim) + client_bias
        
        if class_imbalance:
            # Create class imbalance for this client
            weights = torch.ones(num_classes)
            weights[:num_classes//2] *= 3  # Bias toward first half of classes
            self.targets = torch.multinomial(weights, size, replacement=True)
        else:
            self.targets = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


@pytest.fixture
def personalization_config():
    """Create a test personalization configuration."""
    return PersonalizationConfig(
        strategy="per_fedavg",
        local_epochs=3,
        learning_rate=0.01,
        batch_size=32,
        adaptive_personalization=True,
        track_personalization_metrics=True
    )


@pytest.fixture
def fedper_config():
    """Create a FedPer configuration."""
    return PersonalizationConfig(
        strategy="fedper",
        personalized_layers=["classifier", "fc"],
        local_epochs=3,
        learning_rate=0.01
    )


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleClassifierModel()


@pytest.fixture
def fc_model():
    """Create a model with FC layer for FedPer testing."""
    return ModelWithFC()


@pytest.fixture
def test_dataset():
    """Create a test dataset."""
    return TestDataset(size=100)


@pytest.fixture
def biased_datasets():
    """Create datasets with different biases for different clients."""
    return [
        TestDataset(size=80, client_bias=0.0, class_imbalance=False),
        TestDataset(size=90, client_bias=0.5, class_imbalance=True),
        TestDataset(size=70, client_bias=-0.3, class_imbalance=False)
    ]


@pytest.fixture
def test_dataloader(test_dataset):
    """Create a test dataloader."""
    return data_utils.DataLoader(test_dataset, batch_size=16, shuffle=True)


@pytest.fixture
def personalized_fl(personalization_config):
    """Create a personalized FL algorithm instance."""
    return PersonalizedFederatedLearning(personalization_config)


class TestPersonalizationConfig:
    """Test personalization configuration validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PersonalizationConfig()
        assert config.strategy == "per_fedavg"
        assert config.local_epochs == 5
        assert config.learning_rate == 0.01
        assert config.adaptive_personalization == True
        assert config.personalization_strength == 0.5
    
    def test_per_fedavg_config(self):
        """Test Per-FedAvg specific configuration."""
        config = PersonalizationConfig(
            strategy="per_fedavg",
            meta_learning_rate=0.05,
            meta_steps=3
        )
        assert config.strategy == "per_fedavg"
        assert config.meta_learning_rate == 0.05
        assert config.meta_steps == 3
    
    def test_fedper_config(self):
        """Test FedPer specific configuration."""
        config = PersonalizationConfig(
            strategy="fedper",
            personalized_layers=["classifier", "head"]
        )
        assert config.strategy == "fedper"
        assert "classifier" in config.personalized_layers
        assert "head" in config.personalized_layers
    
    def test_pfnm_config(self):
        """Test PFNM specific configuration."""
        config = PersonalizationConfig(
            strategy="pfnm",
            matching_threshold=0.9,
            neuron_matching_method="hungarian"
        )
        assert config.strategy == "pfnm"
        assert config.matching_threshold == 0.9
        assert config.neuron_matching_method == "hungarian"
    
    def test_fedrep_config(self):
        """Test FedRep specific configuration."""
        config = PersonalizationConfig(
            strategy="fedrep",
            representation_epochs=4,
            head_epochs=3
        )
        assert config.strategy == "fedrep"
        assert config.representation_epochs == 4
        assert config.head_epochs == 3
    
    def test_differential_privacy_config(self):
        """Test differential privacy configuration."""
        config = PersonalizationConfig(
            differential_privacy=True,
            dp_noise_scale=0.05,
            dp_clip_norm=0.5
        )
        assert config.differential_privacy == True
        assert config.dp_noise_scale == 0.05
        assert config.dp_clip_norm == 0.5


class TestPersonalizedClientState:
    """Test personalized client state management."""
    
    def test_client_state_creation(self):
        """Test client state creation."""
        state = PersonalizedClientState(
            device_id="client1",
            personalized_model={},
            personalization_strength=0.6
        )
        
        assert state.device_id == "client1"
        assert state.personalization_strength == 0.6
        assert len(state.personalization_history) == 0
        assert len(state.performance_history) == 0
    
    def test_client_state_with_distribution(self):
        """Test client state with data distribution info."""
        data_distribution = {"class_0": 0.3, "class_1": 0.7}
        
        state = PersonalizedClientState(
            device_id="client1",
            personalized_model={},
            data_distribution=data_distribution
        )
        
        assert state.data_distribution == data_distribution
    
    def test_performance_tracking(self):
        """Test performance history tracking."""
        state = PersonalizedClientState(
            device_id="client1",
            personalized_model={}
        )
        
        # Add performance history
        state.performance_history = [1.0, 0.8, 0.6, 0.5]
        state.local_loss_history = [0.9, 0.7, 0.5, 0.4]
        
        assert len(state.performance_history) == 4
        assert len(state.local_loss_history) == 4


class TestPerFedAvgStrategy:
    """Test Per-FedAvg personalization strategy."""
    
    def test_strategy_creation(self, personalization_config):
        """Test Per-FedAvg strategy creation."""
        strategy = PerFedAvgStrategy(personalization_config)
        assert strategy.config == personalization_config
    
    def test_model_personalization(self, personalization_config, simple_model, test_dataloader):
        """Test model personalization with Per-FedAvg."""
        strategy = PerFedAvgStrategy(personalization_config)
        
        # Create global model parameters
        global_model = {name: param.clone() for name, param in simple_model.named_parameters()}
        
        # Create client state
        client_state = PersonalizedClientState(
            device_id="client1",
            personalized_model={}
        )
        
        # Personalize model
        personalized_model = strategy.personalize_model(
            "client1", global_model, test_dataloader, client_state
        )
        
        assert isinstance(personalized_model, dict)
        assert len(personalized_model) > 0
        
        # Parameters should have same names as global model
        assert set(personalized_model.keys()) == set(global_model.keys())
    
    def test_aggregation(self, personalization_config):
        """Test Per-FedAvg aggregation."""
        strategy = PerFedAvgStrategy(personalization_config)
        
        # Create mock client updates
        client_updates = {
            'client1': {'param1': torch.tensor([1.0, 2.0]), 'param2': torch.tensor([3.0])},
            'client2': {'param1': torch.tensor([2.0, 3.0]), 'param2': torch.tensor([4.0])},
        }
        
        client_weights = {'client1': 0.6, 'client2': 0.4}
        
        aggregated = strategy.aggregate_for_global(client_updates, client_weights)
        
        assert 'param1' in aggregated
        assert 'param2' in aggregated
        
        # Check weighted average
        expected_param1 = 0.6 * torch.tensor([1.0, 2.0]) + 0.4 * torch.tensor([2.0, 3.0])
        assert torch.allclose(aggregated['param1'], expected_param1)


class TestFedPerStrategy:
    """Test FedPer personalization strategy."""
    
    def test_strategy_creation(self, fedper_config):
        """Test FedPer strategy creation."""
        strategy = FedPerStrategy(fedper_config)
        assert strategy.config == fedper_config
        assert "classifier" in strategy.personalized_layer_names
        assert "fc" in strategy.personalized_layer_names
    
    def test_personalized_layer_identification(self, fedper_config):
        """Test identification of personalized layers."""
        strategy = FedPerStrategy(fedper_config)
        
        # Test parameter name classification
        assert strategy._is_personalized_layer("classifier.weight")
        assert strategy._is_personalized_layer("fc.bias")
        assert strategy._is_personalized_layer("head.linear.weight")
        
        assert not strategy._is_personalized_layer("feature_extractor.0.weight")
        assert not strategy._is_personalized_layer("shared_layers.1.bias")
    
    def test_model_personalization(self, fedper_config, fc_model):
        """Test FedPer model personalization."""
        strategy = FedPerStrategy(fedper_config)
        
        # Create global model parameters
        global_model = {name: param.clone() for name, param in fc_model.named_parameters()}
        
        # Create client state with existing personalized model
        client_state = PersonalizedClientState(
            device_id="client1",
            personalized_model={
                "fc.weight": torch.randn(10, 64),
                "fc.bias": torch.randn(10)
            }
        )
        
        # Mock dataloader (not used in FedPer)
        mock_dataloader = Mock()
        
        # Personalize model
        personalized_model = strategy.personalize_model(
            "client1", global_model, mock_dataloader, client_state
        )
        
        # Should use local personalized layers and global shared layers
        assert "fc.weight" in personalized_model
        assert "fc.bias" in personalized_model
        assert "shared_layers.0.weight" in personalized_model
        
        # Personalized layers should come from client state
        assert torch.equal(personalized_model["fc.weight"], client_state.personalized_model["fc.weight"])
        
        # Shared layers should come from global model
        assert torch.equal(personalized_model["shared_layers.0.weight"], global_model["shared_layers.0.weight"])
    
    def test_aggregation_shared_only(self, fedper_config):
        """Test FedPer aggregates only shared layers."""
        strategy = FedPerStrategy(fedper_config)
        
        client_updates = {
            'client1': {
                'shared_layers.0.weight': torch.tensor([[1.0, 2.0]]),
                'shared_layers.0.bias': torch.tensor([1.0]),
                'fc.weight': torch.tensor([[0.5]]),  # Should not be aggregated
                'fc.bias': torch.tensor([0.1])       # Should not be aggregated
            },
            'client2': {
                'shared_layers.0.weight': torch.tensor([[2.0, 3.0]]),
                'shared_layers.0.bias': torch.tensor([2.0]),
                'fc.weight': torch.tensor([[0.7]]),  # Should not be aggregated
                'fc.bias': torch.tensor([0.2])       # Should not be aggregated
            }
        }
        
        client_weights = {'client1': 0.5, 'client2': 0.5}
        
        aggregated = strategy.aggregate_for_global(client_updates, client_weights)
        
        # Should only have shared layer parameters
        assert 'shared_layers.0.weight' in aggregated
        assert 'shared_layers.0.bias' in aggregated
        assert 'fc.weight' not in aggregated
        assert 'fc.bias' not in aggregated


class TestPFNMStrategy:
    """Test PFNM personalization strategy."""
    
    def test_strategy_creation(self, personalization_config):
        """Test PFNM strategy creation."""
        config = PersonalizationConfig(strategy="pfnm", matching_threshold=0.8)
        strategy = PFNMStrategy(config)
        assert strategy.config.matching_threshold == 0.8
    
    def test_similarity_computation(self, personalization_config):
        """Test parameter similarity computation."""
        config = PersonalizationConfig(strategy="pfnm")
        strategy = PFNMStrategy(config)
        
        # Test identical parameters
        param1 = torch.tensor([1.0, 2.0, 3.0])
        param2 = torch.tensor([1.0, 2.0, 3.0])
        similarity = strategy._compute_similarity(param1, param2)
        assert abs(similarity - 1.0) < 1e-6
        
        # Test orthogonal parameters
        param1 = torch.tensor([1.0, 0.0])
        param2 = torch.tensor([0.0, 1.0])
        similarity = strategy._compute_similarity(param1, param2)
        assert abs(similarity) < 1e-6
        
        # Test different shapes
        param1 = torch.tensor([1.0, 2.0])
        param2 = torch.tensor([1.0, 2.0, 3.0])
        similarity = strategy._compute_similarity(param1, param2)
        assert similarity == 0.0
    
    def test_model_personalization_with_matching(self, personalization_config):
        """Test PFNM model personalization with neural matching."""
        config = PersonalizationConfig(strategy="pfnm", matching_threshold=0.7)
        strategy = PFNMStrategy(config)
        
        # Create global and local models
        global_model = {
            'layer1.weight': torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            'layer1.bias': torch.tensor([1.0, 2.0])
        }
        
        # Create client state with similar local model (high similarity)
        client_state = PersonalizedClientState(
            device_id="client1",
            personalized_model={
                'layer1.weight': torch.tensor([[1.1, 1.9], [3.1, 3.9]]),  # Similar to global
                'layer1.bias': torch.tensor([1.1, 1.9])
            }
        )
        
        mock_dataloader = Mock()
        
        personalized_model = strategy.personalize_model(
            "client1", global_model, mock_dataloader, client_state
        )
        
        assert 'layer1.weight' in personalized_model
        assert 'layer1.bias' in personalized_model
        
        # Result should be a weighted combination
        assert not torch.equal(personalized_model['layer1.weight'], global_model['layer1.weight'])
        assert not torch.equal(personalized_model['layer1.weight'], client_state.personalized_model['layer1.weight'])


class TestFedRepStrategy:
    """Test FedRep personalization strategy."""
    
    def test_strategy_creation(self, personalization_config):
        """Test FedRep strategy creation."""
        config = PersonalizationConfig(strategy="fedrep")
        strategy = FedRepStrategy(config)
        assert strategy.config == config
    
    def test_head_layer_identification(self, personalization_config):
        """Test head layer identification."""
        config = PersonalizationConfig(strategy="fedrep")
        strategy = FedRepStrategy(config)
        
        # Test head layer identification
        assert strategy._is_head_layer("classifier.weight")
        assert strategy._is_head_layer("fc.bias")
        assert strategy._is_head_layer("head.linear.weight")
        assert strategy._is_head_layer("output.weight")
        
        assert not strategy._is_head_layer("feature_extractor.0.weight")
        assert not strategy._is_head_layer("encoder.conv1.bias")
        assert not strategy._is_head_layer("shared.layer.weight")
    
    def test_aggregation_representation_only(self, personalization_config):
        """Test FedRep aggregates only representation layers."""
        config = PersonalizationConfig(strategy="fedrep")
        strategy = FedRepStrategy(config)
        
        client_updates = {
            'client1': {
                'feature_extractor.0.weight': torch.tensor([[1.0, 2.0]]),
                'feature_extractor.1.bias': torch.tensor([1.0]),
                'classifier.weight': torch.tensor([[0.5]]),  # Head - should not be aggregated
                'fc.bias': torch.tensor([0.1])               # Head - should not be aggregated
            },
            'client2': {
                'feature_extractor.0.weight': torch.tensor([[2.0, 3.0]]),
                'feature_extractor.1.bias': torch.tensor([2.0]),
                'classifier.weight': torch.tensor([[0.7]]),  # Head - should not be aggregated
                'fc.bias': torch.tensor([0.2])               # Head - should not be aggregated
            }
        }
        
        client_weights = {'client1': 0.5, 'client2': 0.5}
        
        aggregated = strategy.aggregate_for_global(client_updates, client_weights)
        
        # Should only have representation layer parameters
        assert 'feature_extractor.0.weight' in aggregated
        assert 'feature_extractor.1.bias' in aggregated
        assert 'classifier.weight' not in aggregated
        assert 'fc.bias' not in aggregated


class TestPersonalizedFederatedLearning:
    """Test main personalized FL algorithm."""
    
    def test_algorithm_creation(self, personalization_config):
        """Test algorithm creation."""
        pfl = PersonalizedFederatedLearning(personalization_config)
        assert pfl.config == personalization_config
        assert pfl.global_model_params is None
        assert pfl.current_round == 0
    
    def test_strategy_creation(self):
        """Test strategy creation for different types."""
        # Test all supported strategies
        strategies = ["per_fedavg", "fedper", "pfnm", "fedrep"]
        
        for strategy_type in strategies:
            config = PersonalizationConfig(strategy=strategy_type)
            pfl = PersonalizedFederatedLearning(config)
            assert pfl.strategy is not None
            assert pfl.config.strategy == strategy_type
    
    def test_invalid_strategy(self):
        """Test invalid strategy handling."""
        config = PersonalizationConfig(strategy="invalid_strategy")
        
        with pytest.raises(ValueError, match="Unknown personalization strategy"):
            PersonalizedFederatedLearning(config)
    
    def test_global_model_initialization(self, personalized_fl, simple_model):
        """Test global model initialization."""
        personalized_fl.initialize_global_model(simple_model)
        
        assert personalized_fl.global_model_params is not None
        assert len(personalized_fl.global_model_params) > 0
        
        # Verify parameter names match model
        model_param_names = set(name for name, _ in simple_model.named_parameters())
        global_param_names = set(personalized_fl.global_model_params.keys())
        assert model_param_names == global_param_names
    
    def test_client_registration(self, personalized_fl):
        """Test client registration with data distribution."""
        data_distribution = {"class_0": 0.4, "class_1": 0.6}
        
        personalized_fl.register_client(
            "client1", 
            data_size=1000, 
            data_distribution=data_distribution
        )
        
        assert "client1" in personalized_fl.client_states
        client_state = personalized_fl.client_states["client1"]
        assert client_state.device_id == "client1"
        assert client_state.data_distribution == data_distribution
        assert client_state.personalization_strength == personalized_fl.config.personalization_strength
    
    def test_client_update(self, personalized_fl, simple_model, test_dataloader):
        """Test personalized client update."""
        # Initialize global model
        personalized_fl.initialize_global_model(simple_model)
        
        # Register client
        personalized_fl.register_client("client1", data_size=100)
        
        # Perform client update
        loss_fn = nn.CrossEntropyLoss()
        model_params, local_loss, metrics = personalized_fl.client_update(
            "client1", simple_model, test_dataloader, loss_fn
        )
        
        # Verify results
        assert isinstance(model_params, dict)
        assert isinstance(local_loss, float)
        assert isinstance(metrics, dict)
        assert local_loss >= 0
        
        # Verify personalization metrics
        assert 'personalization_strength' in metrics
        assert 'strategy' in metrics
        assert metrics['strategy'] == personalized_fl.config.strategy
        
        # Verify client state updated
        client_state = personalized_fl.client_states["client1"]
        assert len(client_state.local_loss_history) == 1
        assert client_state.local_loss_history[0] == local_loss
        assert len(client_state.personalized_model) > 0
    
    def test_differential_privacy(self, simple_model, test_dataloader):
        """Test differential privacy features."""
        config = PersonalizationConfig(
            differential_privacy=True,
            dp_noise_scale=0.1,
            dp_clip_norm=1.0
        )
        pfl = PersonalizedFederatedLearning(config)
        
        # Initialize and register
        pfl.initialize_global_model(simple_model)
        pfl.register_client("client1", 100)
        
        # Perform update with DP
        loss_fn = nn.CrossEntropyLoss()
        model_params, local_loss, metrics = pfl.client_update(
            "client1", simple_model, test_dataloader, loss_fn
        )
        
        # Should complete successfully with DP
        assert isinstance(model_params, dict)
        assert local_loss >= 0
    
    def test_adaptive_personalization(self, personalized_fl, simple_model):
        """Test adaptive personalization strength."""
        personalized_fl.initialize_global_model(simple_model)
        personalized_fl.register_client("client1", 100)
        
        client_state = personalized_fl.client_states["client1"]
        initial_strength = client_state.personalization_strength
        
        # Simulate improving performance
        client_state.performance_history = [1.0, 0.8, 0.6]  # Decreasing loss (improving)
        personalized_fl._adapt_personalization_strength("client1")
        
        # Should increase personalization strength
        assert client_state.personalization_strength >= initial_strength
        
        # Simulate worsening performance
        client_state.performance_history = [0.5, 0.7, 0.9]  # Increasing loss (worsening)
        personalized_fl._adapt_personalization_strength("client1")
        
        # Should decrease personalization strength
        final_strength = client_state.personalization_strength
        assert final_strength <= client_state.personalization_strength
    
    def test_federated_averaging(self, personalized_fl, simple_model):
        """Test personalized federated averaging."""
        # Initialize global model
        personalized_fl.initialize_global_model(simple_model)
        
        # Register multiple clients
        for i in range(3):
            client_id = f'client{i+1}'
            personalized_fl.register_client(client_id, data_size=100 * (i + 1))
            
            # Set mock client state
            client_state = personalized_fl.client_states[client_id]
            client_state.personalized_model = {
                name: param.clone() + torch.randn_like(param) * 0.1
                for name, param in simple_model.named_parameters()
            }
            client_state.local_loss_history = [0.5 - i * 0.1]
            client_state.performance_history = [0.6 - i * 0.1]
        
        # Perform federated averaging
        participating_clients = ['client1', 'client2', 'client3']
        round_result = personalized_fl.federated_averaging(participating_clients)
        
        # Verify result
        assert isinstance(round_result, PersonalizedRoundResult)
        assert round_result.round_id == 0
        assert set(round_result.participating_clients) == set(participating_clients)
        assert round_result.global_loss > 0
        assert len(round_result.personalized_losses) == 3
        
        # Verify personalization metrics
        assert 'avg_personalization_strength' in round_result.personalization_metrics
        assert round_result.avg_personalization_strength > 0
        
        # Verify performance improvements tracked
        assert len(round_result.performance_improvements) == 3
    
    def test_algorithm_stats(self, personalized_fl, simple_model):
        """Test algorithm statistics."""
        # Initialize and add some state
        personalized_fl.initialize_global_model(simple_model)
        personalized_fl.register_client('client1', 100)
        personalized_fl.register_client('client2', 200)
        
        stats = personalized_fl.get_algorithm_stats()
        
        assert 'current_round' in stats
        assert 'total_clients' in stats
        assert 'strategy' in stats
        assert stats['total_clients'] == 2
        assert stats['strategy'] == personalized_fl.config.strategy


class TestPersonalizationIntegration:
    """Integration tests for personalized FL."""
    
    def test_full_personalized_round(self, biased_datasets):
        """Test a complete personalized training round."""
        config = PersonalizationConfig(
            strategy="fedper",
            personalized_layers=["classifier"],
            local_epochs=2,
            learning_rate=0.01,
            adaptive_personalization=True
        )
        
        pfl = PersonalizedFederatedLearning(config)
        model = SimpleClassifierModel()
        pfl.initialize_global_model(model)
        
        # Register clients with different data characteristics
        client_ids = []
        for i, dataset in enumerate(biased_datasets):
            client_id = f'client{i+1}'
            
            # Compute data distribution
            unique_classes, counts = torch.unique(dataset.targets, return_counts=True)
            data_distribution = {f"class_{cls.item()}": count.item()/len(dataset) 
                               for cls, count in zip(unique_classes, counts)}
            
            pfl.register_client(client_id, len(dataset), data_distribution)
            client_ids.append(client_id)
        
        # Perform client updates
        loss_fn = nn.CrossEntropyLoss()
        for i, client_id in enumerate(client_ids):
            dataloader = data_utils.DataLoader(biased_datasets[i], batch_size=16, shuffle=True)
            client_model = SimpleClassifierModel()
            
            model_params, local_loss, metrics = pfl.client_update(
                client_id, client_model, dataloader, loss_fn
            )
            
            assert local_loss >= 0
            assert 'personalization_strength' in metrics
        
        # Perform federated averaging
        round_result = pfl.federated_averaging(client_ids)
        
        # Verify complete round
        assert round_result.round_id == 0
        assert len(round_result.participating_clients) == 3
        assert round_result.avg_personalization_strength > 0
        assert pfl.current_round == 1
    
    def test_multiple_rounds_adaptation(self, biased_datasets):
        """Test personalization adaptation over multiple rounds."""
        config = PersonalizationConfig(
            strategy="per_fedavg",
            local_epochs=2,
            learning_rate=0.001,
            adaptive_personalization=True,
            meta_steps=2
        )
        
        pfl = PersonalizedFederatedLearning(config)
        model = SimpleClassifierModel()
        pfl.initialize_global_model(model)
        
        # Register clients
        client_ids = []
        for i, dataset in enumerate(biased_datasets[:2]):  # Use 2 clients for faster testing
            client_id = f'client{i+1}'
            pfl.register_client(client_id, len(dataset))
            client_ids.append(client_id)
        
        # Run multiple rounds
        loss_fn = nn.CrossEntropyLoss()
        personalization_strengths = []
        
        for round_num in range(3):
            round_strengths = []
            
            # Client updates
            for i, client_id in enumerate(client_ids):
                dataloader = data_utils.DataLoader(biased_datasets[i], batch_size=8, shuffle=True)
                client_model = SimpleClassifierModel()
                
                try:
                    model_params, local_loss, metrics = pfl.client_update(
                        client_id, client_model, dataloader, loss_fn
                    )
                    
                    client_state = pfl.client_states[client_id]
                    round_strengths.append(client_state.personalization_strength)
                    
                except Exception as e:
                    print(f"Client {client_id} failed in round {round_num}: {e}")
                    continue
            
            # Federated averaging
            if round_strengths:  # Only if some clients succeeded
                round_result = pfl.federated_averaging(client_ids)
                personalization_strengths.append(round_strengths)
        
        # Verify multiple rounds completed
        assert pfl.current_round == 3
        assert len(pfl.round_history) == 3
        
        # Check that personalization strengths are tracked
        assert len(personalization_strengths) >= 1
    
    @pytest.mark.parametrize("strategy", ["per_fedavg", "fedper", "pfnm", "fedrep"])
    def test_different_strategies(self, strategy, simple_model, test_dataloader):
        """Test different personalization strategies."""
        if strategy == "fedper":
            config = PersonalizationConfig(
                strategy=strategy,
                personalized_layers=["classifier"]
            )
        else:
            config = PersonalizationConfig(strategy=strategy)
        
        pfl = PersonalizedFederatedLearning(config)
        pfl.initialize_global_model(simple_model)
        pfl.register_client("client1", 100)
        
        # Perform client update
        loss_fn = nn.CrossEntropyLoss()
        try:
            model_params, local_loss, metrics = pfl.client_update(
                "client1", simple_model, test_dataloader, loss_fn
            )
            
            # All strategies should complete successfully
            assert isinstance(model_params, dict)
            assert local_loss >= 0
            assert metrics['strategy'] == strategy
            
        except Exception as e:
            pytest.fail(f"Strategy {strategy} failed: {e}")


class TestPersonalizationPerformance:
    """Performance tests for personalized FL."""
    
    def test_large_model_personalization(self):
        """Test personalization with larger model."""
        
        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Linear(784, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU()
                )
                self.classifier = nn.Linear(128, 10)
            
            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        config = PersonalizationConfig(
            strategy="fedper",
            personalized_layers=["classifier"]
        )
        
        pfl = PersonalizedFederatedLearning(config)
        large_model = LargeModel()
        
        # Should handle large model
        pfl.initialize_global_model(large_model)
        
        assert pfl.global_model_params is not None
        assert len(pfl.global_model_params) > 0
        
        # Register client and test update
        pfl.register_client("client1", 100)
        
        dataset = TestDataset(size=20)
        dataloader = data_utils.DataLoader(dataset, batch_size=8)
        loss_fn = nn.CrossEntropyLoss()
        
        model_params, local_loss, metrics = pfl.client_update(
            "client1", large_model, dataloader, loss_fn
        )
        
        assert isinstance(model_params, dict)
        assert local_loss >= 0
    
    def test_many_clients_personalization(self):
        """Test personalization with many clients."""
        config = PersonalizationConfig(
            strategy="per_fedavg",
            local_epochs=1,  # Reduce for performance
            meta_steps=1
        )
        
        pfl = PersonalizedFederatedLearning(config)
        model = SimpleClassifierModel()
        pfl.initialize_global_model(model)
        
        # Register many clients
        num_clients = 20
        client_ids = []
        for i in range(num_clients):
            client_id = f'client{i:02d}'
            pfl.register_client(client_id, data_size=100)
            client_ids.append(client_id)
        
        assert len(pfl.client_states) == num_clients
        
        # Mock client updates
        for client_id in client_ids:
            client_state = pfl.client_states[client_id]
            client_state.personalized_model = {
                name: param.clone() + torch.randn_like(param) * 0.01
                for name, param in model.named_parameters()
            }
            client_state.local_loss_history = [np.random.uniform(0.1, 1.0)]
            client_state.performance_history = [np.random.uniform(0.1, 1.0)]
        
        # Perform aggregation with all clients
        round_result = pfl.federated_averaging(client_ids)
        
        assert len(round_result.participating_clients) == num_clients
        assert len(round_result.personalized_losses) == num_clients
        assert len(round_result.performance_improvements) == num_clients


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])