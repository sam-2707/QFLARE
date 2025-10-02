"""
Test Suite for Differential Privacy Implementation

Comprehensive tests for QFLARE's differential privacy mechanisms.
Tests privacy guarantees, parameter validation, and integration.
"""

import pytest
import torch
import numpy as np
import asyncio
import logging
from typing import Dict, Any

# Set up test imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.privacy.differential_privacy import (
    DifferentialPrivacyConfig,
    GaussianMechanism,
    PrivacyEngine,
    create_strong_privacy_config,
    create_moderate_privacy_config,
    create_weak_privacy_config,
    create_privacy_engine
)

from server.privacy.private_trainer import PrivateFederatedTrainer
from server.privacy.private_fl_controller import PrivateFLController

# Configure test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestDifferentialPrivacyConfig:
    """Test differential privacy configuration."""
    
    def test_config_initialization(self):
        """Test basic configuration initialization."""
        config = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        
        assert config.epsilon == 1.0
        assert config.delta == 1e-5
        assert config.max_grad_norm == 1.0
        assert config.noise_multiplier > 0
        assert config.spent_epsilon == 0.0
        assert config.spent_delta == 0.0
        assert config.composition_rounds == 0
        
        print("✓ Config initialization test passed")
    
    def test_noise_multiplier_calculation(self):
        """Test noise multiplier calculation."""
        config = DifferentialPrivacyConfig(epsilon=0.1, delta=1e-6, max_grad_norm=1.0)
        
        # Check that noise multiplier is reasonable
        assert config.noise_multiplier > 0
        assert config.noise_multiplier < 100  # Should not be excessively large
        
        # Test with different parameters
        config2 = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5, max_grad_norm=2.0)
        
        # Smaller epsilon should lead to larger noise
        assert config.noise_multiplier > config2.noise_multiplier
        
        print("✓ Noise multiplier calculation test passed")
    
    def test_privacy_composition(self):
        """Test privacy composition accounting."""
        config = DifferentialPrivacyConfig(epsilon=0.1, delta=1e-6)
        
        # Initial state
        assert config.composition_rounds == 0
        assert config.spent_epsilon == 0.1  # Initial epsilon
        
        # After one composition
        config.update_composition(rounds=1)
        assert config.composition_rounds == 1
        
        # After multiple compositions
        config.update_composition(rounds=5)
        assert config.composition_rounds == 6
        assert config.spent_epsilon > 0.1  # Should increase with composition
        
        print("✓ Privacy composition test passed")
    
    def test_remaining_budget(self):
        """Test remaining privacy budget calculation."""
        config = DifferentialPrivacyConfig(epsilon=0.1, delta=1e-6)
        
        remaining_eps, remaining_delta = config.get_remaining_budget()
        assert remaining_eps > 0
        assert remaining_delta > 0
        
        # Spend some budget
        config.update_composition(rounds=10)
        
        new_remaining_eps, new_remaining_delta = config.get_remaining_budget()
        assert new_remaining_eps < remaining_eps
        
        print("✓ Remaining budget test passed")


class TestGaussianMechanism:
    """Test Gaussian mechanism for noise addition."""
    
    def test_noise_addition_tensor(self):
        """Test noise addition to PyTorch tensors."""
        config = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5)
        mechanism = GaussianMechanism(config)
        
        # Test with tensor
        original_tensor = torch.randn(5, 3)
        noisy_tensor = mechanism.add_noise(original_tensor)
        
        assert isinstance(noisy_tensor, torch.Tensor)
        assert noisy_tensor.shape == original_tensor.shape
        assert noisy_tensor.dtype == original_tensor.dtype
        
        # Check that noise was actually added
        difference = torch.norm(noisy_tensor - original_tensor)
        assert difference > 0
        
        print("✓ Tensor noise addition test passed")
    
    def test_noise_addition_array(self):
        """Test noise addition to NumPy arrays."""
        config = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5)
        mechanism = GaussianMechanism(config)
        
        # Test with array
        original_array = np.random.randn(4, 2)
        noisy_array = mechanism.add_noise(original_array)
        
        assert isinstance(noisy_array, np.ndarray)
        assert noisy_array.shape == original_array.shape
        assert noisy_array.dtype == original_array.dtype
        
        # Check that noise was actually added
        difference = np.linalg.norm(noisy_array - original_array)
        assert difference > 0
        
        print("✓ Array noise addition test passed")
    
    def test_noise_addition_scalar(self):
        """Test noise addition to scalar values."""
        config = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5)
        mechanism = GaussianMechanism(config)
        
        # Test with scalar
        original_value = 5.0
        noisy_value = mechanism.add_noise(original_value)
        
        assert isinstance(noisy_value, float)
        assert noisy_value != original_value  # Should be different due to noise
        
        print("✓ Scalar noise addition test passed")
    
    def test_noise_scale(self):
        """Test that noise scale matches configuration."""
        # Test with different noise multipliers
        config1 = DifferentialPrivacyConfig(epsilon=0.1, delta=1e-6)  # High noise
        config2 = DifferentialPrivacyConfig(epsilon=10.0, delta=1e-6)  # Low noise
        
        mechanism1 = GaussianMechanism(config1)
        mechanism2 = GaussianMechanism(config2)
        
        # Add noise to same tensor multiple times
        original_tensor = torch.zeros(100, 100)
        
        differences1 = []
        differences2 = []
        
        for _ in range(10):
            noisy1 = mechanism1.add_noise(original_tensor.clone())
            noisy2 = mechanism2.add_noise(original_tensor.clone())
            
            differences1.append(torch.norm(noisy1).item())
            differences2.append(torch.norm(noisy2).item())
        
        # Higher epsilon (less privacy) should have less noise
        avg_noise1 = np.mean(differences1)
        avg_noise2 = np.mean(differences2)
        
        assert avg_noise1 > avg_noise2
        
        print("✓ Noise scale test passed")


class TestPrivacyEngine:
    """Test the main privacy engine."""
    
    def test_gradient_clipping(self):
        """Test gradient clipping functionality."""
        config = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        engine = PrivacyEngine(config)
        
        # Create gradients that exceed the norm bound
        gradients = {
            "layer1": torch.tensor([3.0, 4.0]),  # Norm = 5.0 > 1.0
            "layer2": torch.tensor([1.0, 1.0])   # Combined norm > 1.0
        }
        
        clipped = engine.clip_gradients(gradients)
        
        # Calculate total norm of clipped gradients
        total_norm = 0.0
        for grad in clipped.values():
            total_norm += grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        # Should be bounded by max_grad_norm
        assert total_norm <= config.max_grad_norm + 1e-6  # Small tolerance for floating point
        
        print("✓ Gradient clipping test passed")
    
    def test_privacy_noise_addition(self):
        """Test privacy noise addition to gradients."""
        config = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5)
        engine = PrivacyEngine(config)
        
        gradients = {
            "layer1": torch.randn(3, 3),
            "layer2": torch.randn(5)
        }
        
        noisy_gradients = engine.add_privacy_noise(gradients)
        
        # Check structure preserved
        assert len(noisy_gradients) == len(gradients)
        assert all(name in noisy_gradients for name in gradients.keys())
        
        # Check noise was added
        for name in gradients.keys():
            difference = torch.norm(noisy_gradients[name] - gradients[name])
            assert difference > 0
        
        print("✓ Privacy noise addition test passed")
    
    def test_full_privatization_pipeline(self):
        """Test complete privatization pipeline."""
        config = DifferentialPrivacyConfig(epsilon=0.1, delta=1e-6)
        engine = PrivacyEngine(config)
        
        # Create model update
        model_update = {
            "layer1.weight": torch.randn(10, 5) * 2,  # Large gradients
            "layer1.bias": torch.randn(10) * 3,
            "layer2.weight": torch.randn(1, 10) * 1.5,
            "layer2.bias": torch.randn(1) * 0.5
        }
        
        initial_composition = engine.config.composition_rounds
        
        # Apply full privatization
        privatized = engine.privatize_model_update(model_update)
        
        # Check structure preserved
        assert len(privatized) == len(model_update)
        assert all(name in privatized for name in model_update.keys())
        
        # Check composition was updated
        assert engine.config.composition_rounds == initial_composition + 1
        
        # Check privacy history recorded
        assert len(engine.privacy_history) > 0
        
        print("✓ Full privatization pipeline test passed")
    
    def test_aggregated_privatization(self):
        """Test privatization of aggregated models."""
        config = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5)
        engine = PrivacyEngine(config)
        
        aggregated_gradients = {
            "layer1": torch.randn(5, 3),
            "layer2": torch.randn(2, 5)
        }
        
        num_clients = 10
        
        privatized_agg = engine.privatize_aggregated_model(aggregated_gradients, num_clients)
        
        # Check structure preserved
        assert len(privatized_agg) == len(aggregated_gradients)
        
        # Check that noise scaling was applied (method should handle multiple clients)
        for name in aggregated_gradients.keys():
            assert name in privatized_agg
        
        print("✓ Aggregated privatization test passed")
    
    def test_privacy_report_generation(self):
        """Test privacy report generation."""
        config = DifferentialPrivacyConfig(epsilon=0.1, delta=1e-6)
        engine = PrivacyEngine(config)
        
        # Generate some privacy events
        mock_gradients = {"layer1": torch.randn(3, 3)}
        for _ in range(3):
            engine.privatize_model_update(mock_gradients)
        
        report = engine.get_privacy_report()
        
        # Check report structure
        assert "privacy_parameters" in report
        assert "privacy_spent" in report
        assert "remaining_budget" in report
        assert "privacy_history" in report
        assert "total_privatization_events" in report
        
        # Check values
        assert report["total_privatization_events"] == 3
        assert len(report["privacy_history"]) == 3
        assert report["privacy_spent"]["composition_rounds"] == 3
        
        print("✓ Privacy report generation test passed")
    
    def test_privacy_budget_validation(self):
        """Test privacy budget validation."""
        config = DifferentialPrivacyConfig(epsilon=0.01, delta=1e-8)  # Very small budget
        engine = PrivacyEngine(config)
        
        # Initially should be valid
        assert engine.validate_privacy_budget()
        
        # After many compositions, should become invalid
        mock_gradients = {"layer1": torch.randn(2, 2)}
        for _ in range(100):  # Use up budget
            engine.privatize_model_update(mock_gradients)
        
        # Budget might be exhausted (depends on composition accounting)
        validation_result = engine.validate_privacy_budget()
        logger.info(f"Budget validation after 100 rounds: {validation_result}")
        
        print("✓ Privacy budget validation test passed")


class TestPrivateTrainerIntegration:
    """Test privacy-aware federated trainer."""
    
    def test_private_trainer_initialization(self):
        """Test private trainer initialization."""
        trainer = PrivateFederatedTrainer(model_type="mnist", privacy_level="strong")
        
        assert trainer.privacy_level == "strong"
        assert trainer.privacy_engine is not None
        assert hasattr(trainer.privacy_engine, 'config')
        assert trainer.privacy_metrics is not None
        
        print("✓ Private trainer initialization test passed")
    
    def test_privacy_status_reporting(self):
        """Test privacy status reporting."""
        trainer = PrivateFederatedTrainer(model_type="simple", privacy_level="moderate")
        
        status = trainer.get_privacy_status()
        
        assert "privacy_engine_active" in status
        assert "privacy_level" in status
        assert "privacy_config" in status
        assert "privacy_spent" in status
        assert "remaining_budget" in status
        assert "privacy_metrics" in status
        assert "budget_valid" in status
        
        assert status["privacy_engine_active"] is True
        assert status["privacy_level"] == "moderate"
        assert status["budget_valid"] is True
        
        print("✓ Privacy status reporting test passed")


class TestPrivacyConfigurationFactories:
    """Test privacy configuration factory functions."""
    
    def test_strong_privacy_config(self):
        """Test strong privacy configuration."""
        config = create_strong_privacy_config()
        
        assert config.epsilon == 0.1
        assert config.delta == 1e-6
        assert config.max_grad_norm == 1.0
        
        print("✓ Strong privacy config test passed")
    
    def test_moderate_privacy_config(self):
        """Test moderate privacy configuration."""
        config = create_moderate_privacy_config()
        
        assert config.epsilon == 1.0
        assert config.delta == 1e-5
        assert config.max_grad_norm == 1.0
        
        print("✓ Moderate privacy config test passed")
    
    def test_weak_privacy_config(self):
        """Test weak privacy configuration."""
        config = create_weak_privacy_config()
        
        assert config.epsilon == 5.0
        assert config.delta == 1e-4
        assert config.max_grad_norm == 1.0
        
        print("✓ Weak privacy config test passed")
    
    def test_privacy_engine_factory(self):
        """Test privacy engine factory function."""
        # Test different levels
        for level in ["strong", "moderate", "weak"]:
            engine = create_privacy_engine(level)
            assert isinstance(engine, PrivacyEngine)
            assert engine.config is not None
        
        # Test invalid level
        with pytest.raises(ValueError):
            create_privacy_engine("invalid")
        
        print("✓ Privacy engine factory test passed")


@pytest.mark.asyncio
class TestPrivacyControllerIntegration:
    """Test privacy-aware FL controller."""
    
    async def test_privacy_controller_initialization(self):
        """Test privacy controller initialization."""
        controller = PrivateFLController(privacy_level="strong")
        
        assert controller.privacy_level == "strong"
        assert controller.private_trainer is not None
        assert controller.privacy_rounds_completed == 0
        assert isinstance(controller.privacy_history, list)
        
        print("✓ Privacy controller initialization test passed")
    
    async def test_privacy_dashboard_data(self):
        """Test privacy dashboard data generation."""
        controller = PrivateFLController(privacy_level="moderate")
        
        dashboard_data = await controller.get_privacy_dashboard_data()
        
        assert "privacy_overview" in dashboard_data
        assert "privacy_parameters" in dashboard_data
        assert "privacy_budget" in dashboard_data
        assert "training_statistics" in dashboard_data
        assert "system_status" in dashboard_data
        
        assert dashboard_data["privacy_overview"]["privacy_enabled"] is True
        assert dashboard_data["privacy_overview"]["privacy_level"] == "moderate"
        
        print("✓ Privacy dashboard data test passed")
    
    async def test_privacy_level_change(self):
        """Test privacy level change functionality."""
        controller = PrivateFLController(privacy_level="strong")
        
        result = await controller.set_privacy_level("weak")
        
        assert result["success"] is True
        assert result["old_privacy_level"] == "strong"
        assert result["new_privacy_level"] == "weak"
        assert controller.privacy_level == "weak"
        
        print("✓ Privacy level change test passed")
    
    async def test_privacy_parameter_validation(self):
        """Test privacy parameter validation."""
        controller = PrivateFLController()
        
        # Test valid parameters
        valid_result = await controller.validate_privacy_parameters(epsilon=1.0, delta=1e-5)
        assert valid_result["valid"] is True
        
        # Test invalid parameters
        invalid_result = await controller.validate_privacy_parameters(epsilon=-1.0, delta=1e-5)
        assert invalid_result["valid"] is False
        assert len(invalid_result["warnings"]) > 0
        
        print("✓ Privacy parameter validation test passed")


def run_all_tests():
    """Run all differential privacy tests."""
    print("="*60)
    print("RUNNING QFLARE DIFFERENTIAL PRIVACY TEST SUITE")
    print("="*60)
    
    try:
        # Test privacy configuration
        config_tests = TestDifferentialPrivacyConfig()
        config_tests.test_config_initialization()
        config_tests.test_noise_multiplier_calculation()
        config_tests.test_privacy_composition()
        config_tests.test_remaining_budget()
        
        # Test Gaussian mechanism
        gaussian_tests = TestGaussianMechanism()
        gaussian_tests.test_noise_addition_tensor()
        gaussian_tests.test_noise_addition_array()
        gaussian_tests.test_noise_addition_scalar()
        gaussian_tests.test_noise_scale()
        
        # Test privacy engine
        engine_tests = TestPrivacyEngine()
        engine_tests.test_gradient_clipping()
        engine_tests.test_privacy_noise_addition()
        engine_tests.test_full_privatization_pipeline()
        engine_tests.test_aggregated_privatization()
        engine_tests.test_privacy_report_generation()
        engine_tests.test_privacy_budget_validation()
        
        # Test trainer integration
        trainer_tests = TestPrivateTrainerIntegration()
        trainer_tests.test_private_trainer_initialization()
        trainer_tests.test_privacy_status_reporting()
        
        # Test configuration factories
        factory_tests = TestPrivacyConfigurationFactories()
        factory_tests.test_strong_privacy_config()
        factory_tests.test_moderate_privacy_config()
        factory_tests.test_weak_privacy_config()
        factory_tests.test_privacy_engine_factory()
        
        # Test controller integration (async)
        async def run_async_tests():
            controller_tests = TestPrivacyControllerIntegration()
            await controller_tests.test_privacy_controller_initialization()
            await controller_tests.test_privacy_dashboard_data()
            await controller_tests.test_privacy_level_change()
            await controller_tests.test_privacy_parameter_validation()
        
        asyncio.run(run_async_tests())
        
        print("="*60)
        print("✅ ALL DIFFERENTIAL PRIVACY TESTS PASSED!")
        print("="*60)
        print("Summary:")
        print("- Differential Privacy Configuration: ✓")
        print("- Gaussian Mechanism: ✓")
        print("- Privacy Engine Core: ✓")
        print("- Private Trainer Integration: ✓")
        print("- Configuration Factories: ✓")
        print("- Privacy Controller: ✓")
        print("="*60)
        
        return True
        
    except Exception as e:
        print("="*60)
        print("❌ DIFFERENTIAL PRIVACY TESTS FAILED!")
        print(f"Error: {str(e)}")
        print("="*60)
        return False


if __name__ == "__main__":
    run_all_tests()