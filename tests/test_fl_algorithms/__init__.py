"""
FL Algorithms Test Suite

This module contains the test suite initialization for all FL algorithms.
Includes test configuration, common fixtures, and test runner utilities.
"""

import pytest
import sys
import os

# Add server path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'server'))

# Import test modules
from .test_fedprox import *
from .test_fedbn import *
from .test_personalized_fl import *
from .test_adaptive_aggregation import *

# Test configuration
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest for FL algorithms testing."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "fedprox: marks tests specific to FedProx algorithm"
    )
    config.addinivalue_line(
        "markers", "fedbn: marks tests specific to FedBN algorithm"
    )
    config.addinivalue_line(
        "markers", "personalized: marks tests specific to personalized FL"
    )
    config.addinivalue_line(
        "markers", "adaptive: marks tests specific to adaptive aggregation"
    )


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide test data directory."""
    return os.path.join(os.path.dirname(__file__), 'test_data')


@pytest.fixture(scope="session")
def temp_dir():
    """Provide temporary directory for test artifacts."""
    import tempfile
    return tempfile.mkdtemp()


# Common test utilities
class FLTestUtils:
    """Utility class for FL algorithm testing."""
    
    @staticmethod
    def create_test_config(algorithm="fedavg", **kwargs):
        """Create a test configuration for FL algorithms."""
        base_config = {
            'algorithm': algorithm,
            'local_epochs': 2,
            'learning_rate': 0.01,
            'batch_size': 16,
            'test_mode': True
        }
        base_config.update(kwargs)
        return base_config
    
    @staticmethod
    def assert_model_updated(old_params, new_params, tolerance=1e-6):
        """Assert that model parameters have been updated."""
        assert set(old_params.keys()) == set(new_params.keys())
        
        # At least some parameters should have changed
        changed = False
        for name in old_params.keys():
            if not torch.allclose(old_params[name], new_params[name], atol=tolerance):
                changed = True
                break
        
        assert changed, "Model parameters should have been updated"
    
    @staticmethod
    def assert_loss_decreased(initial_loss, final_loss, min_improvement=0.0):
        """Assert that loss has decreased or stayed within tolerance."""
        improvement = initial_loss - final_loss
        assert improvement >= min_improvement, f"Loss should improve by at least {min_improvement}, got {improvement}"
    
    @staticmethod
    def create_mock_device_registry():
        """Create a mock device registry for testing."""
        from unittest.mock import Mock
        
        registry = Mock()
        registry.get_device_info.return_value = {
            'device_id': 'test_device',
            'capabilities': {'memory': '8GB', 'compute': 'GPU'},
            'trust_score': 0.8
        }
        registry.update_device_performance.return_value = None
        registry.is_device_active.return_value = True
        
        return registry


# Export commonly used test utilities
__all__ = [
    'FLTestUtils',
    'pytest_configure',
    'test_data_dir',
    'temp_dir'
]