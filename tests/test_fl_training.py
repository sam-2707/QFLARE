"""
Comprehensive federated learning training tests for QFLARE.
"""

import pytest
import sys
import os
import numpy as np
import base64
import hashlib
import time
from unittest.mock import patch, MagicMock
import tempfile
import json

# Add server path to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

# Mock liboqs to avoid import issues
sys.modules['oqs'] = MagicMock()

# Import with relative path
from fl_core.aggregator import store_model_update, get_global_model
from enclave.mock_enclave import get_secure_enclave, ModelUpdate
from auth.pqcrypto_utils import register_device_keys, verify_model_signature


class TestModelAggregation:
    """Test model aggregation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Register a test device
        self.device_id = "test_device_001"
        kem_public_key = base64.b64encode(b"test_kem_key").decode('utf-8')
        sig_public_key = base64.b64encode(b"test_sig_key").decode('utf-8')
        register_device_keys(self.device_id, kem_public_key, sig_public_key)
    
    def test_store_model_update(self):
        """Test storing a model update."""
        # Create test model weights
        model_weights = np.random.rand(100).tobytes()
        signature = hashlib.sha256(model_weights).hexdigest().encode()
        
        # Store model update
        result = store_model_update(
            device_id=self.device_id,
            model_weights=model_weights,
            signature=signature,
            metadata={"round": 1, "epochs": 10}
        )
        
        assert result is True
    
    def test_store_invalid_model_update(self):
        """Test storing a model update with invalid signature."""
        model_weights = np.random.rand(100).tobytes()
        wrong_signature = b"wrong_signature"
        
        result = store_model_update(
            device_id=self.device_id,
            model_weights=model_weights,
            signature=wrong_signature,
            metadata={"round": 1, "epochs": 10}
        )
        
        assert result is False
    
    def test_store_model_update_unregistered_device(self):
        """Test storing model update from unregistered device."""
        model_weights = np.random.rand(100).tobytes()
        signature = hashlib.sha256(model_weights).hexdigest().encode()
        
        result = store_model_update(
            device_id="unregistered_device",
            model_weights=model_weights,
            signature=signature,
            metadata={"round": 1, "epochs": 10}
        )
        
        assert result is False
    
    def test_get_global_model(self):
        """Test retrieving global model."""
        # First store some model updates
        for i in range(3):
            model_weights = np.random.rand(100).tobytes()
            signature = hashlib.sha256(model_weights).hexdigest().encode()
            
            store_model_update(
                device_id=f"test_device_{i:03d}",
                model_weights=model_weights,
                signature=signature,
                metadata={"round": 1, "epochs": 10}
            )
        
        # Get global model
        global_model = get_global_model()
        
        assert global_model is not None
        assert isinstance(global_model, bytes)
        assert len(global_model) > 0


class TestSecureEnclave:
    """Test secure enclave functionality."""
    
    def test_enclave_initialization(self):
        """Test secure enclave initialization."""
        enclave = get_secure_enclave()
        assert enclave is not None
        
        # Check enclave status
        status = enclave.get_status()
        assert "status" in status
        assert "model_count" in status
        assert "last_aggregation" in status
    
    def test_model_update_processing(self):
        """Test processing model updates in enclave."""
        enclave = get_secure_enclave()
        
        # Create test model update
        model_weights = np.random.rand(100).tobytes()
        signature = hashlib.sha256(model_weights).hexdigest().encode()
        
        model_update = ModelUpdate(
            device_id="test_device_001",
            model_weights=model_weights,
            signature=signature,
            metadata={"round": 1, "epochs": 10}
        )
        
        # Process model update
        result = enclave.process_model_update(model_update)
        assert result is True
        
        # Check that model was stored
        status = enclave.get_status()
        assert status["model_count"] > 0
    
    def test_poisoning_detection(self):
        """Test model poisoning detection."""
        enclave = get_secure_enclave()
        
        # Create normal model update
        normal_weights = np.random.rand(100).tobytes()
        normal_signature = hashlib.sha256(normal_weights).hexdigest().encode()
        
        normal_update = ModelUpdate(
            device_id="normal_device",
            model_weights=normal_weights,
            signature=normal_signature,
            metadata={"round": 1, "epochs": 10}
        )
        
        # Create poisoned model update (very different weights)
        poisoned_weights = (np.random.rand(100) * 1000).tobytes()  # Much larger values
        poisoned_signature = hashlib.sha256(poisoned_weights).hexdigest().encode()
        
        poisoned_update = ModelUpdate(
            device_id="poisoned_device",
            model_weights=poisoned_weights,
            signature=poisoned_signature,
            metadata={"round": 1, "epochs": 10}
        )
        
        # Process normal update
        normal_result = enclave.process_model_update(normal_update)
        assert normal_result is True
        
        # Process poisoned update
        poisoned_result = enclave.process_model_update(poisoned_update)
        # The enclave should detect and reject the poisoned model
        assert poisoned_result is False
    
    def test_federated_averaging(self):
        """Test federated averaging in enclave."""
        enclave = get_secure_enclave()
        
        # Add multiple model updates
        for i in range(3):
            model_weights = np.random.rand(100).tobytes()
            signature = hashlib.sha256(model_weights).hexdigest().encode()
            
            model_update = ModelUpdate(
                device_id=f"test_device_{i:03d}",
                model_weights=model_weights,
                signature=signature,
                metadata={"round": 1, "epochs": 10}
            )
            
            enclave.process_model_update(model_update)
        
        # Perform federated averaging
        aggregated_model = enclave.aggregate_models()
        
        assert aggregated_model is not None
        assert isinstance(aggregated_model, bytes)
        assert len(aggregated_model) > 0
        
        # Check that aggregation was recorded
        status = enclave.get_status()
        assert status["last_aggregation"] is not None


class TestFederatedLearningWorkflow:
    """Test complete federated learning workflow."""
    
    def test_complete_fl_round(self):
        """Test a complete federated learning round."""
        # Initialize enclave
        enclave = get_secure_enclave()
        
        # Simulate multiple devices submitting model updates
        device_ids = ["device_001", "device_002", "device_003"]
        
        for device_id in device_ids:
            # Register device
            kem_public_key = base64.b64encode(f"kem_key_{device_id}".encode()).decode('utf-8')
            sig_public_key = base64.b64encode(f"sig_key_{device_id}".encode()).decode('utf-8')
            register_device_keys(device_id, kem_public_key, sig_public_key)
            
            # Create and submit model update
            model_weights = np.random.rand(100).tobytes()
            signature = hashlib.sha256(model_weights).hexdigest().encode()
            
            # Store model update
            success = store_model_update(
                device_id=device_id,
                model_weights=model_weights,
                signature=signature,
                metadata={"round": 1, "epochs": 10}
            )
            
            assert success is True
        
        # Perform aggregation
        aggregated_model = enclave.aggregate_models()
        assert aggregated_model is not None
        
        # Get global model
        global_model = get_global_model()
        assert global_model is not None
        
        # Verify that global model is the same as aggregated model
        assert global_model == aggregated_model


class TestModelSerialization:
    """Test model serialization and deserialization."""
    
    def test_model_weights_serialization(self):
        """Test serialization of model weights."""
        # Create test model weights
        original_weights = np.random.rand(100)
        weights_bytes = original_weights.tobytes()
        
        # Serialize to base64
        weights_b64 = base64.b64encode(weights_bytes).decode('utf-8')
        
        # Deserialize
        deserialized_bytes = base64.b64decode(weights_b64)
        deserialized_weights = np.frombuffer(deserialized_bytes, dtype=np.float64)
        
        # Verify they're the same
        np.testing.assert_array_equal(original_weights, deserialized_weights)
    
    def test_model_metadata_serialization(self):
        """Test serialization of model metadata."""
        metadata = {
            "round": 1,
            "epochs": 10,
            "device_id": "test_device_001",
            "timestamp": time.time(),
            "model_version": "1.0.0"
        }
        
        # Serialize to JSON
        metadata_json = json.dumps(metadata)
        metadata_bytes = metadata_json.encode('utf-8')
        
        # Deserialize
        deserialized_json = metadata_bytes.decode('utf-8')
        deserialized_metadata = json.loads(deserialized_json)
        
        # Verify they're the same
        assert metadata == deserialized_metadata


class TestErrorHandling:
    """Test error handling in federated learning."""
    
    def test_invalid_model_weights(self):
        """Test handling of invalid model weights."""
        # Try to store empty model weights
        empty_weights = b""
        signature = hashlib.sha256(empty_weights).hexdigest().encode()
        
        result = store_model_update(
            device_id="test_device_001",
            model_weights=empty_weights,
            signature=signature,
            metadata={"round": 1, "epochs": 10}
        )
        
        # Should handle gracefully
        assert result is False
    
    def test_malformed_signature(self):
        """Test handling of malformed signatures."""
        model_weights = np.random.rand(100).tobytes()
        malformed_signature = b"malformed_signature"
        
        result = store_model_update(
            device_id="test_device_001",
            model_weights=model_weights,
            signature=malformed_signature,
            metadata={"round": 1, "epochs": 10}
        )
        
        assert result is False
    
    def test_enclave_error_handling(self):
        """Test error handling in secure enclave."""
        enclave = get_secure_enclave()
        
        # Try to process None model update
        result = enclave.process_model_update(None)
        assert result is False
        
        # Try to aggregate with no models
        aggregated_model = enclave.aggregate_models()
        # Should return None or empty model
        assert aggregated_model is None or len(aggregated_model) == 0


class TestPerformance:
    """Test performance aspects of federated learning."""
    
    def test_large_model_handling(self):
        """Test handling of large model weights."""
        # Create large model weights
        large_weights = np.random.rand(10000).tobytes()  # 80KB
        signature = hashlib.sha256(large_weights).hexdigest().encode()
        
        start_time = time.time()
        
        result = store_model_update(
            device_id="test_device_001",
            model_weights=large_weights,
            signature=signature,
            metadata={"round": 1, "epochs": 10}
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert result is True
        assert processing_time < 1.0  # Should process within 1 second
    
    def test_multiple_model_updates(self):
        """Test handling of multiple model updates."""
        enclave = get_secure_enclave()
        
        start_time = time.time()
        
        # Process multiple model updates
        for i in range(10):
            model_weights = np.random.rand(100).tobytes()
            signature = hashlib.sha256(model_weights).hexdigest().encode()
            
            model_update = ModelUpdate(
                device_id=f"test_device_{i:03d}",
                model_weights=model_weights,
                signature=signature,
                metadata={"round": 1, "epochs": 10}
            )
            
            enclave.process_model_update(model_update)
        
        # Perform aggregation
        aggregated_model = enclave.aggregate_models()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        assert aggregated_model is not None
        assert total_time < 5.0  # Should complete within 5 seconds


if __name__ == "__main__":
    pytest.main([__file__])