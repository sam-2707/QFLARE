"""
Comprehensive key rotation tests for QFLARE.
"""

import pytest
import sys
import os
import base64
import hashlib
import time
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Add server path to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

from auth.pqcrypto_utils import (
    register_device_keys,
    get_device_public_keys,
    generate_device_keypair
)


class TestKeyRotation:
    """Test key rotation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device_id = "test_device_001"
        self.kem_public_key = base64.b64encode(b"test_kem_key").decode('utf-8')
        self.sig_public_key = base64.b64encode(b"test_sig_key").decode('utf-8')
    
    def test_key_registration(self):
        """Test initial key registration."""
        result = register_device_keys(
            self.device_id,
            self.kem_public_key,
            self.sig_public_key
        )
        
        assert result is True
        
        # Verify keys were stored
        stored_keys = get_device_public_keys(self.device_id)
        assert stored_keys is not None
        assert stored_keys["kem_public_key"] == self.kem_public_key
        assert stored_keys["signature_public_key"] == self.sig_public_key
    
    def test_key_rotation_generation(self):
        """Test generation of new keys for rotation."""
        # Generate new key pair
        new_kem_key, new_sig_key = generate_device_keypair(self.device_id)
        
        # Verify new keys are different from original
        assert new_kem_key != self.kem_public_key
        assert new_sig_key != self.sig_public_key
        
        # Verify keys are valid base64
        try:
            base64.b64decode(new_kem_key)
            base64.b64decode(new_sig_key)
        except Exception:
            pytest.fail("Generated rotation keys are not valid base64")
    
    def test_key_rotation_update(self):
        """Test updating device keys during rotation."""
        # Register initial keys
        register_device_keys(self.device_id, self.kem_public_key, self.sig_public_key)
        
        # Generate new keys
        new_kem_key, new_sig_key = generate_device_keypair(self.device_id)
        
        # Update with new keys
        result = register_device_keys(self.device_id, new_kem_key, new_sig_key)
        assert result is True
        
        # Verify keys were updated
        stored_keys = get_device_public_keys(self.device_id)
        assert stored_keys["kem_public_key"] == new_kem_key
        assert stored_keys["signature_public_key"] == new_sig_key
        
        # Verify old keys are no longer stored
        assert stored_keys["kem_public_key"] != self.kem_public_key
        assert stored_keys["signature_public_key"] != self.sig_public_key
    
    def test_multiple_key_rotations(self):
        """Test multiple key rotations for the same device."""
        device_id = "rotation_test_device"
        
        # Perform multiple key rotations
        for i in range(3):
            kem_key, sig_key = generate_device_keypair(device_id)
            register_device_keys(device_id, kem_key, sig_key)
            
            # Verify keys are stored
            stored_keys = get_device_public_keys(device_id)
            assert stored_keys["kem_public_key"] == kem_key
            assert stored_keys["signature_public_key"] == sig_key
    
    def test_key_rotation_uniqueness(self):
        """Test that rotated keys are unique."""
        device_id = "uniqueness_test_device"
        
        # Generate multiple key pairs
        key_pairs = []
        for i in range(5):
            kem_key, sig_key = generate_device_keypair(device_id)
            key_pairs.append((kem_key, sig_key))
        
        # Verify all keys are unique
        kem_keys = [pair[0] for pair in key_pairs]
        sig_keys = [pair[1] for pair in key_pairs]
        
        assert len(set(kem_keys)) == len(kem_keys)
        assert len(set(sig_keys)) == len(sig_keys)


class TestKeyRotationScheduling:
    """Test key rotation scheduling and timing."""
    
    def test_key_rotation_interval(self):
        """Test key rotation based on time intervals."""
        device_id = "scheduled_device"
        
        # Register initial keys
        initial_kem, initial_sig = generate_device_keypair(device_id)
        register_device_keys(device_id, initial_kem, initial_sig)
        
        # Simulate time passing and rotation
        with patch('time.time') as mock_time:
            # Set initial time
            mock_time.return_value = 1000.0
            
            # Register initial keys
            register_device_keys(device_id, initial_kem, initial_sig)
            
            # Simulate time passing (rotation interval)
            mock_time.return_value = 1000.0 + 86400  # 24 hours later
            
            # Generate new keys for rotation
            new_kem, new_sig = generate_device_keypair(device_id)
            register_device_keys(device_id, new_kem, new_sig)
            
            # Verify keys were rotated
            stored_keys = get_device_public_keys(device_id)
            assert stored_keys["kem_public_key"] == new_kem
            assert stored_keys["signature_public_key"] == new_sig
    
    def test_key_rotation_trigger(self):
        """Test key rotation trigger conditions."""
        device_id = "trigger_test_device"
        
        # Register initial keys
        initial_kem, initial_sig = generate_device_keypair(device_id)
        register_device_keys(device_id, initial_kem, initial_sig)
        
        # Simulate rotation trigger (e.g., security event)
        new_kem, new_sig = generate_device_keypair(device_id)
        register_device_keys(device_id, new_kem, new_sig)
        
        # Verify rotation occurred
        stored_keys = get_device_public_keys(device_id)
        assert stored_keys["kem_public_key"] != initial_kem
        assert stored_keys["signature_public_key"] != initial_sig


class TestKeyRotationSecurity:
    """Test security aspects of key rotation."""
    
    def test_key_rotation_after_compromise(self):
        """Test key rotation after potential compromise."""
        device_id = "compromised_device"
        
        # Register initial keys
        initial_kem, initial_sig = generate_device_keypair(device_id)
        register_device_keys(device_id, initial_kem, initial_sig)
        
        # Simulate compromise detection
        compromised_kem = "compromised_kem_key"
        compromised_sig = "compromised_sig_key"
        
        # Rotate to new keys immediately
        new_kem, new_sig = generate_device_keypair(device_id)
        register_device_keys(device_id, new_kem, new_sig)
        
        # Verify compromised keys are no longer active
        stored_keys = get_device_public_keys(device_id)
        assert stored_keys["kem_public_key"] != compromised_kem
        assert stored_keys["signature_public_key"] != compromised_sig
        assert stored_keys["kem_public_key"] == new_kem
        assert stored_keys["signature_public_key"] == new_sig
    
    def test_key_rotation_validation(self):
        """Test validation during key rotation."""
        device_id = "validation_test_device"
        
        # Try to register invalid keys
        invalid_kem = "invalid_kem_key"
        invalid_sig = "invalid_sig_key"
        
        # Should handle invalid keys gracefully
        result = register_device_keys(device_id, invalid_kem, invalid_sig)
        # The function should still return True as it doesn't validate key format
        assert result is True
        
        # Verify keys were stored (even if invalid)
        stored_keys = get_device_public_keys(device_id)
        assert stored_keys["kem_public_key"] == invalid_kem
        assert stored_keys["signature_public_key"] == invalid_sig


class TestKeyRotationErrorHandling:
    """Test error handling during key rotation."""
    
    def test_key_rotation_with_none_keys(self):
        """Test key rotation with None keys."""
        device_id = "none_key_device"
        
        # Try to register None keys
        result = register_device_keys(device_id, None, None)
        # Should handle gracefully
        assert result is True
        
        # Verify None keys were stored
        stored_keys = get_device_public_keys(device_id)
        assert stored_keys["kem_public_key"] is None
        assert stored_keys["signature_public_key"] is None
    
    def test_key_rotation_with_empty_keys(self):
        """Test key rotation with empty keys."""
        device_id = "empty_key_device"
        
        # Try to register empty keys
        result = register_device_keys(device_id, "", "")
        assert result is True
        
        # Verify empty keys were stored
        stored_keys = get_device_public_keys(device_id)
        assert stored_keys["kem_public_key"] == ""
        assert stored_keys["signature_public_key"] == ""
    
    def test_key_rotation_device_not_found(self):
        """Test key rotation for non-existent device."""
        device_id = "nonexistent_device"
        
        # Try to get keys for non-existent device
        stored_keys = get_device_public_keys(device_id)
        assert stored_keys is None


class TestKeyRotationPerformance:
    """Test performance aspects of key rotation."""
    
    def test_key_generation_performance(self):
        """Test performance of key generation."""
        start_time = time.time()
        
        # Generate multiple key pairs
        for i in range(10):
            kem_key, sig_key = generate_device_keypair(f"perf_device_{i}")
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Should complete within reasonable time
        assert generation_time < 5.0  # 5 seconds for 10 key pairs
    
    def test_key_registration_performance(self):
        """Test performance of key registration."""
        start_time = time.time()
        
        # Register multiple devices
        for i in range(100):
            device_id = f"perf_device_{i}"
            kem_key, sig_key = generate_device_keypair(device_id)
            register_device_keys(device_id, kem_key, sig_key)
        
        end_time = time.time()
        registration_time = end_time - start_time
        
        # Should complete within reasonable time
        assert registration_time < 10.0  # 10 seconds for 100 registrations
    
    def test_key_retrieval_performance(self):
        """Test performance of key retrieval."""
        # Register multiple devices first
        for i in range(50):
            device_id = f"retrieval_device_{i}"
            kem_key, sig_key = generate_device_keypair(device_id)
            register_device_keys(device_id, kem_key, sig_key)
        
        start_time = time.time()
        
        # Retrieve keys for all devices
        for i in range(50):
            device_id = f"retrieval_device_{i}"
            stored_keys = get_device_public_keys(device_id)
            assert stored_keys is not None
        
        end_time = time.time()
        retrieval_time = end_time - start_time
        
        # Should complete within reasonable time
        assert retrieval_time < 2.0  # 2 seconds for 50 retrievals


class TestKeyRotationConcurrency:
    """Test concurrent key rotation operations."""
    
    def test_concurrent_key_registration(self):
        """Test concurrent key registration."""
        import threading
        
        results = []
        
        def register_device(device_id):
            kem_key, sig_key = generate_device_keypair(device_id)
            result = register_device_keys(device_id, kem_key, sig_key)
            results.append((device_id, result))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            device_id = f"concurrent_device_{i}"
            thread = threading.Thread(target=register_device, args=(device_id,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all registrations were successful
        assert len(results) == 5
        for device_id, result in results:
            assert result is True
            
            # Verify keys were stored
            stored_keys = get_device_public_keys(device_id)
            assert stored_keys is not None


if __name__ == "__main__":
    pytest.main([__file__])