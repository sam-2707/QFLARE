"""
Comprehensive authentication tests for QFLARE.
"""

import pytest
import sys
import os
import base64
import hashlib
import time
from unittest.mock import patch, MagicMock

# Add server path to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

from auth.pqcrypto_utils import (
    generate_device_keypair,
    validate_enrollment_token,
    register_device_keys,
    generate_session_challenge,
    verify_model_signature,
    revoke_enrollment_token,
    get_device_public_keys
)


class TestDeviceKeyGeneration:
    """Test device key pair generation."""
    
    def test_generate_device_keypair(self):
        """Test that device key pairs are generated correctly."""
        device_id = "test_device_001"
        kem_public_key, sig_public_key = generate_device_keypair(device_id)
        
        # Check that keys are base64 encoded strings
        assert isinstance(kem_public_key, str)
        assert isinstance(sig_public_key, str)
        
        # Check that keys are not empty
        assert len(kem_public_key) > 0
        assert len(sig_public_key) > 0
        
        # Check that keys are valid base64
        try:
            base64.b64decode(kem_public_key)
            base64.b64decode(sig_public_key)
        except Exception:
            pytest.fail("Generated keys are not valid base64")
    
    def test_key_generation_uniqueness(self):
        """Test that generated keys are unique."""
        device_id_1 = "test_device_001"
        device_id_2 = "test_device_002"
        
        kem_1, sig_1 = generate_device_keypair(device_id_1)
        kem_2, sig_2 = generate_device_keypair(device_id_2)
        
        # Keys should be different
        assert kem_1 != kem_2
        assert sig_1 != sig_2


class TestEnrollmentTokenValidation:
    """Test enrollment token validation."""
    
    def test_valid_token_validation(self):
        """Test validation of a valid token."""
        # Create a mock token
        token = "valid_token_123"
        device_id = "test_device_001"
        
        # Mock the token loading to return a valid token
        with patch('auth.pqcrypto_utils._load_tokens') as mock_load:
            mock_load.return_value = {
                token: {
                    "device_id": device_id,
                    "used": False,
                    "expires_at": time.time() + 3600  # 1 hour from now
                }
            }
            
            result = validate_enrollment_token(token, device_id)
            assert result is True
    
    def test_invalid_token_validation(self):
        """Test validation of an invalid token."""
        token = "invalid_token_123"
        device_id = "test_device_001"
        
        with patch('auth.pqcrypto_utils._load_tokens') as mock_load:
            mock_load.return_value = {}
            
            result = validate_enrollment_token(token, device_id)
            assert result is False
    
    def test_expired_token_validation(self):
        """Test validation of an expired token."""
        token = "expired_token_123"
        device_id = "test_device_001"
        
        with patch('auth.pqcrypto_utils._load_tokens') as mock_load:
            mock_load.return_value = {
                token: {
                    "device_id": device_id,
                    "used": False,
                    "expires_at": time.time() - 3600  # 1 hour ago
                }
            }
            
            result = validate_enrollment_token(token, device_id)
            assert result is False
    
    def test_used_token_validation(self):
        """Test validation of an already used token."""
        token = "used_token_123"
        device_id = "test_device_001"
        
        with patch('auth.pqcrypto_utils._load_tokens') as mock_load:
            mock_load.return_value = {
                token: {
                    "device_id": device_id,
                    "used": True,
                    "expires_at": time.time() + 3600
                }
            }
            
            result = validate_enrollment_token(token, device_id)
            assert result is False


class TestDeviceRegistration:
    """Test device registration functionality."""
    
    def test_register_device_keys(self):
        """Test device key registration."""
        device_id = "test_device_001"
        kem_public_key = base64.b64encode(b"test_kem_key").decode('utf-8')
        sig_public_key = base64.b64encode(b"test_sig_key").decode('utf-8')
        
        result = register_device_keys(device_id, kem_public_key, sig_public_key)
        assert result is True
        
        # Verify keys were stored
        stored_keys = get_device_public_keys(device_id)
        assert stored_keys is not None
        assert stored_keys["kem_public_key"] == kem_public_key
        assert stored_keys["signature_public_key"] == sig_public_key
    
    def test_register_nonexistent_device_keys(self):
        """Test getting keys for non-existent device."""
        device_id = "nonexistent_device"
        keys = get_device_public_keys(device_id)
        assert keys is None


class TestSessionChallenge:
    """Test session challenge generation."""
    
    def test_session_challenge_generation(self):
        """Test session challenge generation for registered device."""
        device_id = "test_device_001"
        kem_public_key = base64.b64encode(b"test_kem_key").decode('utf-8')
        sig_public_key = base64.b64encode(b"test_sig_key").decode('utf-8')
        
        # Register device first
        register_device_keys(device_id, kem_public_key, sig_public_key)
        
        # Generate session challenge
        challenge = generate_session_challenge(device_id)
        assert challenge is not None
        assert isinstance(challenge, str)
        assert len(challenge) > 0
    
    def test_session_challenge_unregistered_device(self):
        """Test session challenge for unregistered device."""
        device_id = "unregistered_device"
        challenge = generate_session_challenge(device_id)
        assert challenge is None


class TestModelSignatureVerification:
    """Test model signature verification."""
    
    def test_valid_signature_verification(self):
        """Test verification of valid signature."""
        device_id = "test_device_001"
        kem_public_key = base64.b64encode(b"test_kem_key").decode('utf-8')
        sig_public_key = base64.b64encode(b"test_sig_key").decode('utf-8')
        
        # Register device
        register_device_keys(device_id, kem_public_key, sig_public_key)
        
        # Test data and signature
        test_data = b"test model weights"
        test_signature = hashlib.sha256(test_data).hexdigest().encode()
        
        result = verify_model_signature(device_id, test_data, test_signature)
        assert result is True
    
    def test_invalid_signature_verification(self):
        """Test verification of invalid signature."""
        device_id = "test_device_001"
        kem_public_key = base64.b64encode(b"test_kem_key").decode('utf-8')
        sig_public_key = base64.b64encode(b"test_sig_key").decode('utf-8')
        
        # Register device
        register_device_keys(device_id, kem_public_key, sig_public_key)
        
        # Test data and wrong signature
        test_data = b"test model weights"
        wrong_signature = b"wrong_signature"
        
        result = verify_model_signature(device_id, test_data, wrong_signature)
        assert result is False
    
    def test_signature_verification_unregistered_device(self):
        """Test signature verification for unregistered device."""
        device_id = "unregistered_device"
        test_data = b"test model weights"
        test_signature = b"test_signature"
        
        result = verify_model_signature(device_id, test_data, test_signature)
        assert result is False


class TestTokenRevocation:
    """Test token revocation functionality."""
    
    def test_token_revocation(self):
        """Test successful token revocation."""
        token = "test_token_123"
        
        with patch('auth.pqcrypto_utils._load_tokens') as mock_load, \
             patch('auth.pqcrypto_utils._save_tokens') as mock_save:
            
            mock_load.return_value = {
                token: {
                    "device_id": "test_device",
                    "used": False,
                    "expires_at": time.time() + 3600
                }
            }
            
            result = revoke_enrollment_token(token)
            assert result is True
            
            # Verify save was called
            mock_save.assert_called_once()
    
    def test_revoke_nonexistent_token(self):
        """Test revocation of non-existent token."""
        token = "nonexistent_token"
        
        with patch('auth.pqcrypto_utils._load_tokens') as mock_load:
            mock_load.return_value = {}
            
            result = revoke_enrollment_token(token)
            assert result is False


class TestErrorHandling:
    """Test error handling in authentication functions."""
    
    def test_key_generation_error_handling(self):
        """Test error handling in key generation."""
        with patch('auth.pqcrypto_utils.OQS_AVAILABLE', False):
            device_id = "test_device_001"
            kem_public_key, sig_public_key = generate_device_keypair(device_id)
            
            # Should still return valid keys (fallback)
            assert isinstance(kem_public_key, str)
            assert isinstance(sig_public_key, str)
    
    def test_session_challenge_error_handling(self):
        """Test error handling in session challenge generation."""
        device_id = "test_device_001"
        kem_public_key = base64.b64encode(b"test_kem_key").decode('utf-8')
        sig_public_key = base64.b64encode(b"test_sig_key").decode('utf-8')
        
        register_device_keys(device_id, kem_public_key, sig_public_key)
        
        with patch('auth.pqcrypto_utils.OQS_AVAILABLE', False):
            challenge = generate_session_challenge(device_id)
            assert challenge is not None


if __name__ == "__main__":
    pytest.main([__file__])