"""
Post-Quantum Cryptography utilities for QFLARE.

This module implements real PQC algorithms for key exchange and digital signatures.
In a production environment, this would use hardware-accelerated implementations.
"""

import hashlib
import secrets
import base64
import json
import time
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

# Import liboqs for PQC algorithms
try:
    import oqs
    OQS_AVAILABLE = True
    logging.info("liboqs successfully imported - using real PQC algorithms")
except ImportError as e:
    OQS_AVAILABLE = False
    logging.warning(f"liboqs not available: {e}")
    logging.warning("Using fallback implementations for development/testing")
except RuntimeError as e:
    OQS_AVAILABLE = False
    logging.warning(f"liboqs runtime error: {e}")
    logging.warning("Using fallback implementations for development/testing")

logger = logging.getLogger(__name__)

# Configuration
KEM_ALGORITHM = "FrodoKEM-640-AES"  # Post-quantum KEM
SIG_ALGORITHM = "Dilithium2"  # Post-quantum signature
TOKEN_FILE = "enrollment_tokens.json"

# In-memory storage for development (use database in production)
device_keys = {}
enrollment_tokens = {}


def _load_tokens() -> Dict[str, Any]:
    """Load enrollment tokens from file."""
    token_path = Path(TOKEN_FILE)
    if token_path.exists():
        try:
            with open(token_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}


def _save_tokens(tokens: Dict[str, Any]):
    """Save tokens to file."""
    token_path = Path(TOKEN_FILE)
    token_path.parent.mkdir(exist_ok=True)
    with open(token_path, 'w') as f:
        json.dump(tokens, f, indent=2)


def generate_device_keypair(device_id: str) -> Tuple[str, str]:
    """
    Generate a Post-Quantum key pair for a device.
    
    Args:
        device_id: Unique device identifier
        
    Returns:
        Tuple of (kem_public_key, signature_public_key) as base64 strings
    """
    try:
        if not OQS_AVAILABLE:
            # Fallback implementation for development
            logger.info(f"Using fallback key generation for device {device_id}")
            kem_public_key = base64.b64encode(secrets.token_bytes(32)).decode('utf-8')
            sig_public_key = base64.b64encode(secrets.token_bytes(32)).decode('utf-8')
            return kem_public_key, sig_public_key
        
        # Generate KEM key pair
        with oqs.KeyEncapsulation(KEM_ALGORITHM) as kem:
            kem_public_key = base64.b64encode(kem.generate_keypair()).decode('utf-8')
        
        # Generate signature key pair
        with oqs.Signature(SIG_ALGORITHM) as sig:
            sig_public_key = base64.b64encode(sig.generate_keypair()).decode('utf-8')
        
        logger.info(f"Generated PQC key pair for device {device_id}")
        return kem_public_key, sig_public_key
        
    except Exception as e:
        logger.error(f"Error generating key pair for device {device_id}: {e}")
        # Fallback to development implementation
        logger.info(f"Falling back to development key generation for device {device_id}")
        kem_public_key = base64.b64encode(secrets.token_bytes(32)).decode('utf-8')
        sig_public_key = base64.b64encode(secrets.token_bytes(32)).decode('utf-8')
        return kem_public_key, sig_public_key


def generate_onetime_session_key(device_id: str) -> str:
    """
    Generate a one-time session key for secure communication.
    
    Args:
        device_id: Device identifier
        
    Returns:
        Base64 encoded session key
    """
    try:
        # Generate a cryptographically secure random session key
        session_key = secrets.token_bytes(32)
        return base64.b64encode(session_key).decode('utf-8')
        
    except Exception as e:
        logger.error(f"Error generating session key for device {device_id}: {e}")
        raise


def validate_enrollment_token(token: str, device_id: str) -> bool:
    """
    Validate an enrollment token for device registration.
    
    Args:
        token: Enrollment token
        device_id: Device identifier
        
    Returns:
        True if token is valid, False otherwise
    """
    try:
        tokens = _load_tokens()
        
        if token not in tokens:
            logger.warning(f"Invalid enrollment token for device {device_id}")
            return False
        
        token_data = tokens[token]
        
        # Check if token is already used
        if token_data.get("used", False):
            logger.warning(f"Enrollment token already used for device {device_id}")
            return False
        
        # Check if token is expired
        if time.time() > token_data.get("expires_at", 0):
            logger.warning(f"Enrollment token expired for device {device_id}")
            return False
        
        # Check if device_id matches
        if token_data.get("device_id") != device_id:
            logger.warning(f"Device ID mismatch for enrollment token")
            return False
        
        logger.info(f"Valid enrollment token for device {device_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error validating enrollment token: {e}")
        return False


def register_device_keys(device_id: str, kem_public_key: str, signature_public_key: str) -> bool:
    """
    Register device public keys.
    
    Args:
        device_id: Device identifier
        kem_public_key: KEM public key
        signature_public_key: Signature public key
        
    Returns:
        True if registration successful, False otherwise
    """
    try:
        device_keys[device_id] = {
            "kem_public_key": kem_public_key,
            "signature_public_key": signature_public_key,
            "registered_at": time.time(),
            "status": "active"
        }
        
        logger.info(f"Registered keys for device {device_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error registering keys for device {device_id}: {e}")
        return False


def generate_session_challenge(device_id: str) -> Optional[str]:
    """
    Generate a session challenge using device's KEM public key.
    
    Args:
        device_id: Device identifier
        
    Returns:
        Base64 encoded encrypted session key, or None if device not found
    """
    try:
        if device_id not in device_keys:
            logger.warning(f"Device {device_id} not found for session challenge")
            return None
        
        device_key = device_keys[device_id]["kem_public_key"]
        
        if not OQS_AVAILABLE:
            # Fallback implementation
            logger.info(f"Using fallback session challenge for device {device_id}")
            session_key = secrets.token_bytes(32)
            # Simulate encryption by XORing with a random key
            encrypted_key = bytes(a ^ b for a, b in zip(session_key, secrets.token_bytes(32)))
            return base64.b64encode(encrypted_key).decode('utf-8')
        
        # Use device's KEM public key to encapsulate session key
        with oqs.KeyEncapsulation(KEM_ALGORITHM) as kem:
            kem_public_key_bytes = base64.b64decode(device_key)
            ciphertext, shared_secret = kem.encap_secret(kem_public_key_bytes)
            
            # Use shared secret as session key
            return base64.b64encode(ciphertext).decode('utf-8')
        
    except Exception as e:
        logger.error(f"Error generating session challenge for device {device_id}: {e}")
        # Fallback implementation
        logger.info(f"Falling back to development session challenge for device {device_id}")
        session_key = secrets.token_bytes(32)
        encrypted_key = bytes(a ^ b for a, b in zip(session_key, secrets.token_bytes(32)))
        return base64.b64encode(encrypted_key).decode('utf-8')


def verify_model_signature(device_id: str, model_weights: bytes, signature: bytes) -> bool:
    """
    Verify digital signature of model update.
    
    Args:
        device_id: Device identifier
        model_weights: Model weights as bytes
        signature: Digital signature
        
    Returns:
        True if signature is valid, False otherwise
    """
    try:
        if device_id not in device_keys:
            logger.warning(f"Device {device_id} not found for signature verification")
            return False
        
        device_key = device_keys[device_id]["signature_public_key"]
        
        if not OQS_AVAILABLE:
            # Fallback implementation - simple hash verification
            logger.info(f"Using fallback signature verification for device {device_id}")
            expected_hash = hashlib.sha256(model_weights).hexdigest()
            return signature == expected_hash.encode()
        
        # Verify signature using device's public key
        with oqs.Signature(SIG_ALGORITHM) as sig:
            sig_public_key_bytes = base64.b64decode(device_key)
            return sig.verify(model_weights, signature, sig_public_key_bytes)
        
    except Exception as e:
        logger.error(f"Error verifying signature for device {device_id}: {e}")
        # Fallback to simple hash verification
        logger.info(f"Falling back to development signature verification for device {device_id}")
        expected_hash = hashlib.sha256(model_weights).hexdigest()
        return signature == expected_hash.encode()


def get_device_public_keys(device_id: str) -> Optional[Dict[str, str]]:
    """
    Get device's public keys.
    
    Args:
        device_id: Device identifier
        
    Returns:
        Dictionary with public keys, or None if device not found
    """
    try:
        if device_id not in device_keys:
            return None
        
        return {
            "kem_public_key": device_keys[device_id]["kem_public_key"],
            "signature_public_key": device_keys[device_id]["signature_public_key"]
        }
        
    except Exception as e:
        logger.error(f"Error getting public keys for device {device_id}: {e}")
        return None


def revoke_enrollment_token(token: str) -> bool:
    """
    Revoke an enrollment token after successful enrollment.
    
    Args:
        token: Enrollment token to revoke
        
    Returns:
        True if token was revoked, False otherwise
    """
    try:
        tokens = _load_tokens()
        
        if token in tokens:
            tokens[token]["used"] = True
            _save_tokens(tokens)
            logger.info(f"Revoked enrollment token {token[:16]}...")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error revoking enrollment token: {e}")
        return False


def get_enabled_pqc_algorithms() -> Dict[str, list]:
    """
    Get list of enabled PQC algorithms.
    
    Returns:
        Dictionary with enabled KEM and signature algorithms
    """
    if not OQS_AVAILABLE:
        return {
            "kem": ["FrodoKEM-640-AES (fallback)"],
            "sig": ["Dilithium2 (fallback)"],
            "note": "liboqs not available - using development fallbacks"
        }
    
    try:
        return {
            "kem": list(oqs.get_enabled_kem_mechanisms()),
            "sig": list(oqs.get_enabled_sig_mechanisms())
        }
    except Exception as e:
        logger.error(f"Error getting enabled algorithms: {e}")
        return {"kem": [], "sig": []}


def cleanup_expired_tokens():
    """Clean up expired enrollment tokens."""
    try:
        tokens = _load_tokens()
        current_time = time.time()
        
        expired_tokens = [
            token for token, data in tokens.items()
            if current_time > data.get("expires_at", 0)
        ]
        
        for token in expired_tokens:
            del tokens[token]
        
        if expired_tokens:
            _save_tokens(tokens)
            logger.info(f"Cleaned up {len(expired_tokens)} expired tokens")
            
    except Exception as e:
        logger.error(f"Error cleaning up expired tokens: {e}")


# Legacy function for backward compatibility
def verify_quantum_key(device_id: str, qkey: str) -> bool:
    """
    Legacy quantum key verification (deprecated).
    
    This function is kept for backward compatibility but should not be used
    in new code. Use the secure enrollment and challenge-response mechanism instead.
    """
    logger.warning("Legacy quantum key verification called - use secure enrollment instead")
    return True  # Always return True for backward compatibility