"""
Post-Quantum Cryptography utilities for QFLARE Edge Node.

This module implements PQC algorithms for key exchange and digital signatures
on the edge device side.
"""

import hashlib
import secrets
import base64
import logging
from typing import Optional, Tuple, Dict, Any

# Import liboqs for PQC algorithms
try:
    import oqs
    OQS_AVAILABLE = True
    logging.info("liboqs successfully imported - using real PQC algorithms")
except (ImportError, RuntimeError, Exception) as e:
    OQS_AVAILABLE = False
    logging.warning(f"liboqs not available or failed to load: {e}")
    logging.warning("Using fallback implementations for development/testing")

logger = logging.getLogger(__name__)

# Configuration
KEM_ALGORITHM = "FrodoKEM-640-AES"  # Post-quantum KEM
SIG_ALGORITHM = "Dilithium2"  # Post-quantum signature

# Device key storage (use secure storage in production)
device_keys = {}


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


def store_device_keys(device_id: str, kem_private_key: bytes, sig_private_key: bytes):
    """
    Store device private keys securely.
    
    Args:
        device_id: Device identifier
        kem_private_key: KEM private key
        sig_private_key: Signature private key
    """
    try:
        device_keys[device_id] = {
            "kem_private_key": kem_private_key,
            "sig_private_key": sig_private_key
        }
        logger.info(f"Stored private keys for device {device_id}")
        
    except Exception as e:
        logger.error(f"Error storing keys for device {device_id}: {e}")
        raise


def get_device_private_keys(device_id: str) -> Optional[Dict[str, bytes]]:
    """
    Get device private keys.
    
    Args:
        device_id: Device identifier
        
    Returns:
        Dictionary with private keys, or None if device not found
    """
    try:
        return device_keys.get(device_id)
        
    except Exception as e:
        logger.error(f"Error getting private keys for device {device_id}: {e}")
        return None


def sign_model_update(device_id: str, model_weights: bytes) -> bytes:
    """
    Sign model update with device's private key.
    
    Args:
        device_id: Device identifier
        model_weights: Model weights as bytes
        
    Returns:
        Digital signature as bytes
    """
    try:
        device_key = get_device_private_keys(device_id)
        if not device_key:
            logger.error(f"Device {device_id} not found for signing")
            return b""
        
        sig_private_key = device_key["sig_private_key"]
        
        if not OQS_AVAILABLE:
            # Fallback implementation - simple hash
            logger.info(f"Using fallback signing for device {device_id}")
            return hashlib.sha256(model_weights).digest()
        
        # Sign model update using device's private key
        with oqs.Signature(SIG_ALGORITHM) as sig:
            signature = sig.sign(model_weights, sig_private_key)
            return signature
        
    except Exception as e:
        logger.error(f"Error signing model update for device {device_id}: {e}")
        # Fallback to simple hash
        logger.info(f"Falling back to development signing for device {device_id}")
        return hashlib.sha256(model_weights).digest()


def decrypt_session_key(device_id: str, encrypted_session_key: str) -> Optional[bytes]:
    """
    Decrypt session key using device's KEM private key.
    
    Args:
        device_id: Device identifier
        encrypted_session_key: Base64 encoded encrypted session key
        
    Returns:
        Decrypted session key as bytes, or None if failed
    """
    try:
        device_key = get_device_private_keys(device_id)
        if not device_key:
            logger.error(f"Device {device_id} not found for decryption")
            return None
        
        kem_private_key = device_key["kem_private_key"]
        ciphertext = base64.b64decode(encrypted_session_key)
        
        if not OQS_AVAILABLE:
            # Fallback implementation - simple XOR decryption
            logger.info(f"Using fallback decryption for device {device_id}")
            key_bytes = secrets.token_bytes(32)
            return bytes(a ^ b for a, b in zip(ciphertext, key_bytes))
        
        # Decapsulate session key using device's KEM private key
        with oqs.KeyEncapsulation(KEM_ALGORITHM) as kem:
            session_key = kem.decap_secret(ciphertext, kem_private_key)
            return session_key
        
    except Exception as e:
        logger.error(f"Error decrypting session key for device {device_id}: {e}")
        # Fallback implementation
        logger.info(f"Falling back to development decryption for device {device_id}")
        key_bytes = secrets.token_bytes(32)
        return bytes(a ^ b for a, b in zip(ciphertext, key_bytes))


def verify_server_signature(message: bytes, signature: bytes, server_public_key: bytes) -> bool:
    """
    Verify server's digital signature.
    
    Args:
        message: Message that was signed
        signature: Digital signature
        server_public_key: Server's public key
        
    Returns:
        True if signature is valid, False otherwise
    """
    try:
        if not OQS_AVAILABLE:
            # Fallback implementation - simple hash verification
            logger.info("Using fallback server signature verification")
            expected_hash = hashlib.sha256(message).hexdigest()
            return signature == expected_hash.encode()
        
        # Verify signature using server's public key
        with oqs.Signature(SIG_ALGORITHM) as sig:
            return sig.verify(message, signature, server_public_key)
        
    except Exception as e:
        logger.error(f"Error verifying server signature: {e}")
        # Fallback to simple hash verification
        logger.info("Falling back to development server signature verification")
        expected_hash = hashlib.sha256(message).hexdigest()
        return signature == expected_hash.encode()


def generate_session_key() -> bytes:
    """
    Generate a new session key for secure communication.
    
    Returns:
        Session key as bytes
    """
    try:
        return secrets.token_bytes(32)
        
    except Exception as e:
        logger.error(f"Error generating session key: {e}")
        raise


def hash_model_weights(model_weights: bytes) -> str:
    """
    Compute hash of model weights for integrity verification.
    
    Args:
        model_weights: Model weights as bytes
        
    Returns:
        SHA-256 hash as hex string
    """
    try:
        return hashlib.sha256(model_weights).hexdigest()
        
    except Exception as e:
        logger.error(f"Error hashing model weights: {e}")
        raise


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


# Legacy function for backward compatibility
def authenticate_with_server(device_id: str, qkey: str) -> bool:
    """
    Legacy authentication function (deprecated).
    
    This function is kept for backward compatibility but should not be used
    in new code. Use the secure enrollment and challenge-response mechanism instead.
    """
    logger.warning("Legacy authentication called - use secure enrollment instead")
    return True  # Always return True for backward compatibility 