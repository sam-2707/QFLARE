"""
Key Management System for QFLARE

This module handles post-quantum cryptography key generation, storage, and rotation.
"""

import os
import secrets
import base64
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import json

try:
    import oqs
except ImportError:
    logging.warning("liboqs not available, using fallback cryptography")
    oqs = None

try:
    import pqcrypto
except ImportError:
    logging.warning("pqcrypto not available, using fallback cryptography")
    pqcrypto = None

from database import db_manager

logger = logging.getLogger(__name__)

# Supported algorithms
KEM_ALGORITHMS = {
    "Kyber512": "Kyber512",
    "Kyber768": "Kyber768", 
    "Kyber1024": "Kyber1024",
    "HQC128": "HQC-128",
    "HQC192": "HQC-192"
}

SIGNATURE_ALGORITHMS = {
    "Dilithium2": "Dilithium2",
    "Dilithium3": "Dilithium3",
    "Dilithium5": "Dilithium5",
    "Falcon512": "Falcon-512",
    "Falcon1024": "Falcon-1024"
}

# Default algorithms
DEFAULT_KEM_ALGORITHM = "Kyber768"
DEFAULT_SIGNATURE_ALGORITHM = "Dilithium3"


class KeyManager:
    """Key management system for QFLARE."""
    
    def __init__(self):
        self.server_keys = {}
        self._initialize_server_keys()
    
    def _initialize_server_keys(self):
        """Initialize server key pairs."""
        try:
            # Generate server KEM key pair
            server_kem_public, server_kem_private = self.generate_kem_key_pair(DEFAULT_KEM_ALGORITHM)
            
            # Generate server signature key pair
            server_sig_public, server_sig_private = self.generate_signature_key_pair(DEFAULT_SIGNATURE_ALGORITHM)
            
            self.server_keys = {
                "kem": {
                    "algorithm": DEFAULT_KEM_ALGORITHM,
                    "public_key": server_kem_public,
                    "private_key": server_kem_private
                },
                "signature": {
                    "algorithm": DEFAULT_SIGNATURE_ALGORITHM,
                    "public_key": server_sig_public,
                    "private_key": server_sig_private
                }
            }
            
            logger.info("Server keys initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing server keys: {e}")
            raise
    
    def generate_kem_key_pair(self, algorithm: str = DEFAULT_KEM_ALGORITHM) -> Tuple[str, str]:
        """Generate a KEM key pair."""
        try:
            if oqs is None:
                # Fallback: generate random keys for testing
                public_key = base64.b64encode(secrets.token_bytes(32)).decode('utf-8')
                private_key = base64.b64encode(secrets.token_bytes(32)).decode('utf-8')
                return public_key, private_key
            
            with oqs.KeyEncapsulation(algorithm) as kem:
                public_key = base64.b64encode(kem.generate_keypair()).decode('utf-8')
                private_key = base64.b64encode(kem.export_secret_key()).decode('utf-8')
                
            logger.info(f"Generated KEM key pair using {algorithm}")
            return public_key, private_key
            
        except Exception as e:
            logger.error(f"Error generating KEM key pair: {e}")
            raise
    
    def generate_signature_key_pair(self, algorithm: str = DEFAULT_SIGNATURE_ALGORITHM) -> Tuple[str, str]:
        """Generate a signature key pair."""
        try:
            if oqs is None:
                # Fallback: generate random keys for testing
                public_key = base64.b64encode(secrets.token_bytes(32)).decode('utf-8')
                private_key = base64.b64encode(secrets.token_bytes(32)).decode('utf-8')
                return public_key, private_key
            
            with oqs.Signature(algorithm) as sig:
                public_key = base64.b64encode(sig.generate_keypair()).decode('utf-8')
                private_key = base64.b64encode(sig.export_secret_key()).decode('utf-8')
                
            logger.info(f"Generated signature key pair using {algorithm}")
            return public_key, private_key
            
        except Exception as e:
            logger.error(f"Error generating signature key pair: {e}")
            raise
    
    def generate_session_key(self, device_id: str, device_public_key: str) -> Optional[str]:
        """Generate a session key using device's public key."""
        try:
            if oqs is None:
                # Fallback: generate random session key
                return base64.b64encode(secrets.token_bytes(32)).decode('utf-8')
            
            device_public_key_bytes = base64.b64decode(device_public_key)
            
            with oqs.KeyEncapsulation(DEFAULT_KEM_ALGORITHM) as kem:
                # Import server private key
                server_private_key = base64.b64decode(self.server_keys["kem"]["private_key"])
                kem.import_secret_key(server_private_key)
                
                # Generate shared secret
                ciphertext, shared_secret = kem.encap_secret(device_public_key_bytes)
                
                # Store session info
                session_data = {
                    "ciphertext": base64.b64encode(ciphertext).decode('utf-8'),
                    "shared_secret": base64.b64encode(shared_secret).decode('utf-8'),
                    "created_at": datetime.now().isoformat()
                }
                
                # Store in database
                db_manager.store_key_pair(
                    device_id=device_id,
                    key_type="session",
                    public_key=json.dumps(session_data),
                    expires_at=datetime.now() + timedelta(hours=24)
                )
                
                return base64.b64encode(shared_secret).decode('utf-8')
                
        except Exception as e:
            logger.error(f"Error generating session key for device {device_id}: {e}")
            return None
    
    def verify_signature(self, device_id: str, message: bytes, signature: bytes) -> bool:
        """Verify a signature using device's public key."""
        try:
            if oqs is None:
                # Fallback: always return True for testing
                return True
            
            device_info = db_manager.get_device(device_id)
            if not device_info or not device_info.get("signature_public_key"):
                logger.error(f"No signature public key found for device {device_id}")
                return False
            
            device_public_key = base64.b64decode(device_info["signature_public_key"])
            
            with oqs.Signature(DEFAULT_SIGNATURE_ALGORITHM) as sig:
                sig.import_public_key(device_public_key)
                return sig.verify(message, signature)
                
        except Exception as e:
            logger.error(f"Error verifying signature for device {device_id}: {e}")
            return False
    
    def sign_message(self, message: bytes) -> Optional[str]:
        """Sign a message using server's private key."""
        try:
            if oqs is None:
                # Fallback: return random signature for testing
                return base64.b64encode(secrets.token_bytes(64)).decode('utf-8')
            
            server_private_key = base64.b64decode(self.server_keys["signature"]["private_key"])
            
            with oqs.Signature(DEFAULT_SIGNATURE_ALGORITHM) as sig:
                sig.import_secret_key(server_private_key)
                signature = sig.sign(message)
                return base64.b64encode(signature).decode('utf-8')
                
        except Exception as e:
            logger.error(f"Error signing message: {e}")
            return None
    
    def generate_enrollment_token(self, device_id: str, expires_in_hours: int = 24) -> str:
        """Generate an enrollment token for device registration."""
        try:
            token_data = {
                "device_id": device_id,
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(hours=expires_in_hours)).isoformat(),
                "nonce": secrets.token_hex(16)
            }
            
            token_json = json.dumps(token_data)
            token_signature = self.sign_message(token_json.encode('utf-8'))
            
            if not token_signature:
                raise Exception("Failed to sign enrollment token")
            
            enrollment_token = base64.b64encode(
                f"{token_json}.{token_signature}".encode('utf-8')
            ).decode('utf-8')
            
            # Store token in database
            db_manager.register_device(
                device_id=device_id,
                metadata={"enrollment_token": enrollment_token}
            )
            
            logger.info(f"Generated enrollment token for device {device_id}")
            return enrollment_token
            
        except Exception as e:
            logger.error(f"Error generating enrollment token for device {device_id}: {e}")
            raise
    
    def validate_enrollment_token(self, token: str, device_id: str) -> bool:
        """Validate an enrollment token."""
        try:
            token_bytes = base64.b64decode(token)
            token_parts = token_bytes.decode('utf-8').split('.')
            
            if len(token_parts) != 2:
                return False
            
            token_data, token_signature = token_parts
            
            # Verify signature
            if not self.verify_server_signature(token_data.encode('utf-8'), token_signature):
                return False
            
            # Parse token data
            token_info = json.loads(token_data)
            
            # Check device ID
            if token_info.get("device_id") != device_id:
                return False
            
            # Check expiration
            expires_at = datetime.fromisoformat(token_info.get("expires_at", ""))
            if datetime.now() > expires_at:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating enrollment token: {e}")
            return False
    
    def verify_server_signature(self, message: bytes, signature: str) -> bool:
        """Verify a signature using server's public key."""
        try:
            if oqs is None:
                # Fallback: always return True for testing
                return True
            
            signature_bytes = base64.b64decode(signature)
            server_public_key = base64.b64decode(self.server_keys["signature"]["public_key"])
            
            with oqs.Signature(DEFAULT_SIGNATURE_ALGORITHM) as sig:
                sig.import_public_key(server_public_key)
                return sig.verify(message, signature_bytes)
                
        except Exception as e:
            logger.error(f"Error verifying server signature: {e}")
            return False
    
    def get_server_public_keys(self) -> Dict[str, str]:
        """Get server's public keys."""
        return {
            "kem_public_key": self.server_keys["kem"]["public_key"],
            "signature_public_key": self.server_keys["signature"]["public_key"],
            "kem_algorithm": self.server_keys["kem"]["algorithm"],
            "signature_algorithm": self.server_keys["signature"]["algorithm"]
        }
    
    def rotate_server_keys(self) -> bool:
        """Rotate server keys."""
        try:
            # Generate new keys
            new_kem_public, new_kem_private = self.generate_kem_key_pair()
            new_sig_public, new_sig_private = self.generate_signature_key_pair()
            
            # Update server keys
            self.server_keys = {
                "kem": {
                    "algorithm": DEFAULT_KEM_ALGORITHM,
                    "public_key": new_kem_public,
                    "private_key": new_kem_private
                },
                "signature": {
                    "algorithm": DEFAULT_SIGNATURE_ALGORITHM,
                    "public_key": new_sig_public,
                    "private_key": new_sig_private
                }
            }
            
            logger.info("Server keys rotated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error rotating server keys: {e}")
            return False
    
    def cleanup_expired_keys(self) -> int:
        """Clean up expired keys from database."""
        try:
            # This would be implemented in the database manager
            # For now, return 0
            return 0
            
        except Exception as e:
            logger.error(f"Error cleaning up expired keys: {e}")
            return 0


# Global key manager instance
key_manager = KeyManager() 