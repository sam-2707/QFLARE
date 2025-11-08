"""
QFLARE Key Management System
Secure post-quantum cryptography key generation and exchange
"""

import os
import json
import base64
import secrets
from typing import Tuple, Dict, Any
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Skip liboqs for MVP - use mock crypto for testing
PQC_AVAILABLE = False
print("Info: Using mock crypto for MVP testing (liboqs disabled)")

class QFLAREKeyManager:
    """
    Manages post-quantum cryptographic keys for QFLARE federated learning clients
    """
    
    def __init__(self):
        self.pqc_available = PQC_AVAILABLE
        
    def generate_client_keypair(self, user_id: str, algorithm: str = "Kyber1024") -> Dict[str, Any]:
        """
        Generate a post-quantum key pair for a client
        
        Returns:
            Dict containing public_key, private_key_encrypted, key_metadata
        """
        if self.pqc_available:
            return self._generate_pqc_keypair(user_id, algorithm)
        else:
            return self._generate_mock_keypair(user_id, algorithm)
    
    def _generate_pqc_keypair(self, user_id: str, algorithm: str) -> Dict[str, Any]:
        """Generate actual post-quantum cryptographic keys using liboqs"""
        try:
            # Create KEM (Key Encapsulation Mechanism)
            kem = oqs.KeyEncapsulation(algorithm)
            
            # Generate keypair
            public_key = kem.generate_keypair()
            private_key = kem.export_secret_key()
            
            # Create signature keypair for authentication
            sig = oqs.Signature("Dilithium2")
            sig_public_key = sig.generate_keypair()
            sig_private_key = sig.export_secret_key()
            
            # Encrypt private keys with user-derived key
            user_salt = secrets.token_bytes(32)
            encrypted_private_key = self._encrypt_private_key(private_key, user_id, user_salt)
            encrypted_sig_private_key = self._encrypt_private_key(sig_private_key, user_id, user_salt)
            
            return {
                "user_id": user_id,
                "kem_public_key": base64.b64encode(public_key).decode(),
                "kem_private_key_encrypted": encrypted_private_key,
                "sig_public_key": base64.b64encode(sig_public_key).decode(),
                "sig_private_key_encrypted": encrypted_sig_private_key,
                "algorithm": algorithm,
                "signature_algorithm": "Dilithium2",
                "salt": base64.b64encode(user_salt).decode(),
                "created_at": datetime.utcnow().isoformat(),
                "key_id": f"qflare_{user_id}_{secrets.token_hex(8)}"
            }
            
        except Exception as e:
            print(f"PQC key generation failed: {e}")
            return self._generate_mock_keypair(user_id, algorithm)
    
    def _generate_mock_keypair(self, user_id: str, algorithm: str) -> Dict[str, Any]:
        """Generate mock keys for development/testing"""
        # Generate mock keys that look realistic
        mock_public_kem = secrets.token_bytes(1568)  # Kyber1024 public key size
        mock_private_kem = secrets.token_bytes(3168)  # Kyber1024 private key size
        mock_public_sig = secrets.token_bytes(2592)  # Dilithium2 public key size
        mock_private_sig = secrets.token_bytes(4864)  # Dilithium2 private key size
        
        user_salt = secrets.token_bytes(32)
        encrypted_private_kem = self._encrypt_private_key(mock_private_kem, user_id, user_salt)
        encrypted_private_sig = self._encrypt_private_key(mock_private_sig, user_id, user_salt)
        
        return {
            "user_id": user_id,
            "kem_public_key": base64.b64encode(mock_public_kem).decode(),
            "kem_private_key_encrypted": encrypted_private_kem,
            "sig_public_key": base64.b64encode(mock_public_sig).decode(),
            "sig_private_key_encrypted": encrypted_private_sig,
            "algorithm": f"Mock{algorithm}",
            "signature_algorithm": "MockDilithium2",
            "salt": base64.b64encode(user_salt).decode(),
            "created_at": datetime.utcnow().isoformat(),
            "key_id": f"mock_{user_id}_{secrets.token_hex(8)}"
        }
    
    def _encrypt_private_key(self, private_key: bytes, user_id: str, salt: bytes) -> str:
        """Encrypt private key with user-derived encryption key"""
        # Derive key from user_id (in production, use user password)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(user_id.encode()))
        
        # Encrypt the private key
        f = Fernet(key)
        encrypted = f.encrypt(private_key)
        return base64.b64encode(encrypted).decode()
    
    def decrypt_private_key(self, encrypted_private_key: str, user_id: str, salt: str) -> bytes:
        """Decrypt private key for use"""
        try:
            # Derive the same key
            salt_bytes = base64.b64decode(salt)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt_bytes,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(user_id.encode()))
            
            # Decrypt
            f = Fernet(key)
            encrypted_bytes = base64.b64decode(encrypted_private_key)
            return f.decrypt(encrypted_bytes)
            
        except Exception as e:
            raise ValueError(f"Failed to decrypt private key: {e}")
    
    def create_handshake_token(self, user_id: str, admin_private_key: bytes) -> str:
        """Create a handshake token for client authentication"""
        handshake_data = {
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "expires": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
            "nonce": secrets.token_hex(16)
        }
        
        # In production, sign with admin's private key
        token = base64.b64encode(json.dumps(handshake_data).encode()).decode()
        return token
    
    def verify_handshake_token(self, token: str, admin_public_key: bytes) -> Dict[str, Any]:
        """Verify a handshake token"""
        try:
            data = json.loads(base64.b64decode(token).decode())
            
            # Check expiration
            expires = datetime.fromisoformat(data["expires"])
            if datetime.utcnow() > expires:
                raise ValueError("Token expired")
                
            return data
            
        except Exception as e:
            raise ValueError(f"Invalid handshake token: {e}")

# Global key manager instance
key_manager = QFLAREKeyManager()

def generate_admin_keypair() -> Dict[str, str]:
    """Generate admin keypair for system initialization"""
    return key_manager.generate_client_keypair("admin_system", "Kyber1024")

# Client Registration Flow Helper
class ClientRegistrationFlow:
    """
    Manages the complete client registration and key exchange process
    """
    
    @staticmethod
    def create_registration_request(user_data: dict) -> Dict[str, Any]:
        """Step 1: Client creates registration request"""
        return {
            "user_data": user_data,
            "registration_id": str(secrets.token_hex(16)),
            "timestamp": datetime.utcnow().isoformat(),
            "status": "pending_admin_approval"
        }
    
    @staticmethod
    def admin_approve_and_generate_keys(user_id: str) -> Dict[str, Any]:
        """Step 2: Admin approves user and generates keys"""
        keys = key_manager.generate_client_keypair(user_id)
        
        return {
            "approval_status": "approved",
            "keys_generated": True,
            "key_metadata": {
                "key_id": keys["key_id"],
                "algorithm": keys["algorithm"],
                "created_at": keys["created_at"]
            },
            "handshake_token": key_manager.create_handshake_token(user_id, b"admin_key"),
            "next_steps": "Client should download keys and complete setup"
        }
    
    @staticmethod
    def client_key_download(user_id: str, authentication_token: str) -> Dict[str, Any]:
        """Step 3: Client downloads their keys after approval"""
        # Verify the client is authorized to download these keys
        # In production, verify JWT token and user permissions
        
        return {
            "message": "Keys ready for download",
            "download_instructions": {
                "1": "Store private keys securely on your local device",
                "2": "Use public keys for FL communication",
                "3": "Never share private keys",
                "4": "Report any key compromise immediately"
            }
        }

# Export the registration flow
registration_flow = ClientRegistrationFlow()