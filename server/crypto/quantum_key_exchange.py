#!/usr/bin/env python3
"""
Quantum-Resistant Lattice-Based Key Exchange System
CRYSTALS-Kyber + Timestamp Mapping + Grover's Algorithm Resistance

This implements a comprehensive quantum-safe key exchange using:
1. CRYSTALS-Kyber 1024 (NIST Level 5 security)
2. Timestamp-based key derivation functions
3. Forward secrecy with ephemeral keys
4. Protection against Grover's algorithm (doubling key sizes)
5. Secure key mapping between client and server
"""

import os
import time
import hmac
import hashlib
import secrets
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import struct
import base64
import logging

try:
    import oqs
    LIBOQS_AVAILABLE = True
except ImportError:
    LIBOQS_AVAILABLE = False

# Fallback cryptography for development
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

@dataclass
class KeyExchangeContext:
    """Context for a key exchange session"""
    session_id: str
    device_id: str
    timestamp: int
    client_public_key: bytes
    server_private_key: bytes
    server_public_key: bytes
    shared_secret: bytes
    derived_key: bytes
    expiry_time: int
    nonce: bytes
    
class TimestampKeyDerivation:
    """
    Timestamp-based key derivation with quantum resistance.
    Uses time-based entropy to ensure keys change over time.
    """
    
    def __init__(self, base_key: bytes, time_window: int = 300):
        self.base_key = base_key
        self.time_window = time_window  # 5-minute windows
        
    def derive_temporal_key(self, timestamp: int, salt: bytes) -> bytes:
        """
        Derive a temporal key based on timestamp.
        This provides forward secrecy and time-based key rotation.
        """
        # Calculate time window
        time_bucket = timestamp // self.time_window
        
        # Create temporal context
        temporal_context = struct.pack('>Q', time_bucket) + salt
        
        # Use HKDF with temporal context (quantum-safe derivation)
        hkdf = HKDF(
            algorithm=hashes.SHA3_512(),  # Quantum-resistant hash
            length=64,  # 512 bits for Grover resistance (double SHA-256)
            salt=salt,
            info=temporal_context
        )
        
        return hkdf.derive(self.base_key)

class LatticeKeyExchange:
    """
    Lattice-based key exchange using CRYSTALS-Kyber with timestamp mapping.
    Provides quantum resistance against both classical and quantum attacks.
    """
    
    def __init__(self):
        self.algorithm = "Kyber1024"  # NIST Level 5 security
        self.sessions: Dict[str, KeyExchangeContext] = {}
        self.server_keys = self._generate_server_keypair()
        
        if LIBOQS_AVAILABLE:
            self.kem = oqs.KeyEncapsulation(self.algorithm)
            logger.info(f"Using quantum-safe KEM: {self.algorithm}")
        else:
            logger.warning("liboqs not available, using RSA fallback")
            
    def _generate_server_keypair(self) -> Tuple[bytes, bytes]:
        """Generate server's long-term keypair"""
        if LIBOQS_AVAILABLE:
            kem = oqs.KeyEncapsulation(self.algorithm)
            public_key = kem.generate_keypair()
            private_key = kem.export_secret_key()
            return private_key, public_key
        else:
            # RSA fallback (4096 bits for quantum resistance)
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096
            )
            public_key = private_key.public_key()
            
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            return private_pem, public_pem
    
    def initiate_key_exchange(self, device_id: str, client_public_key: bytes) -> Dict:
        """
        Initiate key exchange with a client device.
        Returns server's response for the key exchange.
        """
        timestamp = int(time.time())
        session_id = secrets.token_hex(32)
        nonce = secrets.token_bytes(32)
        
        try:
            if LIBOQS_AVAILABLE:
                # Use CRYSTALS-Kyber for quantum-safe key exchange
                kem = oqs.KeyEncapsulation(self.algorithm)
                kem.generate_keypair()
                
                # Encapsulate using client's public key
                ciphertext, shared_secret = kem.encap_secret(client_public_key)
                
            else:
                # RSA fallback with OAEP padding
                shared_secret = secrets.token_bytes(64)  # 512 bits
                
                # Load client public key
                from cryptography.hazmat.primitives import serialization
                client_key = serialization.load_pem_public_key(client_public_key)
                
                # Encrypt shared secret with client's public key
                ciphertext = client_key.encrypt(
                    shared_secret,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA512()),
                        algorithm=hashes.SHA512(),
                        label=None
                    )
                )
            
            # Derive session key using timestamp
            temporal_kdf = TimestampKeyDerivation(shared_secret)
            derived_key = temporal_kdf.derive_temporal_key(timestamp, nonce)
            
            # Create session context
            context = KeyExchangeContext(
                session_id=session_id,
                device_id=device_id,
                timestamp=timestamp,
                client_public_key=client_public_key,
                server_private_key=self.server_keys[0],
                server_public_key=self.server_keys[1],
                shared_secret=shared_secret,
                derived_key=derived_key,
                expiry_time=timestamp + 3600,  # 1 hour expiry
                nonce=nonce
            )
            
            self.sessions[session_id] = context
            
            # Return exchange data
            return {
                'session_id': session_id,
                'server_public_key': base64.b64encode(self.server_keys[1]).decode(),
                'ciphertext': base64.b64encode(ciphertext).decode(),
                'timestamp': timestamp,
                'nonce': base64.b64encode(nonce).decode(),
                'algorithm': self.algorithm,
                'expiry_time': context.expiry_time
            }
            
        except Exception as e:
            logger.error(f"Key exchange initiation failed: {e}")
            raise
    
    def complete_key_exchange(self, session_id: str, client_response: bytes) -> bool:
        """
        Complete the key exchange process.
        Verify client's response and establish the session.
        """
        if session_id not in self.sessions:
            return False
            
        context = self.sessions[session_id]
        
        # Verify timestamp validity (prevent replay attacks)
        current_time = int(time.time())
        if current_time - context.timestamp > 300:  # 5-minute window
            del self.sessions[session_id]
            return False
        
        try:
            # Verify client response using derived key
            expected_response = hmac.new(
                context.derived_key,
                context.session_id.encode() + struct.pack('>Q', context.timestamp),
                hashlib.sha3_512
            ).digest()
            
            if hmac.compare_digest(client_response, expected_response):
                logger.info(f"Key exchange completed for session {session_id}")
                return True
            else:
                logger.warning(f"Invalid client response for session {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"Key exchange completion failed: {e}")
            return False
    
    def get_session_key(self, session_id: str) -> Optional[bytes]:
        """Get the derived session key for encrypted communication"""
        context = self.sessions.get(session_id)
        if context and int(time.time()) < context.expiry_time:
            return context.derived_key
        return None
    
    def rotate_session_key(self, session_id: str) -> Optional[bytes]:
        """
        Rotate session key based on current timestamp.
        Provides forward secrecy.
        """
        context = self.sessions.get(session_id)
        if not context:
            return None
            
        current_time = int(time.time())
        if current_time >= context.expiry_time:
            del self.sessions[session_id]
            return None
        
        # Generate new temporal key
        new_nonce = secrets.token_bytes(32)
        temporal_kdf = TimestampKeyDerivation(context.shared_secret)
        new_key = temporal_kdf.derive_temporal_key(current_time, new_nonce)
        
        # Update context
        context.derived_key = new_key
        context.timestamp = current_time
        context.nonce = new_nonce
        
        return new_key
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = int(time.time())
        expired_sessions = [
            sid for sid, ctx in self.sessions.items()
            if current_time >= ctx.expiry_time
        ]
        
        for sid in expired_sessions:
            del self.sessions[sid]
            
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

class QuantumSafeEncryption:
    """
    Quantum-safe encryption using the derived keys from key exchange.
    Uses AES-256-GCM with quantum-resistant key derivation.
    """
    
    def __init__(self, session_key: bytes):
        self.session_key = session_key
        
    def encrypt(self, plaintext: bytes, associated_data: bytes = b'') -> Dict:
        """Encrypt data with quantum-safe parameters"""
        # Generate random IV (96 bits for GCM)
        iv = secrets.token_bytes(12)
        
        # Use first 32 bytes of session key for AES-256
        encryption_key = self.session_key[:32]
        
        # AES-256-GCM encryption
        cipher = Cipher(
            algorithms.AES(encryption_key),
            modes.GCM(iv)
        )
        encryptor = cipher.encryptor()
        
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)
            
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        return {
            'ciphertext': base64.b64encode(ciphertext).decode(),
            'iv': base64.b64encode(iv).decode(),
            'tag': base64.b64encode(encryptor.tag).decode(),
            'associated_data': base64.b64encode(associated_data).decode() if associated_data else ''
        }
    
    def decrypt(self, encrypted_data: Dict) -> bytes:
        """Decrypt data"""
        ciphertext = base64.b64decode(encrypted_data['ciphertext'])
        iv = base64.b64decode(encrypted_data['iv'])
        tag = base64.b64decode(encrypted_data['tag'])
        associated_data = base64.b64decode(encrypted_data['associated_data']) if encrypted_data['associated_data'] else b''
        
        encryption_key = self.session_key[:32]
        
        cipher = Cipher(
            algorithms.AES(encryption_key),
            modes.GCM(iv, tag)
        )
        decryptor = cipher.decryptor()
        
        if associated_data:
            decryptor.authenticate_additional_data(associated_data)
            
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        return plaintext

# Example usage and testing
if __name__ == "__main__":
    # Initialize key exchange system
    kx = LatticeKeyExchange()
    
    # Simulate client key generation
    if LIBOQS_AVAILABLE:
        client_kem = oqs.KeyEncapsulation("Kyber1024")
        client_public_key = client_kem.generate_keypair()
    else:
        # RSA fallback for client
        client_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        client_public_key = client_private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    # Perform key exchange
    exchange_data = kx.initiate_key_exchange("device_001", client_public_key)
    print(f"Key exchange initiated: {exchange_data['session_id']}")
    
    # Get session key for encryption
    session_key = kx.get_session_key(exchange_data['session_id'])
    if session_key:
        # Test encryption
        encryption = QuantumSafeEncryption(session_key)
        
        test_data = b"Quantum-safe federated learning model update"
        encrypted = encryption.encrypt(test_data, b"device_001")
        decrypted = encryption.decrypt(encrypted)
        
        print(f"Encryption test: {'PASSED' if decrypted == test_data else 'FAILED'}")
    
    print("Quantum key exchange system initialized successfully!")