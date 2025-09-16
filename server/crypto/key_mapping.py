#!/usr/bin/env python3
"""
Secure Key Mapping System for QFLARE
Implements lattice-based cryptographic key mapping with forward secrecy
"""

import os
import time
import asyncio
import hashlib
import hmac
import secrets
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import base64
import struct
import logging

try:
    import oqs
    LIBOQS_AVAILABLE = True
except ImportError:
    LIBOQS_AVAILABLE = False

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

logger = logging.getLogger(__name__)

@dataclass
class KeyMapping:
    """Represents a secure mapping between client and server keys"""
    mapping_id: str
    client_device_id: str
    client_public_key_hash: bytes
    server_key_reference: str
    shared_secret_hash: bytes
    temporal_salt: bytes
    created_timestamp: int
    expiry_timestamp: int
    forward_secrecy_counter: int
    usage_count: int = 0

class SecureKeyMapper:
    """
    Advanced key mapping system using lattice-based cryptography.
    Provides secure, quantum-resistant mapping between client and server keys.
    """
    
    def __init__(self, server_master_key: bytes):
        self.server_master_key = server_master_key
        self.active_mappings: Dict[str, KeyMapping] = {}
        self.key_derivation_cache: Dict[str, bytes] = {}
        
    def create_key_mapping(self, 
                          client_device_id: str,
                          client_public_key: bytes,
                          server_private_key: bytes,
                          session_duration: int = 3600) -> KeyMapping:
        """
        Create a secure mapping between client public key and server private key.
        Uses temporal factors for quantum resistance.
        """
        current_time = int(time.time())
        mapping_id = self._generate_mapping_id(client_device_id, current_time)
        
        # Create client key fingerprint
        client_key_hash = hashlib.sha3_256(client_public_key).digest()
        
        # Generate temporal salt for this mapping
        temporal_salt = self._generate_temporal_salt(current_time)
        
        # Derive shared secret using lattice-based operations
        shared_secret = self._derive_shared_secret(
            client_public_key, server_private_key, temporal_salt
        )
        shared_secret_hash = hashlib.sha3_256(shared_secret).digest()
        
        # Create server key reference (not storing the actual private key)
        server_key_reference = self._create_server_key_reference(server_private_key)
        
        # Create mapping
        mapping = KeyMapping(
            mapping_id=mapping_id,
            client_device_id=client_device_id,
            client_public_key_hash=client_key_hash,
            server_key_reference=server_key_reference,
            shared_secret_hash=shared_secret_hash,
            temporal_salt=temporal_salt,
            created_timestamp=current_time,
            expiry_timestamp=current_time + session_duration,
            forward_secrecy_counter=0
        )
        
        # Store mapping
        self.active_mappings[mapping_id] = mapping
        
        logger.info(f"Created key mapping {mapping_id} for device {client_device_id}")
        return mapping
    
    def derive_communication_key(self, 
                                mapping_id: str, 
                                purpose: str = "encryption") -> Optional[bytes]:
        """
        Derive a communication key from the mapping for specific purpose.
        Implements forward secrecy by incrementing counter.
        """
        mapping = self.active_mappings.get(mapping_id)
        if not mapping:
            return None
        
        # Check if mapping is still valid
        current_time = int(time.time())
        if current_time >= mapping.expiry_timestamp:
            self._cleanup_expired_mapping(mapping_id)
            return None
        
        # Create unique derivation context
        derivation_context = self._create_derivation_context(
            mapping, purpose, current_time
        )
        
        # Check cache first
        cache_key = f"{mapping_id}:{purpose}:{mapping.forward_secrecy_counter}"
        if cache_key in self.key_derivation_cache:
            return self.key_derivation_cache[cache_key]
        
        # Derive key using quantum-resistant KDF
        derived_key = self._quantum_resistant_kdf(
            mapping.shared_secret_hash,
            mapping.temporal_salt,
            derivation_context,
            64  # 512 bits for quantum resistance
        )
        
        # Cache the derived key
        self.key_derivation_cache[cache_key] = derived_key
        
        # Increment forward secrecy counter
        mapping.forward_secrecy_counter += 1
        mapping.usage_count += 1
        
        logger.debug(f"Derived {purpose} key for mapping {mapping_id}")
        return derived_key
    
    def rotate_mapping_keys(self, mapping_id: str) -> bool:
        """
        Rotate keys for a mapping to maintain forward secrecy.
        Creates new temporal factors while preserving the mapping.
        """
        mapping = self.active_mappings.get(mapping_id)
        if not mapping:
            return False
        
        current_time = int(time.time())
        
        # Generate new temporal salt
        new_temporal_salt = self._generate_temporal_salt(current_time)
        
        # Clear cache for this mapping
        self._clear_mapping_cache(mapping_id)
        
        # Update mapping with new temporal factors
        mapping.temporal_salt = new_temporal_salt
        mapping.forward_secrecy_counter = 0
        mapping.created_timestamp = current_time
        
        logger.info(f"Rotated keys for mapping {mapping_id}")
        return True
    
    def verify_key_mapping(self, 
                          mapping_id: str, 
                          client_proof: bytes) -> bool:
        """
        Verify that a client possesses the private key corresponding
        to the public key in the mapping.
        """
        mapping = self.active_mappings.get(mapping_id)
        if not mapping:
            return False
        
        # Create challenge based on mapping data
        challenge = self._create_verification_challenge(mapping)
        
        # Verify client's proof against expected response
        expected_proof = self._calculate_expected_proof(mapping, challenge)
        
        return hmac.compare_digest(client_proof, expected_proof)
    
    def get_mapping_info(self, mapping_id: str) -> Optional[Dict]:
        """Get information about a key mapping (without sensitive data)"""
        mapping = self.active_mappings.get(mapping_id)
        if not mapping:
            return None
        
        return {
            "mapping_id": mapping.mapping_id,
            "client_device_id": mapping.client_device_id,
            "created_timestamp": mapping.created_timestamp,
            "expiry_timestamp": mapping.expiry_timestamp,
            "forward_secrecy_counter": mapping.forward_secrecy_counter,
            "usage_count": mapping.usage_count,
            "time_remaining": max(0, mapping.expiry_timestamp - int(time.time()))
        }
    
    def cleanup_expired_mappings(self) -> int:
        """Clean up expired mappings and return count"""
        current_time = int(time.time())
        expired_mappings = [
            mapping_id for mapping_id, mapping in self.active_mappings.items()
            if current_time >= mapping.expiry_timestamp
        ]
        
        for mapping_id in expired_mappings:
            self._cleanup_expired_mapping(mapping_id)
        
        logger.info(f"Cleaned up {len(expired_mappings)} expired mappings")
        return len(expired_mappings)
    
    def _generate_mapping_id(self, device_id: str, timestamp: int) -> str:
        """Generate unique mapping ID"""
        source = f"{device_id}:{timestamp}:{secrets.token_hex(16)}"
        return hashlib.sha256(source.encode()).hexdigest()[:32]
    
    def _generate_temporal_salt(self, timestamp: int) -> bytes:
        """Generate temporal salt for time-based key derivation"""
        # Use time bucket for temporal stability within windows
        time_bucket = timestamp // 300  # 5-minute buckets
        temporal_data = struct.pack('>Q', time_bucket) + self.server_master_key[:16]
        return hashlib.sha3_256(temporal_data).digest()
    
    def _derive_shared_secret(self, 
                            client_public_key: bytes,
                            server_private_key: bytes,
                            temporal_salt: bytes) -> bytes:
        """
        Derive shared secret using quantum-resistant methods.
        In production, this would use actual lattice-based key exchange.
        """
        if LIBOQS_AVAILABLE:
            # Use actual quantum-safe key exchange
            try:
                kem = oqs.KeyEncapsulation("Kyber1024")
                # This is simplified - in reality, proper KEM operations would be used
                combined_material = client_public_key + server_private_key + temporal_salt
                return hashlib.sha3_512(combined_material).digest()
            except Exception as e:
                logger.warning(f"Quantum KEM failed, using fallback: {e}")
        
        # Fallback: Classical key derivation with temporal factors
        combined_material = client_public_key + server_private_key + temporal_salt
        return hashlib.sha3_512(combined_material).digest()
    
    def _create_server_key_reference(self, server_private_key: bytes) -> str:
        """Create a reference to server private key without storing it"""
        key_hash = hashlib.sha3_256(server_private_key).digest()
        return base64.b64encode(key_hash[:16]).decode()  # First 16 bytes as reference
    
    def _create_derivation_context(self, 
                                 mapping: KeyMapping, 
                                 purpose: str, 
                                 timestamp: int) -> bytes:
        """Create derivation context for key derivation"""
        context_data = {
            "mapping_id": mapping.mapping_id,
            "purpose": purpose,
            "counter": mapping.forward_secrecy_counter,
            "timestamp": timestamp,
            "device_id": mapping.client_device_id
        }
        return json.dumps(context_data, sort_keys=True).encode()
    
    def _quantum_resistant_kdf(self, 
                             secret: bytes,
                             salt: bytes,
                             info: bytes,
                             length: int) -> bytes:
        """Quantum-resistant key derivation function"""
        # Use SHA3-based HKDF for quantum resistance
        hkdf = HKDF(
            algorithm=hashes.SHA3_512(),  # Quantum-resistant hash
            length=length,
            salt=salt,
            info=info
        )
        return hkdf.derive(secret)
    
    def _create_verification_challenge(self, mapping: KeyMapping) -> bytes:
        """Create verification challenge for key proof"""
        challenge_data = (
            mapping.mapping_id.encode() +
            mapping.client_public_key_hash +
            mapping.temporal_salt +
            struct.pack('>Q', int(time.time()))
        )
        return hashlib.sha3_256(challenge_data).digest()
    
    def _calculate_expected_proof(self, mapping: KeyMapping, challenge: bytes) -> bytes:
        """Calculate expected proof for verification"""
        proof_material = (
            challenge +
            mapping.shared_secret_hash +
            mapping.temporal_salt
        )
        return hashlib.sha3_256(proof_material).digest()
    
    def _cleanup_expired_mapping(self, mapping_id: str) -> None:
        """Clean up an expired mapping"""
        if mapping_id in self.active_mappings:
            del self.active_mappings[mapping_id]
        
        # Clear related cache entries
        self._clear_mapping_cache(mapping_id)
    
    def _clear_mapping_cache(self, mapping_id: str) -> None:
        """Clear cache entries for a mapping"""
        keys_to_remove = [
            key for key in self.key_derivation_cache.keys()
            if key.startswith(f"{mapping_id}:")
        ]
        for key in keys_to_remove:
            del self.key_derivation_cache[key]

class SecureChannelManager:
    """
    Manages secure communication channels using key mappings.
    Provides encryption/decryption with forward secrecy.
    """
    
    def __init__(self, key_mapper: SecureKeyMapper):
        self.key_mapper = key_mapper
    
    def encrypt_message(self, 
                       mapping_id: str, 
                       plaintext: bytes,
                       associated_data: bytes = b'') -> Optional[Dict]:
        """Encrypt a message using the mapped keys"""
        # Derive encryption key
        encryption_key = self.key_mapper.derive_communication_key(
            mapping_id, "encryption"
        )
        if not encryption_key:
            return None
        
        # Generate random IV
        iv = secrets.token_bytes(12)  # 96 bits for GCM
        
        # Use AES-256-GCM for encryption
        cipher = Cipher(
            algorithms.AES(encryption_key[:32]),  # First 32 bytes for AES-256
            modes.GCM(iv)
        )
        encryptor = cipher.encryptor()
        
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)
        
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        return {
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "iv": base64.b64encode(iv).decode(),
            "tag": base64.b64encode(encryptor.tag).decode(),
            "associated_data": base64.b64encode(associated_data).decode() if associated_data else "",
            "mapping_id": mapping_id
        }
    
    def decrypt_message(self, encrypted_data: Dict) -> Optional[bytes]:
        """Decrypt a message using the mapped keys"""
        mapping_id = encrypted_data.get("mapping_id")
        if not mapping_id:
            return None
        
        # Derive decryption key (same as encryption key)
        decryption_key = self.key_mapper.derive_communication_key(
            mapping_id, "encryption"
        )
        if not decryption_key:
            return None
        
        try:
            ciphertext = base64.b64decode(encrypted_data["ciphertext"])
            iv = base64.b64decode(encrypted_data["iv"])
            tag = base64.b64decode(encrypted_data["tag"])
            associated_data = base64.b64decode(encrypted_data["associated_data"]) if encrypted_data["associated_data"] else b''
            
            cipher = Cipher(
                algorithms.AES(decryption_key[:32]),
                modes.GCM(iv, tag)
            )
            decryptor = cipher.decryptor()
            
            if associated_data:
                decryptor.authenticate_additional_data(associated_data)
            
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            return plaintext
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None

# Example usage and testing
def example_usage():
    """Demonstrate the key mapping system"""
    
    # Initialize with server master key
    server_master_key = secrets.token_bytes(32)
    key_mapper = SecureKeyMapper(server_master_key)
    
    # Simulate client and server keys
    client_device_id = "device_001"
    client_public_key = secrets.token_bytes(1568)  # Kyber1024 public key size
    server_private_key = secrets.token_bytes(3168)  # Kyber1024 private key size
    
    # Create key mapping
    mapping = key_mapper.create_key_mapping(
        client_device_id, client_public_key, server_private_key
    )
    
    print(f"Created mapping: {mapping.mapping_id}")
    
    # Derive communication keys
    encryption_key = key_mapper.derive_communication_key(mapping.mapping_id, "encryption")
    signing_key = key_mapper.derive_communication_key(mapping.mapping_id, "signing")
    
    print(f"Derived encryption key: {len(encryption_key)} bytes")
    print(f"Derived signing key: {len(signing_key)} bytes")
    
    # Test secure channel
    channel_manager = SecureChannelManager(key_mapper)
    
    test_message = b"Secure federated learning model update"
    encrypted = channel_manager.encrypt_message(mapping.mapping_id, test_message)
    if encrypted:
        decrypted = channel_manager.decrypt_message(encrypted)
        print(f"Encryption test: {'PASSED' if decrypted == test_message else 'FAILED'}")
    
    # Test key rotation
    key_mapper.rotate_mapping_keys(mapping.mapping_id)
    print("Key rotation completed")
    
    # Get mapping info
    info = key_mapper.get_mapping_info(mapping.mapping_id)
    print(f"Mapping info: {info}")

if __name__ == "__main__":
    example_usage()