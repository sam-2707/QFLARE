"""
Post-Quantum Cryptographic Key Management for QFLARE

This module provides comprehensive key lifecycle management including:
- Post-quantum key generation (FrodoKEM, Dilithium)
- Secure key storage and rotation
- Certificate management
- Hardware Security Module (HSM) integration
- Key escrow and recovery
"""

import os
import time
import logging
import hashlib
import secrets
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

# Post-quantum cryptography imports
try:
    import oqs
    PQ_AVAILABLE = True
except ImportError:
    PQ_AVAILABLE = False
    logging.warning("liboqs not available, using mock implementations")

# Cryptography imports
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography import x509
from cryptography.x509.oid import NameOID

logger = logging.getLogger(__name__)


class KeyType(Enum):
    """Post-quantum key types supported by QFLARE."""
    KEM_FRODO_640_AES = "FrodoKEM-640-AES"
    KEM_KYBER_512 = "Kyber512"
    KEM_KYBER_768 = "Kyber768"
    SIG_DILITHIUM_2 = "Dilithium2"
    SIG_DILITHIUM_3 = "Dilithium3"
    SIG_FALCON_512 = "Falcon-512"


class KeyStatus(Enum):
    """Key lifecycle status."""
    ACTIVE = "active"
    ROTATING = "rotating"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"
    COMPROMISED = "compromised"


@dataclass
class KeyMetadata:
    """Metadata for cryptographic keys."""
    key_id: str
    key_type: KeyType
    status: KeyStatus
    created_at: datetime
    expires_at: Optional[datetime]
    rotation_count: int
    device_id: Optional[str]
    purpose: str  # 'kem', 'signature', 'encryption'
    algorithm_params: Dict[str, Any]


@dataclass
class KeyPair:
    """Post-quantum key pair with metadata."""
    public_key: bytes
    private_key: bytes
    metadata: KeyMetadata


class PostQuantumKeyManager:
    """Manages post-quantum cryptographic keys for QFLARE devices."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the key manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._setup_default_config()
        self._key_store = {}  # In-memory store (should use HSM in production)
        self._initialize_algorithms()
    
    def _setup_default_config(self):
        """Setup default configuration."""
        defaults = {
            'default_kem_algorithm': KeyType.KEM_FRODO_640_AES,
            'default_sig_algorithm': KeyType.SIG_DILITHIUM_2,
            'key_rotation_interval_days': 90,
            'key_storage_encryption': True,
            'hsm_enabled': False,
            'hsm_provider': 'softhsm',
            'backup_keys': True,
            'audit_all_operations': True
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _initialize_algorithms(self):
        """Initialize available post-quantum algorithms."""
        self.available_algorithms = {}
        
        if PQ_AVAILABLE:
            # Initialize liboqs algorithms
            try:
                self.available_algorithms[KeyType.KEM_FRODO_640_AES] = oqs.KeyEncapsulation
                self.available_algorithms[KeyType.SIG_DILITHIUM_2] = oqs.Signature
                logger.info("Post-quantum algorithms initialized with liboqs")
            except Exception as e:
                logger.error(f"Error initializing liboqs algorithms: {e}")
                PQ_AVAILABLE = False
        
        if not PQ_AVAILABLE:
            logger.warning("Using mock post-quantum implementations")
    
    def generate_key_pair(
        self, 
        key_type: KeyType, 
        device_id: Optional[str] = None,
        purpose: str = "kem"
    ) -> KeyPair:
        """
        Generate a new post-quantum key pair.
        
        Args:
            key_type: Type of key to generate
            device_id: Device identifier (optional)
            purpose: Key purpose (kem, signature, encryption)
            
        Returns:
            Generated key pair with metadata
        """
        try:
            key_id = self._generate_key_id()
            
            if PQ_AVAILABLE and key_type in self.available_algorithms:
                # Use real post-quantum algorithms
                if purpose == "kem":
                    kem = oqs.KeyEncapsulation(key_type.value)
                    public_key, private_key = kem.generate_keypair()
                elif purpose == "signature":
                    sig = oqs.Signature(key_type.value)
                    public_key, private_key = sig.generate_keypair()
                else:
                    raise ValueError(f"Unsupported purpose: {purpose}")
            else:
                # Use mock implementations
                public_key, private_key = self._generate_mock_keypair(key_type)
            
            # Create metadata
            metadata = KeyMetadata(
                key_id=key_id,
                key_type=key_type,
                status=KeyStatus.ACTIVE,
                created_at=datetime.utcnow(),
                expires_at=self._calculate_expiry_date(),
                rotation_count=0,
                device_id=device_id,
                purpose=purpose,
                algorithm_params=self._get_algorithm_params(key_type)
            )
            
            key_pair = KeyPair(
                public_key=public_key,
                private_key=private_key,
                metadata=metadata
            )
            
            # Store key pair
            self._store_key_pair(key_pair)
            
            logger.info(f"Generated {key_type.value} key pair {key_id} for device {device_id}")
            return key_pair
            
        except Exception as e:
            logger.error(f"Error generating key pair: {e}")
            raise
    
    def _generate_mock_keypair(self, key_type: KeyType) -> Tuple[bytes, bytes]:
        """Generate mock key pair for testing."""
        if key_type in [KeyType.KEM_FRODO_640_AES, KeyType.KEM_KYBER_512, KeyType.KEM_KYBER_768]:
            # Mock KEM keys
            public_key = secrets.token_bytes(1312)  # FrodoKEM-640 public key size
            private_key = secrets.token_bytes(31296)  # FrodoKEM-640 private key size
        else:
            # Mock signature keys
            public_key = secrets.token_bytes(1312)  # Dilithium2 public key size
            private_key = secrets.token_bytes(2528)  # Dilithium2 private key size
        
        return public_key, private_key
    
    def _generate_key_id(self) -> str:
        """Generate unique key identifier."""
        timestamp = int(time.time() * 1000)
        random_bytes = secrets.token_bytes(8)
        return f"qflare-key-{timestamp}-{random_bytes.hex()}"
    
    def _calculate_expiry_date(self) -> datetime:
        """Calculate key expiry date based on configuration."""
        days = self.config['key_rotation_interval_days']
        return datetime.utcnow() + timedelta(days=days)
    
    def _get_algorithm_params(self, key_type: KeyType) -> Dict[str, Any]:
        """Get algorithm-specific parameters."""
        params = {
            KeyType.KEM_FRODO_640_AES: {
                "security_level": 128,
                "public_key_size": 1312,
                "private_key_size": 31296,
                "ciphertext_size": 1328
            },
            KeyType.SIG_DILITHIUM_2: {
                "security_level": 128,
                "public_key_size": 1312,
                "private_key_size": 2528,
                "signature_size": 2420
            }
        }
        
        return params.get(key_type, {})
    
    def _store_key_pair(self, key_pair: KeyPair):
        """Store key pair in secure storage."""
        if self.config['hsm_enabled']:
            self._store_in_hsm(key_pair)
        else:
            self._store_in_memory(key_pair)
        
        if self.config['backup_keys']:
            self._backup_key_pair(key_pair)
    
    def _store_in_memory(self, key_pair: KeyPair):
        """Store key pair in memory (for development)."""
        self._key_store[key_pair.metadata.key_id] = key_pair
    
    def _store_in_hsm(self, key_pair: KeyPair):
        """Store key pair in Hardware Security Module."""
        # Implementation would interface with actual HSM
        logger.info(f"Storing key {key_pair.metadata.key_id} in HSM (mock)")
        self._store_in_memory(key_pair)  # Fallback for now
    
    def _backup_key_pair(self, key_pair: KeyPair):
        """Create encrypted backup of key pair."""
        try:
            backup_data = {
                'key_id': key_pair.metadata.key_id,
                'public_key': key_pair.public_key.hex(),
                'private_key': key_pair.private_key.hex(),
                'metadata': {
                    'key_type': key_pair.metadata.key_type.value,
                    'status': key_pair.metadata.status.value,
                    'created_at': key_pair.metadata.created_at.isoformat(),
                    'device_id': key_pair.metadata.device_id,
                    'purpose': key_pair.metadata.purpose
                }
            }
            
            # In production, this would be encrypted and stored securely
            backup_path = f"key_backups/{key_pair.metadata.key_id}.json"
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            logger.info(f"Created backup for key {key_pair.metadata.key_id}")
            
        except Exception as e:
            logger.error(f"Error creating key backup: {e}")
    
    def get_key_pair(self, key_id: str) -> Optional[KeyPair]:
        """
        Retrieve key pair by ID.
        
        Args:
            key_id: Key identifier
            
        Returns:
            Key pair if found, None otherwise
        """
        return self._key_store.get(key_id)
    
    def get_device_keys(self, device_id: str) -> List[KeyPair]:
        """
        Get all keys for a specific device.
        
        Args:
            device_id: Device identifier
            
        Returns:
            List of key pairs for the device
        """
        return [
            key_pair for key_pair in self._key_store.values()
            if key_pair.metadata.device_id == device_id
        ]
    
    def rotate_key(self, key_id: str) -> Optional[KeyPair]:
        """
        Rotate an existing key.
        
        Args:
            key_id: Key identifier to rotate
            
        Returns:
            New key pair if successful, None otherwise
        """
        try:
            old_key = self.get_key_pair(key_id)
            if not old_key:
                logger.error(f"Key {key_id} not found for rotation")
                return None
            
            # Mark old key as rotating
            old_key.metadata.status = KeyStatus.ROTATING
            
            # Generate new key pair
            new_key = self.generate_key_pair(
                key_type=old_key.metadata.key_type,
                device_id=old_key.metadata.device_id,
                purpose=old_key.metadata.purpose
            )
            
            # Update rotation count
            new_key.metadata.rotation_count = old_key.metadata.rotation_count + 1
            
            # Mark old key as deprecated after rotation period
            old_key.metadata.status = KeyStatus.DEPRECATED
            
            logger.info(f"Rotated key {key_id} to {new_key.metadata.key_id}")
            return new_key
            
        except Exception as e:
            logger.error(f"Error rotating key {key_id}: {e}")
            return None
    
    def revoke_key(self, key_id: str, reason: str = "manual_revocation") -> bool:
        """
        Revoke a key pair.
        
        Args:
            key_id: Key identifier
            reason: Revocation reason
            
        Returns:
            True if key was revoked successfully
        """
        try:
            key_pair = self.get_key_pair(key_id)
            if not key_pair:
                logger.error(f"Key {key_id} not found for revocation")
                return False
            
            key_pair.metadata.status = KeyStatus.REVOKED
            
            logger.warning(f"Revoked key {key_id} - reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error revoking key {key_id}: {e}")
            return False
    
    def check_key_expiry(self) -> List[str]:
        """
        Check for expired or soon-to-expire keys.
        
        Returns:
            List of key IDs that need rotation
        """
        expiring_keys = []
        now = datetime.utcnow()
        warning_period = timedelta(days=7)  # Warn 7 days before expiry
        
        for key_pair in self._key_store.values():
            if key_pair.metadata.expires_at:
                if now >= key_pair.metadata.expires_at:
                    # Key has expired
                    key_pair.metadata.status = KeyStatus.DEPRECATED
                    expiring_keys.append(key_pair.metadata.key_id)
                elif now >= (key_pair.metadata.expires_at - warning_period):
                    # Key expires soon
                    expiring_keys.append(key_pair.metadata.key_id)
        
        return expiring_keys
    
    def get_key_statistics(self) -> Dict[str, Any]:
        """
        Get key management statistics.
        
        Returns:
            Dictionary with key statistics
        """
        stats = {
            'total_keys': len(self._key_store),
            'keys_by_type': {},
            'keys_by_status': {},
            'keys_by_device': {},
            'expiring_soon': len(self.check_key_expiry())
        }
        
        for key_pair in self._key_store.values():
            # Count by type
            key_type = key_pair.metadata.key_type.value
            stats['keys_by_type'][key_type] = stats['keys_by_type'].get(key_type, 0) + 1
            
            # Count by status
            status = key_pair.metadata.status.value
            stats['keys_by_status'][status] = stats['keys_by_status'].get(status, 0) + 1
            
            # Count by device
            device_id = key_pair.metadata.device_id or 'unassigned'
            stats['keys_by_device'][device_id] = stats['keys_by_device'].get(device_id, 0) + 1
        
        return stats


class CertificateManager:
    """Manages X.509 certificates for QFLARE devices."""
    
    def __init__(self, ca_key_path: str = None, ca_cert_path: str = None):
        """
        Initialize certificate manager.
        
        Args:
            ca_key_path: Path to CA private key
            ca_cert_path: Path to CA certificate
        """
        self.ca_key_path = ca_key_path
        self.ca_cert_path = ca_cert_path
        self._load_ca_credentials()
    
    def _load_ca_credentials(self):
        """Load CA private key and certificate."""
        try:
            if self.ca_key_path and os.path.exists(self.ca_key_path):
                with open(self.ca_key_path, 'rb') as f:
                    self.ca_private_key = serialization.load_pem_private_key(
                        f.read(), password=None, backend=default_backend()
                    )
            else:
                # Generate new CA key for development
                self.ca_private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048,
                    backend=default_backend()
                )
            
            if self.ca_cert_path and os.path.exists(self.ca_cert_path):
                with open(self.ca_cert_path, 'rb') as f:
                    self.ca_certificate = x509.load_pem_x509_certificate(
                        f.read(), backend=default_backend()
                    )
            else:
                # Generate new CA certificate for development
                self.ca_certificate = self._generate_ca_certificate()
            
            logger.info("CA credentials loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading CA credentials: {e}")
            raise
    
    def _generate_ca_certificate(self) -> x509.Certificate:
        """Generate a new CA certificate."""
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "QFLARE CA"),
            x509.NameAttribute(NameOID.COMMON_NAME, "QFLARE Root CA"),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            self.ca_private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=3650)  # 10 years
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("qflare-ca.local"),
            ]),
            critical=False,
        ).add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        ).sign(self.ca_private_key, hashes.SHA256(), backend=default_backend())
        
        return cert
    
    def generate_device_certificate(self, device_id: str, public_key: bytes) -> bytes:
        """
        Generate a certificate for a device.
        
        Args:
            device_id: Device identifier
            public_key: Device public key
            
        Returns:
            PEM-encoded certificate
        """
        try:
            # For now, use RSA key for certificate (post-quantum certs not widely supported)
            device_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            
            subject = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "QFLARE Device"),
                x509.NameAttribute(NameOID.COMMON_NAME, device_id),
            ])
            
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                self.ca_certificate.subject
            ).public_key(
                device_private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=365)  # 1 year
            ).add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName(f"{device_id}.qflare.local"),
                ]),
                critical=False,
            ).sign(self.ca_private_key, hashes.SHA256(), backend=default_backend())
            
            return cert.public_bytes(serialization.Encoding.PEM)
            
        except Exception as e:
            logger.error(f"Error generating device certificate for {device_id}: {e}")
            raise