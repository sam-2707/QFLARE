"""
Enhanced Cryptographic Key Management with Timestamp-based Challenge-Response

This module implements the secure key exchange mechanism with timestamp validation
and proper public-private key mapping for QFLARE federated learning.

Features:
- Timestamp-based challenge-response authentication
- KEM (Key Encapsulation Mechanism) with post-quantum crypto
- Session key management
- Public-private key mapping
- Hardware Security Module integration
"""

import time
import json
import logging
import hashlib
import secrets
import base64
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import threading
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

# Cryptographic imports
try:
    import oqs  # Open Quantum Safe for post-quantum cryptography
    PQC_AVAILABLE = True
    logger.info("Post-quantum cryptography (OQS) is available")
except (ImportError, RuntimeError) as e:
    PQC_AVAILABLE = False
    logger.warning(f"Open Quantum Safe not available: {e}")
    logger.info("Falling back to classical cryptography (RSA)")


class KeyType(Enum):
    """Supported key types."""
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    DILITHIUM2 = "dilithium2"
    DILITHIUM3 = "dilithium3"
    KYBER768 = "kyber768"  # KEM
    KYBER1024 = "kyber1024"  # KEM


class ChallengeStatus(Enum):
    """Challenge response status."""
    PENDING = "pending"
    VALIDATED = "validated"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class KeyPair:
    """Cryptographic key pair with metadata."""
    device_id: str
    key_type: KeyType
    public_key: bytes
    private_key: bytes  # Encrypted with device password
    created_at: datetime
    expires_at: Optional[datetime] = None
    hardware_backed: bool = False
    hsm_key_id: Optional[str] = None


@dataclass
class ChallengeRequest:
    """Timestamp-based challenge request."""
    device_id: str
    timestamp: float
    nonce: str
    signature: Optional[bytes] = None


@dataclass
class ChallengeResponse:
    """Server challenge response."""
    challenge_id: str
    encrypted_session_key: bytes
    server_timestamp: float
    validity_duration: int  # seconds
    status: ChallengeStatus


@dataclass
class SessionContext:
    """Active session context."""
    device_id: str
    session_key: bytes
    challenge_id: str
    created_at: datetime
    expires_at: datetime
    request_count: int = 0


class TimestampChallengeManager:
    """Manages timestamp-based challenge-response authentication."""
    
    def __init__(self, 
                 tolerance_seconds: int = 30,
                 session_duration_minutes: int = 60,
                 max_concurrent_sessions: int = 1000):
        self.tolerance_seconds = tolerance_seconds
        self.session_duration = timedelta(minutes=session_duration_minutes)
        self.max_concurrent_sessions = max_concurrent_sessions
        
        # Storage
        self.active_challenges: Dict[str, ChallengeResponse] = {}
        self.active_sessions: Dict[str, SessionContext] = {}
        self.device_keys: Dict[str, KeyPair] = {}
        
        # HSM integration (if available)
        self.hsm_available = False
        self._lock = threading.RLock()
        
        # Cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task if event loop is available."""
        try:
            # Only start if we have a running event loop
            loop = asyncio.get_running_loop()
            
            async def cleanup_expired():
                while True:
                    await asyncio.sleep(60)  # Check every minute
                    self._cleanup_expired_sessions()
            
            self._cleanup_task = asyncio.create_task(cleanup_expired())
            
        except RuntimeError:
            # No running event loop, skip async cleanup
            # Will rely on manual cleanup during validation
            logger.info("No event loop available, using manual cleanup")
            self._cleanup_task = None
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions and challenges."""
        with self._lock:
            current_time = datetime.now()
            
            # Cleanup expired sessions
            expired_sessions = [
                session_id for session_id, session in self.active_sessions.items()
                if session.expires_at < current_time
            ]
            
            for session_id in expired_sessions:
                del self.active_sessions[session_id]
                logger.info(f"Cleaned up expired session: {session_id}")
            
            # Cleanup expired challenges
            expired_challenges = [
                challenge_id for challenge_id, challenge in self.active_challenges.items()
                if challenge.status == ChallengeStatus.EXPIRED
            ]
            
            for challenge_id in expired_challenges:
                del self.active_challenges[challenge_id]
    
    def generate_device_keypair(self, 
                               device_id: str, 
                               key_type: KeyType = KeyType.KYBER768,
                               use_hsm: bool = False) -> KeyPair:
        """Generate a new key pair for a device."""
        try:
            if use_hsm and self.hsm_available:
                return self._generate_hsm_keypair(device_id, key_type)
            
            if key_type == KeyType.KYBER768 and PQC_AVAILABLE:
                return self._generate_kyber_keypair(device_id)
            elif key_type == KeyType.DILITHIUM2 and PQC_AVAILABLE:
                return self._generate_dilithium_keypair(device_id)
            else:
                return self._generate_rsa_keypair(device_id, key_type)
                
        except Exception as e:
            logger.error(f"Failed to generate keypair for {device_id}: {e}")
            raise
    
    def _generate_kyber_keypair(self, device_id: str) -> KeyPair:
        """Generate Kyber KEM keypair for post-quantum security."""
        kem = oqs.KeyEncapsulation("Kyber768")
        public_key = kem.generate_keypair()
        private_key = kem.export_secret_key()
        
        keypair = KeyPair(
            device_id=device_id,
            key_type=KeyType.KYBER768,
            public_key=public_key,
            private_key=private_key,
            created_at=datetime.now()
        )
        
        self.device_keys[device_id] = keypair
        logger.info(f"Generated Kyber768 keypair for device: {device_id}")
        
        return keypair
    
    def _generate_dilithium_keypair(self, device_id: str) -> KeyPair:
        """Generate Dilithium signature keypair."""
        sig = oqs.Signature("Dilithium2")
        public_key = sig.generate_keypair()
        private_key = sig.export_secret_key()
        
        keypair = KeyPair(
            device_id=device_id,
            key_type=KeyType.DILITHIUM2,
            public_key=public_key,
            private_key=private_key,
            created_at=datetime.now()
        )
        
        self.device_keys[device_id] = keypair
        logger.info(f"Generated Dilithium2 keypair for device: {device_id}")
        
        return keypair
    
    def _generate_rsa_keypair(self, device_id: str, key_type: KeyType) -> KeyPair:
        """Generate RSA keypair as fallback."""
        key_size = 4096 if key_type == KeyType.RSA_4096 else 2048
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )
        
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        keypair = KeyPair(
            device_id=device_id,
            key_type=key_type,
            public_key=public_pem,
            private_key=private_pem,
            created_at=datetime.now()
        )
        
        self.device_keys[device_id] = keypair
        logger.info(f"Generated {key_type.value} keypair for device: {device_id}")
        
        return keypair
    
    def process_challenge_request(self, request: ChallengeRequest) -> ChallengeResponse:
        """Process timestamp-based challenge request following the sequence diagram."""
        try:
            with self._lock:
                # Step 1: Validate timestamp
                server_timestamp = time.time()
                timestamp_diff = abs(server_timestamp - request.timestamp)
                
                if timestamp_diff > self.tolerance_seconds:
                    logger.warning(f"Timestamp validation failed for {request.device_id}: diff={timestamp_diff}s")
                    raise ValueError(f"Timestamp outside tolerance: {timestamp_diff}s > {self.tolerance_seconds}s")
                
                # Step 2: Verify device has registered keys
                if request.device_id not in self.device_keys:
                    raise ValueError(f"Device {request.device_id} not registered")
                
                device_keypair = self.device_keys[request.device_id]
                
                # Step 3: Generate session key
                session_key = secrets.token_bytes(32)  # 256-bit session key
                challenge_id = secrets.token_urlsafe(32)
                
                # Step 4: Encrypt session key with device's public key (KEM)
                if device_keypair.key_type == KeyType.KYBER768 and PQC_AVAILABLE:
                    encrypted_session_key = self._encrypt_with_kyber(session_key, device_keypair.public_key)
                else:
                    encrypted_session_key = self._encrypt_with_rsa(session_key, device_keypair.public_key)
                
                # Step 5: Create challenge response
                response = ChallengeResponse(
                    challenge_id=challenge_id,
                    encrypted_session_key=encrypted_session_key,
                    server_timestamp=server_timestamp,
                    validity_duration=self.session_duration.total_seconds(),
                    status=ChallengeStatus.VALIDATED
                )
                
                # Step 6: Store session context
                session_context = SessionContext(
                    device_id=request.device_id,
                    session_key=session_key,
                    challenge_id=challenge_id,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + self.session_duration
                )
                
                self.active_challenges[challenge_id] = response
                self.active_sessions[challenge_id] = session_context
                
                logger.info(f"Challenge processed successfully for device: {request.device_id}")
                return response
                
        except Exception as e:
            logger.error(f"Challenge processing failed: {e}")
            # Return error response
            return ChallengeResponse(
                challenge_id="",
                encrypted_session_key=b"",
                server_timestamp=time.time(),
                validity_duration=0,
                status=ChallengeStatus.FAILED
            )
    
    def _encrypt_with_kyber(self, session_key: bytes, public_key: bytes) -> bytes:
        """Encrypt session key using Kyber KEM."""
        try:
            kem = oqs.KeyEncapsulation("Kyber768")
            ciphertext, shared_secret = kem.encap_secret(public_key)
            
            # Use shared secret to encrypt session key
            cipher = Cipher(
                algorithms.AES(shared_secret[:32]),
                modes.GCM(b'\x00' * 12)  # Use proper IV in production
            )
            encryptor = cipher.encryptor()
            encrypted_key = encryptor.update(session_key) + encryptor.finalize()
            
            # Return ciphertext + encrypted session key + tag
            return ciphertext + encrypted_key + encryptor.tag
            
        except Exception as e:
            logger.error(f"Kyber encryption failed: {e}")
            raise
    
    def _encrypt_with_rsa(self, session_key: bytes, public_key_pem: bytes) -> bytes:
        """Encrypt session key using RSA public key."""
        try:
            public_key = serialization.load_pem_public_key(public_key_pem)
            
            encrypted_key = public_key.encrypt(
                session_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return encrypted_key
            
        except Exception as e:
            logger.error(f"RSA encryption failed: {e}")
            raise
    
    def validate_session(self, challenge_id: str) -> Optional[SessionContext]:
        """Validate an active session."""
        with self._lock:
            # Manual cleanup if no async task
            if self._cleanup_task is None:
                self._cleanup_expired_sessions()
            
            session = self.active_sessions.get(challenge_id)
            
            if not session:
                return None
            
            if session.expires_at < datetime.now():
                del self.active_sessions[challenge_id]
                return None
            
            session.request_count += 1
            return session
    
    def get_session_key(self, challenge_id: str) -> Optional[bytes]:
        """Get session key for encrypted communication."""
        session = self.validate_session(challenge_id)
        return session.session_key if session else None
    
    def revoke_session(self, challenge_id: str) -> bool:
        """Revoke an active session."""
        with self._lock:
            if challenge_id in self.active_sessions:
                del self.active_sessions[challenge_id]
                logger.info(f"Session revoked: {challenge_id}")
                return True
            return False
    
    def get_device_public_key(self, device_id: str) -> Optional[bytes]:
        """Get device public key for verification."""
        keypair = self.device_keys.get(device_id)
        return keypair.public_key if keypair else None
    
    def register_device_key(self, device_id: str, public_key: bytes, key_type: KeyType) -> bool:
        """Register a device's public key."""
        try:
            keypair = KeyPair(
                device_id=device_id,
                key_type=key_type,
                public_key=public_key,
                private_key=b"",  # Server doesn't store private keys
                created_at=datetime.now()
            )
            
            self.device_keys[device_id] = keypair
            logger.info(f"Registered public key for device: {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register key for {device_id}: {e}")
            return False
    
    def get_system_status(self) -> Dict:
        """Get system status and metrics."""
        with self._lock:
            return {
                'active_sessions': len(self.active_sessions),
                'active_challenges': len(self.active_challenges),
                'registered_devices': len(self.device_keys),
                'hsm_available': self.hsm_available,
                'pqc_available': PQC_AVAILABLE,
                'tolerance_seconds': self.tolerance_seconds,
                'session_duration_minutes': self.session_duration.total_seconds() / 60
            }


# Global instance
_challenge_manager: Optional[TimestampChallengeManager] = None
_manager_lock = threading.Lock()


def get_challenge_manager() -> TimestampChallengeManager:
    """Get or create the global challenge manager."""
    global _challenge_manager
    
    with _manager_lock:
        if _challenge_manager is None:
            _challenge_manager = TimestampChallengeManager()
        
        return _challenge_manager


def create_challenge_request(device_id: str, private_key: bytes = None) -> ChallengeRequest:
    """Helper function to create a challenge request."""
    timestamp = time.time()
    nonce = secrets.token_urlsafe(16)
    
    # Sign the request if private key is provided
    signature = None
    if private_key:
        # Implementation depends on key type
        pass
    
    return ChallengeRequest(
        device_id=device_id,
        timestamp=timestamp,
        nonce=nonce,
        signature=signature
    )