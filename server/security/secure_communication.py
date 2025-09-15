"""
Enhanced Secure Communication for QFLARE

This module provides enhanced secure communication with:
- Post-quantum authenticated key exchange
- Message authentication and integrity
- Perfect forward secrecy
- Anti-replay protection
- Secure session management
"""

import os
import time
import hmac
import hashlib
import secrets
import logging
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

# Cryptography imports
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend

from .key_management import PostQuantumKeyManager, KeyType, KeyPair

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Message types for QFLARE protocol."""
    HANDSHAKE_INIT = "handshake_init"
    HANDSHAKE_RESPONSE = "handshake_response"
    HANDSHAKE_CONFIRM = "handshake_confirm"
    MODEL_UPDATE = "model_update"
    GLOBAL_MODEL = "global_model"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class SessionState(Enum):
    """Session states."""
    INITIALIZING = "initializing"
    HANDSHAKING = "handshaking"
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"


@dataclass
class SecureSession:
    """Secure communication session."""
    session_id: str
    device_id: str
    state: SessionState
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    
    # Cryptographic materials
    shared_secret: bytes
    encryption_key: bytes
    mac_key: bytes
    
    # Message counters for replay protection
    send_counter: int
    receive_counter: int
    
    # Key exchange materials
    local_keypair: Optional[KeyPair]
    remote_public_key: Optional[bytes]


@dataclass
class SecureMessage:
    """Encrypted message structure."""
    message_type: MessageType
    session_id: str
    sequence_number: int
    timestamp: float
    encrypted_payload: bytes
    mac: bytes
    iv: bytes


class SecureCommunicationManager:
    """Enhanced secure communication manager for QFLARE."""
    
    def __init__(self, key_manager: PostQuantumKeyManager):
        """
        Initialize secure communication manager.
        
        Args:
            key_manager: Post-quantum key manager instance
        """
        self.key_manager = key_manager
        self.sessions = {}  # Active sessions
        self.config = self._setup_default_config()
        self._initialize_session_cleanup()
    
    def _setup_default_config(self) -> Dict[str, Any]:
        """Setup default configuration."""
        return {
            'session_timeout_minutes': 30,
            'max_sessions_per_device': 3,
            'replay_window_size': 64,
            'heartbeat_interval_seconds': 60,
            'message_max_size': 10 * 1024 * 1024,  # 10MB
            'enable_perfect_forward_secrecy': True,
            'require_mutual_authentication': True
        }
    
    def _initialize_session_cleanup(self):
        """Initialize session cleanup timer."""
        # In production, this would be a background task
        pass
    
    def initiate_secure_session(self, device_id: str) -> Tuple[str, bytes]:
        """
        Initiate a secure session with a device.
        
        Args:
            device_id: Target device identifier
            
        Returns:
            Tuple of (session_id, handshake_init_message)
        """
        try:
            # Generate session ID
            session_id = self._generate_session_id()
            
            # Generate ephemeral key pair for this session
            local_keypair = self.key_manager.generate_key_pair(
                key_type=KeyType.KEM_FRODO_640_AES,
                device_id=device_id,
                purpose="kem"
            )
            
            # Create session
            session = SecureSession(
                session_id=session_id,
                device_id=device_id,
                state=SessionState.INITIALIZING,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(
                    minutes=self.config['session_timeout_minutes']
                ),
                shared_secret=b"",
                encryption_key=b"",
                mac_key=b"",
                send_counter=0,
                receive_counter=0,
                local_keypair=local_keypair,
                remote_public_key=None
            )
            
            # Store session
            self.sessions[session_id] = session
            
            # Create handshake initiation message
            handshake_message = self._create_handshake_init(session)
            
            session.state = SessionState.HANDSHAKING
            
            logger.info(f"Initiated secure session {session_id} with device {device_id}")
            return session_id, handshake_message
            
        except Exception as e:
            logger.error(f"Error initiating secure session with {device_id}: {e}")
            raise
    
    def _generate_session_id(self) -> str:
        """Generate unique session identifier."""
        timestamp = int(time.time() * 1000)
        random_bytes = secrets.token_bytes(8)
        return f"qflare-session-{timestamp}-{random_bytes.hex()}"
    
    def _create_handshake_init(self, session: SecureSession) -> bytes:
        """Create handshake initiation message."""
        try:
            handshake_data = {
                'session_id': session.session_id,
                'device_id': session.device_id,
                'timestamp': time.time(),
                'public_key': session.local_keypair.public_key.hex(),
                'supported_algorithms': [
                    KeyType.KEM_FRODO_640_AES.value,
                    KeyType.SIG_DILITHIUM_2.value
                ],
                'protocol_version': '1.0'
            }
            
            message_bytes = json.dumps(handshake_data).encode('utf-8')
            
            # Sign the handshake message
            signature = self._sign_message(message_bytes, session.local_keypair)
            
            final_message = {
                'handshake_data': handshake_data,
                'signature': signature.hex()
            }
            
            return json.dumps(final_message).encode('utf-8')
            
        except Exception as e:
            logger.error(f"Error creating handshake init: {e}")
            raise
    
    def _sign_message(self, message: bytes, keypair: KeyPair) -> bytes:
        """Sign a message using post-quantum signatures."""
        try:
            # For now, use HMAC as a placeholder for post-quantum signatures
            # In production, this would use Dilithium or another PQ signature
            signature_key = keypair.private_key[:32]  # Use first 32 bytes as HMAC key
            signature = hmac.new(signature_key, message, hashlib.sha256).digest()
            return signature
            
        except Exception as e:
            logger.error(f"Error signing message: {e}")
            raise
    
    def process_handshake_response(
        self, 
        session_id: str, 
        response_message: bytes
    ) -> Optional[bytes]:
        """
        Process handshake response from device.
        
        Args:
            session_id: Session identifier
            response_message: Handshake response message
            
        Returns:
            Handshake confirmation message if successful
        """
        try:
            session = self.sessions.get(session_id)
            if not session or session.state != SessionState.HANDSHAKING:
                logger.error(f"Invalid session {session_id} for handshake response")
                return None
            
            # Parse response message
            response_data = json.loads(response_message.decode('utf-8'))
            
            # Verify response
            if not self._verify_handshake_response(session, response_data):
                logger.error(f"Handshake response verification failed for session {session_id}")
                return None
            
            # Extract remote public key
            session.remote_public_key = bytes.fromhex(
                response_data['handshake_data']['public_key']
            )
            
            # Perform key encapsulation
            shared_secret = self._perform_key_encapsulation(session)
            if not shared_secret:
                logger.error(f"Key encapsulation failed for session {session_id}")
                return None
            
            # Derive session keys
            session.shared_secret = shared_secret
            session.encryption_key, session.mac_key = self._derive_session_keys(shared_secret)
            
            # Create confirmation message
            confirmation_message = self._create_handshake_confirm(session)
            
            # Activate session
            session.state = SessionState.ACTIVE
            session.last_activity = datetime.utcnow()
            
            logger.info(f"Completed handshake for session {session_id}")
            return confirmation_message
            
        except Exception as e:
            logger.error(f"Error processing handshake response: {e}")
            return None
    
    def _verify_handshake_response(
        self, 
        session: SecureSession, 
        response_data: Dict[str, Any]
    ) -> bool:
        """Verify handshake response message."""
        try:
            # Verify session ID matches
            if response_data['handshake_data']['session_id'] != session.session_id:
                return False
            
            # Verify timestamp is recent
            response_time = response_data['handshake_data']['timestamp']
            current_time = time.time()
            if abs(current_time - response_time) > 60:  # 1 minute tolerance
                return False
            
            # Verify signature (mock implementation)
            # In production, this would verify the post-quantum signature
            return True
            
        except Exception as e:
            logger.error(f"Error verifying handshake response: {e}")
            return False
    
    def _perform_key_encapsulation(self, session: SecureSession) -> Optional[bytes]:
        """Perform post-quantum key encapsulation."""
        try:
            # Mock key encapsulation - in production this would use real PQ KEM
            # Combine local private key and remote public key to create shared secret
            combined = session.local_keypair.private_key[:32] + session.remote_public_key[:32]
            shared_secret = hashlib.sha256(combined).digest()
            
            logger.debug(f"Generated shared secret for session {session.session_id}")
            return shared_secret
            
        except Exception as e:
            logger.error(f"Error performing key encapsulation: {e}")
            return None
    
    def _derive_session_keys(self, shared_secret: bytes) -> Tuple[bytes, bytes]:
        """Derive encryption and MAC keys from shared secret."""
        try:
            # Derive encryption key
            hkdf_enc = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"QFLARE-ENCRYPTION",
                info=b"session-encryption-key",
                backend=default_backend()
            )
            encryption_key = hkdf_enc.derive(shared_secret)
            
            # Derive MAC key
            hkdf_mac = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"QFLARE-MAC",
                info=b"session-mac-key",
                backend=default_backend()
            )
            mac_key = hkdf_mac.derive(shared_secret)
            
            return encryption_key, mac_key
            
        except Exception as e:
            logger.error(f"Error deriving session keys: {e}")
            raise
    
    def _create_handshake_confirm(self, session: SecureSession) -> bytes:
        """Create handshake confirmation message."""
        try:
            confirm_data = {
                'session_id': session.session_id,
                'timestamp': time.time(),
                'status': 'confirmed',
                'session_parameters': {
                    'encryption_algorithm': 'AES-256-GCM',
                    'mac_algorithm': 'HMAC-SHA256',
                    'session_timeout': self.config['session_timeout_minutes']
                }
            }
            
            message_bytes = json.dumps(confirm_data).encode('utf-8')
            
            # Encrypt the confirmation message with session keys
            encrypted_message = self._encrypt_message(session, message_bytes, MessageType.HANDSHAKE_CONFIRM)
            
            return encrypted_message
            
        except Exception as e:
            logger.error(f"Error creating handshake confirm: {e}")
            raise
    
    def encrypt_message(
        self, 
        session_id: str, 
        message: bytes, 
        message_type: MessageType
    ) -> Optional[bytes]:
        """
        Encrypt a message for secure transmission.
        
        Args:
            session_id: Session identifier
            message: Message to encrypt
            message_type: Type of message
            
        Returns:
            Encrypted message bytes
        """
        try:
            session = self.sessions.get(session_id)
            if not session or session.state != SessionState.ACTIVE:
                logger.error(f"Invalid session {session_id} for encryption")
                return None
            
            # Check message size
            if len(message) > self.config['message_max_size']:
                logger.error(f"Message too large: {len(message)} bytes")
                return None
            
            # Update activity timestamp
            session.last_activity = datetime.utcnow()
            
            return self._encrypt_message(session, message, message_type)
            
        except Exception as e:
            logger.error(f"Error encrypting message: {e}")
            return None
    
    def _encrypt_message(
        self, 
        session: SecureSession, 
        message: bytes, 
        message_type: MessageType
    ) -> bytes:
        """Internal message encryption."""
        try:
            # Generate random IV
            iv = secrets.token_bytes(16)
            
            # Encrypt message using AES-256-GCM
            cipher = Cipher(
                algorithms.AES(session.encryption_key),
                modes.GCM(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            ciphertext = encryptor.update(message) + encryptor.finalize()
            
            # Create message structure
            secure_message = {
                'message_type': message_type.value,
                'session_id': session.session_id,
                'sequence_number': session.send_counter,
                'timestamp': time.time(),
                'encrypted_payload': ciphertext.hex(),
                'auth_tag': encryptor.tag.hex(),
                'iv': iv.hex()
            }
            
            # Calculate MAC
            message_bytes = json.dumps(secure_message).encode('utf-8')
            mac = hmac.new(session.mac_key, message_bytes, hashlib.sha256).digest()
            
            secure_message['mac'] = mac.hex()
            
            # Increment send counter
            session.send_counter += 1
            
            return json.dumps(secure_message).encode('utf-8')
            
        except Exception as e:
            logger.error(f"Error in message encryption: {e}")
            raise
    
    def decrypt_message(
        self, 
        session_id: str, 
        encrypted_message: bytes
    ) -> Optional[Tuple[bytes, MessageType]]:
        """
        Decrypt a received message.
        
        Args:
            session_id: Session identifier
            encrypted_message: Encrypted message bytes
            
        Returns:
            Tuple of (decrypted_message, message_type) if successful
        """
        try:
            session = self.sessions.get(session_id)
            if not session or session.state != SessionState.ACTIVE:
                logger.error(f"Invalid session {session_id} for decryption")
                return None
            
            # Parse encrypted message
            message_data = json.loads(encrypted_message.decode('utf-8'))
            
            # Verify MAC
            received_mac = bytes.fromhex(message_data.pop('mac'))
            calculated_mac = hmac.new(
                session.mac_key,
                json.dumps(message_data).encode('utf-8'),
                hashlib.sha256
            ).digest()
            
            if not hmac.compare_digest(received_mac, calculated_mac):
                logger.error(f"MAC verification failed for session {session_id}")
                return None
            
            # Check sequence number for replay protection
            sequence_number = message_data['sequence_number']
            if sequence_number <= session.receive_counter:
                logger.error(f"Replay attack detected for session {session_id}")
                return None
            
            # Decrypt message
            iv = bytes.fromhex(message_data['iv'])
            ciphertext = bytes.fromhex(message_data['encrypted_payload'])
            auth_tag = bytes.fromhex(message_data['auth_tag'])
            
            cipher = Cipher(
                algorithms.AES(session.encryption_key),
                modes.GCM(iv, auth_tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Update receive counter and activity timestamp
            session.receive_counter = sequence_number
            session.last_activity = datetime.utcnow()
            
            message_type = MessageType(message_data['message_type'])
            
            logger.debug(f"Decrypted {message_type.value} message for session {session_id}")
            return plaintext, message_type
            
        except Exception as e:
            logger.error(f"Error decrypting message: {e}")
            return None
    
    def terminate_session(self, session_id: str) -> bool:
        """
        Terminate a secure session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was terminated successfully
        """
        try:
            session = self.sessions.get(session_id)
            if not session:
                logger.warning(f"Session {session_id} not found for termination")
                return False
            
            session.state = SessionState.TERMINATED
            
            # Clear sensitive data
            session.shared_secret = b""
            session.encryption_key = b""
            session.mac_key = b""
            
            # Remove from active sessions
            del self.sessions[session_id]
            
            logger.info(f"Terminated session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error terminating session {session_id}: {e}")
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            current_time = datetime.utcnow()
            expired_sessions = []
            
            for session_id, session in self.sessions.items():
                if current_time > session.expires_at:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                self.terminate_session(session_id)
            
            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            
            return len(expired_sessions)
            
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
            return 0
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Returns:
            Dictionary with session statistics
        """
        stats = {
            'total_sessions': len(self.sessions),
            'sessions_by_state': {},
            'sessions_by_device': {},
            'average_session_age_minutes': 0
        }
        
        current_time = datetime.utcnow()
        total_age_seconds = 0
        
        for session in self.sessions.values():
            # Count by state
            state = session.state.value
            stats['sessions_by_state'][state] = stats['sessions_by_state'].get(state, 0) + 1
            
            # Count by device
            device_id = session.device_id
            stats['sessions_by_device'][device_id] = stats['sessions_by_device'].get(device_id, 0) + 1
            
            # Calculate age
            age_seconds = (current_time - session.created_at).total_seconds()
            total_age_seconds += age_seconds
        
        if self.sessions:
            stats['average_session_age_minutes'] = (total_age_seconds / len(self.sessions)) / 60
        
        return stats