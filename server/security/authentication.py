"""
Enhanced Authentication System for QFLARE

This module provides comprehensive authentication with:
- Post-quantum device authentication
- JWT token management with rotation
- Multi-factor authentication support
- Device attestation and enrollment
- Role-based access control (RBAC)
"""

import os
import time
import hmac
import hashlib
import secrets
import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import uuid

# JWT and cryptography imports
import jwt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

from .key_management import PostQuantumKeyManager, KeyType, KeyPair
from ..database import DeviceService, AuditService, UserToken

logger = logging.getLogger(__name__)


class AuthenticationMethod(Enum):
    """Authentication methods supported by QFLARE."""
    POST_QUANTUM_SIGNATURE = "pq_signature"
    JWT_TOKEN = "jwt_token"
    MUTUAL_TLS = "mutual_tls"
    DEVICE_CERTIFICATE = "device_certificate"
    ENROLLMENT_TOKEN = "enrollment_token"


class DeviceRole(Enum):
    """Device roles for RBAC."""
    EDGE_NODE = "edge_node"
    AGGREGATOR = "aggregator"
    COORDINATOR = "coordinator"
    ADMIN = "admin"
    OBSERVER = "observer"


class TokenType(Enum):
    """Token types for different purposes."""
    ACCESS_TOKEN = "access"
    REFRESH_TOKEN = "refresh"
    ENROLLMENT_TOKEN = "enrollment"
    API_KEY = "api_key"


@dataclass
class AuthenticationContext:
    """Authentication context for requests."""
    device_id: str
    authenticated: bool
    auth_method: AuthenticationMethod
    roles: Set[DeviceRole]
    permissions: Set[str]
    session_id: Optional[str]
    token_id: Optional[str]
    expires_at: datetime
    metadata: Dict[str, Any]


@dataclass
class DeviceCredentials:
    """Device credentials package."""
    device_id: str
    keypair: KeyPair
    certificate: bytes
    enrollment_token: str
    api_key: str
    created_at: datetime
    expires_at: datetime


class EnhancedAuthenticationManager:
    """Enhanced authentication manager for QFLARE."""
    
    def __init__(self, key_manager: PostQuantumKeyManager):
        """
        Initialize authentication manager.
        
        Args:
            key_manager: Post-quantum key manager instance
        """
        self.key_manager = key_manager
        self.config = self._setup_default_config()
        self._jwt_secret = self._generate_jwt_secret()
        self._enrolled_devices = {}
        self._active_tokens = {}
        self._setup_default_permissions()
    
    def _setup_default_config(self) -> Dict[str, Any]:
        """Setup default configuration."""
        return {
            'jwt_algorithm': 'HS256',
            'access_token_lifetime_minutes': 60,
            'refresh_token_lifetime_days': 30,
            'enrollment_token_lifetime_hours': 24,
            'api_key_lifetime_days': 365,
            'max_tokens_per_device': 5,
            'require_device_attestation': True,
            'enable_token_rotation': True,
            'password_hash_rounds': 100000,
            'enable_mfa': False,
            'rate_limit_attempts': 5,
            'rate_limit_window_minutes': 15
        }
    
    def _generate_jwt_secret(self) -> bytes:
        """Generate JWT signing secret."""
        return secrets.token_bytes(64)
    
    def _setup_default_permissions(self):
        """Setup default role-based permissions."""
        self.role_permissions = {
            DeviceRole.EDGE_NODE: {
                'model:upload', 'model:download', 'training:participate',
                'heartbeat:send', 'status:report'
            },
            DeviceRole.AGGREGATOR: {
                'model:aggregate', 'model:distribute', 'device:manage',
                'training:coordinate', 'metrics:collect'
            },
            DeviceRole.COORDINATOR: {
                'session:create', 'session:manage', 'policy:set',
                'device:enroll', 'device:revoke'
            },
            DeviceRole.ADMIN: {
                'system:admin', 'user:manage', 'audit:view',
                'key:rotate', 'config:update'
            },
            DeviceRole.OBSERVER: {
                'metrics:view', 'status:view', 'audit:view'
            }
        }
    
    def enroll_device(
        self, 
        device_id: str, 
        device_info: Dict[str, Any],
        enrollment_token: Optional[str] = None
    ) -> DeviceCredentials:
        """
        Enroll a new device in the QFLARE system.
        
        Args:
            device_id: Unique device identifier
            device_info: Device information and capabilities
            enrollment_token: Pre-issued enrollment token (if required)
            
        Returns:
            Device credentials package
        """
        try:
            # Validate enrollment token if required
            if self.config['require_device_attestation'] and enrollment_token:
                if not self._validate_enrollment_token(enrollment_token):
                    raise ValueError("Invalid enrollment token")
            
            # Check if device already enrolled
            if device_id in self._enrolled_devices:
                raise ValueError(f"Device {device_id} already enrolled")
            
            # Generate post-quantum key pairs
            kem_keypair = self.key_manager.generate_key_pair(
                key_type=KeyType.KEM_FRODO_640_AES,
                device_id=device_id,
                purpose="kem"
            )
            
            sig_keypair = self.key_manager.generate_key_pair(
                key_type=KeyType.SIG_DILITHIUM_2,
                device_id=device_id,
                purpose="signature"
            )
            
            # Generate device certificate (mock implementation)
            certificate = self._generate_device_certificate(device_id, kem_keypair.public_key)
            
            # Generate API key
            api_key = self._generate_api_key(device_id)
            
            # Create device credentials
            credentials = DeviceCredentials(
                device_id=device_id,
                keypair=sig_keypair,  # Use signature keypair for authentication
                certificate=certificate,
                enrollment_token=enrollment_token or "",
                api_key=api_key,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(
                    days=self.config['api_key_lifetime_days']
                )
            )
            
            # Register device in database
            device_data = {
                **device_info,
                'kem_public_key': kem_keypair.public_key,
                'sig_public_key': sig_keypair.public_key,
                'certificate': certificate,
                'api_key_hash': hashlib.sha256(api_key.encode()).hexdigest()
            }
            
            success = DeviceService.register_device(device_id, device_data)
            if not success:
                raise RuntimeError("Failed to register device in database")
            
            # Store enrollment
            self._enrolled_devices[device_id] = credentials
            
            # Log enrollment event
            AuditService.log_event(
                session=None,
                device_id=device_id,
                event_type='device_enrollment',
                event_description=f'Device {device_id} enrolled successfully',
                event_data=device_info,
                risk_level='medium'
            )
            
            logger.info(f"Device {device_id} enrolled successfully")
            return credentials
            
        except Exception as e:
            logger.error(f"Error enrolling device {device_id}: {e}")
            raise
    
    def _validate_enrollment_token(self, token: str) -> bool:
        """Validate enrollment token."""
        try:
            # Mock validation - in production this would check against database
            return len(token) == 64 and token.startswith('enroll_')
        except Exception as e:
            logger.error(f"Error validating enrollment token: {e}")
            return False
    
    def _generate_device_certificate(self, device_id: str, public_key: bytes) -> bytes:
        """Generate device certificate."""
        # Mock certificate generation
        cert_data = {
            'device_id': device_id,
            'public_key': public_key.hex(),
            'issued_at': datetime.utcnow().isoformat(),
            'issuer': 'QFLARE-CA'
        }
        return json.dumps(cert_data).encode('utf-8')
    
    def _generate_api_key(self, device_id: str) -> str:
        """Generate API key for device."""
        timestamp = int(time.time())
        random_part = secrets.token_hex(16)
        return f"qflare_{device_id}_{timestamp}_{random_part}"
    
    def authenticate_device(
        self, 
        device_id: str, 
        auth_data: Dict[str, Any],
        auth_method: AuthenticationMethod
    ) -> Optional[AuthenticationContext]:
        """
        Authenticate a device using specified method.
        
        Args:
            device_id: Device identifier
            auth_data: Authentication data
            auth_method: Authentication method to use
            
        Returns:
            Authentication context if successful
        """
        try:
            # Check if device is enrolled
            device_info = DeviceService.get_device(device_id)
            if not device_info:
                logger.warning(f"Authentication failed: device {device_id} not enrolled")
                return None
            
            # Check device status
            if device_info['status'] not in ['active', 'training']:
                logger.warning(f"Authentication failed: device {device_id} status is {device_info['status']}")
                return None
            
            # Perform authentication based on method
            if auth_method == AuthenticationMethod.POST_QUANTUM_SIGNATURE:
                return self._authenticate_with_pq_signature(device_id, auth_data)
            elif auth_method == AuthenticationMethod.JWT_TOKEN:
                return self._authenticate_with_jwt(device_id, auth_data)
            elif auth_method == AuthenticationMethod.DEVICE_CERTIFICATE:
                return self._authenticate_with_certificate(device_id, auth_data)
            else:
                logger.error(f"Unsupported authentication method: {auth_method}")
                return None
            
        except Exception as e:
            logger.error(f"Error authenticating device {device_id}: {e}")
            return None
    
    def _authenticate_with_pq_signature(
        self, 
        device_id: str, 
        auth_data: Dict[str, Any]
    ) -> Optional[AuthenticationContext]:
        """Authenticate using post-quantum signature."""
        try:
            # Get device signature key
            device_keys = self.key_manager.get_device_keys(device_id)
            sig_key = next((k for k in device_keys if k.metadata.purpose == "signature"), None)
            
            if not sig_key:
                logger.error(f"No signature key found for device {device_id}")
                return None
            
            # Verify signature
            message = auth_data.get('message', '').encode('utf-8')
            signature = bytes.fromhex(auth_data.get('signature', ''))
            
            # Mock signature verification
            expected_signature = hmac.new(
                sig_key.private_key[:32], message, hashlib.sha256
            ).digest()
            
            if not hmac.compare_digest(signature, expected_signature):
                logger.warning(f"Signature verification failed for device {device_id}")
                return None
            
            # Create authentication context
            return self._create_auth_context(
                device_id=device_id,
                auth_method=AuthenticationMethod.POST_QUANTUM_SIGNATURE,
                roles={DeviceRole.EDGE_NODE}
            )
            
        except Exception as e:
            logger.error(f"Error in PQ signature authentication: {e}")
            return None
    
    def _authenticate_with_jwt(
        self, 
        device_id: str, 
        auth_data: Dict[str, Any]
    ) -> Optional[AuthenticationContext]:
        """Authenticate using JWT token."""
        try:
            token = auth_data.get('token')
            if not token:
                return None
            
            # Verify JWT token
            try:
                payload = jwt.decode(
                    token, 
                    self._jwt_secret, 
                    algorithms=[self.config['jwt_algorithm']]
                )
            except jwt.InvalidTokenError as e:
                logger.warning(f"Invalid JWT token for device {device_id}: {e}")
                return None
            
            # Verify device ID matches
            if payload.get('device_id') != device_id:
                logger.warning(f"Device ID mismatch in JWT token")
                return None
            
            # Check token type
            token_type = payload.get('token_type', TokenType.ACCESS_TOKEN.value)
            if token_type != TokenType.ACCESS_TOKEN.value:
                logger.warning(f"Invalid token type for authentication: {token_type}")
                return None
            
            # Create authentication context
            roles = {DeviceRole(role) for role in payload.get('roles', ['edge_node'])}
            
            return self._create_auth_context(
                device_id=device_id,
                auth_method=AuthenticationMethod.JWT_TOKEN,
                roles=roles,
                token_id=payload.get('jti')
            )
            
        except Exception as e:
            logger.error(f"Error in JWT authentication: {e}")
            return None
    
    def _authenticate_with_certificate(
        self, 
        device_id: str, 
        auth_data: Dict[str, Any]
    ) -> Optional[AuthenticationContext]:
        """Authenticate using device certificate."""
        try:
            certificate = auth_data.get('certificate')
            if not certificate:
                return None
            
            # Mock certificate verification
            # In production, this would verify the certificate chain
            
            return self._create_auth_context(
                device_id=device_id,
                auth_method=AuthenticationMethod.DEVICE_CERTIFICATE,
                roles={DeviceRole.EDGE_NODE}
            )
            
        except Exception as e:
            logger.error(f"Error in certificate authentication: {e}")
            return None
    
    def _create_auth_context(
        self,
        device_id: str,
        auth_method: AuthenticationMethod,
        roles: Set[DeviceRole],
        session_id: Optional[str] = None,
        token_id: Optional[str] = None
    ) -> AuthenticationContext:
        """Create authentication context."""
        # Calculate permissions from roles
        permissions = set()
        for role in roles:
            permissions.update(self.role_permissions.get(role, set()))
        
        # Create context
        context = AuthenticationContext(
            device_id=device_id,
            authenticated=True,
            auth_method=auth_method,
            roles=roles,
            permissions=permissions,
            session_id=session_id,
            token_id=token_id,
            expires_at=datetime.utcnow() + timedelta(
                minutes=self.config['access_token_lifetime_minutes']
            ),
            metadata={}
        )
        
        # Log authentication event
        AuditService.log_event(
            session=None,
            device_id=device_id,
            event_type='authentication',
            event_description=f'Device authenticated using {auth_method.value}',
            event_data={
                'auth_method': auth_method.value,
                'roles': [role.value for role in roles]
            },
            risk_level='low'
        )
        
        return context
    
    def generate_access_token(
        self, 
        device_id: str, 
        roles: List[DeviceRole] = None
    ) -> str:
        """
        Generate JWT access token for device.
        
        Args:
            device_id: Device identifier
            roles: Device roles
            
        Returns:
            JWT access token
        """
        try:
            if roles is None:
                roles = [DeviceRole.EDGE_NODE]
            
            # Create JWT payload
            now = datetime.utcnow()
            payload = {
                'device_id': device_id,
                'token_type': TokenType.ACCESS_TOKEN.value,
                'roles': [role.value for role in roles],
                'iat': now,
                'exp': now + timedelta(minutes=self.config['access_token_lifetime_minutes']),
                'jti': str(uuid.uuid4()),  # Token ID
                'iss': 'qflare-server'
            }
            
            # Generate token
            token = jwt.encode(
                payload, 
                self._jwt_secret, 
                algorithm=self.config['jwt_algorithm']
            )
            
            # Store token info
            self._active_tokens[payload['jti']] = {
                'device_id': device_id,
                'created_at': now,
                'expires_at': payload['exp']
            }
            
            logger.info(f"Generated access token for device {device_id}")
            return token
            
        except Exception as e:
            logger.error(f"Error generating access token: {e}")
            raise
    
    def generate_refresh_token(self, device_id: str) -> str:
        """
        Generate refresh token for device.
        
        Args:
            device_id: Device identifier
            
        Returns:
            JWT refresh token
        """
        try:
            now = datetime.utcnow()
            payload = {
                'device_id': device_id,
                'token_type': TokenType.REFRESH_TOKEN.value,
                'iat': now,
                'exp': now + timedelta(days=self.config['refresh_token_lifetime_days']),
                'jti': str(uuid.uuid4()),
                'iss': 'qflare-server'
            }
            
            token = jwt.encode(
                payload, 
                self._jwt_secret, 
                algorithm=self.config['jwt_algorithm']
            )
            
            logger.info(f"Generated refresh token for device {device_id}")
            return token
            
        except Exception as e:
            logger.error(f"Error generating refresh token: {e}")
            raise
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New access token if successful
        """
        try:
            # Verify refresh token
            payload = jwt.decode(
                refresh_token, 
                self._jwt_secret, 
                algorithms=[self.config['jwt_algorithm']]
            )
            
            # Check token type
            if payload.get('token_type') != TokenType.REFRESH_TOKEN.value:
                logger.warning("Invalid token type for refresh")
                return None
            
            device_id = payload.get('device_id')
            if not device_id:
                logger.warning("Missing device_id in refresh token")
                return None
            
            # Generate new access token
            return self.generate_access_token(device_id)
            
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid refresh token: {e}")
            return None
        except Exception as e:
            logger.error(f"Error refreshing access token: {e}")
            return None
    
    def revoke_token(self, token_id: str) -> bool:
        """
        Revoke a token.
        
        Args:
            token_id: Token identifier (jti)
            
        Returns:
            True if token was revoked successfully
        """
        try:
            if token_id in self._active_tokens:
                del self._active_tokens[token_id]
                logger.info(f"Revoked token {token_id}")
                return True
            else:
                logger.warning(f"Token {token_id} not found for revocation")
                return False
            
        except Exception as e:
            logger.error(f"Error revoking token {token_id}: {e}")
            return False
    
    def check_permission(
        self, 
        context: AuthenticationContext, 
        required_permission: str
    ) -> bool:
        """
        Check if authentication context has required permission.
        
        Args:
            context: Authentication context
            required_permission: Required permission string
            
        Returns:
            True if permission is granted
        """
        if not context.authenticated:
            return False
        
        if context.expires_at < datetime.utcnow():
            return False
        
        return required_permission in context.permissions
    
    def get_authentication_statistics(self) -> Dict[str, Any]:
        """
        Get authentication statistics.
        
        Returns:
            Dictionary with authentication statistics
        """
        stats = {
            'enrolled_devices': len(self._enrolled_devices),
            'active_tokens': len(self._active_tokens),
            'tokens_by_device': {},
            'devices_by_role': {}
        }
        
        # Count tokens by device
        for token_info in self._active_tokens.values():
            device_id = token_info['device_id']
            stats['tokens_by_device'][device_id] = stats['tokens_by_device'].get(device_id, 0) + 1
        
        # This would be enhanced with database queries for role distribution
        
        return stats