"""
Enhanced Security Package for QFLARE

This package provides comprehensive security features including:
- Post-quantum cryptographic key management
- Enhanced secure communication protocols
- Multi-factor device authentication
- Certificate and token management
- Hardware Security Module integration
"""

from .key_management import (
    PostQuantumKeyManager, 
    CertificateManager,
    KeyType, 
    KeyStatus, 
    KeyPair, 
    KeyMetadata
)

from .secure_communication import (
    SecureCommunicationManager,
    SecureSession,
    SecureMessage,
    MessageType,
    SessionState
)

from .authentication import (
    EnhancedAuthenticationManager,
    AuthenticationContext,
    DeviceCredentials,
    AuthenticationMethod,
    DeviceRole,
    TokenType
)

__all__ = [
    # Key Management
    'PostQuantumKeyManager',
    'CertificateManager', 
    'KeyType',
    'KeyStatus',
    'KeyPair',
    'KeyMetadata',
    
    # Secure Communication
    'SecureCommunicationManager',
    'SecureSession',
    'SecureMessage',
    'MessageType',
    'SessionState',
    
    # Authentication
    'EnhancedAuthenticationManager',
    'AuthenticationContext',
    'DeviceCredentials',
    'AuthenticationMethod',
    'DeviceRole',
    'TokenType'
]


class QFLARESecurityManager:
    """Integrated security manager for QFLARE system."""
    
    def __init__(self, config: dict = None):
        """
        Initialize integrated security manager.
        
        Args:
            config: Security configuration dictionary
        """
        self.config = config or {}
        
        # Initialize core security components
        self.key_manager = PostQuantumKeyManager(self.config.get('key_management', {}))
        self.cert_manager = CertificateManager()
        self.comm_manager = SecureCommunicationManager(self.key_manager)
        self.auth_manager = EnhancedAuthenticationManager(self.key_manager)
    
    def initialize_device_security(self, device_id: str, device_info: dict) -> DeviceCredentials:
        """
        Initialize complete security setup for a new device.
        
        Args:
            device_id: Device identifier
            device_info: Device information
            
        Returns:
            Complete device credentials package
        """
        # Enroll device with authentication manager
        credentials = self.auth_manager.enroll_device(device_id, device_info)
        
        # Generate additional certificates if needed
        device_cert = self.cert_manager.generate_device_certificate(
            device_id, 
            credentials.keypair.public_key
        )
        
        return credentials
    
    def establish_secure_session(self, device_id: str) -> tuple:
        """
        Establish secure communication session with device.
        
        Args:
            device_id: Device identifier
            
        Returns:
            Tuple of (session_id, handshake_message)
        """
        return self.comm_manager.initiate_secure_session(device_id)
    
    def authenticate_request(self, device_id: str, auth_data: dict, method: AuthenticationMethod) -> AuthenticationContext:
        """
        Authenticate device request.
        
        Args:
            device_id: Device identifier
            auth_data: Authentication data
            method: Authentication method
            
        Returns:
            Authentication context
        """
        return self.auth_manager.authenticate_device(device_id, auth_data, method)
    
    def get_security_status(self) -> dict:
        """
        Get comprehensive security status.
        
        Returns:
            Dictionary with security status information
        """
        return {
            'key_management': self.key_manager.get_key_statistics(),
            'authentication': self.auth_manager.get_authentication_statistics(),
            'communication': self.comm_manager.get_session_statistics()
        }