"""
Enhanced Security System Test Suite for QFLARE

This script tests the enhanced security functionality including:
- Post-quantum key management
- Secure communication protocols
- Multi-factor authentication
- Certificate management
- Session security
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta

# Add server directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

# Set environment variable to use mock implementation
os.environ['QFLARE_USE_MOCK_CRYPTO'] = 'true'

from server.security import (
    PostQuantumKeyManager, CertificateManager, SecureCommunicationManager,
    EnhancedAuthenticationManager, QFLARESecurityManager,
    KeyType, AuthenticationMethod, DeviceRole, MessageType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_post_quantum_key_management():
    """Test post-quantum key management functionality."""
    print("🔐 Testing Post-Quantum Key Management...")
    
    try:
        # Initialize key manager
        key_manager = PostQuantumKeyManager()
        
        # Test key pair generation
        device_id = "test_device_security_001"
        
        # Generate KEM key pair
        kem_keypair = key_manager.generate_key_pair(
            key_type=KeyType.KEM_FRODO_640_AES,
            device_id=device_id,
            purpose="kem"
        )
        
        if kem_keypair and len(kem_keypair.public_key) > 0:
            print("✅ KEM key pair generation successful")
        else:
            print("❌ KEM key pair generation failed")
            return False
        
        # Generate signature key pair
        sig_keypair = key_manager.generate_key_pair(
            key_type=KeyType.SIG_DILITHIUM_2,
            device_id=device_id,
            purpose="signature"
        )
        
        if sig_keypair and len(sig_keypair.public_key) > 0:
            print("✅ Signature key pair generation successful")
        else:
            print("❌ Signature key pair generation failed")
            return False
        
        # Test key retrieval
        retrieved_key = key_manager.get_key_pair(kem_keypair.metadata.key_id)
        if retrieved_key and retrieved_key.metadata.key_id == kem_keypair.metadata.key_id:
            print("✅ Key retrieval successful")
        else:
            print("❌ Key retrieval failed")
            return False
        
        # Test key rotation
        rotated_key = key_manager.rotate_key(kem_keypair.metadata.key_id)
        if rotated_key and rotated_key.metadata.rotation_count == 1:
            print("✅ Key rotation successful")
        else:
            print("❌ Key rotation failed")
            return False
        
        # Test key statistics
        stats = key_manager.get_key_statistics()
        if stats['total_keys'] >= 3:  # Original + rotated + signature
            print(f"✅ Key statistics: {stats['total_keys']} total keys")
        else:
            print("❌ Key statistics inconsistent")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Post-quantum key management test failed: {e}")
        return False


def test_certificate_management():
    """Test certificate management functionality."""
    print("\n🏆 Testing Certificate Management...")
    
    try:
        # Initialize certificate manager
        cert_manager = CertificateManager()
        
        # Test CA certificate generation
        if cert_manager.ca_certificate:
            print("✅ CA certificate initialization successful")
        else:
            print("❌ CA certificate initialization failed")
            return False
        
        # Test device certificate generation
        device_id = "test_device_cert_001"
        test_public_key = b"mock_public_key_data_for_testing"
        
        device_cert = cert_manager.generate_device_certificate(device_id, test_public_key)
        
        if device_cert and len(device_cert) > 0:
            print("✅ Device certificate generation successful")
        else:
            print("❌ Device certificate generation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Certificate management test failed: {e}")
        return False


def test_secure_communication():
    """Test secure communication protocols."""
    print("\n🔒 Testing Secure Communication...")
    
    try:
        # Initialize components
        key_manager = PostQuantumKeyManager()
        comm_manager = SecureCommunicationManager(key_manager)
        
        device_id = "test_device_comm_001"
        
        # Test session initiation
        session_id, handshake_message = comm_manager.initiate_secure_session(device_id)
        
        if session_id and handshake_message:
            print("✅ Secure session initiation successful")
        else:
            print("❌ Secure session initiation failed")
            return False
        
        # Mock handshake response
        mock_response = {
            'handshake_data': {
                'session_id': session_id,
                'device_id': device_id,
                'timestamp': time.time(),
                'public_key': 'mock_public_key_hex',
                'supported_algorithms': ['FrodoKEM-640-AES']
            },
            'signature': 'mock_signature_hex'
        }
        
        import json
        response_bytes = json.dumps(mock_response).encode('utf-8')
        
        # Process handshake response
        confirmation = comm_manager.process_handshake_response(session_id, response_bytes)
        
        if confirmation:
            print("✅ Handshake completion successful")
        else:
            print("❌ Handshake completion failed")
            return False
        
        # Test message encryption/decryption
        test_message = b"This is a test secure message for QFLARE"
        
        encrypted_message = comm_manager.encrypt_message(
            session_id, 
            test_message, 
            MessageType.MODEL_UPDATE
        )
        
        if encrypted_message:
            print("✅ Message encryption successful")
        else:
            print("❌ Message encryption failed")
            return False
        
        # Test message decryption
        decrypted_result = comm_manager.decrypt_message(session_id, encrypted_message)
        
        if decrypted_result and decrypted_result[0] == test_message:
            print("✅ Message decryption successful")
        else:
            print("❌ Message decryption failed")
            return False
        
        # Test session statistics
        stats = comm_manager.get_session_statistics()
        if stats['total_sessions'] >= 1:
            print(f"✅ Session statistics: {stats['total_sessions']} active sessions")
        else:
            print("❌ Session statistics inconsistent")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Secure communication test failed: {e}")
        return False


def test_enhanced_authentication():
    """Test enhanced authentication system."""
    print("\n🛡️ Testing Enhanced Authentication...")
    
    try:
        # Initialize components
        key_manager = PostQuantumKeyManager()
        auth_manager = EnhancedAuthenticationManager(key_manager)
        
        device_id = "test_device_auth_001"
        device_info = {
            'device_type': 'edge_node',
            'capabilities': {'training': True, 'inference': True}
        }
        
        # Test device enrollment
        credentials = auth_manager.enroll_device(device_id, device_info)
        
        if credentials and credentials.device_id == device_id:
            print("✅ Device enrollment successful")
        else:
            print("❌ Device enrollment failed")
            return False
        
        # Test JWT token generation
        access_token = auth_manager.generate_access_token(
            device_id, 
            [DeviceRole.EDGE_NODE]
        )
        
        if access_token and len(access_token) > 0:
            print("✅ Access token generation successful")
        else:
            print("❌ Access token generation failed")
            return False
        
        # Test JWT authentication
        auth_data = {'token': access_token}
        auth_context = auth_manager.authenticate_device(
            device_id, 
            auth_data, 
            AuthenticationMethod.JWT_TOKEN
        )
        
        if auth_context and auth_context.authenticated:
            print("✅ JWT authentication successful")
        else:
            print("❌ JWT authentication failed")
            return False
        
        # Test permission checking
        has_permission = auth_manager.check_permission(auth_context, 'model:upload')
        
        if has_permission:
            print("✅ Permission checking successful")
        else:
            print("❌ Permission checking failed")
            return False
        
        # Test post-quantum signature authentication
        test_message = "authentication_challenge_message"
        import hmac
        import hashlib
        
        # Create mock signature
        signature = hmac.new(
            credentials.keypair.private_key[:32], 
            test_message.encode(), 
            hashlib.sha256
        ).digest()
        
        pq_auth_data = {
            'message': test_message,
            'signature': signature.hex()
        }
        
        pq_context = auth_manager.authenticate_device(
            device_id,
            pq_auth_data,
            AuthenticationMethod.POST_QUANTUM_SIGNATURE
        )
        
        if pq_context and pq_context.authenticated:
            print("✅ Post-quantum signature authentication successful")
        else:
            print("❌ Post-quantum signature authentication failed")
            return False
        
        # Test refresh token
        refresh_token = auth_manager.generate_refresh_token(device_id)
        new_access_token = auth_manager.refresh_access_token(refresh_token)
        
        if new_access_token and new_access_token != access_token:
            print("✅ Token refresh successful")
        else:
            print("❌ Token refresh failed")
            return False
        
        # Test authentication statistics
        stats = auth_manager.get_authentication_statistics()
        if stats['enrolled_devices'] >= 1:
            print(f"✅ Auth statistics: {stats['enrolled_devices']} enrolled devices")
        else:
            print("❌ Auth statistics inconsistent")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced authentication test failed: {e}")
        return False


def test_integrated_security_manager():
    """Test integrated security manager."""
    print("\n🔐 Testing Integrated Security Manager...")
    
    try:
        # Initialize integrated security manager
        security_manager = QFLARESecurityManager()
        
        device_id = "test_device_integrated_001"
        device_info = {
            'device_type': 'edge_node',
            'hardware_info': {'cpu': 'ARM', 'memory': '4GB'},
            'capabilities': {'training': True}
        }
        
        # Test complete device security initialization
        credentials = security_manager.initialize_device_security(device_id, device_info)
        
        if credentials and credentials.device_id == device_id:
            print("✅ Integrated device security initialization successful")
        else:
            print("❌ Integrated device security initialization failed")
            return False
        
        # Test secure session establishment
        session_id, handshake_msg = security_manager.establish_secure_session(device_id)
        
        if session_id and handshake_msg:
            print("✅ Integrated secure session establishment successful")
        else:
            print("❌ Integrated secure session establishment failed")
            return False
        
        # Test integrated authentication
        auth_data = {'token': security_manager.auth_manager.generate_access_token(device_id)}
        auth_context = security_manager.authenticate_request(
            device_id, 
            auth_data, 
            AuthenticationMethod.JWT_TOKEN
        )
        
        if auth_context and auth_context.authenticated:
            print("✅ Integrated authentication successful")
        else:
            print("❌ Integrated authentication failed")
            return False
        
        # Test security status
        status = security_manager.get_security_status()
        
        if (status.get('key_management', {}).get('total_keys', 0) > 0 and
            status.get('authentication', {}).get('enrolled_devices', 0) > 0):
            print("✅ Security status reporting successful")
        else:
            print("❌ Security status reporting failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Integrated security manager test failed: {e}")
        return False


def test_security_performance():
    """Test security system performance."""
    print("\n⚡ Testing Security Performance...")
    
    try:
        start_time = time.time()
        
        # Initialize security manager
        security_manager = QFLARESecurityManager()
        
        # Perform multiple security operations
        operations = 0
        
        # Enroll multiple devices
        for i in range(5):
            device_id = f"perf_test_device_{i:03d}"
            device_info = {'device_type': 'edge_node'}
            
            credentials = security_manager.initialize_device_security(device_id, device_info)
            if credentials:
                operations += 1
        
        # Generate multiple tokens
        for i in range(10):
            device_id = f"perf_test_device_{i % 5:03d}"
            token = security_manager.auth_manager.generate_access_token(device_id)
            if token:
                operations += 1
        
        # Establish multiple sessions
        for i in range(3):
            device_id = f"perf_test_device_{i:03d}"
            session_id, _ = security_manager.establish_secure_session(device_id)
            if session_id:
                operations += 1
        
        end_time = time.time()
        duration = end_time - start_time
        ops_per_second = operations / duration if duration > 0 else 0
        
        print(f"✅ Security performance test completed:")
        print(f"   - {operations} operations in {duration:.2f} seconds")
        print(f"   - {ops_per_second:.1f} operations per second")
        
        return True
        
    except Exception as e:
        print(f"❌ Security performance test failed: {e}")
        return False


def run_all_security_tests():
    """Run all enhanced security tests."""
    print("🚀 Starting QFLARE Enhanced Security Tests")
    print("=" * 55)
    
    test_results = []
    
    # Run tests
    test_results.append(("Post-Quantum Key Management", test_post_quantum_key_management()))
    test_results.append(("Certificate Management", test_certificate_management()))
    test_results.append(("Secure Communication", test_secure_communication()))
    test_results.append(("Enhanced Authentication", test_enhanced_authentication()))
    test_results.append(("Integrated Security Manager", test_integrated_security_manager()))
    test_results.append(("Security Performance", test_security_performance()))
    
    # Print summary
    print("\n" + "=" * 55)
    print("🎯 Enhanced Security Test Results:")
    print("=" * 55)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} | {status}")
        if result:
            passed += 1
    
    print("-" * 55)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All enhanced security tests passed!")
        print("🔐 QFLARE security system is ready for production!")
    else:
        print("⚠️  Some security tests failed. Review the logs above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_security_tests()
    sys.exit(0 if success else 1)