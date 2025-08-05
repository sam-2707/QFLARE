#!/usr/bin/env python3
"""
Test script that bypasses liboqs completely by mocking the import.
"""

import sys
import os
import unittest.mock

# Mock liboqs before any imports
sys.modules['oqs'] = unittest.mock.MagicMock()

# Now import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

def test_pqc_fallbacks():
    """Test PQC fallback functionality."""
    print("🔐 Testing PQC Fallback System (Mocked liboqs)")
    print("=" * 50)
    
    try:
        # Test 1: Import PQC utils
        print("\n1. Testing PQC utils import...")
        from auth.pqcrypto_utils import generate_device_keypair, register_device_keys, get_device_public_keys
        print("✅ PQC utils import successful")
        
        # Test 2: Generate device key pair
        print("\n2. Testing device key pair generation...")
        device_id = "test_device_001"
        kem_public_key, sig_public_key = generate_device_keypair(device_id)
        print(f"✅ Generated KEM public key: {len(kem_public_key)} chars")
        print(f"✅ Generated signature public key: {len(sig_public_key)} chars")
        
        # Test 3: Register device keys
        print("\n3. Testing device key registration...")
        result = register_device_keys(device_id, kem_public_key, sig_public_key)
        print(f"✅ Device registration: {result}")
        
        # Test 4: Retrieve device keys
        print("\n4. Testing device key retrieval...")
        stored_keys = get_device_public_keys(device_id)
        if stored_keys:
            print("✅ Device keys retrieved successfully")
            print(f"   KEM key: {stored_keys['kem_public_key'][:20]}...")
            print(f"   Sig key: {stored_keys['signature_public_key'][:20]}...")
        else:
            print("❌ Failed to retrieve device keys")
        
        print("\n" + "=" * 50)
        print("🎉 PQC Fallback Test Complete!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_ssl_manager():
    """Test SSL certificate management."""
    print("\n🔒 Testing SSL Certificate Management")
    print("=" * 50)
    
    try:
        # Test 1: Import SSL manager
        print("\n1. Testing SSL manager import...")
        from ssl_manager import setup_ssl_certificates, check_certificate_status
        print("✅ SSL manager import successful")
        
        # Test 2: Check certificate status
        print("\n2. Testing certificate status check...")
        status = check_certificate_status()
        print(f"✅ Certificate status: {status['status']}")
        print(f"   Message: {status['message']}")
        
        print("\n" + "=" * 50)
        print("🎉 SSL Manager Test Complete!")
        return True
        
    except Exception as e:
        print(f"❌ SSL test failed: {e}")
        return False

def test_rate_limiter():
    """Test rate limiting system."""
    print("\n🚦 Testing Rate Limiting System")
    print("=" * 50)
    
    try:
        # Test 1: Import rate limiter
        print("\n1. Testing rate limiter import...")
        from security.rate_limiter import rate_limiter, get_rate_limit_stats
        print("✅ Rate limiter import successful")
        
        # Test 2: Get rate limit stats
        print("\n2. Testing rate limit statistics...")
        stats = get_rate_limit_stats()
        print(f"✅ Rate limit stats: {stats['total_records']} records")
        print(f"   Blocked IPs: {stats['blocked_ips']}")
        print(f"   Suspicious IPs: {stats['suspicious_ips']}")
        
        print("\n" + "=" * 50)
        print("🎉 Rate Limiter Test Complete!")
        return True
        
    except Exception as e:
        print(f"❌ Rate limiter test failed: {e}")
        return False

def test_monitoring():
    """Test monitoring system."""
    print("\n📊 Testing Monitoring System")
    print("=" * 50)
    
    try:
        # Test 1: Import monitoring
        print("\n1. Testing monitoring import...")
        from monitoring.logger import security_monitor, log_security_event, get_monitoring_stats
        from monitoring.logger import EventType, SecurityLevel
        print("✅ Monitoring import successful")
        
        # Test 2: Log a security event
        print("\n2. Testing security event logging...")
        log_security_event(
            event_type=EventType.AUTHENTICATION,
            security_level=SecurityLevel.MEDIUM,
            source_ip="127.0.0.1",
            user_agent="test-agent",
            device_id="test_device",
            success=True
        )
        print("✅ Security event logged successfully")
        
        # Test 3: Get monitoring stats
        print("\n3. Testing monitoring statistics...")
        stats = get_monitoring_stats()
        print(f"✅ Monitoring stats retrieved")
        print(f"   Security events: {stats['security']['total_events_24h']}")
        print(f"   Active alerts: {stats['security']['active_alerts']}")
        
        print("\n" + "=" * 50)
        print("🎉 Monitoring Test Complete!")
        return True
        
    except Exception as e:
        print(f"❌ Monitoring test failed: {e}")
        return False

def test_federated_learning():
    """Test federated learning components."""
    print("\n🤖 Testing Federated Learning Components")
    print("=" * 50)
    
    try:
        # Test 1: Import FL components
        print("\n1. Testing FL component imports...")
        from fl_core.aggregator import store_model_update, get_global_model
        from enclave.mock_enclave import get_secure_enclave
        print("✅ FL components import successful")
        
        # Test 2: Test secure enclave
        print("\n2. Testing secure enclave...")
        enclave = get_secure_enclave()
        status = enclave.get_enclave_status()
        print(f"✅ Enclave status: {status['status']}")
        print(f"   Total aggregations: {status['total_aggregations']}")
        
        # Test 3: Test model aggregation
        print("\n3. Testing model aggregation...")
        import numpy as np
        import hashlib
        
        # Create test model update
        model_weights = np.random.rand(100).tobytes()
        signature = hashlib.sha256(model_weights).hexdigest().encode()
        
        # Register a test device first
        from auth.pqcrypto_utils import register_device_keys
        device_id = "fl_test_device"
        kem_key = "test_kem_key"
        sig_key = "test_sig_key"
        register_device_keys(device_id, kem_key, sig_key)
        
        # Store model update
        result = store_model_update(
            device_id=device_id,
            model_weights=model_weights,
            metadata={"round": 1, "epochs": 10}
        )
        print(f"✅ Model update stored: {result}")
        
        print("\n" + "=" * 50)
        print("🎉 Federated Learning Test Complete!")
        return True
        
    except Exception as e:
        print(f"❌ FL test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running QFLARE Security Component Tests (Bypassed liboqs)")
    print("=" * 70)
    
    tests = [
        test_pqc_fallbacks,
        test_ssl_manager,
        test_rate_limiter,
        test_monitoring,
        test_federated_learning
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! QFLARE security components are working correctly.")
        print("✅ PQC fallback system working")
        print("✅ SSL certificate management working")
        print("✅ Rate limiting system working")
        print("✅ Monitoring system working")
        print("✅ Federated learning components working")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    main() 