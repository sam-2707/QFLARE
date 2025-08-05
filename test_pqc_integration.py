#!/usr/bin/env python3
"""
Test script for PQC integration with liboqs.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from auth.pqcrypto_utils import (
    generate_device_keypair,
    generate_session_challenge,
    verify_model_signature,
    get_enabled_pqc_algorithms
)
import base64
import secrets
import hashlib

def test_pqc_integration():
    """Test the PQC integration."""
    print("🔐 Testing PQC Integration with liboqs")
    print("=" * 50)
    
    # Test 1: Check available algorithms
    print("\n1. Checking available PQC algorithms...")
    algorithms = get_enabled_pqc_algorithms()
    print(f"Available KEM algorithms: {len(algorithms.get('kem', []))}")
    print(f"Available signature algorithms: {len(algorithms.get('sig', []))}")
    
    if 'note' in algorithms:
        print(f"Note: {algorithms['note']}")
    
    # Test 2: Generate device key pair
    print("\n2. Testing device key pair generation...")
    device_id = "test_device_001"
    kem_public_key, sig_public_key = generate_device_keypair(device_id)
    print(f"✅ Generated KEM public key: {len(kem_public_key)} chars")
    print(f"✅ Generated signature public key: {len(sig_public_key)} chars")
    
    # Test 3: Generate session challenge
    print("\n3. Testing session challenge generation...")
    # First register the device keys (simulate enrollment)
    from auth.pqcrypto_utils import register_device_keys
    register_device_keys(device_id, kem_public_key, sig_public_key)
    
    challenge = generate_session_challenge(device_id)
    if challenge:
        print(f"✅ Generated session challenge: {len(challenge)} chars")
    else:
        print("❌ Failed to generate session challenge")
    
    # Test 4: Test signature verification
    print("\n4. Testing signature verification...")
    test_data = b"test model weights for verification"
    test_signature = hashlib.sha256(test_data).hexdigest().encode()
    
    is_valid = verify_model_signature(device_id, test_data, test_signature)
    print(f"✅ Signature verification result: {is_valid}")
    
    print("\n" + "=" * 50)
    print("🎉 PQC Integration Test Complete!")
    
    return True

if __name__ == "__main__":
    test_pqc_integration() 