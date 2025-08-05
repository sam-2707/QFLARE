#!/usr/bin/env python3
"""
Test script for quantum key generation.
"""

import requests
import json

def test_quantum_key_generation():
    """Test quantum key generation endpoint."""
    print("🔐 Testing Quantum Key Generation")
    print("=" * 50)
    
    try:
        # Test the quantum key generation endpoint
        response = requests.get("http://172.18.224.1:8000/api/request_qkey", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Quantum key generation successful!")
            print(f"   Status: {data.get('status')}")
            print(f"   Device ID: {data.get('device_id')}")
            print(f"   KEM Key Length: {len(data.get('kem_public_key', ''))}")
            print(f"   Signature Key Length: {len(data.get('signature_public_key', ''))}")
            print(f"   Message: {data.get('message')}")
            
            # Verify the keys are base64 encoded
            import base64
            try:
                base64.b64decode(data.get('kem_public_key', ''))
                base64.b64decode(data.get('signature_public_key', ''))
                print("✅ Keys are properly base64 encoded")
            except Exception as e:
                print(f"❌ Key encoding error: {e}")
                
        else:
            print(f"❌ Quantum key generation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing quantum key generation: {e}")

if __name__ == "__main__":
    test_quantum_key_generation() 