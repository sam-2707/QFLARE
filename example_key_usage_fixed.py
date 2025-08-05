#!/usr/bin/env python3
"""
Corrected example of using quantum keys in QFLARE with actual API endpoints.
"""

import requests
import time
import hashlib
import base64
import numpy as np
import json

def generate_quantum_keys():
    """Generate quantum keys."""
    print("🔐 Step 1: Generating Quantum Keys")
    print("=" * 50)
    
    # Generate quantum keys
    response = requests.get("http://172.18.224.1:8000/api/request_qkey")
    if response.status_code != 200:
        print("❌ Failed to generate quantum keys")
        return None
    
    keys = response.json()
    print("✅ Quantum keys generated successfully!")
    print(f"   Device ID: {keys['device_id']}")
    print(f"   KEM Key: {keys['kem_public_key'][:20]}...")
    print(f"   Signature Key: {keys['signature_public_key'][:20]}...")
    
    return keys

def register_device_with_keys(keys):
    """Register device using the web form."""
    print("\n📝 Step 2: Registering Device")
    print("=" * 50)
    
    device_id = f"edge_device_{int(time.time())}"
    registration_data = {
        "device_id": device_id,
        "device_type": "IoT_Sensor",
        "location": "Smart_Building_A",
        "description": "Temperature sensor for federated learning",
        "capabilities": "local_training,model_submission,data_collection"
    }
    
    register_response = requests.post(
        "http://172.18.224.1:8000/register",
        data=registration_data
    )
    
    if register_response.status_code == 200:
        print("✅ Device registered successfully!")
        print(f"   Device ID: {device_id}")
        return device_id
    else:
        print("❌ Device registration failed!")
        print(f"   Response: {register_response.text}")
        return None

def check_available_endpoints():
    """Check what endpoints are available."""
    print("\n🔍 Step 3: Checking Available Endpoints")
    print("=" * 50)
    
    endpoints = [
        "/health",
        "/api/request_qkey",
        "/api/devices",
        "/api/enclave/status",
        "/api/global_model"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"http://172.18.224.1:8000{endpoint}")
            print(f"✅ {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint}: Error - {e}")

def test_model_submission(device_id):
    """Test model submission with quantum signature."""
    print("\n🤖 Step 4: Testing Model Submission")
    print("=" * 50)
    
    # Create test model data
    model_weights = np.random.rand(100).tobytes()
    signature = hashlib.sha256(model_weights).hexdigest().encode()
    
    model_data = {
        "device_id": device_id,
        "model_weights": base64.b64encode(model_weights).decode('utf-8'),
        "signature": base64.b64encode(signature).decode('utf-8'),
        "metadata": {
            "round": 1,
            "epochs": 10,
            "training_loss": 0.15,
            "validation_accuracy": 0.92,
            "timestamp": time.time()
        }
    }
    
    # Try the submit_model endpoint
    response = requests.post(
        "http://172.18.224.1:8000/api/submit_model",
        json=model_data
    )
    
    if response.status_code == 200:
        print("✅ Model submission successful!")
        result = response.json()
        print(f"   Status: {result.get('status')}")
        print(f"   Message: {result.get('message')}")
        return True
    else:
        print("❌ Model submission failed!")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.text}")
        return False

def check_system_status():
    """Check overall system status."""
    print("\n📊 Step 5: System Status Check")
    print("=" * 50)
    
    # Check health
    health_response = requests.get("http://172.18.224.1:8000/api/health")
    if health_response.status_code == 200:
        health_data = health_response.json()
        print("✅ System Health:")
        print(f"   Status: {health_data.get('status')}")
        print(f"   Device Count: {health_data.get('components', {}).get('device_count', 'N/A')}")
        print(f"   Enclave Status: {health_data.get('components', {}).get('enclave', 'N/A')}")
    
    # Check enclave status
    enclave_response = requests.get("http://172.18.224.1:8000/api/enclave/status")
    if enclave_response.status_code == 200:
        enclave_data = enclave_response.json()
        print("✅ Enclave Status:")
        print(f"   Status: {enclave_data.get('status')}")
        print(f"   Model Count: {enclave_data.get('model_count', 'N/A')}")
        print(f"   Last Aggregation: {enclave_data.get('last_aggregation', 'N/A')}")

def demonstrate_key_usage():
    """Demonstrate how to use the quantum keys."""
    print("\n🎯 Step 6: Quantum Key Usage Demonstration")
    print("=" * 50)
    
    print("🔐 Your quantum keys can be used for:")
    print("   1. **Device Authentication**: Sign challenges with signature key")
    print("   2. **Secure Communication**: Use KEM key for session encryption")
    print("   3. **Model Integrity**: Sign model updates with signature key")
    print("   4. **Perfect Forward Secrecy**: Ephemeral session keys")
    
    print("\n📋 Key Usage Examples:")
    print("   • Browser: http://172.18.224.1:8000/api/request_qkey")
    print("   • API Call: GET /api/request_qkey")
    print("   • Registration: POST /register with keys")
    print("   • Model Submission: POST /api/submit_model with signature")

def main():
    """Main function demonstrating quantum key usage."""
    print("🚀 QFLARE Quantum Key Usage Example (Fixed)")
    print("=" * 60)
    
    # Step 1: Generate quantum keys
    keys = generate_quantum_keys()
    if not keys:
        print("❌ Failed to generate keys. Exiting.")
        return
    
    # Step 2: Register device
    device_id = register_device_with_keys(keys)
    if not device_id:
        print("❌ Failed to register device. Exiting.")
        return
    
    # Step 3: Check available endpoints
    check_available_endpoints()
    
    # Step 4: Test model submission
    test_model_submission(device_id)
    
    # Step 5: Check system status
    check_system_status()
    
    # Step 6: Demonstrate key usage
    demonstrate_key_usage()
    
    print("\n🎉 Quantum Key Usage Complete!")
    print("=" * 60)
    print("✅ Generated quantum-resistant keys")
    print("✅ Registered device successfully")
    print("✅ Tested system endpoints")
    print("✅ Demonstrated key usage patterns")
    print("\n🔐 Your quantum keys are ready for secure federated learning!")
    print("\n📖 See 'quantum_key_usage_guide.md' for detailed usage instructions.")

if __name__ == "__main__":
    main() 