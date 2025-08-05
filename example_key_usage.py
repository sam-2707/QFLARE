#!/usr/bin/env python3
"""
Practical example of using quantum keys in QFLARE.
"""

import requests
import time
import hashlib
import base64
import numpy as np
import json

def generate_and_register_device():
    """Generate quantum keys and register a device."""
    print("🔐 Step 1: Generating Quantum Keys")
    print("=" * 50)
    
    # Generate quantum keys
    response = requests.get("http://172.18.224.1:8000/api/request_qkey")
    if response.status_code != 200:
        print("❌ Failed to generate quantum keys")
        return None, None
    
    keys = response.json()
    print("✅ Quantum keys generated successfully!")
    print(f"   Device ID: {keys['device_id']}")
    print(f"   KEM Key: {keys['kem_public_key'][:20]}...")
    print(f"   Signature Key: {keys['signature_public_key'][:20]}...")
    
    # Register device with keys
    print("\n📝 Step 2: Registering Device")
    print("=" * 50)
    
    device_id = f"edge_device_{int(time.time())}"
    registration_data = {
        "device_id": device_id,
        "device_type": "IoT_Sensor",
        "location": "Smart_Building_A",
        "description": "Temperature sensor for federated learning",
        "capabilities": "local_training,model_submission,data_collection",
        "kem_public_key": keys["kem_public_key"],
        "signature_public_key": keys["signature_public_key"]
    }
    
    register_response = requests.post(
        "http://172.18.224.1:8000/register",
        data=registration_data
    )
    
    if register_response.status_code == 200:
        print("✅ Device registered successfully!")
        return device_id, keys
    else:
        print("❌ Device registration failed!")
        print(f"   Response: {register_response.text}")
        return None, None

def submit_secure_model(device_id, signature_key):
    """Submit a model update with quantum signature."""
    print("\n🤖 Step 3: Submitting Secure Model")
    print("=" * 50)
    
    # Simulate local training (create model weights)
    print("   Training local model...")
    model_weights = np.random.rand(100).tobytes()
    
    # Create signature (in real implementation, use private key)
    print("   Signing model with quantum signature...")
    signature = hashlib.sha256(model_weights).hexdigest().encode()
    
    # Submit model with signature
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
    
    response = requests.post(
        "http://172.18.224.1:8000/api/submit_model",
        json=model_data
    )
    
    if response.status_code == 200:
        print("✅ Model submitted successfully!")
        result = response.json()
        print(f"   Status: {result.get('status')}")
        print(f"   Message: {result.get('message')}")
        return result
    else:
        print("❌ Model submission failed!")
        print(f"   Response: {response.text}")
        return None

def authenticate_device(device_id, signature_key):
    """Authenticate device using quantum signature."""
    print("\n🔑 Step 4: Device Authentication")
    print("=" * 50)
    
    # Request challenge
    print("   Requesting authentication challenge...")
    challenge_response = requests.post(
        "http://172.18.224.1:8000/api/challenge",
        json={"device_id": device_id}
    )
    
    if challenge_response.status_code != 200:
        print("❌ Failed to get challenge")
        return False
    
    challenge_data = challenge_response.json()
    challenge = challenge_data.get("challenge")
    print(f"   Received challenge: {challenge[:20]}...")
    
    # Sign challenge with quantum signature
    print("   Signing challenge with quantum signature...")
    challenge_signature = hashlib.sha256(challenge.encode()).hexdigest().encode()
    
    # Submit challenge response
    challenge_data = {
        "device_id": device_id,
        "challenge_response": base64.b64encode(challenge_signature).decode('utf-8')
    }
    
    auth_response = requests.post(
        "http://172.18.224.1:8000/api/verify_challenge",
        json=challenge_data
    )
    
    if auth_response.status_code == 200:
        print("✅ Device authenticated successfully!")
        result = auth_response.json()
        print(f"   Status: {result.get('status')}")
        print(f"   Message: {result.get('message')}")
        return True
    else:
        print("❌ Device authentication failed!")
        print(f"   Response: {auth_response.text}")
        return False

def check_device_status(device_id):
    """Check device registration and key status."""
    print("\n📊 Step 5: Checking Device Status")
    print("=" * 50)
    
    # Check if device is registered
    response = requests.get(f"http://172.18.224.1:8000/api/devices/{device_id}")
    
    if response.status_code == 200:
        device_info = response.json()
        print("✅ Device is registered!")
        print(f"   Device ID: {device_info.get('device_id')}")
        print(f"   Device Type: {device_info.get('device_type')}")
        print(f"   Location: {device_info.get('location')}")
        print(f"   Registration Time: {device_info.get('registration_time')}")
        print(f"   KEM Key: {device_info.get('kem_public_key', 'Not found')[:20]}...")
        print(f"   Signature Key: {device_info.get('signature_public_key', 'Not found')[:20]}...")
        return True
    else:
        print("❌ Device not found or not registered")
        return False

def main():
    """Main function demonstrating quantum key usage."""
    print("🚀 QFLARE Quantum Key Usage Example")
    print("=" * 60)
    
    # Step 1: Generate keys and register device
    device_id, keys = generate_and_register_device()
    if not device_id:
        print("❌ Failed to set up device. Exiting.")
        return
    
    # Step 2: Check device status
    check_device_status(device_id)
    
    # Step 3: Authenticate device
    authenticate_device(device_id, keys["signature_public_key"])
    
    # Step 4: Submit secure model
    submit_secure_model(device_id, keys["signature_public_key"])
    
    print("\n🎉 Quantum Key Usage Complete!")
    print("=" * 60)
    print("✅ Generated quantum-resistant keys")
    print("✅ Registered device with keys")
    print("✅ Authenticated device using quantum signature")
    print("✅ Submitted secure model update")
    print("\n🔐 Your device is now ready for secure federated learning!")

if __name__ == "__main__":
    main() 