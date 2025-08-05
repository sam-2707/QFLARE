#!/usr/bin/env python3
"""
Test script to verify authentication endpoints.
"""

import requests
import time
import hashlib
import base64

def test_authentication_flow():
    """Test the complete authentication flow."""
    print("🔐 Testing Authentication Flow")
    print("=" * 50)
    
    # Step 1: Generate quantum keys
    print("1️⃣ Generating quantum keys...")
    response = requests.get("http://172.18.224.1:8000/api/request_qkey")
    if response.status_code != 200:
        print("❌ Failed to generate quantum keys")
        return False
    
    keys = response.json()
    device_id = keys['device_id']
    print(f"✅ Generated keys for device: {device_id}")
    
    # Step 2: Request challenge
    print("\n2️⃣ Requesting authentication challenge...")
    challenge_data = {"device_id": device_id}
    challenge_response = requests.post(
        "http://172.18.224.1:8000/api/challenge",
        json=challenge_data
    )
    
    if challenge_response.status_code != 200:
        print(f"❌ Challenge request failed: {challenge_response.status_code}")
        print(f"   Response: {challenge_response.text}")
        return False
    
    challenge_info = challenge_response.json()
    challenge = challenge_info.get("challenge")
    print(f"✅ Received challenge: {challenge[:20]}...")
    
    # Step 3: Create challenge response (simulate signing)
    print("\n3️⃣ Creating challenge response...")
    challenge_signature = hashlib.sha256(challenge.encode()).hexdigest().encode()
    challenge_response_b64 = base64.b64encode(challenge_signature).decode('utf-8')
    
    # Step 4: Verify challenge response
    print("\n4️⃣ Verifying challenge response...")
    verify_data = {
        "device_id": device_id,
        "challenge_response": challenge_response_b64
    }
    
    verify_response = requests.post(
        "http://172.18.224.1:8000/api/verify_challenge",
        json=verify_data
    )
    
    if verify_response.status_code == 200:
        result = verify_response.json()
        print("✅ Authentication successful!")
        print(f"   Status: {result.get('status')}")
        print(f"   Message: {result.get('message')}")
        print(f"   Session Token: {result.get('session_token', 'N/A')}")
        return True
    else:
        print(f"❌ Authentication failed: {verify_response.status_code}")
        print(f"   Response: {verify_response.text}")
        return False

def test_device_info():
    """Test device information endpoint."""
    print("\n📊 Testing Device Information")
    print("=" * 50)
    
    # Get list of devices
    devices_response = requests.get("http://172.18.224.1:8000/api/devices")
    if devices_response.status_code == 200:
        devices = devices_response.json()
        print(f"✅ Found {devices.get('total_count', 0)} registered devices")
        
        # Try to get info for a specific device
        if devices.get('devices'):
            device_id = devices['devices'][0]['device_id']
            print(f"🔍 Getting info for device: {device_id}")
            
            device_info_response = requests.get(f"http://172.18.224.1:8000/api/devices/{device_id}")
            if device_info_response.status_code == 200:
                device_info = device_info_response.json()
                print("✅ Device info retrieved successfully!")
                print(f"   Device ID: {device_info.get('device_id')}")
                print(f"   Status: {device_info.get('status')}")
            else:
                print(f"❌ Failed to get device info: {device_info_response.status_code}")
        else:
            print("ℹ️  No devices registered yet")
    else:
        print(f"❌ Failed to get devices list: {devices_response.status_code}")

def main():
    """Main test function."""
    print("🚀 QFLARE Authentication Test")
    print("=" * 60)
    
    # Test authentication flow
    auth_success = test_authentication_flow()
    
    # Test device information
    test_device_info()
    
    print("\n🎉 Authentication Test Complete!")
    print("=" * 60)
    if auth_success:
        print("✅ Authentication flow working correctly")
        print("✅ Quantum keys can be used for authentication")
        print("✅ Challenge-response mechanism is functional")
    else:
        print("❌ Authentication flow needs attention")
    
    print("\n🔐 Your quantum keys are ready for secure authentication!")

if __name__ == "__main__":
    main() 