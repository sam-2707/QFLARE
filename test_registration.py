#!/usr/bin/env python3
"""
Test script for device registration functionality.
"""

import requests
import json
import time

def test_registration():
    """Test the device registration functionality."""
    print("🧪 Testing Device Registration")
    print("=" * 50)
    
    base_url = "http://172.18.224.1:8000"
    
    # Test 1: Check if registration page loads
    print("\n1. Testing registration page access...")
    try:
        response = requests.get(f"{base_url}/register", timeout=5)
        if response.status_code == 200:
            print("✅ Registration page loads successfully")
        else:
            print(f"❌ Registration page failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Registration page error: {e}")
        return False
    
    # Test 2: Test device registration
    print("\n2. Testing device registration...")
    try:
        # Test data with simple device ID
        test_device_id = f"test_device_{int(time.time())}"
        
        form_data = {
            "device_id": test_device_id,
            "device_type": "edge",
            "location": "Test Lab",
            "description": "Test device for registration",
            "capabilities": "CPU: 4 cores, Memory: 8GB, Sensors: temperature, humidity"
        }
        
        response = requests.post(f"{base_url}/register", data=form_data, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ Device '{test_device_id}' registered successfully")
            print(f"   Response length: {len(response.text)} characters")
            
            # Check if success message is in response
            if "registered successfully" in response.text.lower():
                print("✅ Success message found in response")
            else:
                print("⚠️  Success message not found in response")
        else:
            print(f"❌ Registration failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return False
    
    # Test 3: Check if device appears in device list
    print("\n3. Testing device list...")
    try:
        response = requests.get(f"{base_url}/devices", timeout=5)
        if response.status_code == 200:
            print("✅ Device list page loads successfully")
            
            # Check if our test device is in the list
            if test_device_id in response.text:
                print(f"✅ Test device '{test_device_id}' found in device list")
            else:
                print(f"⚠️  Test device '{test_device_id}' not found in device list")
        else:
            print(f"❌ Device list failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Device list error: {e}")
    
    # Test 4: Test health endpoint
    print("\n4. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            device_count = health_data.get("device_count", 0)
            print(f"✅ Health check successful - Device count: {device_count}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Device Registration Test Complete!")
    print("\n📋 Next Steps:")
    print("   1. Open http://172.18.224.1:8000/register in your browser")
    print("   2. Fill out the registration form")
    print("   3. Submit the form to register a device")
    print("   4. Check http://172.18.224.1:8000/devices to see registered devices")
    
    return True

if __name__ == "__main__":
    test_registration() 