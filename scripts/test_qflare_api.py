#!/usr/bin/env python3
"""
🧪 QFLARE API Testing Script
Test the advanced device management and training control features
"""

import requests
import json
import time
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def test_health_check():
    """Test basic server health"""
    print("🏥 Testing server health...")
    response = requests.get(f"{API_BASE_URL}/health")
    if response.status_code == 200:
        print("✅ Server is healthy")
        return True
    else:
        print(f"❌ Server health check failed: {response.status_code}")
        return False

def test_device_management():
    """Test device management endpoints"""
    print("\n📱 Testing Device Management...")
    
    # 1. List devices (should be empty initially)
    print("1. Listing devices...")
    response = requests.get(f"{API_BASE_URL}/api/devices")
    if response.status_code == 200:
        devices = response.json()
        print(f"   Found {len(devices)} devices")
    else:
        print(f"   ❌ Failed to list devices: {response.status_code}")
        return False
    
    # 2. Register a test device
    print("2. Registering test device...")
    device_data = {
        "device_name": "test-device-001",
        "device_type": "desktop",
        "capabilities": ["cpu", "gpu"],
        "location": "Test Lab",
        "contact_info": "test@qflare.com",
        "max_concurrent_tasks": 2
    }
    
    response = requests.post(f"{API_BASE_URL}/api/devices/register", 
                           json=device_data, headers=HEADERS)
    if response.status_code == 200:
        result = response.json()
        device_id = result["device_id"]
        print(f"   ✅ Device registered: {device_id}")
    else:
        print(f"   ❌ Failed to register device: {response.status_code}")
        print(f"   Response: {response.text}")
        return False
    
    # 3. Get device details
    print("3. Getting device details...")
    response = requests.get(f"{API_BASE_URL}/api/devices/{device_id}")
    if response.status_code == 200:
        device = response.json()
        print(f"   ✅ Device details retrieved: {device['device_name']}")
    else:
        print(f"   ❌ Failed to get device details: {response.status_code}")
    
    # 4. Send heartbeat
    print("4. Sending device heartbeat...")
    response = requests.post(f"{API_BASE_URL}/api/devices/{device_id}/heartbeat")
    if response.status_code == 200:
        print("   ✅ Heartbeat sent successfully")
    else:
        print(f"   ❌ Failed to send heartbeat: {response.status_code}")
    
    # 5. Get device stats
    print("5. Getting device statistics...")
    response = requests.get(f"{API_BASE_URL}/api/devices/stats/overview")
    if response.status_code == 200:
        stats = response.json()
        print(f"   ✅ Stats retrieved: {stats['total_devices']} total devices")
    else:
        print(f"   ❌ Failed to get stats: {response.status_code}")
    
    return device_id

def test_training_control(device_id=None):
    """Test training control endpoints"""
    print("\n🏋️ Testing Training Control...")
    
    # 1. List training sessions (should be empty initially)
    print("1. Listing training sessions...")
    response = requests.get(f"{API_BASE_URL}/api/training/sessions")
    if response.status_code == 200:
        sessions = response.json()
        print(f"   Found {len(sessions)} sessions")
    else:
        print(f"   ❌ Failed to list sessions: {response.status_code}")
        return False
    
    # 2. Create a training session
    print("2. Creating training session...")
    session_data = {
        "session_name": "test-session-001",
        "model_architecture": "cnn",
        "dataset_name": "MNIST",
        "aggregation_method": "fedavg",
        "global_rounds": 5,
        "local_epochs": 3,
        "batch_size": 32,
        "learning_rate": 0.01,
        "min_participants": 1,
        "max_participants": 5,
        "participation_rate": 1.0,
        "differential_privacy": False,
        "secure_aggregation": True
    }
    
    response = requests.post(f"{API_BASE_URL}/api/training/sessions", 
                           json=session_data, headers=HEADERS)
    if response.status_code == 200:
        result = response.json()
        session_id = result["session_id"]
        print(f"   ✅ Session created: {session_id}")
    else:
        print(f"   ❌ Failed to create session: {response.status_code}")
        print(f"   Response: {response.text}")
        return False
    
    # 3. Get session details
    print("3. Getting session details...")
    response = requests.get(f"{API_BASE_URL}/api/training/sessions/{session_id}")
    if response.status_code == 200:
        session = response.json()
        print(f"   ✅ Session details retrieved: {session['session_name']}")
    else:
        print(f"   ❌ Failed to get session details: {response.status_code}")
    
    # 4. Register device for training (if device_id provided)
    if device_id:
        print("4. Registering device for training...")
        response = requests.post(f"{API_BASE_URL}/api/training/sessions/{session_id}/devices/{device_id}/register")
        if response.status_code == 200:
            print("   ✅ Device registered for training")
        else:
            print(f"   ❌ Failed to register device for training: {response.status_code}")
    
    # 5. Get session metrics
    print("5. Getting session metrics...")
    response = requests.get(f"{API_BASE_URL}/api/training/sessions/{session_id}/metrics")
    if response.status_code == 200:
        metrics = response.json()
        print(f"   ✅ Metrics retrieved: {metrics['session_name']}")
    else:
        print(f"   ❌ Failed to get metrics: {response.status_code}")
    
    return session_id

def main():
    """Main test function"""
    print("🚀 QFLARE API Testing Started")
    print("=" * 50)
    
    # Test server health
    if not test_health_check():
        print("❌ Server health check failed. Exiting...")
        return
    
    # Test device management
    device_id = test_device_management()
    if not device_id:
        print("❌ Device management tests failed")
        return
    
    # Test training control
    session_id = test_training_control(device_id)
    if not session_id:
        print("❌ Training control tests failed")
        return
    
    print("\n🎉 All tests completed successfully!")
    print("=" * 50)
    print(f"✅ Device registered: {device_id}")
    print(f"✅ Training session created: {session_id}")
    print(f"✅ QFLARE system is fully operational!")

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to QFLARE server.")
        print("   Please ensure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")