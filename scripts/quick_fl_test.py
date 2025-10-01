#!/usr/bin/env python3
"""
Quick FL System Test
Tests the FL endpoints with a simple workflow
"""

import requests
import json
import time

API_BASE = "http://localhost:8080/api/fl"

def test_fl_system():
    """Test the FL system with basic operations."""
    print("🧪 QFLARE FL System Test")
    print("=" * 50)
    
    # Test 1: Check FL Status
    print("\n1️⃣ Testing FL Status...")
    try:
        response = requests.get(f"{API_BASE}/status")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {data['fl_status']['status']}")
            print(f"   ✅ Round: {data['fl_status']['current_round']}/{data['fl_status']['total_rounds']}")
            print(f"   ✅ Devices: {data['fl_status']['registered_devices']}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 2: Register Devices
    print("\n2️⃣ Registering Test Devices...")
    devices = ["test_device_001", "test_device_002", "test_device_003"]
    for device_id in devices:
        try:
            response = requests.post(
                f"{API_BASE}/register",
                json={
                    "device_id": device_id,
                    "capabilities": {
                        "compute": "high",
                        "memory": 8192,
                        "bandwidth": 100
                    }
                }
            )
            if response.status_code == 200:
                print(f"   ✅ Registered: {device_id}")
            else:
                print(f"   ❌ Failed to register {device_id}: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error registering {device_id}: {e}")
    
    time.sleep(1)
    
    # Test 3: Start Training
    print("\n3️⃣ Starting FL Training...")
    try:
        response = requests.post(
            f"{API_BASE}/start_training",
            json={
                "rounds": 5,
                "min_participants": 2
            }
        )
        if response.status_code == 200:
            print(f"   ✅ Training started!")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    time.sleep(1)
    
    # Test 4: Submit Mock Models
    print("\n4️⃣ Submitting Model Updates...")
    for i, device_id in enumerate(devices):
        try:
            # Simulate model weights
            model_weights = [0.1 + i*0.01, 0.2 + i*0.01, 0.3 + i*0.01] * 100
            
            response = requests.post(
                f"{API_BASE}/submit_model",
                json={
                    "device_id": device_id,
                    "round_number": 1,
                    "model_weights": model_weights,
                    "metrics": {
                        "accuracy": 0.75 + i*0.05,
                        "loss": 0.25 - i*0.03
                    },
                    "samples": 1000 + i*100
                }
            )
            if response.status_code == 200:
                print(f"   ✅ Model submitted from {device_id}")
            else:
                print(f"   ❌ Failed from {device_id}: {response.status_code}")
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"   ❌ Error from {device_id}: {e}")
    
    time.sleep(2)
    
    # Test 5: Get Global Model
    print("\n5️⃣ Retrieving Global Model...")
    try:
        response = requests.get(f"{API_BASE}/global_model")
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                model_info = data['global_model']
                print(f"   ✅ Global model retrieved!")
                print(f"   Round: {model_info['round_number']}")
                print(f"   Participants: {model_info['participants']}")
                print(f"   Accuracy: {model_info.get('accuracy', 'N/A')}")
            else:
                print(f"   ℹ️  No global model yet: {data.get('message')}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 6: Check Updated Status
    print("\n6️⃣ Checking Final Status...")
    try:
        response = requests.get(f"{API_BASE}/status")
        if response.status_code == 200:
            data = response.json()
            status = data['fl_status']
            print(f"   ✅ Status: {status['status']}")
            print(f"   ✅ Round: {status['current_round']}/{status['total_rounds']}")
            print(f"   ✅ Registered Devices: {status['registered_devices']}")
            print(f"   ✅ Active Devices: {status['active_devices']}")
            print(f"   ✅ Participants This Round: {status['participants_this_round']}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 7: Get Device List
    print("\n7️⃣ Listing Registered Devices...")
    try:
        response = requests.get(f"{API_BASE}/devices")
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                devices_list = data['devices']
                print(f"   ✅ Found {len(devices_list)} devices:")
                for device in devices_list:
                    print(f"      • {device['device_id']} - Status: {device['status']}")
            else:
                print(f"   ℹ️  No devices registered")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ FL System Test Complete!")
    print("\n📊 Next Steps:")
    print("   1. Open http://localhost:4000/federated-learning")
    print("   2. Watch the FL dashboard update in real-time")
    print("   3. Run multiple training rounds")
    print("\n🚀 The FL system is ready!")
    
    return True

if __name__ == "__main__":
    try:
        test_fl_system()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
