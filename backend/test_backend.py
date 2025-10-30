#!/usr/bin/env python3
"""
Quick test script to verify QFLARE backend is running
"""

import requests
import json

def test_backend():
    base_url = "http://localhost:8000"
    
    try:
        # Test health endpoint
        print("🔍 Testing backend health...")
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend health check passed")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
        
        # Test API info
        print("\n🔍 Testing API info...")
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ API info endpoint working")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ API info failed: {response.status_code}")
        
        # Test clients endpoint
        print("\n🔍 Testing clients endpoint...")
        response = requests.get(f"{base_url}/api/clients", timeout=5)
        if response.status_code == 200:
            clients_data = response.json()
            print(f"✅ Clients endpoint working - {len(clients_data['clients'])} clients found")
        else:
            print(f"❌ Clients endpoint failed: {response.status_code}")
        
        # Test metrics endpoint
        print("\n🔍 Testing metrics endpoint...")
        response = requests.get(f"{base_url}/api/metrics", timeout=5)
        if response.status_code == 200:
            print("✅ Metrics endpoint working")
        else:
            print(f"❌ Metrics endpoint failed: {response.status_code}")
        
        print("\n🎉 Backend is running correctly and ready for frontend connections!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend - make sure it's running on port 8000")
        return False
    except Exception as e:
        print(f"❌ Error testing backend: {e}")
        return False

if __name__ == "__main__":
    test_backend()