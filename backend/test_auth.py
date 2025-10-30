#!/usr/bin/env python3
"""
Quick test script to verify authentication endpoints are working
"""

import requests
import json

def test_authentication():
    """Test the authentication endpoints"""
    base_url = "http://localhost:8000"
    
    print("🔍 Testing QFLARE Authentication Endpoints")
    print("=" * 50)
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test 2: Admin login
    try:
        login_data = {
            "username": "admin",
            "password": "admin123"
        }
        response = requests.post(f"{base_url}/api/auth/login", json=login_data)
        print(f"✅ Admin login: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Token: {data['access_token'][:20]}...")
            print(f"   User: {data['user']['username']} ({data['user']['role']})")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Admin login failed: {e}")
    
    # Test 3: User login
    try:
        login_data = {
            "username": "user",
            "password": "user123"
        }
        response = requests.post(f"{base_url}/api/auth/login", json=login_data)
        print(f"✅ User login: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Token: {data['access_token'][:20]}...")
            print(f"   User: {data['user']['username']} ({data['user']['role']})")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ User login failed: {e}")
    
    # Test 4: Invalid credentials
    try:
        login_data = {
            "username": "invalid",
            "password": "invalid"
        }
        response = requests.post(f"{base_url}/api/auth/login", json=login_data)
        print(f"✅ Invalid login test: {response.status_code}")
        if response.status_code == 401:
            print("   ✅ Correctly rejected invalid credentials")
        else:
            print(f"   ⚠️ Unexpected response: {response.text}")
    except Exception as e:
        print(f"❌ Invalid login test failed: {e}")
    
    # Test 5: List available endpoints
    try:
        response = requests.get(f"{base_url}/api/docs")
        print(f"✅ API Documentation: {response.status_code}")
        if response.status_code == 200:
            print("   📚 API docs available at http://localhost:8000/api/docs")
    except Exception as e:
        print(f"ℹ️  API docs: {e}")

if __name__ == "__main__":
    test_authentication()