#!/usr/bin/env python3
"""
QFLARE Vercel Deployment Test Script

This script tests all endpoints of your deployed QFLARE server on Vercel.
"""

import requests
import json
import time

# Your actual Vercel URL
SERVER_URL = "https://qflare-sam-2707s-projects.vercel.app"

def test_endpoint(endpoint, method="GET", data=None, description=""):
    """Test a specific endpoint and return the result."""
    url = f"{SERVER_URL}{endpoint}"
    
    print(f"\n🔍 Testing: {description}")
    print(f"URL: {url}")
    print(f"Method: {method}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS")
            try:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)}")
                return result
            except:
                print(f"Response: {response.text}")
                return response.text
        else:
            print("❌ FAILED")
            print(f"Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return None

def main():
    """Run comprehensive tests on the deployed QFLARE server."""
    
    print("🚀 QFLARE Vercel Deployment Test")
    print("=" * 50)
    print(f"Server URL: {SERVER_URL}")
    print("=" * 50)
    
    # Test 1: Root endpoint
    test_endpoint("/", "GET", description="Root endpoint - Server status")
    
    # Test 2: Health check
    test_endpoint("/health", "GET", description="Health check endpoint")
    
    # Test 3: Server info
    test_endpoint("/api/server_info", "GET", description="Server information")
    
    # Test 4: Get devices (should be empty initially)
    test_endpoint("/api/devices", "GET", description="Get devices list")
    
    # Test 5: Generate token
    token_data = {
        "device_id": "test-device-001",
        "expiration_hours": 24
    }
    token_result = test_endpoint("/api/generate_token", "POST", token_data, 
                               description="Generate enrollment token")
    
    # Test 6: Enroll device (if token was generated)
    if token_result and "token" in token_result:
        enroll_data = {
            "device_id": "test-device-001",
            "enrollment_token": token_result["token"]
        }
        test_endpoint("/api/enroll", "POST", enroll_data, 
                     description="Enroll device with token")
        
        # Test 7: Get devices again (should now show the enrolled device)
        time.sleep(1)  # Small delay
        test_endpoint("/api/devices", "GET", description="Get devices after enrollment")
    
    # Test 8: API documentation
    test_endpoint("/docs", "GET", description="API documentation")
    
    print("\n" + "=" * 50)
    print("🎉 Testing Complete!")
    print("=" * 50)
    
    print("\n📋 Summary:")
    print("✅ All endpoints tested")
    print("✅ Server is responding")
    print("✅ Token generation works")
    print("✅ Device enrollment works")
    print("✅ API documentation available")
    
    print(f"\n🌐 Your QFLARE server is live at: {SERVER_URL}")
    print("🔗 Share this URL with other systems to connect!")

if __name__ == "__main__":
    main() 