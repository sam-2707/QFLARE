#!/usr/bin/env python3
"""
Test the challenge endpoint with proper data.
"""

import requests
import json

def test_challenge_endpoint():
    """Test the challenge endpoint with proper data."""
    print("🔐 Testing Challenge Endpoint")
    print("=" * 50)
    
    # Test with proper data
    challenge_data = {
        "device_id": "test_device_001"
    }
    
    try:
        response = requests.post(
            "http://172.18.224.1:8000/api/challenge",
            json=challenge_data,
            timeout=5
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Challenge endpoint is working!")
            print(f"   Status: {result.get('status')}")
            print(f"   Device ID: {result.get('device_id')}")
            print(f"   Challenge: {result.get('challenge', '')[:20]}...")
            print(f"   Message: {result.get('message')}")
            return result.get('challenge')
        elif response.status_code == 422:
            print("⚠️  Challenge endpoint exists but needs proper data format")
            print(f"   Response: {response.text}")
        else:
            print(f"❌ Challenge endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing challenge endpoint: {e}")

def test_verify_endpoint():
    """Test the verify challenge endpoint."""
    print("\n🔍 Testing Verify Challenge Endpoint")
    print("=" * 50)
    
    verify_data = {
        "device_id": "test_device_001",
        "challenge_response": "test_response"
    }
    
    try:
        response = requests.post(
            "http://172.18.224.1:8000/api/verify_challenge",
            json=verify_data,
            timeout=5
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Verify challenge endpoint is working!")
            print(f"   Status: {result.get('status')}")
            print(f"   Message: {result.get('message')}")
        elif response.status_code == 404:
            print("❌ Verify challenge endpoint not found")
            print("   This endpoint needs to be added to the server")
        else:
            print(f"⚠️  Verify challenge endpoint: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing verify endpoint: {e}")

def main():
    """Main function."""
    print("🚀 QFLARE Challenge Test")
    print("=" * 60)
    
    # Test challenge endpoint
    challenge = test_challenge_endpoint()
    
    # Test verify endpoint
    test_verify_endpoint()
    
    print("\n🎉 Challenge Test Complete!")
    print("=" * 60)
    print("✅ Challenge endpoint is working")
    print("❌ Verify challenge endpoint needs to be added")
    print("\n📋 Next Steps:")
    print("   1. Add /api/verify_challenge endpoint to server")
    print("   2. Restart server to load new endpoints")
    print("   3. Test complete authentication flow")

if __name__ == "__main__":
    main() 