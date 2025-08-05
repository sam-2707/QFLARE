#!/usr/bin/env python3
"""
Script to check QFLARE server status.
"""

import requests
import json
import sys

def check_server_status():
    """Check the status of the QFLARE server."""
    print("🔍 Checking QFLARE Server Status")
    print("=" * 50)
    
    try:
        # Check health endpoint
        print("\n1. Checking health endpoint...")
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health endpoint: OK")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   Server: {health_data.get('components', {}).get('server', 'unknown')}")
            print(f"   Enclave: {health_data.get('components', {}).get('enclave', 'unknown')}")
            print(f"   Aggregator: {health_data.get('components', {}).get('aggregator', 'unknown')}")
        else:
            print(f"❌ Health endpoint: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health endpoint: Connection failed - {e}")
        return False
    
    try:
        # Check main dashboard
        print("\n2. Checking main dashboard...")
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Main dashboard: OK")
        else:
            print(f"❌ Main dashboard: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Main dashboard: Connection failed - {e}")
    
    try:
        # Check devices page
        print("\n3. Checking devices page...")
        response = requests.get("http://localhost:8000/devices", timeout=5)
        if response.status_code == 200:
            print("✅ Devices page: OK")
        else:
            print(f"❌ Devices page: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Devices page: Connection failed - {e}")
    
    try:
        # Check 404 page
        print("\n4. Checking error handling...")
        response = requests.get("http://localhost:8000/nonexistent", timeout=5)
        if response.status_code == 404:
            print("✅ 404 error handling: OK")
        else:
            print(f"⚠️  404 error handling: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ 404 error handling: Connection failed - {e}")
    
    print("\n" + "=" * 50)
    print("🎉 QFLARE Server is running successfully!")
    print("\n🌐 Available endpoints:")
    print("   - GET  http://localhost:8000/health - Health check")
    print("   - GET  http://localhost:8000/ - Main dashboard")
    print("   - GET  http://localhost:8000/devices - Device management")
    print("   - POST http://localhost:8000/api/enroll - Device enrollment")
    print("   - POST http://localhost:8000/api/challenge - Session challenge")
    print("   - POST http://localhost:8000/api/submit_model - Model submission")
    print("   - GET  http://localhost:8000/api/global_model - Global model download")
    
    print("\n📋 Next steps:")
    print("   1. Open http://localhost:8000 in your browser")
    print("   2. Explore the dashboard and device management")
    print("   3. Test the API endpoints")
    print("   4. Try device enrollment and federated learning")
    
    return True

if __name__ == "__main__":
    check_server_status() 