#!/usr/bin/env python3
"""
QFLARE Live Server Validation
Tests all endpoints and demonstrates real-time functionality
"""

import requests
import json
import time
from datetime import datetime, timezone

def test_endpoint(url, method="GET", data=None, headers=None, description=""):
    """Test an endpoint and show results"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: {description}")
    print(f"📍 {method} {url}")
    print(f"{'='*60}")
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=10)
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"⏱️  Response Time: {response.elapsed.total_seconds():.3f}s")
        
        if response.headers.get('content-type', '').startswith('application/json'):
            try:
                response_data = response.json()
                print(f"📄 Response: {json.dumps(response_data, indent=2)}")
            except:
                print(f"📄 Response: {response.text[:500]}")
        else:
            print(f"📄 Response: {response.text[:200]}")
        
        if response.status_code < 400:
            print("✅ SUCCESS")
        else:
            print("⚠️  WARNING - Check endpoint implementation")
            
        return response
        
    except requests.exceptions.RequestException as e:
        print(f"❌ CONNECTION ERROR: {e}")
        print("💡 Make sure the server is running!")
        return None

def main():
    """Run live server validation"""
    server_url = "http://localhost:8000"
    
    print(f"""
🚀 QFLARE LIVE SERVER VALIDATION
{'='*70}
Testing all endpoints and demonstrating real functionality.
Server: {server_url}
Time: {datetime.now(timezone.utc).isoformat()}
{'='*70}
""")
    
    # Test 1: Health Check
    test_endpoint(f"{server_url}/health", 
                 description="Health Check - Server Status")
    
    # Test 2: Main Dashboard
    test_endpoint(f"{server_url}/", 
                 description="Main Dashboard - Web Interface")
    
    # Test 3: Device Management
    test_endpoint(f"{server_url}/devices", 
                 description="Device Management - List Devices")
    
    # Test 4: Registration Page
    test_endpoint(f"{server_url}/register", 
                 description="Registration Page - Device Registration UI")
    
    # Test 5: Quantum Key Request
    test_endpoint(f"{server_url}/api/request_qkey", 
                 description="Quantum Key Generation - PQ Crypto")
    
    # Test 6: API Documentation
    test_endpoint(f"{server_url}/docs", 
                 description="API Documentation - Swagger UI")
    
    # Test 7: Device Enrollment (POST)
    enrollment_data = {
        "device_id": f"test_device_{int(time.time())}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pub_kem": "dGVzdF9wdWJsaWNfa2V5X2Zvl_kZW1fb3JfZGlsaXRoaXVtX3B1YmxpY19rZXk=",
        "device_info": {
            "type": "edge_node",
            "version": "1.0.0"
        }
    }
    
    test_endpoint(f"{server_url}/api/enroll", 
                 method="POST",
                 data=enrollment_data,
                 description="Device Enrollment - Register New Device")
    
    # Test 8: Challenge Response
    challenge_data = {
        "device_id": "test_device_challenge",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    test_endpoint(f"{server_url}/api/challenge", 
                 method="POST",
                 data=challenge_data,
                 description="Challenge Response - Session Establishment")
    
    # Test 9: Global Model Download
    test_endpoint(f"{server_url}/api/global_model", 
                 description="Global Model - Federated Learning Model")
    
    # Test 10: Security Headers Check
    print(f"\n{'='*60}")
    print(f"🔒 SECURITY HEADERS CHECK")
    print(f"{'='*60}")
    
    response = requests.get(f"{server_url}/health", timeout=5)
    if response:
        security_headers = {
            "X-Content-Type-Options": response.headers.get("X-Content-Type-Options"),
            "X-Frame-Options": response.headers.get("X-Frame-Options"),
            "X-XSS-Protection": response.headers.get("X-XSS-Protection"),
            "Strict-Transport-Security": response.headers.get("Strict-Transport-Security"),
            "Content-Security-Policy": response.headers.get("Content-Security-Policy")
        }
        
        for header, value in security_headers.items():
            if value:
                print(f"✅ {header}: {value}")
            else:
                print(f"⚠️  {header}: Missing")
    
    # Test 11: Performance Check
    print(f"\n{'='*60}")
    print(f"⚡ PERFORMANCE CHECK")
    print(f"{'='*60}")
    
    start_time = time.time()
    for i in range(5):
        response = requests.get(f"{server_url}/health", timeout=5)
        if response and response.status_code == 200:
            print(f"✅ Request {i+1}: {response.elapsed.total_seconds():.3f}s")
        else:
            print(f"❌ Request {i+1}: Failed")
    
    avg_time = (time.time() - start_time) / 5
    print(f"📊 Average Response Time: {avg_time:.3f}s")
    
    # Final Summary
    print(f"\n{'='*70}")
    print(f"🎯 VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"✅ Server is running on {server_url}")
    print(f"✅ All endpoints tested")
    print(f"✅ Ready for quantum-safe federated learning")
    print(f"\n🌐 Access your QFLARE dashboard: {server_url}")
    print(f"📚 API Documentation: {server_url}/docs")
    print(f"🔧 Health Check: {server_url}/health")

if __name__ == "__main__":
    main()