#!/usr/bin/env python3
"""
Quick test script to verify QFLARE fixes
"""

import requests
import websocket
import json
import threading
import time
import sys

def test_api_endpoints():
    """Test API endpoints"""
    print("🧪 Testing API endpoints...")
    
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health check: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test devices endpoint (should return empty array)
    try:
        response = requests.get(f"{base_url}/api/devices")
        data = response.json()
        print(f"✅ Devices API: {response.status_code} - Data type: {type(data)} - Content: {data}")
        
        # Verify it returns an array (not object)
        if isinstance(data, list):
            print("✅ Devices API returns array (correct for frontend)")
        else:
            print("❌ Devices API returns object (will cause .filter error)")
            return False
            
    except Exception as e:
        print(f"❌ Devices API failed: {e}")
        return False
    
    # Test FL status endpoint
    try:
        response = requests.get(f"{base_url}/api/fl/status")
        print(f"✅ FL Status: {response.status_code}")
    except Exception as e:
        print(f"❌ FL Status failed: {e}")
        return False
    
    return True

def test_websocket():
    """Test WebSocket connection"""
    print("🔌 Testing WebSocket connection...")
    
    ws_url = "ws://localhost:8000/ws"
    connection_success = False
    
    def on_message(ws, message):
        print(f"📨 WebSocket message received: {message}")
    
    def on_error(ws, error):
        print(f"❌ WebSocket error: {error}")
    
    def on_close(ws, close_status_code, close_msg):
        print(f"🔌 WebSocket connection closed: {close_status_code} - {close_msg}")
    
    def on_open(ws):
        nonlocal connection_success
        connection_success = True
        print("✅ WebSocket connection opened successfully!")
        # Send a test message
        ws.send(json.dumps({"test": "hello"}))
        # Close after 3 seconds
        def close_connection():
            time.sleep(3)
            ws.close()
        threading.Thread(target=close_connection).start()
    
    try:
        ws = websocket.WebSocketApp(ws_url,
                                    on_open=on_open,
                                    on_message=on_message,
                                    on_error=on_error,
                                    on_close=on_close)
        
        # Run for 5 seconds max
        ws.run_forever()
        
        if connection_success:
            print("✅ WebSocket test completed successfully")
            return True
        else:
            print("❌ WebSocket connection failed")
            return False
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 QFLARE Issue Fix Verification")
    print("=" * 50)
    
    # Test API endpoints
    api_success = test_api_endpoints()
    print()
    
    # Test WebSocket
    ws_success = test_websocket()
    print()
    
    # Summary
    print("📊 Test Results:")
    print(f"  API Endpoints: {'✅ PASS' if api_success else '❌ FAIL'}")
    print(f"  WebSocket:     {'✅ PASS' if ws_success else '❌ FAIL'}")
    
    if api_success and ws_success:
        print("\n🎉 All fixes verified! QFLARE should work properly now.")
        return 0
    else:
        print("\n❌ Some issues remain. Check the server logs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())