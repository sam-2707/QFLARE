#!/usr/bin/env python3
"""
QFLARE Application Integration Test

This script tests the complete authentication and data flow
between the frontend and backend components.
"""

import requests
import time
import json
from concurrent.futures import ThreadPoolExecutor
import websocket
import threading

class QFLARESystemTest:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:3000"
        self.results = []
        
    def test_backend_health(self):
        """Test if backend server is responsive"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.results.append({
                    "test": "Backend Health Check",
                    "status": "✅ PASS", 
                    "details": f"Server version: {data.get('version', 'unknown')}"
                })
                return True
            else:
                self.results.append({
                    "test": "Backend Health Check", 
                    "status": "❌ FAIL", 
                    "details": f"HTTP {response.status_code}"
                })
                return False
        except Exception as e:
            self.results.append({
                "test": "Backend Health Check", 
                "status": "❌ FAIL", 
                "details": str(e)
            })
            return False

    def test_authentication(self):
        """Test authentication endpoints"""
        test_cases = [
            {"username": "admin", "password": "admin123", "expected_role": "admin"},
            {"username": "user", "password": "user123", "expected_role": "user"},
            {"username": "invalid", "password": "invalid", "expected_status": 401}
        ]
        
        for case in test_cases:
            try:
                response = requests.post(
                    f"{self.base_url}/api/auth/login",
                    json={"username": case["username"], "password": case["password"]},
                    timeout=5
                )
                
                if "expected_status" in case:
                    # Test for invalid credentials
                    if response.status_code == case["expected_status"]:
                        self.results.append({
                            "test": f"Auth Test - Invalid Credentials",
                            "status": "✅ PASS",
                            "details": "Correctly rejected invalid login"
                        })
                    else:
                        self.results.append({
                            "test": f"Auth Test - Invalid Credentials",
                            "status": "❌ FAIL",
                            "details": f"Expected {case['expected_status']}, got {response.status_code}"
                        })
                else:
                    # Test for valid credentials
                    if response.status_code == 200:
                        data = response.json()
                        user_role = data.get("user", {}).get("role", "")
                        if user_role == case["expected_role"]:
                            self.results.append({
                                "test": f"Auth Test - {case['username']}",
                                "status": "✅ PASS",
                                "details": f"Role: {user_role}, Token: {data.get('access_token', '')[:20]}..."
                            })
                        else:
                            self.results.append({
                                "test": f"Auth Test - {case['username']}",
                                "status": "❌ FAIL",
                                "details": f"Expected role {case['expected_role']}, got {user_role}"
                            })
                    else:
                        self.results.append({
                            "test": f"Auth Test - {case['username']}",
                            "status": "❌ FAIL",
                            "details": f"HTTP {response.status_code}: {response.text}"
                        })
                        
            except Exception as e:
                self.results.append({
                    "test": f"Auth Test - {case['username']}",
                    "status": "❌ FAIL",
                    "details": str(e)
                })

    def test_api_endpoints(self):
        """Test various API endpoints"""
        endpoints = [
            "/api/system/info",
            "/api/metrics", 
            "/api/clients",
            "/api/training/status",
            "/api/config"
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    self.results.append({
                        "test": f"API Endpoint - {endpoint}",
                        "status": "✅ PASS",
                        "details": f"Response size: {len(response.text)} bytes"
                    })
                else:
                    self.results.append({
                        "test": f"API Endpoint - {endpoint}",
                        "status": "⚠️  WARN",
                        "details": f"HTTP {response.status_code}"
                    })
            except Exception as e:
                self.results.append({
                    "test": f"API Endpoint - {endpoint}",
                    "status": "❌ FAIL",
                    "details": str(e)
                })

    def test_websocket_connection(self):
        """Test WebSocket connectivity"""
        ws_url = "ws://localhost:8000/ws/test_client"
        connection_success = False
        message_received = False
        
        def on_message(ws, message):
            nonlocal message_received
            message_received = True
            
        def on_open(ws):
            nonlocal connection_success
            connection_success = True
            ws.send(json.dumps({"type": "ping", "client_id": "test_client"}))
            
        def on_error(ws, error):
            pass
            
        try:
            ws = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error
            )
            
            # Run WebSocket in a thread with timeout
            def run_ws():
                ws.run_forever()
                
            thread = threading.Thread(target=run_ws)
            thread.daemon = True
            thread.start()
            
            # Wait up to 3 seconds for connection
            time.sleep(3)
            ws.close()
            
            if connection_success:
                self.results.append({
                    "test": "WebSocket Connection",
                    "status": "✅ PASS",
                    "details": f"Connected successfully, Message received: {message_received}"
                })
            else:
                self.results.append({
                    "test": "WebSocket Connection",
                    "status": "❌ FAIL", 
                    "details": "Could not establish connection"
                })
                
        except Exception as e:
            self.results.append({
                "test": "WebSocket Connection",
                "status": "❌ FAIL",
                "details": str(e)
            })

    def test_frontend_accessibility(self):
        """Test if frontend is accessible"""
        try:
            response = requests.get(self.frontend_url, timeout=5)
            if response.status_code == 200:
                if "QFLARE" in response.text:
                    self.results.append({
                        "test": "Frontend Accessibility",
                        "status": "✅ PASS",
                        "details": "React app is running and QFLARE content detected"
                    })
                else:
                    self.results.append({
                        "test": "Frontend Accessibility",
                        "status": "⚠️  WARN",
                        "details": "Page loaded but QFLARE content not found"
                    })
            else:
                self.results.append({
                    "test": "Frontend Accessibility",
                    "status": "❌ FAIL",
                    "details": f"HTTP {response.status_code}"
                })
        except Exception as e:
            self.results.append({
                "test": "Frontend Accessibility",
                "status": "❌ FAIL",
                "details": str(e)
            })

    def run_all_tests(self):
        """Run comprehensive system tests"""
        print("🔍 QFLARE System Integration Test")
        print("=" * 50)
        
        # Test backend first
        if self.test_backend_health():
            self.test_authentication()
            self.test_api_endpoints()
            self.test_websocket_connection()
        
        # Test frontend
        self.test_frontend_accessibility()
        
        # Display results
        print("\n📊 Test Results:")
        print("-" * 50)
        
        pass_count = 0
        fail_count = 0
        warn_count = 0
        
        for result in self.results:
            print(f"{result['status']} {result['test']}")
            print(f"   {result['details']}")
            
            if "✅" in result['status']:
                pass_count += 1
            elif "❌" in result['status']:
                fail_count += 1
            else:
                warn_count += 1
        
        print("\n" + "=" * 50)
        print(f"Summary: {pass_count} passed, {warn_count} warnings, {fail_count} failed")
        
        if fail_count == 0:
            print("🎉 All critical tests passed! QFLARE system is ready.")
        else:
            print("⚠️  Some tests failed. Check the backend server and try again.")

if __name__ == "__main__":
    tester = QFLARESystemTest()
    tester.run_all_tests()