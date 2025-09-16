#!/usr/bin/env python3
"""
QFLARE Quantum Key Exchange Testing Suite
Comprehensive testing and validation tools for quantum-safe cryptography
"""

import asyncio
import json
import time
import requests
import secrets
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add server directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

class QuantumKeyExchangeTester:
    """Comprehensive testing suite for quantum key exchange system"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_devices = []
        self.test_sessions = []
        self.test_results = {
            "test_start": datetime.utcnow().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": [],
            "performance_metrics": {}
        }
    
    def log_test(self, test_name: str, success: bool, details: Dict = None):
        """Log test result"""
        self.test_results["tests_run"] += 1
        if success:
            self.test_results["tests_passed"] += 1
            print(f"✅ {test_name}: PASSED")
        else:
            self.test_results["tests_failed"] += 1
            print(f"❌ {test_name}: FAILED")
            if details:
                self.test_results["errors"].append({
                    "test": test_name,
                    "details": details,
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        if details:
            print(f"   Details: {details}")
    
    def test_system_status(self) -> bool:
        """Test basic system connectivity and status"""
        try:
            response = self.session.get(f"{self.base_url}/api/system/status")
            if response.status_code == 200:
                data = response.json()
                success = data.get("status") == "operational"
                self.log_test("System Status Check", success, data)
                return success
            else:
                self.log_test("System Status Check", False, {"status_code": response.status_code})
                return False
        except Exception as e:
            self.log_test("System Status Check", False, {"error": str(e)})
            return False
    
    def test_device_registration(self, device_count: int = 5) -> bool:
        """Test device registration functionality"""
        success_count = 0
        
        for i in range(device_count):
            try:
                device_data = {
                    "device_id": f"test_device_{secrets.token_hex(4)}",
                    "device_type": ["EDGE_NODE", "MOBILE_DEVICE", "IOT_SENSOR", "SERVER"][i % 4]
                }
                
                response = self.session.post(
                    f"{self.base_url}/api/test/simulate_device_registration",
                    json=device_data
                )
                
                if response.status_code == 200:
                    device_info = response.json()
                    self.test_devices.append(device_info)
                    success_count += 1
                    print(f"   Device {device_data['device_id']} registered successfully")
                else:
                    print(f"   Failed to register device {device_data['device_id']}: {response.status_code}")
            
            except Exception as e:
                print(f"   Error registering device: {e}")
        
        success = success_count == device_count
        self.log_test(f"Device Registration ({device_count} devices)", success, {
            "registered": success_count,
            "expected": device_count
        })
        return success
    
    def test_key_exchange(self) -> bool:
        """Test quantum key exchange functionality"""
        if not self.test_devices:
            self.log_test("Key Exchange Test", False, {"error": "No test devices available"})
            return False
        
        success_count = 0
        total_tests = len(self.test_devices)
        
        for device in self.test_devices:
            try:
                start_time = time.time()
                
                response = self.session.post(
                    f"{self.base_url}/api/test/simulate_key_exchange",
                    json={"device_id": device["device_id"]}
                )
                
                exchange_time = time.time() - start_time
                
                if response.status_code == 200:
                    session_info = response.json()
                    self.test_sessions.append(session_info)
                    success_count += 1
                    
                    # Store performance metrics
                    if "exchange_times" not in self.test_results["performance_metrics"]:
                        self.test_results["performance_metrics"]["exchange_times"] = []
                    self.test_results["performance_metrics"]["exchange_times"].append(exchange_time)
                    
                    print(f"   Key exchange for {device['device_id']}: {exchange_time:.3f}s")
                else:
                    print(f"   Key exchange failed for {device['device_id']}: {response.status_code}")
            
            except Exception as e:
                print(f"   Error in key exchange for {device['device_id']}: {e}")
        
        success = success_count == total_tests
        avg_time = sum(self.test_results["performance_metrics"].get("exchange_times", [0])) / max(success_count, 1)
        
        self.log_test(f"Key Exchange ({total_tests} exchanges)", success, {
            "successful": success_count,
            "total": total_tests,
            "average_time": f"{avg_time:.3f}s"
        })
        return success
    
    def test_threat_simulation(self) -> bool:
        """Test security threat simulation"""
        threat_types = ["QUANTUM_ATTACK", "ANOMALOUS_BEHAVIOR", "KEY_COMPROMISE"]
        severities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        
        success_count = 0
        total_tests = len(threat_types) * len(severities)
        
        for threat_type in threat_types:
            for severity in severities:
                try:
                    threat_data = {
                        "threat_type": threat_type,
                        "severity": severity,
                        "device_id": self.test_devices[0]["device_id"] if self.test_devices else None
                    }
                    
                    response = self.session.post(
                        f"{self.base_url}/api/test/simulate_threat",
                        json=threat_data
                    )
                    
                    if response.status_code == 200:
                        success_count += 1
                        print(f"   Threat simulation: {threat_type} ({severity}) - Success")
                    else:
                        print(f"   Threat simulation: {threat_type} ({severity}) - Failed: {response.status_code}")
                
                except Exception as e:
                    print(f"   Error simulating {threat_type} ({severity}): {e}")
        
        success = success_count == total_tests
        self.log_test(f"Threat Simulation ({total_tests} scenarios)", success, {
            "successful": success_count,
            "total": total_tests
        })
        return success
    
    def test_stress_test(self) -> bool:
        """Test system under stress conditions"""
        try:
            stress_params = {
                "num_devices": 10,
                "exchanges_per_device": 3
            }
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/test/stress_test",
                json=stress_params
            )
            stress_time = time.time() - start_time
            
            if response.status_code == 200:
                results = response.json()
                success_rate = results.get("success_rate", 0)
                success = success_rate >= 0.8  # 80% success rate threshold
                
                self.test_results["performance_metrics"]["stress_test"] = {
                    "total_time": stress_time,
                    "success_rate": success_rate,
                    "devices_created": results.get("devices_created", 0),
                    "exchanges_completed": results.get("exchanges_completed", 0)
                }
                
                self.log_test("Stress Test", success, {
                    "success_rate": f"{success_rate:.1%}",
                    "total_time": f"{stress_time:.3f}s",
                    "devices": results.get("devices_created", 0),
                    "exchanges": results.get("exchanges_completed", 0)
                })
                return success
            else:
                self.log_test("Stress Test", False, {"status_code": response.status_code})
                return False
        
        except Exception as e:
            self.log_test("Stress Test", False, {"error": str(e)})
            return False
    
    def test_quantum_resistance_demo(self) -> bool:
        """Test quantum resistance information endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/api/test/quantum_resistance_demo")
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate expected fields
                required_fields = ["algorithms", "security_levels", "temporal_features", "resistance_against"]
                all_present = all(field in data for field in required_fields)
                
                # Validate algorithms
                algorithms = data.get("algorithms", {})
                expected_algorithms = ["key_exchange", "signatures", "hashing"]
                algorithms_present = all(alg in algorithms for alg in expected_algorithms)
                
                success = all_present and algorithms_present
                self.log_test("Quantum Resistance Demo", success, {
                    "fields_present": all_present,
                    "algorithms_present": algorithms_present,
                    "algorithms": algorithms
                })
                return success
            else:
                self.log_test("Quantum Resistance Demo", False, {"status_code": response.status_code})
                return False
        
        except Exception as e:
            self.log_test("Quantum Resistance Demo", False, {"error": str(e)})
            return False
    
    def test_api_endpoints(self) -> bool:
        """Test all API endpoints for basic functionality"""
        endpoints = [
            "/api/system/status",
            "/api/keys/statistics", 
            "/api/sessions/active",
            "/api/events/recent",
            "/api/devices",
            "/api/test/quantum_resistance_demo"
        ]
        
        success_count = 0
        
        for endpoint in endpoints:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}")
                if response.status_code == 200:
                    success_count += 1
                    print(f"   {endpoint}: OK")
                else:
                    print(f"   {endpoint}: FAILED ({response.status_code})")
            except Exception as e:
                print(f"   {endpoint}: ERROR ({e})")
        
        success = success_count == len(endpoints)
        self.log_test(f"API Endpoints ({len(endpoints)} endpoints)", success, {
            "accessible": success_count,
            "total": len(endpoints)
        })
        return success
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run complete test suite"""
        print("🚀 Starting QFLARE Quantum Key Exchange Test Suite")
        print("=" * 60)
        
        # Run all tests
        tests = [
            self.test_system_status,
            self.test_api_endpoints,
            self.test_device_registration,
            self.test_key_exchange,
            self.test_threat_simulation,
            self.test_quantum_resistance_demo,
            self.test_stress_test
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"❌ Test {test.__name__} crashed: {e}")
                self.test_results["tests_failed"] += 1
                self.test_results["errors"].append({
                    "test": test.__name__,
                    "details": {"crash": str(e)},
                    "timestamp": datetime.utcnow().isoformat()
                })
            print()  # Add spacing between tests
        
        # Calculate final results
        self.test_results["test_end"] = datetime.utcnow().isoformat()
        self.test_results["success_rate"] = (
            self.test_results["tests_passed"] / max(self.test_results["tests_run"], 1)
        )
        
        # Performance summary
        if "exchange_times" in self.test_results["performance_metrics"]:
            times = self.test_results["performance_metrics"]["exchange_times"]
            self.test_results["performance_metrics"]["exchange_summary"] = {
                "min_time": min(times),
                "max_time": max(times),
                "avg_time": sum(times) / len(times),
                "total_exchanges": len(times)
            }
        
        return self.test_results
    
    def print_summary(self):
        """Print test summary"""
        results = self.test_results
        
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Tests Run: {results['tests_run']}")
        print(f"Tests Passed: {results['tests_passed']}")
        print(f"Tests Failed: {results['tests_failed']}")
        print(f"Success Rate: {results['success_rate']:.1%}")
        print()
        
        if "exchange_summary" in results["performance_metrics"]:
            perf = results["performance_metrics"]["exchange_summary"]
            print("⚡ PERFORMANCE METRICS")
            print("-" * 30)
            print(f"Total Key Exchanges: {perf['total_exchanges']}")
            print(f"Average Time: {perf['avg_time']:.3f}s")
            print(f"Min Time: {perf['min_time']:.3f}s")
            print(f"Max Time: {perf['max_time']:.3f}s")
            print()
        
        if results["errors"]:
            print("🔍 ERRORS")
            print("-" * 30)
            for error in results["errors"][:5]:  # Show first 5 errors
                print(f"• {error['test']}: {error['details']}")
            if len(results["errors"]) > 5:
                print(f"... and {len(results['errors']) - 5} more errors")
            print()
        
        if results["success_rate"] >= 0.9:
            print("🎉 EXCELLENT! Quantum key exchange system is working perfectly!")
        elif results["success_rate"] >= 0.7:
            print("✅ GOOD! Most tests passed. Check errors for improvements.")
        else:
            print("⚠️  NEEDS ATTENTION! Several tests failed. Review system configuration.")

def main():
    """Main testing function"""
    print("🔐 QFLARE Quantum Key Exchange Testing Suite")
    print("Testing quantum-safe cryptography system...\n")
    
    # Check if dashboard is running
    tester = QuantumKeyExchangeTester()
    
    print("🔍 Checking if dashboard is running on http://localhost:8002...")
    if not tester.test_system_status():
        print("\n❌ Dashboard is not accessible!")
        print("Please start the dashboard first:")
        print("   python quantum_dashboard.py")
        return
    
    print("✅ Dashboard is running!")
    print()
    
    # Run comprehensive tests
    results = tester.run_comprehensive_test()
    
    # Print summary
    tester.print_summary()
    
    # Save results to file
    with open("quantum_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"📁 Detailed results saved to: quantum_test_results.json")

if __name__ == "__main__":
    main()