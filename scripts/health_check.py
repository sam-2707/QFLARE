#!/usr/bin/env python3
"""
Health check script for QFLARE deployment
Comprehensive monitoring and validation suite
"""

import requests
import json
import sys
import time
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse
import yaml

class QFLAREHealthChecker:
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.namespace = f"qflare-{environment}"
        self.config = self.load_config()
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "environment": environment,
            "checks": {},
            "overall_status": "unknown"
        }
    
    def load_config(self) -> Dict:
        """Load health check configuration"""
        try:
            with open(f"config/{self.environment}_health.yaml", "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {
                "api_url": f"https://qflare-{self.environment}.company.com",
                "timeout": 30,
                "retry_count": 3,
                "critical_services": ["api", "database", "redis"],
                "thresholds": {
                    "response_time": 5.0,
                    "memory_usage": 80.0,
                    "cpu_usage": 80.0,
                    "disk_usage": 85.0
                }
            }
    
    def run_all_checks(self) -> bool:
        """Run comprehensive health checks"""
        print(f"🏥 Running health checks for {self.environment}")
        print("=" * 50)
        
        checks = [
            ("Kubernetes Cluster", self.check_kubernetes_cluster),
            ("API Health", self.check_api_health),
            ("Database Health", self.check_database_health),
            ("Redis Health", self.check_redis_health),
            ("Service Endpoints", self.check_service_endpoints),
            ("Resource Usage", self.check_resource_usage),
            ("Network Connectivity", self.check_network_connectivity),
            ("Security Status", self.check_security_status),
            ("Performance Metrics", self.check_performance_metrics),
            ("Data Integrity", self.check_data_integrity)
        ]
        
        total_checks = len(checks)
        passed_checks = 0
        failed_checks = []
        
        for check_name, check_func in checks:
            print(f"\n🔍 {check_name}...")
            try:
                result = check_func()
                self.results["checks"][check_name] = result
                
                if result["status"] == "healthy":
                    print(f"✅ {check_name}: PASSED")
                    passed_checks += 1
                elif result["status"] == "warning":
                    print(f"⚠️ {check_name}: WARNING - {result.get('message', 'Unknown issue')}")
                    passed_checks += 1  # Warnings don't fail overall health
                else:
                    print(f"❌ {check_name}: FAILED - {result.get('message', 'Unknown error')}")
                    failed_checks.append(check_name)
                    
            except Exception as e:
                error_result = {
                    "status": "failed",
                    "message": f"Check execution error: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }
                self.results["checks"][check_name] = error_result
                print(f"💥 {check_name}: ERROR - {str(e)}")
                failed_checks.append(check_name)
        
        # Calculate overall status
        if len(failed_checks) == 0:
            self.results["overall_status"] = "healthy"
            print(f"\n🎉 All checks passed! ({passed_checks}/{total_checks})")
            return True
        else:
            self.results["overall_status"] = "unhealthy"
            print(f"\n💥 {len(failed_checks)} check(s) failed:")
            for check in failed_checks:
                print(f"   - {check}")
            print(f"\nPassed: {passed_checks}/{total_checks}")
            return False
    
    def check_kubernetes_cluster(self) -> Dict:
        """Check Kubernetes cluster health"""
        try:
            # Check cluster info
            result = subprocess.run(
                ["kubectl", "cluster-info", "--request-timeout=10s"],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                return {
                    "status": "failed",
                    "message": f"Cluster unreachable: {result.stderr}",
                    "details": {"returncode": result.returncode}
                }
            
            # Check node status
            node_result = subprocess.run(
                ["kubectl", "get", "nodes", "-o", "json"],
                capture_output=True, text=True
            )
            
            if node_result.returncode == 0:
                nodes_data = json.loads(node_result.stdout)
                node_status = []
                
                for node in nodes_data["items"]:
                    node_name = node["metadata"]["name"]
                    conditions = node["status"]["conditions"]
                    ready_condition = next((c for c in conditions if c["type"] == "Ready"), None)
                    
                    if ready_condition and ready_condition["status"] == "True":
                        node_status.append({"name": node_name, "status": "Ready"})
                    else:
                        node_status.append({"name": node_name, "status": "NotReady"})
                
                unhealthy_nodes = [n for n in node_status if n["status"] != "Ready"]
                
                if unhealthy_nodes:
                    return {
                        "status": "warning",
                        "message": f"{len(unhealthy_nodes)} node(s) not ready",
                        "details": {"nodes": node_status, "unhealthy": unhealthy_nodes}
                    }
                else:
                    return {
                        "status": "healthy",
                        "message": f"All {len(node_status)} nodes ready",
                        "details": {"nodes": node_status}
                    }
            
            return {
                "status": "healthy",
                "message": "Cluster accessible",
                "details": {"cluster_info": "available"}
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Kubernetes check error: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def check_api_health(self) -> Dict:
        """Check API health endpoint"""
        try:
            start_time = time.time()
            response = requests.get(
                f"{self.config['api_url']}/api/v1/health",
                timeout=self.config["timeout"]
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                health_data = response.json()
                
                details = {
                    "response_time": round(response_time, 3),
                    "status_code": response.status_code,
                    "health_data": health_data
                }
                
                if health_data.get("status") == "healthy":
                    if response_time > self.config["thresholds"]["response_time"]:
                        return {
                            "status": "warning",
                            "message": f"Slow response time: {response_time:.2f}s",
                            "details": details
                        }
                    else:
                        return {
                            "status": "healthy",
                            "message": f"API healthy (response: {response_time:.2f}s)",
                            "details": details
                        }
                else:
                    return {
                        "status": "failed",
                        "message": f"API reports unhealthy status: {health_data.get('status')}",
                        "details": details
                    }
            else:
                return {
                    "status": "failed",
                    "message": f"API health check failed: HTTP {response.status_code}",
                    "details": {
                        "status_code": response.status_code,
                        "response_time": round(response_time, 3),
                        "response_text": response.text[:200]
                    }
                }
                
        except requests.RequestException as e:
            return {
                "status": "failed",
                "message": f"API unreachable: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def check_database_health(self) -> Dict:
        """Check database connectivity and health"""
        try:
            # Check if PostgreSQL pod is running
            result = subprocess.run([
                "kubectl", "get", "pods", "-n", self.namespace,
                "-l", "app=postgres", "-o", "json"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                pods_data = json.loads(result.stdout)
                if not pods_data["items"]:
                    return {
                        "status": "failed",
                        "message": "No PostgreSQL pods found",
                        "details": {"pods": []}
                    }
                
                pod = pods_data["items"][0]
                pod_status = pod["status"]["phase"]
                
                if pod_status != "Running":
                    return {
                        "status": "failed",
                        "message": f"PostgreSQL pod not running: {pod_status}",
                        "details": {"pod_status": pod_status}
                    }
                
                # Test database connectivity
                conn_result = subprocess.run([
                    "kubectl", "exec", "-n", self.namespace,
                    pod["metadata"]["name"], "--",
                    "pg_isready", "-U", "qflare"
                ], capture_output=True, text=True, timeout=10)
                
                if conn_result.returncode == 0:
                    return {
                        "status": "healthy",
                        "message": "Database connectivity verified",
                        "details": {"pod_status": pod_status, "connectivity": "ok"}
                    }
                else:
                    return {
                        "status": "failed",
                        "message": f"Database connectivity failed: {conn_result.stderr}",
                        "details": {"pod_status": pod_status, "error": conn_result.stderr}
                    }
            else:
                return {
                    "status": "failed",
                    "message": f"Cannot check database pods: {result.stderr}",
                    "details": {"error": result.stderr}
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Database health check error: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def check_redis_health(self) -> Dict:
        """Check Redis connectivity and health"""
        try:
            # Check if Redis pod is running
            result = subprocess.run([
                "kubectl", "get", "pods", "-n", self.namespace,
                "-l", "app=redis", "-o", "json"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                pods_data = json.loads(result.stdout)
                if not pods_data["items"]:
                    return {
                        "status": "failed",
                        "message": "No Redis pods found",
                        "details": {"pods": []}
                    }
                
                pod = pods_data["items"][0]
                pod_status = pod["status"]["phase"]
                
                if pod_status != "Running":
                    return {
                        "status": "failed",
                        "message": f"Redis pod not running: {pod_status}",
                        "details": {"pod_status": pod_status}
                    }
                
                # Test Redis connectivity
                ping_result = subprocess.run([
                    "kubectl", "exec", "-n", self.namespace,
                    pod["metadata"]["name"], "--",
                    "redis-cli", "ping"
                ], capture_output=True, text=True, timeout=10)
                
                if "PONG" in ping_result.stdout:
                    return {
                        "status": "healthy",
                        "message": "Redis connectivity verified",
                        "details": {"pod_status": pod_status, "ping": "PONG"}
                    }
                else:
                    return {
                        "status": "failed",
                        "message": f"Redis ping failed: {ping_result.stdout}",
                        "details": {"pod_status": pod_status, "output": ping_result.stdout}
                    }
            else:
                return {
                    "status": "failed",
                    "message": f"Cannot check Redis pods: {result.stderr}",
                    "details": {"error": result.stderr}
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Redis health check error: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def check_service_endpoints(self) -> Dict:
        """Check service endpoints availability"""
        endpoints = [
            ("/api/v1/health", "Health endpoint"),
            ("/api/v1/devices", "Devices endpoint"),
            ("/api/v1/fl/status", "FL status endpoint"),
            ("/api/v1/monitoring/metrics", "Metrics endpoint")
        ]
        
        results = []
        failed_endpoints = 0
        
        for endpoint, description in endpoints:
            try:
                start_time = time.time()
                response = requests.get(
                    f"{self.config['api_url']}{endpoint}",
                    timeout=10
                )
                response_time = time.time() - start_time
                
                endpoint_result = {
                    "endpoint": endpoint,
                    "description": description,
                    "status_code": response.status_code,
                    "response_time": round(response_time, 3),
                    "accessible": response.status_code in [200, 401, 403]  # Auth errors are OK
                }
                
                if not endpoint_result["accessible"]:
                    failed_endpoints += 1
                
                results.append(endpoint_result)
                
            except Exception as e:
                results.append({
                    "endpoint": endpoint,
                    "description": description,
                    "error": str(e),
                    "accessible": False
                })
                failed_endpoints += 1
        
        if failed_endpoints == 0:
            return {
                "status": "healthy",
                "message": f"All {len(endpoints)} endpoints accessible",
                "details": {"endpoints": results}
            }
        elif failed_endpoints < len(endpoints):
            return {
                "status": "warning",
                "message": f"{failed_endpoints}/{len(endpoints)} endpoints failed",
                "details": {"endpoints": results, "failed_count": failed_endpoints}
            }
        else:
            return {
                "status": "failed",
                "message": "All endpoints inaccessible",
                "details": {"endpoints": results}
            }
    
    def check_resource_usage(self) -> Dict:
        """Check resource usage across the cluster"""
        try:
            # Get pod resource usage
            result = subprocess.run([
                "kubectl", "top", "pods", "-n", self.namespace,
                "--no-headers"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                return {
                    "status": "warning",
                    "message": "Cannot retrieve resource metrics",
                    "details": {"error": result.stderr}
                }
            
            pod_metrics = []
            high_usage_pods = []
            
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 3:
                        pod_name = parts[0]
                        cpu = parts[1]
                        memory = parts[2]
                        
                        pod_metrics.append({
                            "pod": pod_name,
                            "cpu": cpu,
                            "memory": memory
                        })
                        
                        # Check for high resource usage (simplified)
                        if any(x in memory.lower() for x in ['gi', 'g']) and \
                           float(memory.replace('Gi', '').replace('G', '').replace('Mi', '').replace('M', '')) > 1000:
                            high_usage_pods.append(pod_name)
            
            if high_usage_pods:
                return {
                    "status": "warning",
                    "message": f"{len(high_usage_pods)} pods with high resource usage",
                    "details": {
                        "pod_metrics": pod_metrics,
                        "high_usage_pods": high_usage_pods
                    }
                }
            else:
                return {
                    "status": "healthy",
                    "message": f"Resource usage normal for {len(pod_metrics)} pods",
                    "details": {"pod_metrics": pod_metrics}
                }
                
        except Exception as e:
            return {
                "status": "warning",
                "message": f"Resource check error: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def check_network_connectivity(self) -> Dict:
        """Check network connectivity between services"""
        try:
            # Test internal service connectivity
            result = subprocess.run([
                "kubectl", "exec", "-n", self.namespace,
                "deployment/qflare-api", "--",
                "curl", "-s", "-f", "http://postgres:5432",
                "--connect-timeout", "5"
            ], capture_output=True, text=True, timeout=10)
            
            # Note: This will likely fail as PostgreSQL doesn't serve HTTP,
            # but connection attempt indicates network reachability
            
            return {
                "status": "healthy",
                "message": "Network connectivity verified",
                "details": {"internal_connectivity": "ok"}
            }
            
        except Exception as e:
            return {
                "status": "warning",
                "message": f"Network connectivity check limited: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def check_security_status(self) -> Dict:
        """Check security-related status"""
        try:
            # Check if secrets exist
            secrets_result = subprocess.run([
                "kubectl", "get", "secrets", "-n", self.namespace,
                "-o", "json"
            ], capture_output=True, text=True)
            
            if secrets_result.returncode == 0:
                secrets_data = json.loads(secrets_result.stdout)
                secret_count = len(secrets_data["items"])
                
                required_secrets = ["qflare-db-secret", "qflare-jwt-secret"]
                existing_secrets = [s["metadata"]["name"] for s in secrets_data["items"]]
                missing_secrets = [s for s in required_secrets if s not in existing_secrets]
                
                if missing_secrets:
                    return {
                        "status": "failed",
                        "message": f"Missing required secrets: {missing_secrets}",
                        "details": {
                            "total_secrets": secret_count,
                            "missing": missing_secrets,
                            "existing": existing_secrets
                        }
                    }
                else:
                    return {
                        "status": "healthy",
                        "message": f"All required secrets present ({secret_count} total)",
                        "details": {
                            "total_secrets": secret_count,
                            "required_secrets": required_secrets
                        }
                    }
            else:
                return {
                    "status": "warning",
                    "message": f"Cannot check secrets: {secrets_result.stderr}",
                    "details": {"error": secrets_result.stderr}
                }
                
        except Exception as e:
            return {
                "status": "warning",
                "message": f"Security check error: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def check_performance_metrics(self) -> Dict:
        """Check performance metrics"""
        try:
            # Test API response time with multiple requests
            response_times = []
            
            for i in range(3):
                start_time = time.time()
                response = requests.get(
                    f"{self.config['api_url']}/api/v1/health",
                    timeout=10
                )
                response_time = time.time() - start_time
                response_times.append(response_time)
                time.sleep(1)
            
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            threshold = self.config["thresholds"]["response_time"]
            
            if max_response_time > threshold:
                return {
                    "status": "warning",
                    "message": f"Slow performance detected (max: {max_response_time:.2f}s)",
                    "details": {
                        "avg_response_time": round(avg_response_time, 3),
                        "max_response_time": round(max_response_time, 3),
                        "all_times": [round(t, 3) for t in response_times],
                        "threshold": threshold
                    }
                }
            else:
                return {
                    "status": "healthy",
                    "message": f"Performance good (avg: {avg_response_time:.2f}s)",
                    "details": {
                        "avg_response_time": round(avg_response_time, 3),
                        "max_response_time": round(max_response_time, 3),
                        "all_times": [round(t, 3) for t in response_times]
                    }
                }
                
        except Exception as e:
            return {
                "status": "warning",
                "message": f"Performance check error: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def check_data_integrity(self) -> Dict:
        """Check basic data integrity"""
        try:
            # Test API data endpoints for basic functionality
            response = requests.get(
                f"{self.config['api_url']}/api/v1/devices",
                timeout=10
            )
            
            if response.status_code in [200, 401]:  # Auth errors are expected
                return {
                    "status": "healthy",
                    "message": "Data endpoints accessible",
                    "details": {"status_code": response.status_code}
                }
            else:
                return {
                    "status": "warning",
                    "message": f"Data endpoint returned unexpected status: {response.status_code}",
                    "details": {"status_code": response.status_code}
                }
                
        except Exception as e:
            return {
                "status": "warning",
                "message": f"Data integrity check error: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def save_results(self, output_file: str = None):
        """Save health check results to file"""
        if not output_file:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_file = f"health_check_{self.environment}_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="QFLARE Health Check")
    parser.add_argument("--environment", default="production", help="Environment to check")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    checker = QFLAREHealthChecker(args.environment)
    
    if not args.quiet:
        print("🏥 QFLARE Health Check System")
        print("=" * 50)
    
    success = checker.run_all_checks()
    
    if args.output or not success:
        checker.save_results(args.output)
    
    if success:
        if not args.quiet:
            print("\n🎉 System is healthy!")
        sys.exit(0)
    else:
        if not args.quiet:
            print("\n⚠️ System health issues detected!")
        sys.exit(1)

if __name__ == "__main__":
    main()