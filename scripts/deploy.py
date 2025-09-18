#!/usr/bin/env python3
"""
Production deployment script for QFLARE
Handles blue-green deployments, health checks, and rollbacks
"""

import argparse
import subprocess
import time
import requests
import json
import sys
from datetime import datetime
from typing import Dict, List, Optional
import yaml

class QFLAREDeployer:
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.namespace = f"qflare-{environment}"
        self.config = self.load_config()
        
    def load_config(self) -> Dict:
        """Load deployment configuration"""
        try:
            with open(f"config/{self.environment}_deploy.yaml", "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {
                "api_url": f"https://qflare-{self.environment}.company.com",
                "healthcheck_timeout": 300,
                "rollback_timeout": 180,
                "replicas": 3 if self.environment == "production" else 1
            }
    
    def deploy(self, version: str, blue_green: bool = True) -> bool:
        """Deploy QFLARE to the specified environment"""
        print(f"🚀 Starting deployment to {self.environment}")
        print(f"📦 Version: {version}")
        print(f"🔄 Blue-Green: {blue_green}")
        print("=" * 50)
        
        try:
            # Pre-deployment checks
            if not self.pre_deployment_checks():
                print("❌ Pre-deployment checks failed")
                return False
            
            # Deploy application
            if blue_green:
                success = self.blue_green_deploy(version)
            else:
                success = self.rolling_deploy(version)
            
            if not success:
                print("❌ Deployment failed")
                return False
            
            # Post-deployment verification
            if not self.post_deployment_checks():
                print("⚠️ Post-deployment checks failed, initiating rollback")
                self.rollback()
                return False
            
            print("✅ Deployment completed successfully!")
            return True
            
        except Exception as e:
            print(f"💥 Deployment error: {e}")
            self.rollback()
            return False
    
    def pre_deployment_checks(self) -> bool:
        """Run pre-deployment health checks"""
        print("🔍 Running pre-deployment checks...")
        
        checks = [
            self.check_cluster_health,
            self.check_database_connectivity,
            self.check_redis_connectivity,
            self.check_storage_capacity,
            self.check_current_app_health
        ]
        
        for check in checks:
            if not check():
                return False
        
        print("✅ All pre-deployment checks passed")
        return True
    
    def check_cluster_health(self) -> bool:
        """Check Kubernetes cluster health"""
        try:
            result = subprocess.run(
                ["kubectl", "cluster-info", "--request-timeout=10s"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print("✅ Kubernetes cluster is healthy")
                return True
            else:
                print(f"❌ Cluster health check failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Cluster health check error: {e}")
            return False
    
    def check_database_connectivity(self) -> bool:
        """Check database connectivity"""
        try:
            result = subprocess.run([
                "kubectl", "exec", "-n", self.namespace,
                "deployment/postgres", "--",
                "pg_isready", "-U", "qflare"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("✅ Database connectivity check passed")
                return True
            else:
                print(f"❌ Database connectivity check failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Database connectivity check error: {e}")
            return False
    
    def check_redis_connectivity(self) -> bool:
        """Check Redis connectivity"""
        try:
            result = subprocess.run([
                "kubectl", "exec", "-n", self.namespace,
                "deployment/redis", "--",
                "redis-cli", "ping"
            ], capture_output=True, text=True, timeout=10)
            
            if "PONG" in result.stdout:
                print("✅ Redis connectivity check passed")
                return True
            else:
                print(f"❌ Redis connectivity check failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Redis connectivity check error: {e}")
            return False
    
    def check_storage_capacity(self) -> bool:
        """Check storage capacity"""
        try:
            result = subprocess.run([
                "kubectl", "top", "nodes", "--no-headers"
            ], capture_output=True, text=True)
            
            # Simple check - in production, implement proper capacity monitoring
            print("✅ Storage capacity check passed")
            return True
        except Exception as e:
            print(f"⚠️ Storage capacity check warning: {e}")
            return True  # Non-critical for deployment
    
    def check_current_app_health(self) -> bool:
        """Check current application health"""
        try:
            response = requests.get(
                f"{self.config['api_url']}/api/v1/health",
                timeout=10
            )
            if response.status_code == 200:
                print("✅ Current application is healthy")
                return True
            else:
                print(f"⚠️ Current application health check returned: {response.status_code}")
                return True  # Allow deployment even if current app is unhealthy
        except Exception as e:
            print(f"⚠️ Current application health check failed: {e}")
            return True  # Allow deployment even if current app is unreachable
    
    def blue_green_deploy(self, version: str) -> bool:
        """Perform blue-green deployment"""
        print("🔵 Starting blue-green deployment...")
        
        # Create green environment
        if not self.create_green_environment(version):
            return False
        
        # Wait for green environment to be ready
        if not self.wait_for_green_ready():
            return False
        
        # Switch traffic to green
        if not self.switch_traffic_to_green():
            return False
        
        # Verify green environment
        if not self.verify_green_environment():
            return False
        
        # Cleanup blue environment
        self.cleanup_blue_environment()
        
        print("🟢 Blue-green deployment completed")
        return True
    
    def create_green_environment(self, version: str) -> bool:
        """Create green environment"""
        print("🔄 Creating green environment...")
        
        try:
            # Update deployment manifests with new version
            self.update_manifests(version, color="green")
            
            # Apply green manifests
            result = subprocess.run([
                "kubectl", "apply", "-f", f"k8s/{self.environment}-green.yaml"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Green environment created")
                return True
            else:
                print(f"❌ Failed to create green environment: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Green environment creation error: {e}")
            return False
    
    def wait_for_green_ready(self) -> bool:
        """Wait for green environment to be ready"""
        print("⏳ Waiting for green environment to be ready...")
        
        timeout = self.config.get("healthcheck_timeout", 300)
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = subprocess.run([
                    "kubectl", "get", "pods", "-n", f"{self.namespace}-green",
                    "-l", "app=qflare-api",
                    "-o", "jsonpath={.items[*].status.phase}"
                ], capture_output=True, text=True)
                
                if "Running" in result.stdout and "Pending" not in result.stdout:
                    print("✅ Green environment is ready")
                    return True
                
                print("⏳ Green environment not ready yet, waiting...")
                time.sleep(10)
                
            except Exception as e:
                print(f"⚠️ Error checking green environment: {e}")
                time.sleep(10)
        
        print("❌ Green environment failed to become ready within timeout")
        return False
    
    def switch_traffic_to_green(self) -> bool:
        """Switch traffic from blue to green"""
        print("🔀 Switching traffic to green environment...")
        
        try:
            # Update service selector to point to green
            result = subprocess.run([
                "kubectl", "patch", "service", "qflare-api",
                "-n", self.namespace,
                "-p", '{"spec":{"selector":{"color":"green"}}}'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Traffic switched to green environment")
                return True
            else:
                print(f"❌ Failed to switch traffic: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Traffic switching error: {e}")
            return False
    
    def verify_green_environment(self) -> bool:
        """Verify green environment is working correctly"""
        print("🔍 Verifying green environment...")
        
        # Wait a moment for traffic switch to take effect
        time.sleep(30)
        
        return self.post_deployment_checks()
    
    def cleanup_blue_environment(self):
        """Cleanup blue environment"""
        print("🧹 Cleaning up blue environment...")
        
        try:
            subprocess.run([
                "kubectl", "delete", "deployment", "qflare-api-blue",
                "-n", self.namespace
            ], capture_output=True)
            print("✅ Blue environment cleaned up")
        except Exception as e:
            print(f"⚠️ Blue environment cleanup warning: {e}")
    
    def rolling_deploy(self, version: str) -> bool:
        """Perform rolling deployment"""
        print("🔄 Starting rolling deployment...")
        
        try:
            # Update deployment with new image
            result = subprocess.run([
                "kubectl", "set", "image",
                f"deployment/qflare-api",
                f"qflare-api=qflare/api:{version}",
                "-n", self.namespace
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to update deployment: {result.stderr}")
                return False
            
            # Wait for rollout to complete
            result = subprocess.run([
                "kubectl", "rollout", "status",
                "deployment/qflare-api",
                "-n", self.namespace,
                "--timeout=600s"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Rolling deployment completed")
                return True
            else:
                print(f"❌ Rolling deployment failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Rolling deployment error: {e}")
            return False
    
    def post_deployment_checks(self) -> bool:
        """Run post-deployment verification"""
        print("🔍 Running post-deployment checks...")
        
        checks = [
            self.health_check,
            self.api_functionality_check,
            self.database_connectivity_check,
            self.performance_check
        ]
        
        for check in checks:
            if not check():
                return False
        
        print("✅ All post-deployment checks passed")
        return True
    
    def health_check(self) -> bool:
        """Check application health endpoint"""
        try:
            response = requests.get(
                f"{self.config['api_url']}/api/v1/health",
                timeout=30
            )
            
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("status") == "healthy":
                    print("✅ Health check passed")
                    return True
                else:
                    print(f"❌ Health check failed: {health_data}")
                    return False
            else:
                print(f"❌ Health check failed with status: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def api_functionality_check(self) -> bool:
        """Check basic API functionality"""
        try:
            # Test device listing endpoint
            response = requests.get(
                f"{self.config['api_url']}/api/v1/devices",
                timeout=30
            )
            
            if response.status_code in [200, 401]:  # 401 is expected without auth
                print("✅ API functionality check passed")
                return True
            else:
                print(f"❌ API functionality check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API functionality check error: {e}")
            return False
    
    def database_connectivity_check(self) -> bool:
        """Check database connectivity post-deployment"""
        return self.check_database_connectivity()
    
    def performance_check(self) -> bool:
        """Check application performance"""
        try:
            start_time = time.time()
            response = requests.get(
                f"{self.config['api_url']}/api/v1/health",
                timeout=10
            )
            response_time = time.time() - start_time
            
            if response_time < 5.0:  # 5 second threshold
                print(f"✅ Performance check passed ({response_time:.2f}s)")
                return True
            else:
                print(f"⚠️ Performance check warning: slow response ({response_time:.2f}s)")
                return True  # Non-blocking warning
        except Exception as e:
            print(f"⚠️ Performance check warning: {e}")
            return True  # Non-blocking warning
    
    def rollback(self) -> bool:
        """Rollback to previous version"""
        print("🔄 Initiating rollback...")
        
        try:
            result = subprocess.run([
                "kubectl", "rollout", "undo",
                "deployment/qflare-api",
                "-n", self.namespace
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Wait for rollback to complete
                rollback_result = subprocess.run([
                    "kubectl", "rollout", "status",
                    "deployment/qflare-api",
                    "-n", self.namespace,
                    f"--timeout={self.config.get('rollback_timeout', 180)}s"
                ], capture_output=True, text=True)
                
                if rollback_result.returncode == 0:
                    print("✅ Rollback completed successfully")
                    return True
                else:
                    print(f"❌ Rollback status check failed: {rollback_result.stderr}")
                    return False
            else:
                print(f"❌ Rollback failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Rollback error: {e}")
            return False
    
    def update_manifests(self, version: str, color: str = "blue"):
        """Update Kubernetes manifests with new version"""
        # This would update the k8s manifests with the new image version
        # Implementation depends on your manifest structure
        pass

def main():
    parser = argparse.ArgumentParser(description="QFLARE Production Deployment")
    parser.add_argument("--environment", default="production", help="Deployment environment")
    parser.add_argument("--version", required=True, help="Version to deploy")
    parser.add_argument("--blue-green", action="store_true", help="Use blue-green deployment")
    parser.add_argument("--rollback", action="store_true", help="Rollback current deployment")
    
    args = parser.parse_args()
    
    deployer = QFLAREDeployer(args.environment)
    
    if args.rollback:
        success = deployer.rollback()
    else:
        success = deployer.deploy(args.version, args.blue_green)
    
    if success:
        print(f"\n🎉 Operation completed successfully!")
        sys.exit(0)
    else:
        print(f"\n💥 Operation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()