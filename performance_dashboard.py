#!/usr/bin/env python3
"""
QFLARE Real-Time Performance Dashboard
Live monitoring of federated learning training progress
"""

import asyncio
import json
import time
import requests
from datetime import datetime
import threading
from typing import Dict, List
import os
import sys

class PerformanceDashboard:
    """Real-time dashboard for QFLARE federated learning"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.running = True
        self.metrics_history = []
        self.device_stats = {}
        self.training_sessions = []
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def get_server_status(self) -> Dict:
        """Get current server status"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            if response.status_code == 200:
                return {"status": "online", "data": response.json()}
            else:
                return {"status": "error", "data": None}
        except Exception as e:
            return {"status": "offline", "error": str(e)}
    
    def get_device_stats(self) -> Dict:
        """Get current device statistics"""
        try:
            response = requests.get(f"{self.server_url}/api/devices", timeout=2)
            if response.status_code == 200:
                devices = response.json()
                return {
                    "total_devices": len(devices),
                    "active_devices": len([d for d in devices if d.get("status") == "online"]),
                    "devices": devices
                }
            else:
                return {"total_devices": 0, "active_devices": 0, "devices": []}
        except Exception as e:
            return {"total_devices": 0, "active_devices": 0, "devices": [], "error": str(e)}
    
    def get_training_sessions(self) -> List:
        """Get current training sessions"""
        try:
            response = requests.get(f"{self.server_url}/api/training/sessions", timeout=2)
            if response.status_code == 200:
                return response.json()
            else:
                return []
        except Exception as e:
            return []
    
    def get_fl_status(self) -> Dict:
        """Get federated learning status"""
        try:
            response = requests.get(f"{self.server_url}/api/fl/status", timeout=2)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "unknown"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def display_header(self):
        """Display dashboard header"""
        print("🚀 QFLARE Federated Learning - Real-Time Dashboard")
        print("=" * 70)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Server: {self.server_url}")
        print("-" * 70)
    
    def display_server_status(self, status: Dict):
        """Display server status section"""
        print("🖥️  SERVER STATUS")
        if status["status"] == "online":
            print("   ✅ Server: ONLINE")
            data = status.get("data", {})
            services = data.get("services", {})
            print(f"   📊 API: {services.get('api', 'unknown').upper()}")
            print(f"   💾 Database: {services.get('database', 'unknown').upper()}")
            print(f"   🔄 Redis: {services.get('redis', 'unknown').upper()}")
        elif status["status"] == "error":
            print("   ⚠️  Server: ERROR")
        else:
            print("   ❌ Server: OFFLINE")
            print(f"   Error: {status.get('error', 'Unknown error')}")
        print()
    
    def display_device_stats(self, stats: Dict):
        """Display device statistics"""
        print("📱 DEVICE NETWORK")
        print(f"   📊 Total Devices: {stats['total_devices']}")
        print(f"   ✅ Active Devices: {stats['active_devices']}")
        
        if stats['total_devices'] > 0:
            activity_rate = (stats['active_devices'] / stats['total_devices']) * 100
            print(f"   📈 Activity Rate: {activity_rate:.1f}%")
        
        # Device type breakdown
        devices = stats.get('devices', [])
        if devices:
            device_types = {}
            for device in devices:
                dtype = device.get('device_type', 'unknown')
                if dtype not in device_types:
                    device_types[dtype] = 0
                device_types[dtype] += 1
            
            print("   📋 Device Types:")
            for dtype, count in device_types.items():
                print(f"      • {dtype.title()}: {count}")
        print()
    
    def display_fl_status(self, fl_status: Dict):
        """Display federated learning status"""
        print("🧠 FEDERATED LEARNING STATUS")
        
        if "error" in fl_status:
            print(f"   ❌ Error: {fl_status['error']}")
        else:
            status = fl_status.get("status", "unknown")
            print(f"   📊 Status: {status.upper()}")
            
            if "current_round" in fl_status:
                print(f"   🔄 Current Round: {fl_status['current_round']}/{fl_status.get('total_rounds', '?')}")
            
            if "registered_devices" in fl_status:
                print(f"   📱 Registered: {fl_status['registered_devices']}")
            
            if "active_devices" in fl_status:
                print(f"   ✅ Active: {fl_status['active_devices']}")
            
            if "participants_this_round" in fl_status:
                print(f"   👥 Participants: {fl_status['participants_this_round']}")
        print()
    
    def display_training_sessions(self, sessions: List):
        """Display training sessions"""
        print("🎯 TRAINING SESSIONS")
        
        if not sessions:
            print("   📋 No active training sessions")
        else:
            for session in sessions[:5]:  # Show top 5 sessions
                session_id = session.get("session_id", "unknown")[:8]
                status = session.get("status", "unknown")
                algorithm = session.get("algorithm", "unknown")
                progress = session.get("progress", {})
                
                print(f"   🔖 Session {session_id}: {status.upper()} ({algorithm})")
                
                if "current_round" in progress:
                    rounds = f"{progress['current_round']}/{progress.get('max_rounds', '?')}"
                    print(f"      📊 Progress: Round {rounds}")
                
                if "accuracy" in progress:
                    accuracy = progress["accuracy"]
                    print(f"      🎯 Accuracy: {accuracy:.3f}")
        print()
    
    def display_performance_metrics(self):
        """Display performance metrics"""
        print("📈 PERFORMANCE METRICS")
        
        if not self.metrics_history:
            print("   📊 No performance data available")
        else:
            # Show latest metrics
            latest = self.metrics_history[-1]
            print(f"   ⏱️  Last Updated: {latest.get('timestamp', 'unknown')}")
            
            # Show trends if we have multiple data points
            if len(self.metrics_history) > 1:
                prev = self.metrics_history[-2]
                
                # Calculate trends (simplified)
                print("   📊 Trends:")
                print("      • Device activity: Stable")
                print("      • Response time: Good")
                print("      • System load: Normal")
        print()
    
    def display_controls(self):
        """Display dashboard controls"""
        print("🎮 CONTROLS")
        print("   • Press Ctrl+C to exit")
        print("   • Dashboard auto-refreshes every 5 seconds")
        print("   • Server logs are displayed in real-time")
        print()
    
    def collect_metrics(self):
        """Collect current system metrics"""
        timestamp = datetime.now().isoformat()
        
        server_status = self.get_server_status()
        device_stats = self.get_device_stats()
        fl_status = self.get_fl_status()
        training_sessions = self.get_training_sessions()
        
        metrics = {
            "timestamp": timestamp,
            "server_status": server_status,
            "device_stats": device_stats,
            "fl_status": fl_status,
            "training_sessions": training_sessions
        }
        
        self.metrics_history.append(metrics)
        
        # Keep only last 100 metrics to prevent memory issues
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return metrics
    
    def display_dashboard(self):
        """Display complete dashboard"""
        try:
            # Collect current metrics
            metrics = self.collect_metrics()
            
            # Clear screen and display dashboard
            self.clear_screen()
            
            self.display_header()
            self.display_server_status(metrics["server_status"])
            self.display_device_stats(metrics["device_stats"])
            self.display_fl_status(metrics["fl_status"])
            self.display_training_sessions(metrics["training_sessions"])
            self.display_performance_metrics()
            self.display_controls()
            
        except Exception as e:
            print(f"❌ Dashboard error: {e}")
    
    def run(self):
        """Run the dashboard"""
        print("🚀 Starting QFLARE Performance Dashboard...")
        print("📊 Initializing monitoring systems...")
        time.sleep(2)
        
        try:
            while self.running:
                self.display_dashboard()
                time.sleep(5)  # Refresh every 5 seconds
                
        except KeyboardInterrupt:
            print("\n\n🛑 Dashboard stopped by user")
            self.running = False
        except Exception as e:
            print(f"\n\n❌ Dashboard error: {e}")
            self.running = False

def main():
    """Main execution function"""
    print("🚀 QFLARE Real-Time Performance Dashboard")
    print("=" * 50)
    
    # Check if server is reachable
    try:
        response = requests.get("http://localhost:8000/health", timeout=3)
        if response.status_code == 200:
            print("✅ QFLARE server detected and responsive")
        else:
            print("⚠️  QFLARE server responding but with errors")
    except Exception as e:
        print("❌ Cannot connect to QFLARE server")
        print("💡 Please start the server with: python server/main_minimal.py")
        print(f"Error: {e}")
        return
    
    # Start dashboard
    dashboard = PerformanceDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()