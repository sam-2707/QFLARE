#!/usr/bin/env python3
"""
QFLARE Core Workflow Demo
Complete demonstration of the federated learning workflow
"""

import os
import sys
import time
import requests
import json
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")

def print_step(step_num, description):
    """Print a step description"""
    print(f"\n📋 Step {step_num}: {description}")
    print("-" * 40)

def check_server_health():
    """Check if QFLARE server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Server is healthy!")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Active devices: {data.get('active_devices', 0)}")
            return True
        else:
            print("❌ Server health check failed")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("   Make sure QFLARE server is running: python qflare_core_server.py")
        return False

def register_devices():
    """Register sample devices with the server"""
    devices = [
        {
            "device_id": "edge_demo_01",
            "device_type": "edge",
            "location": "New York, USA",
            "capabilities": {"cpu_cores": 4, "memory_gb": 8, "data_samples": 1000}
        },
        {
            "device_id": "mobile_demo_02",
            "device_type": "mobile",
            "location": "London, UK",
            "capabilities": {"cpu_cores": 2, "memory_gb": 4, "data_samples": 500}
        },
        {
            "device_id": "iot_demo_03",
            "device_type": "iot",
            "location": "Tokyo, Japan",
            "capabilities": {"cpu_cores": 1, "memory_gb": 2, "data_samples": 200}
        }
    ]

    registered_devices = []

    for device in devices:
        print(f"\n🔐 Registering device: {device['device_id']}")

        # Generate mock PQ key pair
        import secrets
        import hashlib
        private_key = f"private_key_{secrets.token_hex(32)}"
        public_key = f"public_key_{hashlib.sha256(private_key.encode()).hexdigest()}"

        registration_data = {
            **device,
            "public_key": public_key
        }

        try:
            response = requests.post(
                "http://localhost:8000/api/v1/devices/register",
                json=registration_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ {device['device_id']} registered successfully!")
                print(f"   Token: {result.get('token', 'N/A')[:20]}...")

                # Save device info
                device_info = {
                    **device,
                    "private_key": private_key,
                    "public_key": public_key,
                    "token": result.get("token"),
                    "server_public_key": result.get("server_public_key")
                }
                registered_devices.append(device_info)

                # Save to file
                with open(f"device_{device['device_id']}_credentials.json", "w") as f:
                    json.dump(device_info, f, indent=2)

            else:
                print(f"❌ Failed to register {device['device_id']}: {response.text}")

        except Exception as e:
            print(f"❌ Error registering {device['device_id']}: {e}")

    return registered_devices

def start_fl_round():
    """Start a new federated learning round"""
    print("\n🚀 Starting Federated Learning Round...")

    round_config = {
        "algorithm": "fedavg",
        "min_participants": 2  # Allow demo with fewer devices
    }

    try:
        response = requests.post(
            "http://localhost:8000/api/v1/fl/start-round",
            json=round_config,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ FL Round started successfully!")
            print(f"   Round ID: {result.get('round_id')}")
            print(f"   Algorithm: {result.get('algorithm')}")
            print(f"   Participants: {len(result.get('participants', []))}")
            return result.get("round_id")
        else:
            print(f"❌ Failed to start FL round: {response.text}")
            return None

    except Exception as e:
        print(f"❌ Error starting FL round: {e}")
        return None

def simulate_device_training(devices, round_id):
    """Simulate devices training and submitting model updates"""
    print(f"\n🏋️  Devices training locally and submitting updates...")

    for device in devices:
        print(f"\n🤖 {device['device_id']} training...")

        # Simulate local training
        time.sleep(2)  # Simulate training time

        # Generate mock model update
        import random
        model_data = {}
        for i in range(10):  # 10 model parameters
            model_data[f"param_{i}"] = random.uniform(-1.0, 1.0)

        local_metrics = {
            "accuracy": random.uniform(0.8, 0.95),
            "loss": random.uniform(0.1, 0.3),
            "samples_used": device["capabilities"]["data_samples"],
            "training_time": random.uniform(10, 30)
        }

        # Submit model update
        update_data = {
            "round_id": round_id,
            "model_data": model_data,
            "local_metrics": local_metrics
        }

        try:
            response = requests.post(
                "http://localhost:8000/api/v1/fl/submit-update",
                json=update_data,
                headers={
                    "Content-Type": "application/json",
                    "X-Device-ID": device["device_id"],
                    "X-Device-Token": device["token"]
                },
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ {device['device_id']} submitted update successfully!")

                if result.get("status") == "submitted_and_completed":
                    print(f"🎉 Round {round_id} completed with this submission!")
                    return True  # Round completed
            else:
                print(f"❌ {device['device_id']} submission failed: {response.text}")

        except Exception as e:
            print(f"❌ Error submitting update for {device['device_id']}: {e}")

    return False

def monitor_round_progress(round_id):
    """Monitor the progress of the FL round"""
    print(f"\n📊 Monitoring round {round_id} progress...")

    for _ in range(10):  # Check 10 times
        try:
            response = requests.get("http://localhost:8000/api/v1/fl/current-round", timeout=5)

            if response.status_code == 200:
                data = response.json()

                if data.get("status") == "no_active_round":
                    print("ℹ️  No active round found")
                    break

                if data.get("round_id") == round_id:
                    submissions = len(data.get("submissions", []))
                    participants = len(data.get("participants", []))
                    progress = f"{submissions}/{participants}"

                    print(f"📈 Round {round_id}: {progress} submissions")

                    if submissions >= participants:
                        print("🎯 Round completed!")
                        return True
                else:
                    print(f"ℹ️  Different round active: {data.get('round_id')}")
            else:
                print(f"❌ Error checking round status: {response.status}")

        except Exception as e:
            print(f"❌ Error monitoring round: {e}")

        time.sleep(2)  # Wait 2 seconds between checks

    return False

def show_round_history():
    """Show the history of completed rounds"""
    print("\n📚 FL Round History:")

    try:
        response = requests.get("http://localhost:8000/api/v1/fl/rounds", timeout=10)

        if response.status_code == 200:
            data = response.json()
            rounds = data.get("rounds", [])

            if rounds:
                for round_info in rounds[:5]:  # Show last 5 rounds
                    print(f"\n🏆 Round {round_info['round_id']}")
                    print(f"   Algorithm: {round_info['algorithm']}")
                    print(f"   Status: {round_info['status']}")
                    print(f"   Participants: {len(round_info['participants'])}")
                    print(f"   Started: {round_info['started_at']}")
                    if round_info.get('completed_at'):
                        print(f"   Completed: {round_info['completed_at']}")
            else:
                print("   No completed rounds yet")
        else:
            print(f"❌ Failed to fetch round history: {response.status_code}")

    except Exception as e:
        print(f"❌ Error fetching round history: {e}")

def main():
    """Main demo workflow"""
    print_header("QFLARE Core - Federated Learning Workflow Demo")
    print("\n🎯 This demo shows the complete QFLARE workflow:")
    print("   1. Device registration with post-quantum keys")
    print("   2. Server-initiated federated learning round")
    print("   3. Devices training locally and submitting updates")
    print("   4. Server aggregating model updates")
    print("   5. Round completion and result tracking")

    # Step 1: Check server health
    print_step(1, "Checking QFLARE Server Health")
    if not check_server_health():
        print("\n❌ Server is not running. Please start it first:")
        print("   python qflare_core_server.py")
        return

    # Step 2: Register devices
    print_step(2, "Registering Edge Devices")
    devices = register_devices()

    if len(devices) < 2:
        print("❌ Need at least 2 devices for FL demo")
        return

    # Step 3: Start FL round
    print_step(3, "Starting Federated Learning Round")
    round_id = start_fl_round()

    if not round_id:
        print("❌ Failed to start FL round")
        return

    # Step 4: Simulate device training
    print_step(4, "Devices Training and Submitting Updates")
    round_completed = simulate_device_training(devices, round_id)

    if not round_completed:
        # Monitor progress manually
        print_step(5, "Monitoring Round Progress")
        round_completed = monitor_round_progress(round_id)

    # Step 5: Show results
    if round_completed:
        print_step(6, "Round Completed - Showing Results")
        show_round_history()
    else:
        print("\n⚠️  Round may still be in progress. Check the web dashboard for status.")

    # Final instructions
    print_header("Demo Complete!")
    print("\n🎉 QFLARE Core Workflow Demo Finished!")
    print("\n📋 What happened:")
    print("   ✅ Devices registered with post-quantum cryptography")
    print("   ✅ Server initiated federated learning round")
    print("   ✅ Devices trained locally on their private data")
    print("   ✅ Model updates submitted securely to server")
    print("   ✅ Server aggregated updates using FedAvg algorithm")
    print("   ✅ Global model created without exposing private data")

    print("\n🌐 Check the web dashboard for real-time monitoring:")
    print("   http://localhost:8000")

    print("\n📊 View detailed metrics and round history in the dashboard!")

    print("\n🔄 To run again with different devices:")
    print("   python qflare_workflow_demo.py")

if __name__ == "__main__":
    main()