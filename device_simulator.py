#!/usr/bin/env python3
"""
QFLARE Device Simulator
Simulates edge devices that register and participate in federated learning
"""

import os
import sys
import asyncio
import logging
import json
import secrets
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib
import base64

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimulatedDevice:
    """Simulates an edge device participating in federated learning"""

    def __init__(self, device_id: str, device_type: str = "edge",
                 location: str = "Unknown", server_url: str = "http://localhost:8000"):
        self.device_id = device_id
        self.device_type = device_type
        self.location = location
        self.server_url = server_url

        # Generate mock post-quantum key pair
        self.private_key = f"private_key_{secrets.token_hex(32)}"
        self.public_key = f"public_key_{hashlib.sha256(self.private_key.encode()).hexdigest()}"

        # Device state
        self.registered = False
        self.token: Optional[str] = None
        self.local_model = self._initialize_local_model()
        self.training_data = self._generate_training_data()

        logger.info(f"Device {device_id} initialized ({device_type})")

    def _initialize_local_model(self) -> Dict[str, float]:
        """Initialize a simple local model (neural network weights)"""
        # Simulate a simple neural network with random weights
        model = {}
        for i in range(10):  # 10 layers
            for j in range(5):  # 5 neurons per layer
                model[f"layer_{i}_weight_{j}"] = random.uniform(-1.0, 1.0)
                model[f"layer_{i}_bias_{j}"] = random.uniform(-0.5, 0.5)
        return model

    def _generate_training_data(self) -> List[Dict[str, Any]]:
        """Generate simulated training data"""
        # Simulate different types of data based on device type
        if self.device_type == "mobile":
            # Mobile device: user behavior data
            data = [
                {"feature1": random.random(), "feature2": random.random(), "label": random.randint(0, 1)}
                for _ in range(random.randint(50, 200))
            ]
        elif self.device_type == "iot":
            # IoT device: sensor data
            data = [
                {"temperature": random.uniform(20, 30), "humidity": random.uniform(40, 80), "motion": random.randint(0, 1)}
                for _ in range(random.randint(100, 500))
            ]
        else:  # edge device
            # Edge device: general ML data
            data = [
                {"input": [random.random() for _ in range(10)], "output": random.randint(0, 9)}
                for _ in range(random.randint(100, 300))
            ]

        return data

    async def register_with_server(self) -> bool:
        """Register this device with the QFLARE server"""
        import aiohttp

        registration_data = {
            "device_id": self.device_id,
            "public_key": self.public_key,
            "device_type": self.device_type,
            "location": self.location,
            "capabilities": {
                "cpu_cores": random.randint(2, 8),
                "memory_gb": random.randint(4, 16),
                "storage_gb": random.randint(32, 256),
                "supported_algorithms": ["fedavg", "fedprox"],
                "data_samples": len(self.training_data)
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.server_url}/api/v1/devices/register",
                    json=registration_data,
                    headers={"Content-Type": "application/json"}
                ) as response:

                    if response.status == 200:
                        result = await response.json()
                        self.token = result.get("token")
                        self.registered = True

                        logger.info(f"✅ Device {self.device_id} registered successfully")
                        logger.info(f"   Token: {self.token[:20]}...")
                        return True
                    else:
                        error = await response.text()
                        logger.error(f"❌ Registration failed: {error}")
                        return False

        except Exception as e:
            logger.error(f"❌ Registration error: {e}")
            return False

    async def check_for_training_task(self) -> Optional[Dict[str, Any]]:
        """Check if there's an active training round to participate in"""
        if not self.registered or not self.token:
            logger.warning(f"Device {self.device_id} not registered")
            return None

        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.server_url}/api/v1/fl/current-round",
                    headers={
                        "X-Device-ID": self.device_id,
                        "X-Device-Token": self.token
                    }
                ) as response:

                    if response.status == 200:
                        round_info = await response.json()
                        if round_info.get("status") == "active":
                            logger.info(f"🎯 Training round available: {round_info['round_id']}")
                            return round_info
                        else:
                            logger.debug(f"No active training round for {self.device_id}")
                            return None
                    else:
                        logger.debug(f"Failed to check training round: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Error checking training task: {e}")
            return None

    def train_local_model(self, round_id: str) -> Dict[str, Any]:
        """Train the local model on device data"""
        logger.info(f"🏋️  Device {self.device_id} starting local training...")

        # Simulate training process
        initial_loss = random.uniform(2.0, 5.0)
        final_loss = initial_loss * random.uniform(0.1, 0.5)  # Improve by 50-90%

        # Update model weights (simulate training)
        updated_model = {}
        for key, value in self.local_model.items():
            # Add some noise to simulate learning
            noise = random.uniform(-0.1, 0.1)
            updated_model[key] = value + noise

        # Calculate training metrics
        training_metrics = {
            "samples_used": len(self.training_data),
            "epochs": random.randint(5, 20),
            "batch_size": random.randint(16, 64),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "accuracy": random.uniform(0.7, 0.95),
            "training_time_seconds": random.uniform(30, 120)
        }

        logger.info(f"Training completed - Loss: {initial_loss:.2f} -> {final_loss:.2f}")
        return {
            "round_id": round_id,
            "device_id": self.device_id,
            "model_data": updated_model,
            "local_metrics": training_metrics,
            "training_completed_at": datetime.now().isoformat()
        }

    async def submit_model_update(self, training_result: Dict[str, Any]) -> bool:
        """Submit the trained model update to the server"""
        if not self.registered or not self.token:
            logger.warning(f"Device {self.device_id} not registered")
            return False

        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.server_url}/api/v1/fl/submit-update",
                    json=training_result,
                    headers={
                        "X-Device-ID": self.device_id,
                        "X-Device-Token": self.token,
                        "Content-Type": "application/json"
                    }
                ) as response:

                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ Model update submitted by {self.device_id}")

                        if result.get("status") == "submitted_and_completed":
                            logger.info(f"🎉 Round {training_result['round_id']} completed!")

                        return True
                    else:
                        error = await response.text()
                        logger.error(f"❌ Submission failed: {error}")
                        return False

        except Exception as e:
            logger.error(f"❌ Submission error: {e}")
            return False

    async def participate_in_federated_learning(self):
        """Main loop for participating in federated learning"""
        logger.info(f"🔄 Device {self.device_id} starting FL participation loop...")

        while True:
            try:
                # Check for training tasks
                round_info = await self.check_for_training_task()

                if round_info:
                    logger.info(f"📋 Participating in round {round_info['round_id']}")

                    # Train local model
                    training_result = self.train_local_model(round_info['round_id'])

                    # Submit update
                    success = await self.submit_model_update(training_result)

                    if success:
                        logger.info(f"✅ Successfully participated in round {round_info['round_id']}")
                    else:
                        logger.error(f"❌ Failed to participate in round {round_info['round_id']}")

                    # Wait before checking again
                    await asyncio.sleep(random.uniform(10, 30))
                else:
                    # No active round, wait and check again
                    await asyncio.sleep(random.uniform(5, 15))

            except Exception as e:
                logger.error(f"Error in FL participation loop: {e}")
                await asyncio.sleep(10)

async def run_device_simulation():
    """Run multiple device simulations"""
    logger.info("🚀 Starting QFLARE Device Simulation...")

    # Create different types of devices
    devices = [
        SimulatedDevice("edge_001", "edge", "New York, USA"),
        SimulatedDevice("mobile_002", "mobile", "London, UK"),
        SimulatedDevice("iot_003", "iot", "Tokyo, Japan"),
        SimulatedDevice("edge_004", "edge", "Berlin, Germany"),
        SimulatedDevice("mobile_005", "mobile", "Sydney, Australia"),
    ]

    # Register all devices
    logger.info("📝 Registering devices with server...")
    registration_tasks = [device.register_with_server() for device in devices]
    await asyncio.gather(*registration_tasks)

    # Start FL participation for all devices
    logger.info("🎓 Starting federated learning participation...")
    fl_tasks = [device.participate_in_federated_learning() for device in devices]
    await asyncio.gather(*fl_tasks)

def create_device_registration_script():
    """Create a simple script for manual device registration"""
    script_content = '''#!/usr/bin/env python3
"""
Manual Device Registration Script
Use this to register a single device with the QFLARE server
"""

import requests
import json
import secrets
import hashlib

def register_device(device_id, device_type="edge", location="Unknown"):
    """Register a device manually"""

    # Generate mock PQ key pair
    private_key = f"private_key_{secrets.token_hex(32)}"
    public_key = f"public_key_{hashlib.sha256(private_key.encode()).hexdigest()}"

    registration_data = {
        "device_id": device_id,
        "public_key": public_key,
        "device_type": device_type,
        "location": location,
        "capabilities": {
            "cpu_cores": 4,
            "memory_gb": 8,
            "storage_gb": 128,
            "supported_algorithms": ["fedavg", "fedprox"],
            "data_samples": 1000
        }
    }

    try:
        response = requests.post(
            "http://localhost:8000/api/v1/devices/register",
            json=registration_data,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Device {device_id} registered successfully!")
            print(f"   Token: {result.get('token', 'N/A')}")
            print(f"   Status: {result.get('status', 'unknown')}")

            # Save credentials
            with open(f"device_{device_id}_credentials.json", "w") as f:
                json.dump({
                    "device_id": device_id,
                    "private_key": private_key,
                    "public_key": public_key,
                    "token": result.get("token"),
                    "server_public_key": result.get("server_public_key")
                }, f, indent=2)

            print(f"   Credentials saved to device_{device_id}_credentials.json")
        else:
            print(f"❌ Registration failed: {response.text}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python register_device.py <device_id> [device_type] [location]")
        print("Example: python register_device.py my_device edge 'New York, USA'")
        sys.exit(1)

    device_id = sys.argv[1]
    device_type = sys.argv[2] if len(sys.argv) > 2 else "edge"
    location = sys.argv[3] if len(sys.argv) > 3 else "Unknown"

    register_device(device_id, device_type, location)
'''

    with open("register_device.py", "w") as f:
        f.write(script_content)

    logger.info("📝 Created register_device.py for manual device registration")

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="QFLARE Device Simulator")
    parser.add_argument("--server", default="http://localhost:8000",
                       help="QFLARE server URL")
    parser.add_argument("--devices", type=int, default=5,
                       help="Number of devices to simulate")
    parser.add_argument("--create-script", action="store_true",
                       help="Create manual registration script and exit")

    args = parser.parse_args()

    if args.create_script:
        create_device_registration_script()
        logger.info("📄 Manual registration script created: register_device.py")
        return

    # Create device simulation script
    create_device_registration_script()

    # Run the simulation
    asyncio.run(run_device_simulation())

if __name__ == "__main__":
    main()