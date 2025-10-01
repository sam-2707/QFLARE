"""
QFLARE Federated Learning Demo

This script demonstrates the complete FL workflow:
1. Start server
2. Register devices
3. Start FL training round
4. Simulate edge device participation
5. Show results
"""

import requests
import time
import json
import subprocess
import threading
import logging
from typing import List, Dict, Any
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FLDemo:
    """Complete FL demonstration."""
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url
        self.session = requests.Session()
        self.demo_devices = []
    
    def wait_for_server(self, timeout: int = 60) -> bool:
        """Wait for server to be available."""
        logger.info("Waiting for server to be available...")
        
        for i in range(timeout):
            try:
                response = self.session.get(f"{self.server_url}/")
                if response.status_code == 200:
                    logger.info("✅ Server is available")
                    return True
            except:
                pass
            time.sleep(1)
        
        logger.error("❌ Server not available")
        return False
    
    def register_demo_devices(self, num_devices: int = 5) -> List[str]:
        """Register demo devices for FL."""
        logger.info(f"Registering {num_devices} demo devices...")
        
        device_ids = []
        
        for i in range(num_devices):
            device_id = f"fl_demo_device_{i+1:03d}"
            device_ids.append(device_id)
            
            # Register device with mock data
            registration_data = {
                "device_id": device_id,
                "device_type": "edge_computer",
                "organization": "QFLARE Demo",
                "contact_email": f"demo{i+1}@qflare.ai",
                "phone_number": f"+1555000{i+1:04d}",
                "use_case": "federated_learning_demo",
                "key_exchange_method": "qr_otp"
            }
            
            try:
                response = self.session.post(
                    f"{self.server_url}/api/secure_register",
                    data=registration_data
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        logger.info(f"✅ Registered device: {device_id}")
                        
                        # Auto-verify device (demo only)
                        self.auto_verify_device(device_id)
                    else:
                        logger.warning(f"⚠️ Registration failed for {device_id}: {data}")
                else:
                    logger.warning(f"⚠️ Registration failed for {device_id}: {response.status_code}")
            
            except Exception as e:
                logger.error(f"❌ Error registering {device_id}: {e}")
        
        self.demo_devices = device_ids
        logger.info(f"✅ Registered {len(device_ids)} devices")
        return device_ids
    
    def auto_verify_device(self, device_id: str):
        """Auto-verify device for demo (simplified)."""
        try:
            # In a real scenario, this would require proper OTP verification
            # For demo, we'll mark device as enrolled directly
            verification_data = {"otp": "123456"}  # Demo OTP
            
            response = self.session.post(
                f"{self.server_url}/api/secure_verify/{device_id}",
                data=verification_data
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    logger.info(f"✅ Auto-verified device: {device_id}")
                else:
                    logger.warning(f"⚠️ Verification failed for {device_id}")
            
        except Exception as e:
            logger.error(f"❌ Error verifying {device_id}: {e}")
    
    def check_fl_status(self) -> Dict[str, Any]:
        """Check FL system status."""
        try:
            response = self.session.get(f"{self.server_url}/api/fl/status")
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return data.get("fl_status", {})
            return {}
        except Exception as e:
            logger.error(f"Error checking FL status: {e}")
            return {}
    
    def start_fl_round(self, target_participants: int = 3) -> bool:
        """Start a new FL training round."""
        logger.info(f"Starting FL round with {target_participants} participants...")
        
        try:
            form_data = {
                "target_participants": str(target_participants),
                "local_epochs": "5",
                "learning_rate": "0.01"
            }
            
            response = self.session.post(
                f"{self.server_url}/api/fl/start_round",
                data=form_data
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    logger.info(f"✅ FL round started: {data['round_info']['round_number']}")
                    logger.info(f"Participants: {data['round_info']['participants']}")
                    return True
            
            logger.error(f"❌ Failed to start FL round: {response.text}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error starting FL round: {e}")
            return False
    
    def simulate_device_participation(self, device_id: str) -> bool:
        """Simulate device participation in FL round."""
        try:
            # Simulate local training
            time.sleep(random.uniform(2, 5))  # Training time
            
            # Generate mock training results
            training_loss = random.uniform(0.5, 2.0)
            num_samples = random.randint(100, 500)
            
            # Create mock model data
            model_data = json.dumps({
                "device_id": device_id,
                "trained_parameters": [random.random() for _ in range(100)],
                "timestamp": time.time()
            })
            
            # Submit model update
            form_data = {
                "device_id": device_id,
                "training_loss": str(training_loss),
                "local_epochs": "5",
                "num_samples": str(num_samples),
                "model_data": model_data
            }
            
            response = self.session.post(
                f"{self.server_url}/api/fl/submit_model",
                data=form_data
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    logger.info(f"✅ {device_id} submitted model (loss: {training_loss:.4f})")
                    return True
            
            logger.error(f"❌ {device_id} failed to submit model: {response.text}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error in device participation for {device_id}: {e}")
            return False
    
    def run_complete_demo(self):
        """Run complete FL demonstration."""
        logger.info("🚀 Starting QFLARE Federated Learning Demo")
        logger.info("=" * 60)
        
        # Step 1: Wait for server
        if not self.wait_for_server():
            logger.error("❌ Demo failed: Server not available")
            return
        
        # Step 2: Register demo devices
        device_ids = self.register_demo_devices(5)
        if not device_ids:
            logger.error("❌ Demo failed: No devices registered")
            return
        
        time.sleep(2)  # Wait for registration to complete
        
        # Step 3: Check initial FL status
        fl_status = self.check_fl_status()
        logger.info(f"FL Status: {fl_status.get('status', 'unknown')}")
        logger.info(f"Active devices: {fl_status.get('active_devices', 0)}")
        
        # Step 4: Start FL round
        if not self.start_fl_round(3):
            logger.error("❌ Demo failed: Could not start FL round")
            return
        
        time.sleep(1)
        
        # Step 5: Simulate device participation
        logger.info("🤖 Simulating device participation...")
        participating_devices = device_ids[:3]  # First 3 devices participate
        
        # Run participation in parallel
        threads = []
        for device_id in participating_devices:
            thread = threading.Thread(
                target=self.simulate_device_participation,
                args=(device_id,)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all devices to complete
        for thread in threads:
            thread.join()
        
        # Step 6: Check final results
        time.sleep(2)
        final_status = self.check_fl_status()
        logger.info("📊 Final FL Status:")
        logger.info(f"  Round: {final_status.get('current_round', 0)}")
        logger.info(f"  Status: {final_status.get('status', 'unknown')}")
        logger.info(f"  Training History: {len(final_status.get('training_history', []))} rounds")
        
        # Step 7: Show training history
        training_history = final_status.get('training_history', [])
        if training_history:
            logger.info("📈 Training History:")
            for round_info in training_history[-3:]:  # Last 3 rounds
                logger.info(f"  Round {round_info.get('round', 'N/A')}: "
                          f"{round_info.get('participants', 0)} participants, "
                          f"avg loss: {round_info.get('avg_loss', 0):.4f}")
        
        logger.info("=" * 60)
        logger.info("✅ QFLARE Federated Learning Demo Completed!")
        logger.info(f"🌐 View dashboard at: {self.server_url}/fl-dashboard")
        logger.info(f"📊 View React UI at: http://localhost:3000/fl")


def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QFLARE FL Demo")
    parser.add_argument("--server-url", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--devices", type=int, default=5, help="Number of demo devices")
    
    args = parser.parse_args()
    
    # Run demo
    demo = FLDemo(server_url=args.server_url)
    demo.run_complete_demo()


if __name__ == "__main__":
    main()