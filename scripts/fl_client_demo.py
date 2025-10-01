"""
QFLARE Edge Node FL Client

A simple FL client that can participate in federated learning rounds.
This is a simplified version for demonstration purposes.
"""

import requests
import time
import json
import logging
import random
import numpy as np
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleFLClient:
    """
    Simple FL client that can participate in training rounds.
    """
    
    def __init__(self, 
                 device_id: str,
                 server_url: str = "http://localhost:8080",
                 auto_participate: bool = True):
        """
        Initialize FL client.
        
        Args:
            device_id: Unique device identifier
            server_url: Server URL
            auto_participate: Whether to automatically participate in rounds
        """
        self.device_id = device_id
        self.server_url = server_url
        self.auto_participate = auto_participate
        self.session = requests.Session()
        
        logger.info(f"FL Client initialized: {device_id}")
    
    def check_fl_status(self) -> Optional[Dict[str, Any]]:
        """Check current FL status."""
        try:
            response = self.session.get(f"{self.server_url}/api/fl/status")
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return data.get("fl_status")
            return None
        except Exception as e:
            logger.error(f"Error checking FL status: {e}")
            return None
    
    def participate_in_round(self) -> bool:
        """
        Participate in current training round if selected.
        
        Returns:
            True if participated successfully, False otherwise
        """
        try:
            # Check if we're selected for current round
            fl_status = self.check_fl_status()
            if not fl_status:
                return False
            
            if fl_status["status"] != "training":
                logger.debug(f"No active training round (status: {fl_status['status']})")
                return False
            
            # Check if we're a participant
            # Note: This is simplified - in real implementation, 
            # the server would notify selected devices
            
            # Simulate local training
            logger.info(f"Starting local training for device {self.device_id}")
            training_result = self.simulate_local_training()
            
            # Submit model update
            return self.submit_model_update(training_result)
            
        except Exception as e:
            logger.error(f"Error participating in round: {e}")
            return False
    
    def simulate_local_training(self) -> Dict[str, Any]:
        """
        Simulate local training process.
        
        Returns:
            Training results dictionary
        """
        # Simulate training time
        training_time = random.uniform(2, 5)
        time.sleep(training_time)
        
        # Simulate training metrics
        initial_loss = random.uniform(2.0, 3.0)
        final_loss = initial_loss - random.uniform(0.1, 0.5)
        
        # Simulate model parameters (very simplified)
        model_params = {
            "conv1_weight": np.random.randn(32, 1, 3, 3).tolist(),
            "conv1_bias": np.random.randn(32).tolist(),
            "fc_weight": np.random.randn(10, 128).tolist(),
            "fc_bias": np.random.randn(10).tolist()
        }
        
        result = {
            "training_loss": final_loss,
            "local_epochs": 5,
            "num_samples": random.randint(100, 500),
            "model_data": json.dumps(model_params),
            "training_time": training_time
        }
        
        logger.info(f"Local training completed - Loss: {final_loss:.4f}, Samples: {result['num_samples']}")
        return result
    
    def submit_model_update(self, training_result: Dict[str, Any]) -> bool:
        """
        Submit model update to server.
        
        Args:
            training_result: Training results from local training
            
        Returns:
            True if submission successful, False otherwise
        """
        try:
            # Prepare form data
            form_data = {
                "device_id": self.device_id,
                "training_loss": training_result["training_loss"],
                "local_epochs": training_result["local_epochs"],
                "num_samples": training_result["num_samples"],
                "model_data": training_result["model_data"]
            }
            
            # Submit to server
            response = self.session.post(
                f"{self.server_url}/api/fl/submit_model",
                data=form_data
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    logger.info(f"Model update submitted successfully")
                    logger.info(f"Round completion: {data['submission_info']['submitted_count']}/{data['submission_info']['total_participants']}")
                    return True
            
            logger.error(f"Model submission failed: {response.status_code} - {response.text}")
            return False
            
        except Exception as e:
            logger.error(f"Error submitting model update: {e}")
            return False
    
    def get_global_model(self) -> Optional[Dict[str, Any]]:
        """
        Download global model from server.
        
        Returns:
            Global model dictionary or None if failed
        """
        try:
            response = self.session.get(
                f"{self.server_url}/api/fl/global_model",
                params={"device_id": self.device_id}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return data.get("global_model")
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting global model: {e}")
            return None
    
    def run_client_loop(self, check_interval: int = 30):
        """
        Run client loop to automatically participate in rounds.
        
        Args:
            check_interval: How often to check for new rounds (seconds)
        """
        logger.info(f"Starting FL client loop (checking every {check_interval}s)")
        
        while True:
            try:
                if self.auto_participate:
                    fl_status = self.check_fl_status()
                    if fl_status and fl_status["status"] == "training":
                        # Try to participate (simplified logic)
                        self.participate_in_round()
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Client loop stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in client loop: {e}")
                time.sleep(check_interval)


def main():
    """Main function to run FL client."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QFLARE FL Client")
    parser.add_argument("--device-id", default="edge_device_001", help="Device ID")
    parser.add_argument("--server-url", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--auto", action="store_true", default=True, help="Auto-participate in rounds")
    parser.add_argument("--check-interval", type=int, default=30, help="Check interval in seconds")
    
    args = parser.parse_args()
    
    # Create FL client
    client = SimpleFLClient(
        device_id=args.device_id,
        server_url=args.server_url,
        auto_participate=args.auto
    )
    
    # Test connection
    fl_status = client.check_fl_status()
    if fl_status:
        logger.info(f"Connected to FL server - Status: {fl_status['status']}")
        logger.info(f"Current round: {fl_status['current_round']}")
    else:
        logger.error("Failed to connect to FL server")
        return
    
    # Run client loop
    try:
        client.run_client_loop(args.check_interval)
    except KeyboardInterrupt:
        logger.info("FL client stopped")


if __name__ == "__main__":
    main()