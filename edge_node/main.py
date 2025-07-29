"""
QFLARE Edge Node - Main Application

This is the main application for edge devices participating in federated learning.
"""

import asyncio
import logging
import time
import base64
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from secure_comm import authenticate_with_server, establish_secure_session
from trainer import train_local_model
from data_loader import load_local_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SERVER_URL = os.getenv("SERVER_URL", "https://localhost:8000")
DEVICE_ID = os.getenv("DEVICE_ID", "edge_device_001")
ENROLLMENT_TOKEN = os.getenv("ENROLLMENT_TOKEN", None)
VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() == "true"

# Retry configuration for network requests
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("http://", adapter)
session.mount("https://", adapter)

if not VERIFY_SSL:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def wait_for_server(url: str, timeout: int = 60) -> bool:
    """
    Wait for server to become available.
    
    Args:
        url: Server URL
        timeout: Timeout in seconds
        
    Returns:
        True if server is available, False otherwise
    """
    logger.info(f"Waiting for server at {url}")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = session.get(f"{url}/health", verify=VERIFY_SSL, timeout=5)
            if response.status_code == 200:
                logger.info("Server is available")
                return True
        except Exception as e:
            logger.debug(f"Server not ready yet: {e}")
        
        time.sleep(2)
    
    logger.error("Server not available within timeout")
    return False


async def enroll_device() -> bool:
    """
    Enroll device with the server using secure enrollment token.
    
    Returns:
        True if enrollment successful, False otherwise
    """
    try:
        if not ENROLLMENT_TOKEN:
            logger.error("No enrollment token provided")
            return False
        
        # Generate device key pair
        from auth.pqcrypto_utils import generate_device_keypair
        kem_public_key, sig_public_key = generate_device_keypair(DEVICE_ID)
        
        # Prepare enrollment request
        enrollment_data = {
            "device_id": DEVICE_ID,
            "enrollment_token": ENROLLMENT_TOKEN,
            "kem_public_key": kem_public_key,
            "signature_public_key": sig_public_key
        }
        
        # Send enrollment request
        response = session.post(
            f"{SERVER_URL}/api/enroll",
            json=enrollment_data,
            verify=VERIFY_SSL,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Device enrolled successfully: {result}")
            return True
        else:
            logger.error(f"Enrollment failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error during enrollment: {e}")
        return False


async def request_session_challenge() -> Optional[str]:
    """
    Request a session challenge from the server.
    
    Returns:
        Encrypted session key, or None if failed
    """
    try:
        challenge_data = {"device_id": DEVICE_ID}
        
        response = session.post(
            f"{SERVER_URL}/api/challenge",
            json=challenge_data,
            verify=VERIFY_SSL,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info("Session challenge received")
            return result.get("challenge")
        else:
            logger.error(f"Challenge request failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error requesting session challenge: {e}")
        return None


async def download_global_model() -> Optional[bytes]:
    """
    Download the current global model from the server.
    
    Returns:
        Global model weights as bytes, or None if failed
    """
    try:
        response = session.get(
            f"{SERVER_URL}/api/global_model",
            verify=VERIFY_SSL,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            model_weights_b64 = result.get("model_weights")
            if model_weights_b64:
                model_weights = base64.b64decode(model_weights_b64)
                logger.info("Global model downloaded successfully")
                return model_weights
            else:
                logger.warning("No global model available")
                return None
        else:
            logger.error(f"Model download failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error downloading global model: {e}")
        return None


async def submit_model_update(model_weights: bytes, metadata: Dict[str, Any] = None) -> bool:
    """
    Submit model update to the server.
    
    Args:
        model_weights: Model weights as bytes
        metadata: Additional metadata
        
    Returns:
        True if submission successful, False otherwise
    """
    try:
        # Sign the model update
        from auth.pqcrypto_utils import sign_model_update
        signature = sign_model_update(DEVICE_ID, model_weights)
        
        # Prepare submission data
        submission_data = {
            "device_id": DEVICE_ID,
            "model_weights": base64.b64encode(model_weights).decode('utf-8'),
            "signature": base64.b64encode(signature).decode('utf-8'),
            "metadata": metadata or {}
        }
        
        response = session.post(
            f"{SERVER_URL}/api/submit_model",
            json=submission_data,
            verify=VERIFY_SSL,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Model update submitted successfully: {result}")
            return True
        else:
            logger.error(f"Model submission failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error submitting model update: {e}")
        return False


async def main():
    """
    Main federated learning loop for edge device.
    """
    logger.info(f"Starting QFLARE edge node: {DEVICE_ID}")
    
    # Wait for server to be available
    if not wait_for_server(SERVER_URL):
        logger.error("Cannot connect to server")
        return
    
    # Enroll device if not already enrolled
    if ENROLLMENT_TOKEN:
        logger.info("Enrolling device with server")
        if not await enroll_device():
            logger.error("Device enrollment failed")
            return
    else:
        logger.info("No enrollment token provided, assuming device is already enrolled")
    
    # Main federated learning loop
    round_number = 0
    while True:
        try:
            round_number += 1
            logger.info(f"Starting federated learning round {round_number}")
            
            # Request session challenge
            challenge = await request_session_challenge()
            if not challenge:
                logger.error("Failed to get session challenge")
                await asyncio.sleep(30)
                continue
            
            # Download global model
            global_model = await download_global_model()
            if not global_model:
                logger.warning("No global model available, skipping round")
                await asyncio.sleep(60)
                continue
            
            # Train local model
            logger.info("Training local model")
            local_data = load_local_data()
            if not local_data:
                logger.warning("No local data available, skipping round")
                await asyncio.sleep(60)
                continue
            
            trained_model = train_local_model(global_model, local_data)
            if not trained_model:
                logger.error("Local training failed")
                await asyncio.sleep(60)
                continue
            
            # Submit model update
            metadata = {
                "round": round_number,
                "device_id": DEVICE_ID,
                "timestamp": time.time(),
                "data_samples": len(local_data)
            }
            
            if await submit_model_update(trained_model, metadata):
                logger.info(f"Round {round_number} completed successfully")
            else:
                logger.error(f"Round {round_number} failed")
            
            # Wait before next round
            await asyncio.sleep(300)  # 5 minutes between rounds
            
        except KeyboardInterrupt:
            logger.info("Shutting down edge node")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            await asyncio.sleep(60)


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the main loop
    asyncio.run(main())
