#!/usr/bin/env python3
"""
Manual Device Registration Script for QFLARE
"""

import requests
import json
import secrets
import hashlib
import sys

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
            return True
        else:
            print(f"❌ Registration failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Registration error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python register_device.py <device_id> [device_type] [location]")
        print("Example: python register_device.py edge-node-002 edge 'San Francisco, USA'")
        sys.exit(1)

    device_id = sys.argv[1]
    device_type = sys.argv[2] if len(sys.argv) > 2 else "edge"
    location = sys.argv[3] if len(sys.argv) > 3 else "Unknown"

    success = register_device(device_id, device_type, location)
    sys.exit(0 if success else 1)
