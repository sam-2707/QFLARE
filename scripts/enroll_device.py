#!/usr/bin/env python3
"""
Device Enrollment Script for QFLARE

This script allows a new device to enroll with the QFLARE server using
a secure one-time enrollment token.
"""

import sys
from pathlib import Path
import os
import argparse
import requests
import base64
import json
from typing import Optional, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from edge_node.auth.pqcrypto_utils import generate_device_keypair


def enroll_device(server_url: str, device_id: str, enrollment_token: str, 
                 verify_ssl: bool = True) -> bool:
    """
    Enroll a device with the QFLARE server.
    
    Args:
        server_url: Server URL
        device_id: Device identifier
        enrollment_token: One-time enrollment token
        verify_ssl: Whether to verify SSL certificates
        
    Returns:
        True if enrollment successful, False otherwise
    """
    try:
        print(f"Enrolling device {device_id} with server {server_url}")
        
        # Generate device key pair
        print("Generating Post-Quantum key pair...")
        kem_public_key, sig_public_key = generate_device_keypair(device_id)
        
        # Prepare enrollment request
        enrollment_data = {
            "device_id": device_id,
            "enrollment_token": enrollment_token,
            "kem_public_key": kem_public_key,
            "signature_public_key": sig_public_key
        }
        
        # Send enrollment request
        print("Sending enrollment request...")
        response = requests.post(
            f"{server_url}/api/enroll",
            json=enrollment_data,
            verify=verify_ssl,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Device enrolled successfully!")
            print(f"   Device ID: {result.get('device_id')}")
            print(f"   Status: {result.get('status')}")
            if result.get('message'):
                print(f"   Message: {result.get('message')}")
            return True
        else:
            print(f"❌ Enrollment failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server at {server_url}")
        print("   Please check that the server is running and accessible")
        return False
    except requests.exceptions.SSLError:
        print(f"❌ SSL certificate verification failed")
        print("   Use --no-verify-ssl to skip SSL verification (not recommended for production)")
        return False
    except Exception as e:
        print(f"❌ Enrollment error: {e}")
        return False


def test_connection(server_url: str, verify_ssl: bool = True) -> bool:
    """
    Test connection to the server.
    
    Args:
        server_url: Server URL
        verify_ssl: Whether to verify SSL certificates
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        print(f"Testing connection to {server_url}...")
        response = requests.get(
            f"{server_url}/health",
            verify=verify_ssl,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Server is accessible")
            return True
        else:
            print(f"❌ Server returned status code {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server at {server_url}")
        return False
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Enroll a device with the QFLARE server"
    )
    parser.add_argument(
        "--server-url", 
        default="https://localhost:8000",
        help="Server URL (default: https://localhost:8000)"
    )
    parser.add_argument(
        "--device-id", 
        required=True,
        help="Device identifier"
    )
    parser.add_argument(
        "--enrollment-token", 
        required=True,
        help="One-time enrollment token"
    )
    parser.add_argument(
        "--no-verify-ssl", 
        action="store_true",
        help="Skip SSL certificate verification (not recommended for production)"
    )
    parser.add_argument(
        "--test-connection", 
        action="store_true",
        help="Test connection to server before enrollment"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.device_id or not args.enrollment_token:
        print("❌ Device ID and enrollment token are required")
        sys.exit(1)
    
    # Test connection if requested
    if args.test_connection:
        if not test_connection(args.server_url, not args.no_verify_ssl):
            sys.exit(1)
    
    # Perform enrollment
    success = enroll_device(
        args.server_url,
        args.device_id,
        args.enrollment_token,
        not args.no_verify_ssl
    )
    
    if success:
        print("\n🎉 Device enrollment completed successfully!")
        print("\nNext steps:")
        print("1. Start the edge node with the same device ID")
        print("2. The device will automatically participate in federated learning")
        print("3. Monitor the server dashboard for device activity")
        sys.exit(0)
    else:
        print("\n❌ Device enrollment failed")
        print("\nTroubleshooting:")
        print("1. Check that the server is running and accessible")
        print("2. Verify the enrollment token is valid and not expired")
        print("3. Ensure the device ID is unique")
        print("4. Check server logs for more details")
        sys.exit(1)


if __name__ == "__main__":
    main()