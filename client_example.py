#!/usr/bin/env python3
"""
QFLARE Client Example

This script demonstrates how other systems can connect to the QFLARE server.
"""

import requests
import json
import time
from typing import Dict, Any, Optional

class QFLAREClient:
    """Client for connecting to QFLARE server."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'QFLARE-Client/1.0.0'
        })
    
    def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        try:
            response = self.session.get(f"{self.server_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information."""
        try:
            response = self.session.get(f"{self.server_url}/api/server_info")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def get_devices(self) -> Dict[str, Any]:
        """Get list of registered devices."""
        try:
            response = self.session.get(f"{self.server_url}/api/devices")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def generate_token(self, device_id: str, expiration_hours: int = 24) -> Dict[str, Any]:
        """Generate enrollment token for device."""
        try:
            data = {
                "device_id": device_id,
                "expiration_hours": expiration_hours
            }
            response = self.session.post(f"{self.server_url}/api/generate_token", json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def update_device_status(self, device_id: str, status: str) -> Dict[str, Any]:
        """Update device status."""
        try:
            data = {"status": status}
            response = self.session.put(f"{self.server_url}/api/devices/{device_id}/status", json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def rotate_keys(self) -> Dict[str, Any]:
        """Rotate server keys."""
        try:
            response = self.session.post(f"{self.server_url}/api/rotate_keys")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}


def main():
    """Main function demonstrating client usage."""
    
    # Configuration
    SERVER_URL = "http://localhost:8000"  # Change to your server URL
    
    print("=" * 60)
    print("🔌 QFLARE Client Example")
    print("=" * 60)
    
    # Create client
    client = QFLAREClient(SERVER_URL)
    
    # Test server connection
    print(f"\n📡 Connecting to server: {SERVER_URL}")
    
    # Health check
    print("\n🏥 Health Check:")
    health = client.health_check()
    if "error" in health:
        print(f"❌ Error: {health['error']}")
        return
    else:
        print(f"✅ Status: {health.get('status', 'unknown')}")
        print(f"📊 Statistics: {health.get('statistics', {})}")
    
    # Get server info
    print("\n📋 Server Information:")
    info = client.get_server_info()
    if "error" in info:
        print(f"❌ Error: {info['error']}")
    else:
        server_info = info.get('server_info', {})
        print(f"   Name: {server_info.get('name', 'Unknown')}")
        print(f"   Version: {server_info.get('version', 'Unknown')}")
        print(f"   Host: {server_info.get('host', 'Unknown')}")
        print(f"   Port: {server_info.get('port', 'Unknown')}")
        
        security = info.get('security', {})
        print(f"   KEM Algorithm: {security.get('kem_algorithm', 'Unknown')}")
        print(f"   Signature Algorithm: {security.get('signature_algorithm', 'Unknown')}")
    
    # Get devices
    print("\n📱 Registered Devices:")
    devices = client.get_devices()
    if "error" in devices:
        print(f"❌ Error: {devices['error']}")
    else:
        device_list = devices.get('devices', [])
        if device_list:
            for device in device_list:
                status_color = "🟢" if device.get('status') == 'active' else "🔴"
                print(f"   {status_color} {device.get('device_id', 'Unknown')} - {device.get('status', 'unknown')}")
        else:
            print("   No devices registered")
    
    # Generate test token
    print("\n🔑 Generate Test Token:")
    test_device_id = f"test-device-{int(time.time())}"
    token_result = client.generate_token(test_device_id, 1)  # 1 hour expiration
    
    if "error" in token_result:
        print(f"❌ Error: {token_result['error']}")
    else:
        print(f"✅ Token generated for device: {test_device_id}")
        print(f"   Token: {token_result.get('token', 'Unknown')[:50]}...")
        print(f"   Expires: {token_result.get('expires_at', 'Unknown')}")
    
    # Test device status update
    print("\n🔄 Test Device Status Update:")
    if "error" not in token_result:
        status_result = client.update_device_status(test_device_id, "active")
        if "error" in status_result:
            print(f"❌ Error: {status_result['error']}")
        else:
            print(f"✅ Device status updated: {status_result.get('new_status', 'Unknown')}")
    
    print("\n" + "=" * 60)
    print("✅ Client example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main() 