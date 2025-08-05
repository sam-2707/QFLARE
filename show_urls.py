#!/usr/bin/env python3
"""
Script to show all available URLs for accessing the QFLARE server.
"""

import socket
import subprocess
import os

def get_ip_addresses():
    """Get all available IP addresses."""
    ip_addresses = []
    
    try:
        # Get hostname
        hostname = socket.gethostname()
        
        # Get localhost
        ip_addresses.append(("localhost", "127.0.0.1"))
        
        # Get all IP addresses
        result = subprocess.run(['ipconfig'], capture_output=True, text=True, shell=True)
        lines = result.stdout.split('\n')
        
        for line in lines:
            if 'IPv4 Address' in line and ':' in line:
                ip = line.split(':')[-1].strip().strip('.')
                if ip and ip != '127.0.0.1':
                    ip_addresses.append((f"IP {len(ip_addresses)}", ip))
                    
    except Exception as e:
        print(f"Error getting IP addresses: {e}")
        ip_addresses.append(("localhost", "127.0.0.1"))
    
    return ip_addresses

def show_urls():
    """Show all available URLs for the QFLARE server."""
    print("🌐 QFLARE Server - Available Access URLs")
    print("=" * 60)
    
    ip_addresses = get_ip_addresses()
    port = 8000
    
    print(f"\n🚀 Server is running on port {port}")
    print("📋 You can access it using any of these URLs:")
    print()
    
    # Main dashboard URLs
    print("📊 MAIN DASHBOARD:")
    for name, ip in ip_addresses:
        print(f"   {name:12} → http://{ip}:{port}/")
    
    print()
    print("🔍 HEALTH CHECK:")
    for name, ip in ip_addresses:
        print(f"   {name:12} → http://{ip}:{port}/health")
    
    print()
    print("📱 DEVICE MANAGEMENT:")
    for name, ip in ip_addresses:
        print(f"   {name:12} → http://{ip}:{port}/devices")
    
    print()
    print("📝 DEVICE REGISTRATION:")
    for name, ip in ip_addresses:
        print(f"   {name:12} → http://{ip}:{port}/register")
    
    print()
    print("🔧 API ENDPOINTS:")
    for name, ip in ip_addresses:
        print(f"   {name:12} → http://{ip}:{port}/api/enroll (POST)")
        print(f"   {name:12} → http://{ip}:{port}/api/challenge (POST)")
        print(f"   {name:12} → http://{ip}:{port}/api/submit_model (POST)")
        break  # Only show once for API endpoints
    
    print()
    print("💡 RECOMMENDATIONS:")
    print("   • For personal use: http://localhost:8000/")
    print("   • For network access: Use your IP address URLs")
    print("   • For mobile testing: Use IP address from other devices")
    
    print()
    print("🔒 SECURITY NOTES:")
    print("   • localhost = Only accessible from this computer")
    print("   • IP addresses = Accessible from other devices on network")
    print("   • Use firewall rules for production deployment")
    
    return ip_addresses

if __name__ == "__main__":
    show_urls() 