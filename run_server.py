#!/usr/bin/env python3
"""
QFLARE Server Startup Script

This script starts the QFLARE server and provides connection information.
"""

import os
import sys
import socket
import uvicorn
from pathlib import Path
from dotenv import load_dotenv

# Add server directory to path
server_dir = Path(__file__).parent / "server"
sys.path.insert(0, str(server_dir))

def get_local_ip():
    """Get the local IP address."""
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"

def main():
    """Main startup function."""
    print("=" * 60)
    print("🚀 QFLARE Server - Quantum-Resistant Federated Learning")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    
    # Get local IP for external access
    local_ip = get_local_ip()
    
    print(f"\n📋 Configuration:")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Local IP: {local_ip}")
    
    # Check if database exists
    db_path = Path("qflare.db")
    if db_path.exists():
        print(f"   Database: {db_path} (exists)")
    else:
        print(f"   Database: {db_path} (will be created)")
    
    print(f"\n🌐 Server URLs:")
    print(f"   Local:     http://localhost:{port}")
    print(f"   Network:   http://{local_ip}:{port}")
    
    if host == "0.0.0.0":
        print(f"   External:  http://{local_ip}:{port}")
    
    print(f"\n📱 Device Connection:")
    print(f"   Other systems can connect to: http://{local_ip}:{port}")
    print(f"   API Documentation: http://{local_ip}:{port}/docs")
    print(f"   Health Check: http://{local_ip}:{port}/health")
    
    print(f"\n🔐 Security Features:")
    print(f"   • Post-quantum cryptography")
    print(f"   • Secure device enrollment")
    print(f"   • Key rotation and management")
    print(f"   • Rate limiting and protection")
    
    print(f"\n📊 Management Interface:")
    print(f"   Dashboard: http://{local_ip}:{port}/")
    print(f"   Devices:   http://{local_ip}:{port}/devices")
    
    print(f"\n⏳ Starting server...")
    print("=" * 60)
    
    try:
        # Start the server
        uvicorn.run(
            "server.main:app",
            host=host,
            port=port,
            reload=False,  # Set to True for development
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 