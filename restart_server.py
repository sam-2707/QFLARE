#!/usr/bin/env python3
"""
Script to restart the QFLARE server cleanly.
"""

import sys
import os
import subprocess
import time
import psutil

def kill_process_on_port(port):
    """Kill any process using the specified port."""
    print(f"🔍 Checking for processes on port {port}...")
    
    try:
        # Find processes using the port
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                connections = proc.info['connections']
                for conn in connections:
                    if conn.laddr.port == port:
                        print(f"🔄 Killing process {proc.info['name']} (PID: {proc.info['pid']}) on port {port}")
                        proc.kill()
                        time.sleep(1)
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        print(f"✅ No processes found on port {port}")
        return False
    except Exception as e:
        print(f"⚠️  Could not check processes: {e}")
        return False

def restart_server():
    """Restart the QFLARE server."""
    print("🚀 Restarting QFLARE Server...")
    print("=" * 50)
    
    # Kill any existing processes on port 8000
    kill_process_on_port(8000)
    
    # Wait a moment
    time.sleep(2)
    
    try:
        # Change to server directory
        server_dir = os.path.join(os.path.dirname(__file__), 'server')
        os.chdir(server_dir)
        print(f"✅ Changed to server directory: {server_dir}")
        
        # Mock liboqs before imports
        from unittest.mock import MagicMock
        sys.modules['oqs'] = MagicMock()
        
        # Import and start the server
        from server.main import app
        import uvicorn
        
        print("✅ Server app imported successfully")
        print("✅ Starting server on http://localhost:8000")
        print("📋 Available endpoints:")
        print("   - GET  /health - Health check")
        print("   - GET  / - Main dashboard")
        print("   - GET  /devices - Device management")
        print("   - POST /api/enroll - Device enrollment")
        print("   - POST /api/challenge - Session challenge")
        print("   - POST /api/submit_model - Model submission")
        print("   - GET  /api/global_model - Global model download")
        print("\n🔧 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Start the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ Failed to restart server: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    restart_server() 