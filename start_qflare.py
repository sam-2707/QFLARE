#!/usr/bin/env python3
"""
QFLARE Server Startup Script
Run this from the project root directory.
"""

import sys
import os
from unittest.mock import MagicMock

# Mock liboqs to avoid import issues
sys.modules['oqs'] = MagicMock()

def start_qflare_server():
    """Start the QFLARE server from project root."""
    print("🚀 Starting QFLARE Server...")
    print("=" * 50)
    
    try:
        # Add server directory to Python path
        project_root = os.path.dirname(__file__)
        server_dir = os.path.join(project_root, 'server')
        
        # Add both project root and server directory to Python path
        sys.path.insert(0, project_root)
        sys.path.insert(0, server_dir)
        
        print(f"✅ Added paths to Python: {project_root}, {server_dir}")
        
        # Change to server directory for file operations
        os.chdir(server_dir)
        print(f"✅ Changed to server directory: {server_dir}")
        
        # Import the main app
        from main import app
        print("✅ Server app imported successfully")
        
        # Import uvicorn
        import uvicorn
        
        print("✅ Starting server on http://localhost:8000")
        print("📋 Available endpoints:")
        print("   - GET  /health - Health check")
        print("   - GET  / - Main dashboard")
        print("   - GET  /devices - Device management")
        print("   - GET  /register - Device registration")
        print("   - GET  /api/request_qkey - Quantum key generation")
        print("   - POST /api/enroll - Device enrollment")
        print("   - POST /api/challenge - Session challenge")
        print("   - POST /api/submit_model - Model submission")
        print("   - GET  /api/global_model - Global model download")
        print("\n🔧 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Start the server
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    start_qflare_server() 