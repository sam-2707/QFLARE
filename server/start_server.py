#!/usr/bin/env python3
"""
Simple server startup script that bypasses liboqs issues.
"""

import sys
import os
import unittest.mock

# Mock liboqs before any imports
sys.modules['oqs'] = unittest.mock.MagicMock()

def start_server():
    """Start the QFLARE server."""
    print("🚀 Starting QFLARE Server...")
    print("=" * 50)
    
    try:
        # Import the main app (using relative imports since we're in server directory)
        from main import app
        print("✅ Server app imported successfully")
        
        # Import uvicorn
        import uvicorn
        
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
        print(f"❌ Failed to start server: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    start_server() 