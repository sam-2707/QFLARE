#!/usr/bin/env python3
"""
QFLARE Simple Startup Script
Starts the QFLARE server with basic functionality for production deployment
"""

import os
import sys
import asyncio
import uvicorn
from pathlib import Path

# Add the server directory to Python path
server_dir = Path(__file__).parent / "server"
sys.path.insert(0, str(server_dir))

async def main():
    """Main entry point for QFLARE server"""
    print("🚀 Starting QFLARE Production Server...")
    
    # Set environment variables
    os.environ['QFLARE_MODE'] = 'production'
    os.environ['PYTHONPATH'] = str(Path(__file__).parent)
    
    # Configuration
    config = {
        "app": "server.main:app",
        "host": "0.0.0.0",
        "port": 8000,
        "log_level": "info",
        "access_log": True,
        "loop": "asyncio",
    }
    
    print(f"🌐 Server will start on http://{config['host']}:{config['port']}")
    print("📊 Dashboard available at: http://localhost:8000/dashboard")
    print("🔧 API documentation: http://localhost:8000/docs")
    print("")
    
    # Start the server
    try:
        await uvicorn.run(**config)
    except KeyboardInterrupt:
        print("\n👋 QFLARE server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())