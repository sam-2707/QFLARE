#!/usr/bin/env python3
"""
Simple start script for cloud deployment
"""

import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port_str = os.getenv("PORT", "8000")
    
    # Handle PORT environment variable properly
    try:
        port = int(port_str)
    except ValueError:
        print(f"Warning: Invalid PORT value '{port_str}', using default 8000")
        port = 8000
    
    print(f"🚀 Starting QFLARE Server on {host}:{port}")
    
    # Start the server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    ) 