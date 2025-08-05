#!/usr/bin/env python3
"""
Simple script to start the QFLARE server.
"""

import sys
import os
from unittest.mock import MagicMock

# Mock liboqs
sys.modules['oqs'] = MagicMock()

# Change to server directory
os.chdir('server')

# Import and run
import sys
sys.path.append('.')
from main import app
import uvicorn

print("🚀 Starting QFLARE Server...")
print("📍 Server will be available at:")
print("   - http://localhost:8000")
print("   - http://172.18.224.1:8000")
print("   - http://0.0.0.0:8000")
print("\n🔧 Press Ctrl+C to stop")

uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") 