#!/usr/bin/env python3
"""
QFLARE Development Server Startup Script

This script starts both the backend API server and provides instructions
for starting the frontend development server.
"""

import subprocess
import sys
import time
import requests
import json
from pathlib import Path

def check_backend_health(max_retries=30):
    """Check if backend is healthy"""
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
        print(f"Waiting for backend... ({i+1}/{max_retries})")
    return False

def start_backend():
    """Start the backend server"""
    print("🚀 Starting QFLARE Backend Server...")
    
    backend_dir = Path(__file__).parent
    python_exe = backend_dir.parent / "qflare-env" / "Scripts" / "python.exe"
    
    # Start backend server without reload to prevent restart issues
    cmd = [
        str(python_exe),
        "-c",
        """
import uvicorn
import sys
import os
sys.path.insert(0, os.getcwd())
from professional_backend import app
uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="info")
"""
    ]
    
    return subprocess.Popen(cmd, cwd=backend_dir)

def main():
    print("=" * 60)
    print("🛡️  QFLARE Development Environment Setup")
    print("=" * 60)
    
    # Start backend
    backend_process = start_backend()
    
    # Wait for backend to be healthy
    print("\n⏳ Waiting for backend to start...")
    if check_backend_health():
        print("✅ Backend is running successfully on http://localhost:8000")
        
        # Test API endpoints
        try:
            response = requests.get("http://localhost:8000/api/clients")
            clients_data = response.json()
            print(f"✅ API working - {len(clients_data['clients'])} mock clients loaded")
        except Exception as e:
            print(f"⚠️  API test warning: {e}")
    else:
        print("❌ Backend failed to start properly")
        backend_process.terminate()
        return
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("=" * 60)
    print("\n1. Backend Server: ✅ RUNNING")
    print("   URL: http://localhost:8000")
    print("   API Docs: http://localhost:8000/api/docs")
    print("   Health: http://localhost:8000/health")
    
    print("\n2. Frontend Server: ⏳ Start with:")
    print("   cd frontend")
    print("   npm start")
    print("   Then visit: http://localhost:3000")
    
    print("\n3. Documentation Portal:")
    print("   cd docs/portal/build")
    print("   python -m http.server 8080")
    print("   Then visit: http://localhost:8080")
    
    print("\n📊 Backend API Endpoints:")
    print("   GET  /health              - Health check")
    print("   GET  /api/clients         - List all clients")
    print("   GET  /api/metrics         - Current metrics") 
    print("   GET  /api/training/status - Training status")
    print("   POST /api/training/start  - Start training")
    print("   POST /api/training/stop   - Stop training")
    print("   WS   /ws                  - WebSocket for real-time updates")
    
    print(f"\n🔧 Backend Process ID: {backend_process.pid}")
    print("   Press Ctrl+C to stop the backend server")
    
    try:
        # Keep backend running
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down backend server...")
        backend_process.terminate()
        backend_process.wait()
        print("✅ Backend server stopped")

if __name__ == "__main__":
    main()