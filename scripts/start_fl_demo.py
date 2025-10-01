"""
QFLARE FL Demo Startup Script

This script starts the server and runs the complete FL demonstration.
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def start_server():
    """Start the QFLARE server."""
    print("🚀 Starting QFLARE server...")
    
    # Get script directory
    script_dir = Path(__file__).parent.parent
    server_dir = script_dir / "server"
    
    # Start server
    server_process = subprocess.Popen([
        sys.executable, "simple_server.py"
    ], cwd=server_dir)
    
    return server_process

def run_fl_demo():
    """Run the FL demonstration."""
    print("🤖 Running FL demonstration...")
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Run FL demo
    demo_process = subprocess.run([
        sys.executable, "fl_demo_complete.py"
    ], cwd=script_dir)
    
    return demo_process.returncode == 0

def main():
    """Main function."""
    print("=" * 60)
    print("🔬 QFLARE Federated Learning Demo")
    print("=" * 60)
    
    try:
        # Start server
        server_process = start_server()
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(10)
        
        # Run demo
        success = run_fl_demo()
        
        if success:
            print("✅ Demo completed successfully!")
            print("🌐 Server is still running for further exploration")
            print("📊 Visit http://localhost:8080/fl-dashboard to see FL dashboard")
            print("⚛️ Visit http://localhost:3000/fl for React UI (if frontend is running)")
            print("\nPress Ctrl+C to stop the server...")
            
            # Keep server running
            try:
                server_process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
                server_process.terminate()
                server_process.wait()
        else:
            print("❌ Demo failed")
            server_process.terminate()
            server_process.wait()
    
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted")
        if 'server_process' in locals():
            server_process.terminate()
            server_process.wait()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        if 'server_process' in locals():
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    main()