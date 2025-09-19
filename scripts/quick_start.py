#!/usr/bin/env python3
"""
QFLARE Quick Start Guide
Everything you need to run and demonstrate QFLARE
"""

import subprocess
import sys
import os
import time
import requests

def print_header(title):
    """Print a nice header"""
    print(f"\n{'=' * 60}")
    print(f"🚀 {title}")
    print(f"{'=' * 60}")

def check_server_running():
    """Check if QFLARE server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=3)
        return response.status_code == 200
    except:
        return False

def main():
    """Quick start guide for QFLARE"""
    print(f"""
🌟 QFLARE QUANTUM-SAFE FEDERATED LEARNING
{'=' * 80}
Complete demonstration guide for quantum-safe AI system

📋 WHAT YOU'LL SEE:
✅ Post-quantum cryptography (CRYSTALS-Kyber-1024, Dilithium-2)
✅ Admin token generation and device enrollment
✅ User self-registration flows
✅ Quantum key exchange and secure communication
✅ Federated learning with quantum security
✅ Real-time system monitoring

🎯 Let's get started!
{'=' * 80}
""")

    # Step 1: Check if server is running
    print_header("STEP 1: SERVER STATUS CHECK")
    
    if check_server_running():
        print("✅ QFLARE server is already running!")
        print("🌐 Server URL: http://localhost:8000")
    else:
        print("⚠️  QFLARE server is not running. Let's start it!")
        print("\n🔧 Starting QFLARE server...")
        print("📝 Run this command in another terminal:")
        print("   cd D:\\QFLARE_Project_Structure")
        print("   python start_qflare.py")
        print("\n⏳ Waiting for you to start the server...")
        
        while not check_server_running():
            print(".", end="", flush=True)
            time.sleep(2)
        
        print("\n✅ Server is now running!")
    
    # Step 2: Validate server endpoints
    print_header("STEP 2: SERVER VALIDATION")
    print("🔧 Running server validation script...")
    print("📝 Command: python scripts/validate_server.py")
    
    try:
        subprocess.run([sys.executable, "scripts/validate_server.py"], 
                      cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    except Exception as e:
        print(f"⚠️  Could not run validation: {e}")
        print("💡 You can run it manually: python scripts/validate_server.py")
    
    # Step 3: Security demonstration
    print_header("STEP 3: SECURITY STRENGTH DEMONSTRATION")
    print("🛡️  Show how strong QFLARE's security is!")
    print("📝 This demonstrates quantum-safe cryptography and security features")
    
    choice = input("\n🔒 Run security demonstration now? (y/n): ").lower().strip()
    
    if choice in ['y', 'yes']:
        print("\n🛡️  Starting security demonstration...")
        print("📝 Command: python scripts/simple_security_demo.py")
        
        try:
            subprocess.run([sys.executable, "scripts/simple_security_demo.py"], 
                          cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        except Exception as e:
            print(f"⚠️  Could not run security demo: {e}")
            print("💡 You can run it manually: python scripts/simple_security_demo.py")
    else:
        print("📝 No problem! You can run the security demo anytime:")
        print("   python scripts/simple_security_demo.py")
    
    # Step 4: Interactive demonstration
    print_header("STEP 4: INTERACTIVE DEMONSTRATION")
    print("🎭 Now for the complete interactive demonstration!")
    print("📝 This will show you all the quantum-safe authentication flows")
    
    choice = input("\n🤔 Run interactive demo now? (y/n): ").lower().strip()
    
    if choice in ['y', 'yes']:
        print("\n🚀 Starting interactive demonstration...")
        print("📝 Command: python scripts/interactive_demo.py")
        
        try:
            subprocess.run([sys.executable, "scripts/interactive_demo.py"], 
                          cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        except Exception as e:
            print(f"⚠️  Could not run demo: {e}")
            print("💡 You can run it manually: python scripts/interactive_demo.py")
    else:
        print("📝 No problem! You can run the demo anytime:")
        print("   python scripts/interactive_demo.py")
    
    # Step 5: Web interface exploration
    print_header("STEP 5: WEB INTERFACE EXPLORATION")
    print("🌐 QFLARE provides several web interfaces:")
    print(f"   📊 Main Dashboard:     http://localhost:8000/")
    print(f"   📱 Device Management:  http://localhost:8000/devices")
    print(f"   👤 User Registration:  http://localhost:8000/register")
    print(f"   📚 API Documentation:  http://localhost:8000/docs")
    print(f"   💚 Health Check:       http://localhost:8000/health")
    
    # Step 6: Additional resources
    print_header("STEP 6: ADDITIONAL RESOURCES")
    print("📚 Learn more about QFLARE:")
    print("   📖 README.md - Complete project overview")
    print("   🏗️  PROJECT_STRUCTURE.md - Code organization")
    print("   📊 PROJECT_STATUS.md - Current status")
    print("   🔑 quantum_key_overview.md - Cryptography details")
    print("   🎯 quantum_key_usage_guide.md - Usage examples")
    
    print("\n🎯 AVAILABLE DEMONSTRATION SCRIPTS:")
    print("   🔍 python scripts/validate_server.py       - Test all endpoints")
    print("   🎭 python scripts/interactive_demo.py      - Complete walkthrough")
    print("   🛡️  python scripts/simple_security_demo.py  - Security strength (RECOMMENDED)")
    print("   🔬 python scripts/security_demonstration.py - Advanced security testing")
    print("   🧪 python scripts/crypto_analysis.py       - Quantum crypto analysis")
    print("   📜 python scripts/compliance_validator.py  - Standards compliance")
    print("   📖 python scripts/quick_start.py           - This guide")
    
    # Final summary
    print(f"\n{'🎉' * 60}")
    print("QFLARE QUICK START COMPLETE!")
    print(f"{'🎉' * 60}")
    print(f"""
✅ Your quantum-safe federated learning system is ready!

🌟 KEY FEATURES DEMONSTRATED:
   🔐 Post-quantum cryptography (NIST standards)
   🤖 Federated learning with privacy preservation
   🔑 Quantum-safe key exchange and authentication
   📱 Device enrollment and user registration
   🌐 Web-based management interface

🚀 Next Steps:
   1. Explore the web interface at http://localhost:8000
   2. Read the documentation in the docs/ folder
   3. Try the API endpoints using the /docs interface
   4. Experiment with the quantum cryptography features

💡 Need help? Check TROUBLESHOOTING.md or run the demos again!
""")

if __name__ == "__main__":
    main()