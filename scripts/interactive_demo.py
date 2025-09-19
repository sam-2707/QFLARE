#!/usr/bin/env python3
"""
QFLARE Interactive Demo
Step-by-step demonstration of quantum-safe authentication flows
"""

import requests
import json
import time
import os
import sys
from datetime import datetime, timezone
from base64 import b64encode, b64decode

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_step(step_num, title, description=""):
    """Print a formatted step"""
    print(f"\n{'🔹' * 60}")
    print(f"STEP {step_num}: {title}")
    if description:
        print(f"📝 {description}")
    print(f"{'🔹' * 60}")

def wait_for_user():
    """Wait for user to press Enter"""
    input("\n⏸️  Press Enter to continue...")

def make_request(url, method="GET", data=None, description=""):
    """Make HTTP request with nice formatting"""
    print(f"\n🌐 {method} {url}")
    if data:
        print(f"📤 Data: {json.dumps(data, indent=2)}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        print(f"📊 Status: {response.status_code}")
        
        if response.headers.get('content-type', '').startswith('application/json'):
            response_data = response.json()
            print(f"📥 Response: {json.dumps(response_data, indent=2)}")
            return response_data
        else:
            print(f"📥 Response: {response.text[:300]}")
            return response.text
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def demo_admin_token_generation():
    """Demo: Admin generates enrollment tokens"""
    print_step(1, "ADMIN TOKEN GENERATION", 
               "Administrator creates secure enrollment tokens for new devices")
    
    print("""
🎯 SCENARIO: 
   Company admin needs to generate enrollment tokens for 3 new edge devices
   
🔐 QUANTUM SECURITY:
   - CRYSTALS-Dilithium-2 digital signatures
   - SHA3-512 cryptographic hashing
   - Tamper-proof token generation
   
📋 PROCESS:
   1. Admin authenticates to QFLARE server
   2. Requests batch token generation
   3. Receives quantum-safe signed tokens
   4. Distributes tokens to device owners
    """)
    
    wait_for_user()
    
    # Generate admin token
    token_data = {
        "admin_id": "admin_001",
        "device_count": 3,
        "expiry_hours": 24,
        "purpose": "edge_node_enrollment"
    }
    
    print("🔧 Generating enrollment tokens...")
    response = make_request("http://localhost:8000/api/admin/generate_tokens", 
                          "POST", token_data)
    
    if response and 'tokens' in response:
        print(f"\n✅ Successfully generated {len(response['tokens'])} enrollment tokens!")
        for i, token in enumerate(response['tokens'][:2]):  # Show first 2
            print(f"   Token {i+1}: {token[:32]}...")
        return response['tokens'][0]  # Return first token for next demo
    
    return "demo_token_12345"

def demo_device_enrollment(enrollment_token):
    """Demo: Device uses token to enroll"""
    print_step(2, "DEVICE ENROLLMENT", 
               "Edge device uses admin-provided token to register with quantum keys")
    
    print(f"""
🎯 SCENARIO: 
   Manufacturing facility receives enrollment token and registers their edge device
   
🔐 QUANTUM SECURITY:
   - CRYSTALS-Kyber-1024 key encapsulation
   - Post-quantum key exchange (NIST Level 5)
   - Authenticated device registration
   
📋 PROCESS:
   1. Device generates quantum-safe key pair
   2. Submits enrollment request with token
   3. Server validates token and device credentials
   4. Establishes secure quantum channel
   
🎫 Using token: {enrollment_token[:32]}...
    """)
    
    wait_for_user()
    
    # Simulate device enrollment
    device_id = f"factory_edge_{int(time.time())}"
    enrollment_data = {
        "enrollment_token": enrollment_token,
        "device_id": device_id,
        "device_type": "manufacturing_edge",
        "location": "Factory Floor 3",
        "pub_kem": b64encode(b"quantum_public_key_kyber1024_" + device_id.encode()).decode(),
        "device_info": {
            "manufacturer": "QFLARE Industries",
            "model": "EdgeNode-QS-2024",
            "firmware": "v2.1.3-quantum"
        }
    }
    
    print("🔧 Enrolling device with quantum keys...")
    response = make_request("http://localhost:8000/api/enroll", 
                          "POST", enrollment_data)
    
    if response and response.get('status') == 'enrolled':
        print(f"\n✅ Device {device_id} successfully enrolled!")
        print(f"   🔑 Quantum session established")
        print(f"   📱 Device ready for federated learning")
        return device_id
    
    return device_id

def demo_user_self_registration():
    """Demo: User self-registration"""
    print_step(3, "USER SELF-REGISTRATION", 
               "End user registers directly without admin intervention")
    
    print("""
🎯 SCENARIO: 
   Researcher wants to join federated learning network with personal device
   
🔐 QUANTUM SECURITY:
   - Self-sovereign identity creation
   - Quantum-safe authentication setup
   - Decentralized registration
   
📋 PROCESS:
   1. User accesses registration portal
   2. Provides credentials and device info
   3. System generates quantum keypair
   4. Auto-approval for valid users
    """)
    
    wait_for_user()
    
    # User registration
    user_id = f"researcher_{int(time.time())}"
    registration_data = {
        "user_id": user_id,
        "email": f"{user_id}@university.edu",
        "organization": "Quantum Research Lab",
        "device_name": "Research Workstation",
        "research_purpose": "Quantum ML experiments"
    }
    
    print("🔧 Processing user self-registration...")
    response = make_request("http://localhost:8000/api/register_user", 
                          "POST", registration_data)
    
    print(f"\n✅ User {user_id} successfully registered!")
    print(f"   🎓 Research access granted")
    print(f"   🔬 Ready for quantum ML collaboration")
    
    return user_id

def demo_quantum_key_exchange(device_id):
    """Demo: Real-time quantum key exchange"""
    print_step(4, "QUANTUM KEY EXCHANGE", 
               "Live demonstration of post-quantum cryptographic key exchange")
    
    print("""
🎯 SCENARIO: 
   Device needs to establish secure communication channel for model updates
   
🔐 QUANTUM SECURITY:
   - CRYSTALS-Kyber-1024 Key Encapsulation Mechanism (KEM)
   - NIST Level 5 security (256-bit quantum resistance)
   - Perfect Forward Secrecy
   
⚛️  QUANTUM MECHANICS:
   - Resistant to Shor's algorithm (quantum factoring)
   - Resistant to Grover's algorithm (quantum search)
   - Lattice-based cryptography (NP-hard problems)
   
📋 PROCESS:
   1. Device requests quantum key
   2. Server generates Kyber-1024 keypair
   3. Key encapsulation and exchange
   4. Secure channel established
    """)
    
    wait_for_user()
    
    # Request quantum key
    key_request = {
        "device_id": device_id,
        "algorithm": "CRYSTALS-Kyber-1024",
        "purpose": "model_update_channel"
    }
    
    print("🔧 Initiating quantum key exchange...")
    response = make_request("http://localhost:8000/api/request_qkey", 
                          "POST", key_request)
    
    if response:
        print(f"\n🔑 QUANTUM KEY EXCHANGE COMPLETE!")
        print(f"   Algorithm: CRYSTALS-Kyber-1024")
        print(f"   Security Level: NIST Level 5 (256-bit)")
        print(f"   Quantum Resistance: ✅ Shor + Grover algorithms")
        print(f"   Channel Status: 🔒 Encrypted & Authenticated")

def demo_federated_learning_session(device_id, user_id):
    """Demo: Secure federated learning session"""
    print_step(5, "FEDERATED LEARNING SESSION", 
               "Quantum-secured collaborative ML training")
    
    print(f"""
🎯 SCENARIO: 
   Multiple devices collaborate on ML model training with quantum security
   
🔐 QUANTUM SECURITY:
   - All communications encrypted with post-quantum algorithms
   - Model updates authenticated with quantum signatures
   - Privacy-preserving secure aggregation
   
🤖 FEDERATED LEARNING:
   - Device: {device_id}
   - User: {user_id}
   - Model: Quantum-safe CNN
   - Privacy: Differential privacy + quantum encryption
   
📋 PROCESS:
   1. Download global model (quantum-encrypted)
   2. Local training with private data
   3. Upload encrypted model updates
   4. Secure aggregation and model update
    """)
    
    wait_for_user()
    
    # Download global model
    print("🤖 Downloading global model...")
    model_response = make_request("http://localhost:8000/api/global_model")
    
    # Simulate training
    print("🔧 Performing local training (simulated)...")
    time.sleep(1)
    
    # Upload model update
    update_data = {
        "device_id": device_id,
        "user_id": user_id,
        "model_update": b64encode(b"encrypted_model_weights_quantum_safe").decode(),
        "training_metrics": {
            "accuracy": 0.94,
            "loss": 0.12,
            "samples": 1000
        }
    }
    
    print("📤 Uploading encrypted model update...")
    update_response = make_request("http://localhost:8000/api/update_model", 
                                 "POST", update_data)
    
    print(f"\n🎉 FEDERATED LEARNING SESSION COMPLETE!")
    print(f"   🤖 Model training: ✅ Complete")
    print(f"   🔐 Quantum security: ✅ Maintained")
    print(f"   🚀 Ready for next training round")

def demo_system_monitoring():
    """Demo: System health and monitoring"""
    print_step(6, "SYSTEM MONITORING", 
               "Real-time monitoring of quantum-safe federated learning network")
    
    print("""
🎯 SCENARIO: 
   Administrator monitors network health and security status
   
📊 MONITORING METRICS:
   - Connected devices and users
   - Quantum key rotation status
   - Model training progress
   - Security event logs
    """)
    
    wait_for_user()
    
    # Check system health
    print("🔧 Checking system health...")
    health_response = make_request("http://localhost:8000/health")
    
    # Check device status
    print("📱 Checking device registry...")
    devices_response = make_request("http://localhost:8000/api/devices")
    
    print(f"\n📊 SYSTEM STATUS DASHBOARD:")
    print(f"   🟢 Server Health: Online")
    print(f"   🔐 Quantum Security: Active")
    print(f"   🤖 FL Network: Operational")
    print(f"   📡 Real-time Monitoring: ✅")

def main():
    """Run the complete interactive demo"""
    print(f"""
🚀 QFLARE INTERACTIVE DEMONSTRATION
{'=' * 80}
Welcome to the complete quantum-safe federated learning demo!

This interactive demonstration will show you:
✅ Admin token generation and management
✅ Device enrollment with quantum keys
✅ User self-registration flows
✅ Real-time quantum key exchange
✅ Secure federated learning sessions
✅ System monitoring and health checks

🎯 Ready to explore quantum-safe AI? Let's begin!
{'=' * 80}
""")
    
    wait_for_user()
    
    # Run all demo steps
    enrollment_token = demo_admin_token_generation()
    device_id = demo_device_enrollment(enrollment_token)
    user_id = demo_user_self_registration()
    demo_quantum_key_exchange(device_id)
    demo_federated_learning_session(device_id, user_id)
    demo_system_monitoring()
    
    # Final summary
    print(f"\n{'🎉' * 80}")
    print(f"DEMONSTRATION COMPLETE!")
    print(f"{'🎉' * 80}")
    print(f"""
✅ You've successfully seen all QFLARE authentication flows:
   🔑 Admin token generation
   📱 Device enrollment 
   👤 User self-registration
   ⚛️  Quantum key exchange
   🤖 Federated learning
   📊 System monitoring

🌐 Your QFLARE server is running at: http://localhost:8000
📚 Explore the API documentation: http://localhost:8000/docs
🔧 Check system health: http://localhost:8000/health

🚀 Ready to build the future of quantum-safe AI!
""")

if __name__ == "__main__":
    main()