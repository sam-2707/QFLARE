# 🔐 Quantum Key Usage Guide for QFLARE

## 📋 Overview
The quantum keys you generate are used for secure federated learning communication. Here's how to use them:

## 🎯 **Key Types & Usage**

### 1. **KEM Key (FrodoKEM-640-AES)**
- **Purpose**: Key Encapsulation Mechanism for secure key exchange
- **Usage**: Encrypting session keys for secure communication
- **When Used**: During device enrollment and session establishment

### 2. **Signature Key (Dilithium2)**
- **Purpose**: Digital signatures for authentication and integrity
- **Usage**: Signing model updates and verifying device identity
- **When Used**: Model submission and device authentication

## 🚀 **How to Use the Keys**

### **Step 1: Generate Keys**
```bash
# Browser access
http://172.18.224.1:8000/api/request_qkey

# Or via API
curl http://172.18.224.1:8000/api/request_qkey
```

### **Step 2: Register Device with Keys**
```python
import requests

# Your generated keys
response = requests.get("http://172.18.224.1:8000/api/request_qkey")
keys = response.json()

# Register device with keys
device_data = {
    "device_id": "my_edge_device_001",
    "kem_public_key": keys["kem_public_key"],
    "signature_public_key": keys["signature_public_key"]
}

register_response = requests.post(
    "http://172.18.224.1:8000/api/enroll",
    json=device_data
)
```

### **Step 3: Use Keys for Secure Communication**

#### **A. Device Enrollment**
```python
# 1. Generate quantum keys
keys = requests.get("http://172.18.224.1:8000/api/request_qkey").json()

# 2. Register device
enrollment_data = {
    "device_id": "edge_device_001",
    "device_type": "IoT_Sensor",
    "location": "Building_A_Floor_2",
    "description": "Temperature sensor for FL training",
    "capabilities": ["local_training", "model_submission"],
    "kem_public_key": keys["kem_public_key"],
    "signature_public_key": keys["signature_public_key"]
}

response = requests.post(
    "http://172.18.224.1:8000/api/enroll",
    json=enrollment_data
)
```

#### **B. Model Submission (Secure)**
```python
import numpy as np
import hashlib
import base64

# 1. Create model update
model_weights = np.random.rand(100).tobytes()
model_hash = hashlib.sha256(model_weights).digest()

# 2. Sign with your signature key
# (In real implementation, you'd use the private key)
signature = hashlib.sha256(model_hash).hexdigest().encode()

# 3. Submit model with signature
model_data = {
    "device_id": "edge_device_001",
    "model_weights": base64.b64encode(model_weights).decode('utf-8'),
    "signature": base64.b64encode(signature).decode('utf-8'),
    "metadata": {
        "round": 1,
        "epochs": 10,
        "timestamp": time.time()
    }
}

response = requests.post(
    "http://172.18.224.1:8000/api/submit_model",
    json=model_data
)
```

#### **C. Session Challenge (Authentication)**
```python
# 1. Request session challenge
challenge_response = requests.post(
    "http://172.18.224.1:8000/api/challenge",
    json={"device_id": "edge_device_001"}
)

challenge = challenge_response.json()["challenge"]

# 2. Solve challenge with your signature key
# (In real implementation, you'd use the private key)
challenge_signature = hashlib.sha256(challenge.encode()).hexdigest().encode()

# 3. Submit challenge response
challenge_data = {
    "device_id": "edge_device_001",
    "challenge_response": base64.b64encode(challenge_signature).decode('utf-8')
}

auth_response = requests.post(
    "http://172.18.224.1:8000/api/verify_challenge",
    json=challenge_data
)
```

## 🔧 **Practical Examples**

### **Example 1: Complete Device Setup**
```python
import requests
import time

def setup_device():
    """Complete device setup with quantum keys."""
    
    # 1. Generate quantum keys
    print("🔐 Generating quantum keys...")
    keys_response = requests.get("http://172.18.224.1:8000/api/request_qkey")
    keys = keys_response.json()
    
    device_id = f"edge_device_{int(time.time())}"
    
    # 2. Register device
    print(f"📝 Registering device: {device_id}")
    registration_data = {
        "device_id": device_id,
        "device_type": "IoT_Sensor",
        "location": "Smart_Building_A",
        "description": "Temperature and humidity sensor for FL",
        "capabilities": ["local_training", "model_submission", "data_collection"],
        "kem_public_key": keys["kem_public_key"],
        "signature_public_key": keys["signature_public_key"]
    }
    
    register_response = requests.post(
        "http://172.18.224.1:8000/register",
        data=registration_data
    )
    
    if register_response.status_code == 200:
        print("✅ Device registered successfully!")
        return device_id, keys
    else:
        print("❌ Device registration failed!")
        return None, None

# Usage
device_id, keys = setup_device()
```

### **Example 2: Secure Model Training**
```python
def submit_secure_model(device_id, signature_key):
    """Submit a model update with quantum signature."""
    
    # Simulate local training
    import numpy as np
    model_weights = np.random.rand(100).tobytes()
    
    # Create signature (in real implementation, use private key)
    signature = hashlib.sha256(model_weights).hexdigest().encode()
    
    # Submit model
    model_data = {
        "device_id": device_id,
        "model_weights": base64.b64encode(model_weights).decode('utf-8'),
        "signature": base64.b64encode(signature).decode('utf-8'),
        "metadata": {
            "round": 1,
            "epochs": 10,
            "training_loss": 0.15,
            "validation_accuracy": 0.92
        }
    }
    
    response = requests.post(
        "http://172.18.224.1:8000/api/submit_model",
        json=model_data
    )
    
    if response.status_code == 200:
        print("✅ Model submitted successfully!")
        return response.json()
    else:
        print("❌ Model submission failed!")
        return None

# Usage
result = submit_secure_model(device_id, keys["signature_public_key"])
```

## 🛡️ **Security Features**

### **1. Quantum Resistance**
- **FrodoKEM-640-AES**: Resistant to quantum attacks
- **Dilithium2**: Post-quantum digital signatures
- **Perfect Forward Secrecy**: Session keys are ephemeral

### **2. Authentication Flow**
```
Device → Generate Keys → Register → Get Challenge → Sign Response → Authenticated
```

### **3. Model Integrity**
```
Local Training → Sign Model → Submit → Server Verifies → Aggregation
```

## 📊 **Monitoring Your Keys**

### **Check Device Status**
```bash
# View all registered devices
http://172.18.224.1:8000/devices

# Check device info via API
curl http://172.18.224.1:8000/api/devices
```

### **Verify Key Registration**
```python
import requests

def check_device_keys(device_id):
    """Check if device keys are properly registered."""
    response = requests.get(f"http://172.18.224.1:8000/api/devices/{device_id}")
    
    if response.status_code == 200:
        device_info = response.json()
        print(f"✅ Device {device_id} is registered")
        print(f"   KEM Key: {device_info.get('kem_public_key', 'Not found')[:20]}...")
        print(f"   Signature Key: {device_info.get('signature_public_key', 'Not found')[:20]}...")
        return True
    else:
        print(f"❌ Device {device_id} not found")
        return False
```

## 🎯 **Next Steps**

1. **Register your device** with the generated keys
2. **Participate in federated learning** rounds
3. **Submit signed model updates** for aggregation
4. **Monitor your contribution** to the global model

## 🔗 **Quick Links**

- **Dashboard**: `http://172.18.224.1:8000/`
- **Device Registration**: `http://172.18.224.1:8000/register`
- **Quantum Key Generation**: `http://172.18.224.1:8000/api/request_qkey`
- **Device Management**: `http://172.18.224.1:8000/devices`

---

**Your quantum keys are now ready for secure federated learning!** 🔐✨ 