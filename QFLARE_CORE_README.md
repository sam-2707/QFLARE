# QFLARE Core - Federated Learning Coordinator

## 🎯 **Core Concept: Server-Hosted Federated Learning**

QFLARE Core implements the complete federated learning workflow where **you are the server/host** managing a network of edge devices that participate in collaborative machine learning while keeping their data private.

## 🏗️ **System Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   QFLARE Core   │    │   Edge Devices  │    │   Coordinator   │
│     Server      │◄──►│   (Mobile/IoT)  │◄──►│   (FedAvg)     │
│                 │    │                 │    │                 │
│ • Device Reg    │    │ • Local Training│    │ • Model Agg    │
│ • Auth & Sec    │    │ • Model Updates │    │ • Round Mgmt   │
│ • Web Dashboard │    │ • Secure Comm   │    │ • Progress Trk │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 **Core Workflow**

### **Phase 1: Device Registration**
```bash
# Devices register with the server using post-quantum cryptography
POST /api/v1/devices/register
{
  "device_id": "edge_001",
  "public_key": "pq_public_key_...",
  "device_type": "edge",
  "location": "New York, USA",
  "capabilities": {
    "cpu_cores": 4,
    "data_samples": 1000
  }
}
```

### **Phase 2: Server Initiates FL Round**
```bash
# Server starts a new federated learning round
POST /api/v1/fl/start-round
{
  "algorithm": "fedavg",
  "min_participants": 3
}
```

### **Phase 3: Devices Participate**
```python
# Devices check for active rounds
GET /api/v1/fl/current-round

# Train locally on device data
# Submit model updates securely
POST /api/v1/fl/submit-update
{
  "round_id": "round_20250915_143000",
  "model_data": {...},
  "local_metrics": {...}
}
```

### **Phase 4: Server Aggregates**
```python
# Server aggregates model updates using FedAvg
global_model = fedavg_aggregate(round_id)

# Round completes when enough devices participate
```

## 🛠️ **Quick Start**

### **1. Start the Core Server**
```bash
# Start Redis (required)
docker run -d -p 6379:6379 redis:7-alpine

# Start QFLARE Core Server
python qflare_core_server.py
```

### **2. Register Devices**
```bash
# Option A: Use the device simulator
python device_simulator.py

# Option B: Manual registration
python register_device.py my_device edge "New York, USA"
```

### **3. Start Federated Learning**
```bash
# Open web dashboard
open http://localhost:8000

# Click "Start New Round" in the dashboard
# Or use the API directly
```

## 📋 **API Endpoints**

### **Device Management**
- `POST /api/v1/devices/register` - Register new device
- `GET /api/v1/devices` - List all devices
- `GET /health` - System health check

### **Federated Learning**
- `POST /api/v1/fl/start-round` - Start new FL round
- `GET /api/v1/fl/current-round` - Get active round info
- `POST /api/v1/fl/submit-update` - Submit model update (device auth required)
- `GET /api/v1/fl/rounds` - Get round history

### **Security**
- `POST /api/v1/security/rotate-keys` - Rotate PQ keys
- All device endpoints require authentication headers:
  - `X-Device-ID: device_id`
  - `X-Device-Token: auth_token`

## 🔐 **Security Features**

### **Post-Quantum Cryptography**
- Device registration uses PQ key pairs
- Secure communication channels
- Key rotation capabilities

### **Authentication & Authorization**
- Device token-based authentication
- Server validates device credentials
- Secure model update submission

### **Privacy Protection**
- Data never leaves devices
- Only model updates are shared
- Differential privacy ready

## 🎮 **Demo Scenarios**

### **Scenario 1: IoT Sensor Network**
```bash
# Register IoT devices
python register_device.py sensor_01 iot "Factory Floor A"
python register_device.py sensor_02 iot "Factory Floor B"

# Start FL round for predictive maintenance
# Devices train on local sensor data
# Server aggregates for global predictive model
```

### **Scenario 2: Mobile User Behavior**
```bash
# Register mobile devices
python register_device.py mobile_01 mobile "New York"
python register_device.py mobile_02 mobile "London"

# Start FL round for recommendation system
# Devices train on user behavior locally
# Server creates global recommendation model
```

## 📊 **Monitoring & Dashboard**

### **Real-time Metrics**
- Active devices and participation
- Round progress and completion
- Model performance tracking
- System health monitoring

### **Web Dashboard Features**
- Device registration management
- FL round control and monitoring
- Performance metrics visualization
- System logs and alerts

## 🔧 **Configuration**

### **Environment Variables**
```bash
DATABASE_URL=sqlite:///data/qflare_core.db
REDIS_URL=redis://localhost:6379/0
QFLARE_JWT_SECRET=your-secret-key
QFLARE_SGX_MODE=SIM  # SIM or HW
QFLARE_LOG_LEVEL=INFO
```

### **Database Schema**
- `devices` - Device registry with PQ keys
- `fl_rounds` - FL round tracking
- `model_updates` - Device model submissions

## 🚀 **Advanced Usage**

### **Custom Aggregation Algorithms**
```python
# Implement your own FL algorithm
def custom_aggregate(round_id: str) -> Dict[str, Any]:
    # Get device updates
    updates = get_round_updates(round_id)
    
    # Custom aggregation logic
    global_model = your_custom_algorithm(updates)
    
    return global_model
```

### **Device Simulator Customization**
```python
# Create custom device types
device = SimulatedDevice("custom_01", "custom", "Location")
device.training_data = your_custom_data
device.local_model = your_initial_model
```

## 📈 **Scaling Considerations**

### **Production Deployment**
- Use production Redis cluster
- Implement load balancing
- Add monitoring and alerting
- Configure backup and recovery

### **Performance Optimization**
- Batch model updates
- Optimize aggregation algorithms
- Implement model compression
- Use efficient communication protocols

## 🐛 **Troubleshooting**

### **Common Issues**
1. **Redis Connection Failed**
   ```bash
   # Ensure Redis is running
   docker run -d -p 6379:6379 redis:7-alpine
   ```

2. **Device Registration Failed**
   - Check server is running on port 8000
   - Verify device_id is unique
   - Check network connectivity

3. **FL Round Not Starting**
   - Ensure minimum devices are registered
   - Check device connectivity
   - Verify Redis is accessible

## 🎯 **Next Steps**

1. **Implement Real PQ Crypto** - Replace mock keys with actual post-quantum algorithms
2. **Add Model Validation** - Implement model verification and anomaly detection
3. **Enhance Privacy** - Add differential privacy and secure aggregation
4. **Production Deployment** - Container orchestration and scaling
5. **Advanced Algorithms** - FedProx, SCAFFOLD, personalized FL

---

## 📞 **Support**

- **Web Dashboard**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Logs**: Check `data/logs/` directory

**QFLARE Core** - Your gateway to privacy-preserving collaborative AI! 🚀