#!/usr/bin/env python3
"""
QFLARE Quantum Key Exchange Visualization Dashboard
Real-time monitoring and testing interface for the quantum-safe system
"""

import os
import sys
import asyncio
import json
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Add the server directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our quantum crypto system
try:
    from server.crypto.quantum_key_exchange import LatticeKeyExchange, QuantumSafeEncryption
    from server.crypto.key_manager import QuantumKeyManager
    from server.crypto.key_mapping import SecureKeyMapper, SecureChannelManager
    from server.security.security_monitor import SecurityMonitor, SecurityEventType, EventSeverity, ThreatLevel
except ImportError as e:
    print(f"Import error: {e}")
    # Create mock classes for development
    class LatticeKeyExchange:
        def __init__(self): pass
        def initiate_key_exchange(self, device_id, public_key): 
            return {"session_id": secrets.token_hex(16), "algorithm": "Kyber1024-Mock"}

app = FastAPI(title="QFLARE Quantum Dashboard", version="1.0.0")

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global state for demo
class DashboardState:
    def __init__(self):
        self.key_exchange = LatticeKeyExchange()
        self.active_sessions = {}
        self.security_events = []
        self.device_registry = {}
        self.key_statistics = {
            "total_keys": 0,
            "active_sessions": 0,
            "key_exchanges": 0,
            "threat_level": "LOW"
        }
        self.websocket_connections = []
        
    def add_websocket(self, websocket):
        self.websocket_connections.append(websocket)
        
    def remove_websocket(self, websocket):
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)
    
    async def broadcast_update(self, data):
        """Broadcast update to all connected websockets"""
        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(json.dumps(data))
            except:
                disconnected.append(websocket)
        
        # Remove disconnected websockets
        for ws in disconnected:
            self.remove_websocket(ws)

dashboard_state = DashboardState()

@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("quantum_dashboard.html", {
        "request": request,
        "title": "QFLARE Quantum Dashboard"
    })

@app.get("/api/system/status")
async def get_system_status():
    """Get overall system status"""
    return {
        "status": "operational",
        "quantum_ready": True,
        "active_devices": len(dashboard_state.device_registry),
        "active_sessions": len(dashboard_state.active_sessions),
        "security_level": dashboard_state.key_statistics["threat_level"],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/keys/statistics")
async def get_key_statistics():
    """Get comprehensive key statistics"""
    stats = dashboard_state.key_statistics.copy()
    stats.update({
        "active_mappings": len(dashboard_state.active_sessions),
        "recent_events": len([e for e in dashboard_state.security_events if 
                            datetime.fromisoformat(e["timestamp"]) > datetime.utcnow() - timedelta(hours=1)]),
        "device_count": len(dashboard_state.device_registry)
    })
    return stats

@app.post("/api/test/simulate_device_registration")
async def simulate_device_registration(device_data: Dict):
    """Simulate a device registration"""
    device_id = device_data.get("device_id", f"device_{secrets.token_hex(4)}")
    device_type = device_data.get("device_type", "EDGE_NODE")
    
    # Simulate device registration
    device_info = {
        "device_id": device_id,
        "device_type": device_type,
        "registered_at": datetime.utcnow().isoformat(),
        "status": "ACTIVE",
        "trust_score": 1.0,
        "public_key": base64.b64encode(secrets.token_bytes(1568)).decode(),  # Kyber1024 size
        "capabilities": {
            "quantum_ready": True,
            "algorithms": ["Kyber1024", "Dilithium2"]
        }
    }
    
    dashboard_state.device_registry[device_id] = device_info
    
    # Log security event
    event = {
        "event_id": secrets.token_hex(8),
        "event_type": "DEVICE_REGISTRATION",
        "severity": "INFO",
        "device_id": device_id,
        "timestamp": datetime.utcnow().isoformat(),
        "data": device_info
    }
    dashboard_state.security_events.append(event)
    
    # Broadcast update
    await dashboard_state.broadcast_update({
        "type": "device_registered",
        "device": device_info,
        "event": event
    })
    
    return device_info

@app.post("/api/test/simulate_key_exchange")
async def simulate_key_exchange(exchange_data: Dict):
    """Simulate a quantum key exchange"""
    device_id = exchange_data.get("device_id")
    if not device_id or device_id not in dashboard_state.device_registry:
        raise HTTPException(status_code=400, detail="Device not found")
    
    device = dashboard_state.device_registry[device_id]
    
    # Simulate key exchange
    try:
        client_public_key = base64.b64decode(device["public_key"])
        exchange_result = dashboard_state.key_exchange.initiate_key_exchange(
            device_id, client_public_key
        )
        
        # Create session info
        session_info = {
            "session_id": exchange_result["session_id"],
            "device_id": device_id,
            "algorithm": exchange_result["algorithm"],
            "initiated_at": datetime.utcnow().isoformat(),
            "expires_at": datetime.utcfromtimestamp(exchange_result["expiry_time"]).isoformat(),
            "status": "ACTIVE",
            "temporal_mapping": {
                "time_window": 300,
                "timestamp": exchange_result["timestamp"],
                "nonce": exchange_result["nonce"]
            }
        }
        
        dashboard_state.active_sessions[exchange_result["session_id"]] = session_info
        dashboard_state.key_statistics["key_exchanges"] += 1
        dashboard_state.key_statistics["active_sessions"] = len(dashboard_state.active_sessions)
        
        # Log security event
        event = {
            "event_id": secrets.token_hex(8),
            "event_type": "KEY_EXCHANGE",
            "severity": "INFO",
            "device_id": device_id,
            "session_id": exchange_result["session_id"],
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "algorithm": exchange_result["algorithm"],
                "quantum_safe": True,
                "temporal_mapping": True
            }
        }
        dashboard_state.security_events.append(event)
        
        # Broadcast update
        await dashboard_state.broadcast_update({
            "type": "key_exchange",
            "session": session_info,
            "event": event
        })
        
        return session_info
        
    except Exception as e:
        # Log error event
        error_event = {
            "event_id": secrets.token_hex(8),
            "event_type": "SECURITY_VIOLATION",
            "severity": "ERROR",
            "device_id": device_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"error": str(e), "context": "key_exchange_failed"}
        }
        dashboard_state.security_events.append(error_event)
        
        await dashboard_state.broadcast_update({
            "type": "error",
            "event": error_event
        })
        
        raise HTTPException(status_code=500, detail=f"Key exchange failed: {e}")

@app.post("/api/test/simulate_threat")
async def simulate_threat(threat_data: Dict):
    """Simulate a security threat for testing"""
    threat_type = threat_data.get("threat_type", "ANOMALOUS_BEHAVIOR")
    severity = threat_data.get("severity", "MEDIUM")
    device_id = threat_data.get("device_id")
    
    threat_scenarios = {
        "QUANTUM_ATTACK": {
            "description": "Potential quantum attack detected",
            "indicators": ["unusual_timing", "excessive_key_requests", "algorithm_downgrade"]
        },
        "ANOMALOUS_BEHAVIOR": {
            "description": "Unusual device behavior pattern",
            "indicators": ["timing_correlation", "suspicious_patterns"]
        },
        "KEY_COMPROMISE": {
            "description": "Possible key compromise detected",
            "indicators": ["nonce_reuse", "invalid_signatures"]
        }
    }
    
    scenario = threat_scenarios.get(threat_type, threat_scenarios["ANOMALOUS_BEHAVIOR"])
    
    # Create threat event
    threat_event = {
        "event_id": secrets.token_hex(8),
        "event_type": threat_type,
        "severity": severity,
        "threat_level": "HIGH" if severity in ["ERROR", "CRITICAL"] else "MEDIUM",
        "device_id": device_id,
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "description": scenario["description"],
            "indicators": scenario["indicators"],
            "simulated": True
        }
    }
    
    dashboard_state.security_events.append(threat_event)
    dashboard_state.key_statistics["threat_level"] = threat_event["threat_level"]
    
    # Broadcast threat alert
    await dashboard_state.broadcast_update({
        "type": "threat_detected",
        "event": threat_event,
        "alert": True
    })
    
    return threat_event

@app.get("/api/sessions/active")
async def get_active_sessions():
    """Get all active key exchange sessions"""
    current_time = datetime.utcnow()
    active = {}
    
    for session_id, session in dashboard_state.active_sessions.items():
        expires_at = datetime.fromisoformat(session["expires_at"])
        if expires_at > current_time:
            session["time_remaining"] = int((expires_at - current_time).total_seconds())
            active[session_id] = session
        else:
            # Mark as expired
            session["status"] = "EXPIRED"
    
    # Update active sessions count
    dashboard_state.key_statistics["active_sessions"] = len(active)
    
    return active

@app.get("/api/events/recent")
async def get_recent_events(hours: int = 24):
    """Get recent security events"""
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    
    recent_events = [
        event for event in dashboard_state.security_events
        if datetime.fromisoformat(event["timestamp"]) > cutoff_time
    ]
    
    # Sort by timestamp (newest first)
    recent_events.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return recent_events[:100]  # Limit to 100 events

@app.get("/api/devices")
async def get_devices():
    """Get all registered devices"""
    return dashboard_state.device_registry

@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket for real-time dashboard updates"""
    await websocket.accept()
    dashboard_state.add_websocket(websocket)
    
    try:
        # Send initial data
        await websocket.send_text(json.dumps({
            "type": "initial_data",
            "statistics": dashboard_state.key_statistics,
            "devices": dashboard_state.device_registry,
            "active_sessions": await get_active_sessions(),
            "recent_events": await get_recent_events(1)
        }))
        
        # Keep connection alive
        while True:
            # Send periodic updates
            await asyncio.sleep(5)
            await websocket.send_text(json.dumps({
                "type": "heartbeat",
                "timestamp": datetime.utcnow().isoformat(),
                "statistics": dashboard_state.key_statistics
            }))
            
    except WebSocketDisconnect:
        dashboard_state.remove_websocket(websocket)

# Advanced testing endpoints
@app.post("/api/test/stress_test")
async def run_stress_test(test_params: Dict):
    """Run a stress test with multiple devices and key exchanges"""
    num_devices = test_params.get("num_devices", 10)
    exchanges_per_device = test_params.get("exchanges_per_device", 5)
    
    results = {
        "test_id": secrets.token_hex(8),
        "started_at": datetime.utcnow().isoformat(),
        "devices_created": 0,
        "exchanges_completed": 0,
        "errors": []
    }
    
    try:
        # Create devices
        for i in range(num_devices):
            device_data = {
                "device_id": f"stress_test_device_{i:03d}",
                "device_type": "EDGE_NODE"
            }
            await simulate_device_registration(device_data)
            results["devices_created"] += 1
        
        # Perform key exchanges
        for device_id in list(dashboard_state.device_registry.keys())[-num_devices:]:
            for j in range(exchanges_per_device):
                try:
                    await simulate_key_exchange({"device_id": device_id})
                    results["exchanges_completed"] += 1
                except Exception as e:
                    results["errors"].append(f"Device {device_id}: {str(e)}")
        
        results["completed_at"] = datetime.utcnow().isoformat()
        results["success_rate"] = results["exchanges_completed"] / (num_devices * exchanges_per_device)
        
        # Broadcast test results
        await dashboard_state.broadcast_update({
            "type": "stress_test_completed",
            "results": results
        })
        
        return results
        
    except Exception as e:
        results["error"] = str(e)
        results["completed_at"] = datetime.utcnow().isoformat()
        return results

@app.get("/api/test/quantum_resistance_demo")
async def quantum_resistance_demo():
    """Demonstrate quantum resistance features"""
    demo_data = {
        "algorithms": {
            "key_exchange": "CRYSTALS-Kyber 1024",
            "signatures": "CRYSTALS-Dilithium 2",
            "hashing": "SHA3-512"
        },
        "security_levels": {
            "classical_equivalent": "256-bit AES",
            "quantum_security": "NIST Level 5",
            "grover_resistance": "512-bit effective"
        },
        "temporal_features": {
            "time_window": "5 minutes",
            "key_rotation": "Automatic",
            "forward_secrecy": "Perfect"
        },
        "resistance_against": [
            "Shor's Algorithm (factoring)",
            "Grover's Algorithm (searching)",
            "Classical cryptanalysis",
            "Side-channel attacks",
            "Timing attacks"
        ]
    }
    
    return demo_data

if __name__ == "__main__":
    print("Starting QFLARE Quantum Dashboard...")
    print("Dashboard will be available at: http://localhost:8002")
    print("Quantum key exchange system ready for testing")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )