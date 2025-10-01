#!/usr/bin/env python3
"""
QFLARE Simple Server for Secure Registration Demo
Minimal server to demonstrate secure key exchange without database complexity
"""

import os
import time
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

import uvicorn
from sms_utils import send_sms_otp

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import FL components (with fallbacks)
try:
    from fl_core.fl_controller import FLController
    from fl_core.model_aggregator import FederatedAveraging
    from fl_core.security import ModelValidator, SecurityMonitor
    FL_AVAILABLE = True
    logger.info("✅ FL components loaded successfully")
except ImportError as e:
    logger.warning(f"FL components not available: {e}")
    FL_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(title="QFLARE Secure Registration API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:4000", "http://127.0.0.1:4000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Templates
templates = Jinja2Templates(directory="templates")

# Simple in-memory storage for demo
devices_registry = {}
device_counter = 1

# Initialize FL components
fl_controller = None
model_validator = None
security_monitor = None
fl_state = {
    "current_round": 0,
    "total_rounds": 10,
    "status": "idle",  # idle, training, aggregating, completed
    "participants": {},
    "global_model": None,
    "round_start_time": None,
    "training_history": []
}

if FL_AVAILABLE:
    try:
        fl_controller = FLController(min_participants=2, max_participants=10)
        model_validator = ModelValidator()
        security_monitor = SecurityMonitor()
        logger.info("✅ FL system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize FL system: {e}")
        FL_AVAILABLE = False

# Initialize secure key exchange
secure_key_exchange = None

try:
    from secure_key_exchange import get_secure_key_exchange
    secure_key_exchange = get_secure_key_exchange()
    logger.info("✅ Secure key exchange initialized")
except Exception as e:
    logger.warning(f"⚠️ Secure key exchange not available: {e}")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with navigation to secure registration."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>QFLARE Secure Registration Demo</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #1e40af, #3b82f6); color: white; min-height: 100vh; }}
            .container {{ max-width: 800px; margin: 0 auto; text-align: center; }}
            .card {{ background: rgba(255,255,255,0.1); padding: 40px; border-radius: 20px; backdrop-filter: blur(10px); }}
            .btn {{ background: #06b6d4; color: white; padding: 15px 30px; border: none; border-radius: 10px; text-decoration: none; display: inline-block; margin: 10px; font-weight: bold; }}
            .btn:hover {{ background: #0891b2; }}
            .status {{ background: rgba(16, 185, 129, 0.2); padding: 20px; border-radius: 10px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>🔐 QFLARE Secure Registration</h1>
                <p>Demonstrating quantum-safe key exchange with MITM attack prevention</p>
                
                <div class="status">
                    <h3>✅ System Status</h3>
                    <p>Secure Key Exchange: {'Active' if secure_key_exchange else 'Unavailable'}</p>
                    <p>Registered Devices: {len(devices_registry)}</p>
                    <p>Server Time: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <a href="/secure-register" class="btn">🚀 Start Secure Registration</a>
                <a href="/devices" class="btn">📱 View Devices</a>
                <a href="/fl-dashboard" class="btn">🤖 FL Dashboard</a>
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/secure-register", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def secure_register_form(request: Request):
    """Serve the secure registration page."""
    try:
        return templates.TemplateResponse(
            "secure_register.html", 
            {"request": request, "timestamp": int(time.time())}
        )
    except Exception as e:
        logger.error(f"Error serving secure register template: {e}")
        return HTMLResponse(f"<h1>Template Error</h1><p>Could not load secure registration form: {e}</p>")


# @app.post("/api/secure_register")  # DISABLED - Using JSON version below
# @limiter.limit("5/minute")
async def process_secure_registration_OLD(request: Request):
    """Process secure registration request."""
    try:
        form = await request.form()
        
        # Extract form data
        device_id = form.get("device_id", "").strip()
        device_type = form.get("device_type", "").strip()
        organization = form.get("organization", "").strip()
        contact_email = form.get("contact_email", "").strip()
        use_case = form.get("use_case", "").strip()
        key_exchange_method = form.get("key_exchange_method", "").strip()
        
        # Validate required fields
        if not all([device_id, device_type, organization, contact_email, use_case, key_exchange_method]):
            return templates.TemplateResponse(
                "secure_register.html",
                {
                    "request": request,
                    "error": "All fields are required",
                    "timestamp": int(time.time())
                }
            )
        
        # Check if secure key exchange is available
        if not secure_key_exchange:
            return templates.TemplateResponse(
                "secure_register.html",
                {
                    "request": request,
                    "error": "Secure key exchange service is not available.",
                    "timestamp": int(time.time())
                }
            )
        
        # Prepare user request data
        user_request = {
            "device_id": device_id,
            "device_type": device_type,
            "organization": organization,
            "email": contact_email,
            "use_case": use_case,
            "contact_info": form.get("contact_info", ""),
            "secure_contact": form.get("secure_contact", ""),
            "pgp_public_key": form.get("pgp_public_key", ""),
            "delivery_address": form.get("delivery_address", ""),
            "delivery_method": form.get("delivery_method", "")
        }
        
        # Process based on selected method
        result = None
        success_message = ""
        
        if key_exchange_method == "qr_otp":
            result = secure_key_exchange.method_1_qr_code_with_otp(user_request)
            # Send OTP via SMS if phone number provided
            phone_number = form.get("phone_number", "").strip()
            otp_sent = False
            sms_error = None
            if phone_number and result.get('otp'):
                try:
                    send_sms_otp(phone_number, result['otp'])
                    otp_sent = True
                except Exception as sms_exc:
                    sms_error = str(sms_exc)
            success_message = f"""
            🔐 <strong>QR Code + OTP Method Initiated</strong><br>
            • QR Code file: {result['qr_code_file']}<br>
            • One-Time Password: <strong>{result['otp']}</strong><br>
            • Expires in: {result['expires_in']//60} minutes<br>
            <br>
            {'✅ OTP sent to your mobile number.' if otp_sent else ''}
            {f'<span style="color:red">⚠️ SMS error: {sms_error}</span>' if sms_error else ''}
            <br>
            📱 <strong>Next Steps:</strong><br>
            1. Admin will provide QR code via secure channel<br>
            2. You will receive OTP via SMS<br>
            3. Scan QR code and enter OTP to decrypt your quantum keys<br>
            <a href="/secure-verify/{device_id}" style="color: #06b6d4;">→ Continue to Verification</a>
            """
            
        elif key_exchange_method == "pgp_email":
            if not user_request['pgp_public_key']:
                return templates.TemplateResponse(
                    "secure_register.html",
                    {
                        "request": request,
                        "error": "PGP public key is required for this method",
                        "timestamp": int(time.time())
                    }
                )
            result = secure_key_exchange.method_2_secure_email_with_pgp(user_request)
            success_message = f"""
            📧 <strong>PGP Email Method Initiated</strong><br>
            • Encrypted keys prepared for: {contact_email}<br>
            • Encryption: Your PGP public key<br>
            <br>
            📬 <strong>Next Steps:</strong><br>
            1. Admin will send encrypted keys to your email<br>
            2. Decrypt the email using your PGP private key<br>
            3. Install the quantum keys on your device
            """
            
        elif key_exchange_method == "totp":
            result = secure_key_exchange.method_3_totp_based_exchange(user_request)
            success_message = f"""
            🔐 <strong>TOTP Method Initiated</strong><br>
            • TOTP Secret: <strong>{result['totp_secret'][:8]}...</strong><br>
            • Device ID: {result['device_id']}<br>
            <br>
            📱 <strong>Next Steps:</strong><br>
            1. Admin will provide TOTP secret via secure channel<br>
            2. Set up authenticator app with the secret<br>
            3. Use current TOTP code to authenticate and receive keys<br>
            <a href="/secure-verify/{device_id}" style="color: #06b6d4;">→ Continue to Verification</a>
            """
            
        elif key_exchange_method == "physical_token":
            result = secure_key_exchange.method_4_physical_token_exchange(user_request)
            success_message = f"""
            🔑 <strong>Physical Token Method Initiated</strong><br>
            • Token File: {result['token_file']}<br>
            • Token PIN: <strong>{result['token_pin'][:4]}****</strong><br>
            • Delivery: Physical delivery required<br>
            <br>
            📦 <strong>Next Steps:</strong><br>
            1. Physical token will be delivered to your address<br>
            2. Enter the token PIN to decrypt your quantum keys<br>
            3. Install keys on your device<br>
            <a href="/secure-verify/{device_id}" style="color: #06b6d4;">→ Continue to Verification</a>
            """
            
        elif key_exchange_method == "blockchain":
            result = secure_key_exchange.method_5_blockchain_verification(user_request)
            success_message = f"""
            ⛓️ <strong>Blockchain Verification Method Initiated</strong><br>
            • Transaction: {result['tx_hash'][:16]}...<br>
            • Block Height: {result['block_height']}<br>
            • Key Fingerprint: {result['key_fingerprint'][:16]}...<br>
            <br>
            🔗 <strong>Next Steps:</strong><br>
            1. Keys will be delivered via HTTPS<br>
            2. Verify key fingerprint against blockchain record<br>
            3. Install verified keys on your device
            """
        
        else:
            return templates.TemplateResponse(
                "secure_register.html",
                {
                    "request": request,
                    "error": "Invalid key exchange method selected",
                    "timestamp": int(time.time())
                }
            )
        
        # Log the registration attempt
        logger.info(f"Secure registration initiated: {device_id} ({key_exchange_method})")
        
        return templates.TemplateResponse(
            "secure_register.html",
            {
                "request": request,
                "success": success_message,
                "method_used": key_exchange_method,
                "device_id": device_id,
                "timestamp": int(time.time())
            }
        )
        
    except Exception as e:
        logger.error(f"Error in secure registration: {e}")
        return templates.TemplateResponse(
            "secure_register.html",
            {
                "request": request,
                "error": f"Registration failed: {str(e)}",
                "timestamp": int(time.time())
            }
        )


@app.get("/secure-verify/{device_id}")
@limiter.limit("10/minute")
async def secure_verification_page(request: Request, device_id: str):
    """Serve verification page."""
    if not secure_key_exchange:
        raise HTTPException(status_code=503, detail="Secure key exchange service not available")
    
    if device_id not in secure_key_exchange.pending_registrations:
        raise HTTPException(status_code=404, detail="No pending registration found for this device")
    
    registration = secure_key_exchange.pending_registrations[device_id]
    method = registration['method']
    
    return templates.TemplateResponse(
        "secure_verify.html",
        {
            "request": request,
            "device_id": device_id,
            "method": method,
            "timestamp": int(time.time())
        }
    )


# @app.post("/api/secure_verify/{device_id}")  # DISABLED - Using JSON version below
# @limiter.limit("5/minute")
# async def process_secure_verification(request: Request, device_id: str):
#     """Process verification and deliver quantum keys."""
#     try:
#         if not secure_key_exchange:
#             raise HTTPException(status_code=503, detail="Secure key exchange service not available")
#         
#         form = await request.form()
#         verification_data = {}
#         
#         # Extract verification data based on method
#         if "otp" in form:
#             verification_data["otp"] = form.get("otp")
#         if "totp_code" in form:
#             verification_data["totp_code"] = form.get("totp_code")
#         if "token_pin" in form:
#             verification_data["token_pin"] = form.get("token_pin")
#         
#         # Verify and deliver keys
#         quantum_keys = secure_key_exchange.verify_and_deliver_keys(device_id, verification_data)
#         
#         # Register device in simple registry
#         global device_counter
#         devices_registry[device_id] = {
#             "id": device_counter,
#             "device_id": device_id,
#             "device_type": "quantum_secured",
#             "status": "enrolled",
#             "quantum_keys_active": True,
#             "enrolled_at": time.strftime('%Y-%m-%d %H:%M:%S'),
#             "security_level": "Post-Quantum"
#         }
#         device_counter += 1
#         
#         logger.info(f"✅ Device {device_id} successfully verified and enrolled with quantum keys")
#         
#         return templates.TemplateResponse(
#             "secure_verify.html",
#             {
#                 "request": request,
#                 "device_id": device_id,
#                 "success": "✅ Verification successful! Your quantum keys have been delivered and your device is now enrolled in the QFLARE network.",
#                 "quantum_keys_active": True,
#                 "timestamp": int(time.time())
#             }
#         )
#         
#     except ValueError as e:
#         return templates.TemplateResponse(
#             "secure_verify.html",
#             {
#                 "request": request,
#                 "device_id": device_id,
#                 "error": str(e),
#                 "timestamp": int(time.time())
#             }
#         )
#     except Exception as e:
#         logger.error(f"Error in secure verification: {e}")
#         return templates.TemplateResponse(
#             "secure_verify.html",
#             {
#                 "request": request,
#                 "device_id": device_id,
#                 "error": f"Verification failed: {str(e)}",
#                 "timestamp": int(time.time())
#             }
#         )


@app.get("/devices", response_class=HTMLResponse)
async def devices_page(request: Request):
    """Show registered devices."""
    device_list = ""
    for device_id, device in devices_registry.items():
        status_color = "#10b981" if device['quantum_keys_active'] else "#ef4444"
        device_list += f"""
        <div style="background: rgba(255,255,255,0.1); padding: 20px; margin: 10px 0; border-radius: 10px;">
            <h4>📱 {device['device_id']}</h4>
            <p><strong>Type:</strong> {device['device_type']}</p>
            <p><strong>Status:</strong> <span style="color: {status_color};">●</span> {device['status']}</p>
            <p><strong>Security:</strong> {device['security_level']}</p>
            <p><strong>Enrolled:</strong> {device['enrolled_at']}</p>
        </div>
        """
    
    if not device_list:
        device_list = "<p>No devices registered yet. <a href='/secure-register' style='color: #06b6d4;'>Register the first device</a></p>"
    
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>QFLARE - Registered Devices</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #1e40af, #3b82f6); color: white; min-height: 100vh; }}
            .container {{ max-width: 1000px; margin: 0 auto; }}
            .card {{ background: rgba(255,255,255,0.1); padding: 40px; border-radius: 20px; backdrop-filter: blur(10px); }}
            .btn {{ background: #06b6d4; color: white; padding: 10px 20px; border: none; border-radius: 10px; text-decoration: none; display: inline-block; margin: 10px; }}
            .btn:hover {{ background: #0891b2; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>📱 Registered Devices</h1>
                <p>Total devices: {len(devices_registry)}</p>
                
                {device_list}
                
                <div style="margin-top: 30px;">
                    <a href="/" class="btn">🏠 Home</a>
                    <a href="/secure-register" class="btn">➕ Add Device</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """)


# =============================================================================
# JSON API ENDPOINTS FOR FRONTEND
# =============================================================================

@app.get("/api/devices", response_class=JSONResponse)
@limiter.limit("30/minute")
async def get_devices(request: Request):
    """Get all registered devices as JSON."""
    devices_list = []
    for device_id, device in devices_registry.items():
        # Determine status based on registration and verification state
        if device.get("status") == "pending":
            status = "pending"
        elif device.get("quantum_keys_active"):
            status = "online"
        else:
            status = "offline"
            
        devices_list.append({
            "id": device.get("id", 0),
            "device_id": device_id,
            "device_type": device.get("device_type", "Unknown"),
            "organization": device.get("organization", "Unknown"),
            "contact_email": device.get("contact_email", ""),
            "use_case": device.get("use_case", ""),
            "status": status,
            "security_level": device.get("security_level", "Standard"),
            "enrolled_at": device.get("enrolled_at") or device.get("registered_at", "Unknown"),
            "registered_at": device.get("registered_at", "Unknown"),
            "last_seen": time.strftime('%Y-%m-%d %H:%M:%S'),
        })
    
    return JSONResponse({
        "devices": devices_list,
        "total": len(devices_list),
        "online": len([d for d in devices_list if d["status"] == "online"]),
        "offline": len([d for d in devices_list if d["status"] == "offline"]),
        "pending": len([d for d in devices_list if d["status"] == "pending"]),
    })


@app.get("/api/devices/{device_id}", response_class=JSONResponse)
@limiter.limit("30/minute")
async def get_device_details(request: Request, device_id: str):
    """Get specific device details."""
    if device_id not in devices_registry:
        raise HTTPException(status_code=404, detail="Device not found")
    
    device = devices_registry[device_id]
    return JSONResponse({
        "device_id": device_id,
        "device_type": device.get("device_type", "Unknown"),
        "organization": device.get("organization", "Unknown"),
        "status": "online" if device.get("quantum_keys_active") else "offline",
        "security_level": device.get("security_level", "Standard"),
        "enrolled_at": device.get("enrolled_at", "Unknown"),
        "last_seen": time.strftime('%Y-%m-%d %H:%M:%S'),
        "quantum_keys_active": device.get("quantum_keys_active", False),
    })


@app.delete("/api/devices/{device_id}", response_class=JSONResponse)
@limiter.limit("10/minute")
async def delete_device(request: Request, device_id: str):
    """Delete a device."""
    if device_id not in devices_registry:
        raise HTTPException(status_code=404, detail="Device not found")
    
    del devices_registry[device_id]
    logger.info(f"Device {device_id} deleted")
    
    return JSONResponse({"message": f"Device {device_id} deleted successfully"})


@app.put("/api/devices/{device_id}/status", response_class=JSONResponse)
@limiter.limit("20/minute")
async def update_device_status(request: Request, device_id: str, status: str = Form(...)):
    """Update device status."""
    if device_id not in devices_registry:
        raise HTTPException(status_code=404, detail="Device not found")
    
    devices_registry[device_id]["status"] = status
    logger.info(f"Device {device_id} status updated to {status}")
    
    return JSONResponse({"message": f"Device {device_id} status updated to {status}"})


@app.get("/api/admin/metrics", response_class=JSONResponse)
@limiter.limit("20/minute")
async def get_system_metrics(request: Request):
    """Get system metrics for admin dashboard."""
    total_devices = len(devices_registry)
    online_devices = len([d for d in devices_registry.values() if d.get("quantum_keys_active")])
    
    return JSONResponse({
        "totalDevices": total_devices,
        "onlineDevices": online_devices,
        "offlineDevices": total_devices - online_devices,
        "securityLevel": 98,
        "systemUptime": 99.9,
        "quantumResistance": 100,
        "encryptionStrength": "256-bit",
        "keyExchangeSecurity": 98,
    })


@app.get("/api/admin/logs", response_class=JSONResponse)
@limiter.limit("20/minute")
async def get_activity_logs(request: Request):
    """Get activity logs for admin dashboard."""
    # Mock activity logs
    logs = [
        {
            "id": "1",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() - 300)),
            "type": "registration",
            "message": "New device registration initiated",
            "severity": "success",
        },
        {
            "id": "2",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() - 600)),
            "type": "verification",
            "message": "OTP verification completed successfully",
            "severity": "success",
        },
        {
            "id": "3",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() - 900)),
            "type": "security",
            "message": "Quantum key rotation completed",
            "severity": "info",
        },
        {
            "id": "4",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() - 1200)),
            "type": "system",
            "message": "System health check passed",
            "severity": "info",
        },
    ]
    
    return JSONResponse({"logs": logs})


@app.get("/api/admin/security-report", response_class=JSONResponse)
@limiter.limit("5/minute")
async def generate_security_report(request: Request):
    """Generate security report."""
    report = {
        "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
        "quantum_resistance": {
            "level": "100%",
            "algorithms": ["CRYSTALS-Kyber-1024", "CRYSTALS-Dilithium-2"],
            "key_strength": "256-bit",
        },
        "device_security": {
            "total_devices": len(devices_registry),
            "secure_devices": len([d for d in devices_registry.values() if d.get("quantum_keys_active")]),
            "security_compliance": "100%",
        },
        "threat_assessment": {
            "quantum_threat_resistance": "High",
            "mitm_protection": "Enabled",
            "key_exchange_security": "Multi-factor",
        },
        "recommendations": [
            "Continue regular key rotation schedule",
            "Monitor device health metrics",
            "Maintain quantum-safe algorithm updates",
        ],
    }
    
    return JSONResponse(report)


# Updated registration endpoint to return JSON for API calls
@app.post("/api/secure_register", response_class=JSONResponse)
@limiter.limit("5/minute")
async def api_secure_registration(request: Request):
    """API version of secure registration that returns JSON."""
    try:
        form = await request.form()
        
        # Extract form data
        device_id = form.get("device_id", "").strip()
        device_type = form.get("device_type", "").strip()
        organization = form.get("organization", "").strip()
        contact_email = form.get("contact_email", "").strip()
        phone_number = form.get("phone_number", "").strip()
        use_case = form.get("use_case", "").strip()
        key_exchange_method = form.get("key_exchange_method", "qr_otp").strip()
        
        # Validate required fields
        if not all([device_id, device_type, organization, contact_email, use_case]):
            raise HTTPException(status_code=400, detail="All fields are required")
        
        # Check if secure key exchange is available
        if not secure_key_exchange:
            raise HTTPException(status_code=503, detail="Secure key exchange service is not available")
        
        # Prepare user request data
        user_request = {
            "device_id": device_id,
            "device_type": device_type,
            "organization": organization,
            "email": contact_email,
            "use_case": use_case,
            "phone_number": phone_number,
        }
        
        # Process registration
        result = secure_key_exchange.method_1_qr_code_with_otp(user_request)
        
        # Send OTP via SMS if phone number provided
        otp_sent = False
        sms_error = None
        if phone_number and result.get('otp'):
            try:
                send_sms_otp(phone_number, result['otp'])
                otp_sent = True
                logger.info(f"OTP sent to {phone_number} for device {device_id}")
            except Exception as sms_exc:
                sms_error = str(sms_exc)
                logger.warning(f"Failed to send SMS OTP: {sms_error}")
        
        # Store device in registry with "pending" status
        global device_counter
        devices_registry[device_id] = {
            "id": device_counter,
            "device_id": device_id,
            "device_type": device_type,
            "organization": organization,
            "contact_email": contact_email,
            "phone_number": phone_number,
            "use_case": use_case,
            "status": "pending",
            "quantum_keys_active": False,
            "enrolled_at": None,
            "registered_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "security_level": "Pending Verification"
        }
        device_counter += 1
        
        # Log the registration attempt
        logger.info(f"Secure registration initiated: {device_id} ({key_exchange_method})")
        
        return JSONResponse({
            "success": True,
            "message": "Registration initiated successfully",
            "device_id": device_id,
            "otp_sent": otp_sent,
            "sms_error": sms_error,
            "expires_in": result.get('expires_in', 1800),
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in secure registration: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


# Updated verification endpoint to return JSON for API calls
@app.post("/api/secure_verify/{device_id}", response_class=JSONResponse)
@limiter.limit("5/minute")
async def api_secure_verification(request: Request, device_id: str):
    """API version of secure verification that returns JSON."""
    try:
        if not secure_key_exchange:
            raise HTTPException(status_code=503, detail="Secure key exchange service not available")
        
        form = await request.form()
        otp = form.get("otp", "").strip()
        
        if not otp:
            raise HTTPException(status_code=400, detail="OTP is required")
        
        # Verify OTP
        quantum_keys = secure_key_exchange.verify_and_deliver_keys(device_id, {"otp": otp})
        
        # Update existing device in registry or create new one
        if device_id in devices_registry:
            # Update existing device
            devices_registry[device_id].update({
                "status": "enrolled",
                "quantum_keys_active": True,
                "enrolled_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "security_level": "Post-Quantum"
            })
        else:
            # Create new device entry (fallback)
            global device_counter
            devices_registry[device_id] = {
                "id": device_counter,
                "device_id": device_id,
                "device_type": "quantum_secured",
                "status": "enrolled",
                "quantum_keys_active": True,
                "enrolled_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "security_level": "Post-Quantum"
            }
            device_counter += 1
        
        logger.info(f"✅ Device {device_id} successfully verified and enrolled with quantum keys")
        
        return JSONResponse({
            "success": True,
            "message": "Verification successful! Your quantum keys have been delivered and your device is now enrolled in the QFLARE network.",
            "device_id": device_id,
            "quantum_keys_active": True,
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in secure verification: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


# =====================================
# FEDERATED LEARNING ENDPOINTS
# =====================================

@app.get("/api/fl/status")
@limiter.limit("30/minute")
async def get_fl_status(request: Request):
    """Get current federated learning status."""
    if not FL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Federated Learning not available")
    
    try:
        active_devices = [d for d in devices_registry.values() if d.get("status") == "enrolled"]
        
        return JSONResponse({
            "success": True,
            "fl_status": {
                "available": FL_AVAILABLE,
                "current_round": fl_state["current_round"],
                "total_rounds": fl_state["total_rounds"],
                "status": fl_state["status"],
                "registered_devices": len(devices_registry),
                "active_devices": len(active_devices),
                "participants_this_round": len(fl_state["participants"]),
                "round_start_time": fl_state["round_start_time"],
                "training_history": fl_state["training_history"][-5:]  # Last 5 rounds
            }
        })
    except Exception as e:
        logger.error(f"Error getting FL status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fl/start_round")
@limiter.limit("5/minute")
async def start_fl_round(
    request: Request,
    target_participants: int = Form(default=3),
    local_epochs: int = Form(default=5),
    learning_rate: float = Form(default=0.01)
):
    """Start a new federated learning training round."""
    if not FL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Federated Learning not available")
    
    try:
        if fl_state["status"] != "idle":
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot start new round. Current status: {fl_state['status']}"
            )
        
        # Get active devices
        active_devices = [d for d in devices_registry.values() if d.get("status") == "enrolled"]
        
        if len(active_devices) < target_participants:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough active devices. Need {target_participants}, have {len(active_devices)}"
            )
        
        # Select participants (first N active devices for demo)
        selected_devices = active_devices[:target_participants]
        
        # Initialize round
        fl_state["current_round"] += 1
        fl_state["status"] = "training"
        fl_state["round_start_time"] = time.strftime('%Y-%m-%d %H:%M:%S')
        fl_state["participants"] = {
            device["device_id"]: {
                "device_id": device["device_id"],
                "device_type": device.get("device_type", "unknown"),
                "status": "selected",
                "model_submitted": False,
                "submission_time": None,
                "model_size": 0,
                "training_loss": None
            }
            for device in selected_devices
        }
        
        logger.info(f"Started FL round {fl_state['current_round']} with {len(selected_devices)} participants")
        
        return JSONResponse({
            "success": True,
            "message": f"Training round {fl_state['current_round']} started",
            "round_info": {
                "round_number": fl_state["current_round"],
                "participants": list(fl_state["participants"].keys()),
                "target_participants": target_participants,
                "local_epochs": local_epochs,
                "learning_rate": learning_rate,
                "deadline": "30 minutes from now"
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting FL round: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fl/submit_model")
@limiter.limit("10/minute")
async def submit_model_update(request: Request):
    """Submit a local model update from an edge device."""
    if not FL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Federated Learning not available")
    
    try:
        form = await request.form()
        device_id = form.get("device_id")
        training_loss = float(form.get("training_loss", 0.0))
        local_epochs = int(form.get("local_epochs", 5))
        num_samples = int(form.get("num_samples", 100))
        
        # For demo, we'll accept a simple model data string
        model_data = form.get("model_data", "")
        
        if not device_id:
            raise HTTPException(status_code=400, detail="device_id is required")
        
        # Validate device is participating
        if device_id not in fl_state["participants"]:
            raise HTTPException(
                status_code=403,
                detail="Device not selected for current training round"
            )
        
        if fl_state["status"] != "training":
            raise HTTPException(
                status_code=400,
                detail=f"Not accepting submissions. Current status: {fl_state['status']}"
            )
        
        # Store model update
        participant = fl_state["participants"][device_id]
        participant.update({
            "status": "submitted",
            "model_submitted": True,
            "submission_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "model_size": len(model_data),
            "training_loss": training_loss,
            "local_epochs": local_epochs,
            "num_samples": num_samples,
            "model_data": model_data.encode()
        })
        
        logger.info(f"Received model update from {device_id}: {len(model_data)} bytes, loss: {training_loss}")
        
        # Check if all participants have submitted
        submitted_count = sum(1 for p in fl_state["participants"].values() if p["model_submitted"])
        total_participants = len(fl_state["participants"])
        
        response_data = {
            "success": True,
            "message": "Model update received successfully",
            "submission_info": {
                "device_id": device_id,
                "model_size": len(model_data),
                "training_loss": training_loss,
                "submitted_count": submitted_count,
                "total_participants": total_participants,
                "round_complete": submitted_count == total_participants
            }
        }
        
        # Trigger aggregation if all models received
        if submitted_count == total_participants:
            await trigger_model_aggregation()
            response_data["message"] += " All models received, starting aggregation."
        
        return JSONResponse(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting model update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fl/global_model")
@limiter.limit("20/minute")
async def get_global_model(request: Request, device_id: str):
    """Download the latest global model (simplified for demo)."""
    if not FL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Federated Learning not available")
    
    try:
        # Validate device is registered
        device_ids = [d["device_id"] for d in devices_registry.values()]
        
        if device_id not in device_ids:
            raise HTTPException(status_code=403, detail="Device not registered")
        
        # Return model info (simplified for demo)
        model_info = {
            "model_type": "SimpleCNN",
            "round": fl_state["current_round"],
            "parameters": {
                "conv1.weight": "tensor([32, 1, 3, 3])",
                "conv1.bias": "tensor([32])",  
                "conv2.weight": "tensor([64, 32, 3, 3])",
                "conv2.bias": "tensor([64])",
                "fc1.weight": "tensor([128, 3136])",
                "fc1.bias": "tensor([128])",
                "fc2.weight": "tensor([10, 128])",
                "fc2.bias": "tensor([10])"
            },
            "metadata": {
                "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "aggregation_method": "FederatedAveraging",
                "participants": len(fl_state["participants"]) if fl_state["participants"] else 0
            }
        }
        
        return JSONResponse({
            "success": True,
            "global_model": model_info,
            "message": f"Global model for round {fl_state['current_round']}"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving global model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fl/training_history")
@limiter.limit("30/minute")
async def get_training_history(request: Request):
    """Get federated learning training history."""
    if not FL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Federated Learning not available")
    
    return JSONResponse({
        "success": True,
        "training_history": fl_state["training_history"],
        "current_round": fl_state["current_round"],
        "total_rounds": fl_state["total_rounds"],
        "system_status": fl_state["status"]
    })


@app.post("/api/fl/reset")
@limiter.limit("2/minute")
async def reset_fl_system(request: Request):
    """Reset the federated learning system (admin only)."""
    if not FL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Federated Learning not available")
    
    try:
        fl_state.update({
            "current_round": 0,
            "status": "idle",
            "participants": {},
            "global_model": None,
            "round_start_time": None,
            "training_history": []
        })
        
        logger.info("FL system reset successfully")
        
        return JSONResponse({
            "success": True,
            "message": "Federated learning system reset successfully"
        })
        
    except Exception as e:
        logger.error(f"Error resetting FL system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# FL Dashboard endpoint
@app.get("/fl-dashboard", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def fl_dashboard(request: Request):
    """Serve FL dashboard page."""
    try:
        active_devices = [d for d in devices_registry.values() if d.get("status") == "enrolled"]
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>QFLARE - Federated Learning Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #1e40af, #3b82f6); color: white; min-height: 100vh; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .card {{ background: rgba(255,255,255,0.1); padding: 30px; border-radius: 20px; backdrop-filter: blur(10px); margin: 20px 0; }}
                .btn {{ background: #06b6d4; color: white; padding: 15px 30px; border: none; border-radius: 10px; text-decoration: none; display: inline-block; margin: 10px; font-weight: bold; cursor: pointer; }}
                .btn:hover {{ background: #0891b2; }}
                .btn-danger {{ background: #ef4444; }}
                .btn-danger:hover {{ background: #dc2626; }}
                .status {{ padding: 20px; border-radius: 10px; margin: 20px 0; }}
                .status-idle {{ background: rgba(34, 197, 94, 0.2); }}
                .status-training {{ background: rgba(234, 179, 8, 0.2); }}
                .status-aggregating {{ background: rgba(168, 85, 247, 0.2); }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .metric {{ text-align: center; padding: 20px; background: rgba(255,255,255,0.05); border-radius: 10px; }}
                .history {{ max-height: 300px; overflow-y: auto; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 10px; }}
                #fl-status {{ margin: 20px 0; }}
                form {{ display: inline-block; margin: 10px; }}
                input, select {{ padding: 10px; margin: 5px; border-radius: 5px; border: none; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="card">
                    <h1>🤖 QFLARE Federated Learning Dashboard</h1>
                    <p>Manage and monitor federated learning training rounds</p>
                    
                    <div id="fl-status" class="status status-{fl_state['status']}">
                        <h3>System Status: {fl_state['status'].upper()}</h3>
                        <div class="grid">
                            <div class="metric">
                                <h4>Current Round</h4>
                                <div style="font-size: 2em; font-weight: bold;">{fl_state['current_round']}</div>
                            </div>
                            <div class="metric">
                                <h4>Active Devices</h4>
                                <div style="font-size: 2em; font-weight: bold;">{len(active_devices)}</div>
                            </div>
                            <div class="metric">
                                <h4>Participants</h4>
                                <div style="font-size: 2em; font-weight: bold;">{len(fl_state['participants'])}</div>
                            </div>
                            <div class="metric">
                                <h4>Submitted Models</h4>
                                <div style="font-size: 2em; font-weight: bold;">{sum(1 for p in fl_state['participants'].values() if p.get('model_submitted', False))}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h3>🚀 Start Training Round</h3>
                        <form action="/api/fl/start_round" method="post" style="display: block;">
                            <label>Target Participants:</label>
                            <input type="number" name="target_participants" value="3" min="2" max="10" required>
                            
                            <label>Local Epochs:</label>
                            <input type="number" name="local_epochs" value="5" min="1" max="20" required>
                            
                            <label>Learning Rate:</label>
                            <input type="number" name="learning_rate" value="0.01" step="0.001" min="0.001" max="1" required>
                            
                            <button type="submit" class="btn" {'disabled' if fl_state['status'] != 'idle' else ''}>
                                Start New Round
                            </button>
                        </form>
                    </div>
                    
                    <div class="card">
                        <h3>📊 Training History</h3>
                        <div class="history">
                            {'<p>No training history yet. Start your first round!</p>' if not fl_state['training_history'] else ''}
                            {''.join([f"<p><strong>Round {h.get('round', 'N/A')}</strong> - {h.get('participants', 0)} participants, Avg Loss: {h.get('avg_loss', 0):.4f}</p>" for h in fl_state['training_history'][-10:]])}
                        </div>
                    </div>
                    
                    <div class="card">
                        <h3>🔧 System Controls</h3>
                        <a href="/api/fl/status" class="btn">🔄 Refresh Status</a>
                        <a href="/api/fl/training_history" class="btn">📈 View History</a>
                        <form action="/api/fl/reset" method="post" style="display: inline;">
                            <button type="submit" class="btn btn-danger" onclick="return confirm('Are you sure you want to reset the FL system?')">
                                🔄 Reset System
                            </button>
                        </form>
                    </div>
                    
                    <div class="card">
                        <h3>📱 Active Devices</h3>
                        {'<p>No active devices. Register devices first!</p>' if not active_devices else ''}
                        {''.join([f"<p><strong>{d['device_id']}</strong> - {d.get('device_type', 'Unknown')} ({'✅ Enrolled' if d.get('status') == 'enrolled' else '⏳ Pending'})</p>" for d in active_devices[:10]])}
                        <p><a href="/devices" class="btn">View All Devices</a></p>
                    </div>
                    
                    <div class="card" style="text-align: center;">
                        <p><a href="/" class="btn">🏠 Back to Home</a></p>
                    </div>
                </div>
            </div>
            
            <script>
                // Auto-refresh status every 30 seconds
                setTimeout(function() {{
                    window.location.reload();
                }}, 30000);
            </script>
        </body>
        </html>
        """
    
    except Exception as e:
        logger.error(f"Error rendering FL dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to render FL dashboard")


# Helper function for model aggregation
async def trigger_model_aggregation():
    """Trigger model aggregation after all participants submit."""
    try:
        fl_state["status"] = "aggregating"
        logger.info("Starting model aggregation...")
        
        # Simulate aggregation process
        time.sleep(2)  # Simulate processing time
        
        # Calculate round statistics
        participants = fl_state["participants"]
        avg_loss = sum(p["training_loss"] for p in participants.values() if p["model_submitted"]) / len(participants)
        total_samples = sum(p["num_samples"] for p in participants.values() if p["model_submitted"])
        
        # Store aggregated model (simplified for demo)
        fl_state["global_model"] = {
            "round": fl_state["current_round"],
            "participants": len(participants),
            "avg_loss": avg_loss,
            "total_samples": total_samples,
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Record training history
        round_history = {
            "round": fl_state["current_round"],
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "participants": len(participants),
            "avg_loss": avg_loss,
            "total_samples": total_samples
        }
        fl_state["training_history"].append(round_history)
        
        # Reset for next round
        fl_state["status"] = "idle"
        fl_state["participants"] = {}
        
        logger.info(f"Model aggregation completed for round {fl_state['current_round']}")
        
    except Exception as e:
        logger.error(f"Error in model aggregation: {e}")
        fl_state["status"] = "error"


# =======================
# FEDERATED LEARNING ENDPOINTS
# =======================

@app.get("/api/fl/status")
@limiter.limit("60/minute")
async def fl_status(request: Request):
    """Get FL system status."""
    try:
        if fl_controller:
            status = fl_controller.get_status()
            return {"success": True, "fl_status": status}
        else:
            return {"success": False, "message": "FL system not available"}
    except Exception as e:
        logger.error(f"Error getting FL status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fl/register")
@limiter.limit("10/minute")
async def fl_register_device(request: Request):
    """Register a device for federated learning."""
    try:
        data = await request.json()
        device_id = data.get("device_id")
        capabilities = data.get("capabilities", {})
        
        if not device_id:
            raise HTTPException(status_code=400, detail="device_id is required")
        
        if fl_controller:
            success = fl_controller.register_device(device_id, capabilities)
            if success:
                return {"success": True, "message": f"Device {device_id} registered for FL"}
            else:
                raise HTTPException(status_code=400, detail="Failed to register device")
        else:
            raise HTTPException(status_code=503, detail="FL system not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering device for FL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fl/submit_model")
@limiter.limit("30/minute")
async def fl_submit_model(request: Request):
    """Submit a trained model update."""
    try:
        data = await request.json()
        device_id = data.get("device_id")
        
        if not device_id:
            raise HTTPException(status_code=400, detail="device_id is required")
        
        if fl_controller:
            success = fl_controller.submit_model(data)
            if success:
                return {"success": True, "message": "Model submitted successfully"}
            else:
                raise HTTPException(status_code=400, detail="Failed to submit model")
        else:
            raise HTTPException(status_code=503, detail="FL system not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fl/global_model")
@limiter.limit("60/minute")
async def fl_get_global_model(request: Request):
    """Get the current global model."""
    try:
        if fl_controller:
            model = fl_controller.get_global_model()
            if model:
                return {"success": True, "global_model": model}
            else:
                return {"success": False, "message": "No global model available yet"}
        else:
            raise HTTPException(status_code=503, detail="FL system not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting global model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fl/start_training")
@limiter.limit("5/minute")
async def fl_start_training(request: Request):
    """Start a new FL training round."""
    try:
        data = await request.json()
        rounds = data.get("rounds", 10)
        min_participants = data.get("min_participants", 2)
        
        if fl_controller:
            success = fl_controller.start_training(rounds, min_participants)
            if success:
                return {"success": True, "message": f"Training started for {rounds} rounds"}
            else:
                raise HTTPException(status_code=400, detail="Failed to start training")
        else:
            raise HTTPException(status_code=503, detail="FL system not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fl/stop_training")
@limiter.limit("5/minute")
async def fl_stop_training(request: Request):
    """Stop the current FL training."""
    try:
        if fl_controller:
            fl_controller.stop_training()
            return {"success": True, "message": "Training stopped"}
        else:
            raise HTTPException(status_code=503, detail="FL system not available")
    except Exception as e:
        logger.error(f"Error stopping training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fl/devices")
@limiter.limit("60/minute")
async def fl_list_devices(request: Request):
    """List all registered FL devices."""
    try:
        if fl_controller:
            devices = fl_controller.list_devices()
            return {"success": True, "devices": devices}
        else:
            raise HTTPException(status_code=503, detail="FL system not available")
    except Exception as e:
        logger.error(f"Error listing devices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fl/metrics")
@limiter.limit("60/minute")
async def fl_get_metrics(request: Request):
    """Get FL training metrics."""
    try:
        if fl_controller:
            metrics = fl_controller.get_metrics()
            return {"success": True, "metrics": metrics}
        else:
            raise HTTPException(status_code=503, detail="FL system not available")
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("🚀 Starting QFLARE Secure Registration Demo Server...")
    print("📍 Navigate to: http://localhost:8080")
    print("🔐 Secure Registration: http://localhost:8080/secure-register")
    
    uvicorn.run(
        "simple_server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )