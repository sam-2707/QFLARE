
"""
QFLARE Server - Main Application

This is the main FastAPI application for the QFLARE federated learning server.
Now with persistent database storage.
"""

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
import os
import time
from pathlib import Path
from datetime import datetime

from api.routes import router as api_router
from fl_core.client_manager import register_client
from registry import register_device, get_registered_devices
from database import init_database, close_database

# Import secure key exchange
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from secure_key_exchange import SecureKeyExchange

# Configure logging
# 
# # Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# --- CORRECTED ORDER ---
# 1. Initialize Limiter first
limiter = Limiter(key_func=get_remote_address)

# 2. Initialize FastAPI app
app = FastAPI(
    title="QFLARE Server",
    description="Quantum-Resistant Federated Learning Server",
    version="1.0.0"
)

# 3. Apply middleware and state to the app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.on_event("startup")
async def startup_event():
    """Initialize database on server startup."""
    try:
        # Initialize database with default SQLite configuration
        db_config = {
            "database_type": "sqlite",
            "sqlite_path": "qflare.db"
        }
        await init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup database connections on server shutdown."""
    try:
        await close_database()
        logger.info("Database connections cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up database: {e}")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- END CORRECTION ---

# Define BASE_DIR as the directory containing this file
BASE_DIR = Path(__file__).resolve().parent

# Configure templates (absolute path)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Mount static files (absolute path)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Include API routes
app.include_router(api_router, prefix="/api", tags=["api"])


@app.get("/", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def root(request: Request):
    """Main landing page."""
    try:
        device_count = len(get_registered_devices())
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "device_count": device_count}
        )
    except Exception as e:
        logger.error(f"Error rendering root page: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/register-v2", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def register_v2_form(request: Request):
    """Serves the new interactive registration page with a cache-busting timestamp."""
    return templates.TemplateResponse(
        "register_v2.html", 
        {"request": request, "timestamp": int(time.time())} # Add the timestamp here
    )

@app.get("/enroll_dashboard", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def enroll_dashboard(request: Request):
    """Serves the device enrollment dashboard."""
    return templates.TemplateResponse(
        "enroll_dashboard.html",
        {"request": request}
    )
    
@app.get("/devices", response_class=HTMLResponse)
@limiter.limit("30/minute")
async def list_devices(request: Request):
    """Display list of registered devices."""
    try:
        devices = get_registered_devices()
        return templates.TemplateResponse(
            "devices.html", 
            {"request": request, "devices": devices}
        )
    except Exception as e:
        logger.error(f"Error rendering devices page: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/status", response_class=HTMLResponse)
@limiter.limit("30/minute")
async def system_status_page(request: Request):
    """Serves the main system status and monitoring page."""
    return templates.TemplateResponse("status.html", {"request": request})

@app.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Health check endpoint."""
    try:
        from fl_core.aggregator import get_aggregation_status
        from enclave.mock_enclave import get_secure_enclave
        
        # Check component status
        aggregation_status = get_aggregation_status()
        enclave_status = get_secure_enclave().get_enclave_status()
        
        return {
            "status": "healthy",
            "components": {
                "server": "healthy",
                "enclave": enclave_status.get("status", "unknown"),
                "aggregator": "healthy" if aggregation_status else "error"
            },
            "device_count": len(get_registered_devices()),
            "aggregation_status": aggregation_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/status")
@limiter.limit("30/minute")
async def system_status(request: Request):
    """Get system status information."""
    try:
        from registry import get_device_statistics
        from fl_core.aggregator import get_aggregation_status
        
        device_stats = get_device_statistics()
        aggregation_status = get_aggregation_status()
        
        return {
            "system_status": "operational",
            "device_statistics": device_stats,
            "aggregation_status": aggregation_status,
            "security_features": {
                "secure_enrollment": True,
                "post_quantum_crypto": True,
                "secure_enclave": True,
                "poisoning_defense": True
            }
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Error getting system status")


# Legacy endpoints for backward compatibility (deprecated)
@app.get("/register-v2", response_class=HTMLResponse)
async def register_v2_form(request: Request):
    return templates.TemplateResponse("register_v2.html", {"request": request})


@app.get("/authenticate", response_class=HTMLResponse)
async def authenticate_form(request: Request):
    return templates.TemplateResponse("authenticate.html", {"request": request})


@app.get("/register", response_class=HTMLResponse)
@limiter.limit("5/minute")
async def register_form(request: Request):
    """
    Legacy registration form (deprecated).
    
    This endpoint is kept for backward compatibility but should not be used.
    Use the secure enrollment process instead.
    """
    logger.warning("Legacy registration form accessed - use secure enrollment instead")
    return templates.TemplateResponse(
        "register.html", 
        {"request": request, "message": "This registration method is deprecated. Use secure enrollment instead."}
    )


@app.post("/register", response_class=HTMLResponse)
@limiter.limit("5/minute")
async def register_device_legacy(request: Request):
    """
    Legacy device registration (deprecated).
    
    This endpoint is kept for backward compatibility but should not be used.
    Use the secure enrollment process instead.
    """
    logger.warning("Legacy device registration called - use secure enrollment instead")
    
    try:
        # Parse form data
        form_data = await request.form()
        device_id = form_data.get("device_id")
        device_type = form_data.get("device_type", "edge")
        location = form_data.get("location", "")
        description = form_data.get("description", "")
        capabilities = form_data.get("capabilities", "")
        
        if not device_id:
            return templates.TemplateResponse(
                "register.html", 
                {
                    "request": request, 
                    "error": "Device ID is required",
                    "device_id": device_id,
                    "device_type": device_type,
                    "location": location,
                    "description": description,
                    "capabilities": capabilities
                }
            )
        
        # Register device with legacy method
        device_info = {
            "registration_method": "legacy",
            "device_type": device_type,
            "location": location,
            "description": description,
            "capabilities": capabilities,
            "registration_time": datetime.now().isoformat()
        }
        
        success = register_device(device_id, device_info)
        
        if success:
            return templates.TemplateResponse(
                "register.html", 
                {
                    "request": request, 
                    "message": f"Device '{device_id}' registered successfully!",
                    "device_id": "",
                    "device_type": "edge",
                    "location": "",
                    "description": "",
                    "capabilities": ""
                }
            )
        else:
            return templates.TemplateResponse(
                "register.html", 
                {
                    "request": request, 
                    "error": f"Failed to register device '{device_id}'. Device may already exist.",
                    "device_id": device_id,
                    "device_type": device_type,
                    "location": location,
                    "description": description,
                    "capabilities": capabilities
                }
            )
    except Exception as e:
        logger.error(f"Error in legacy device registration: {e}")
        return templates.TemplateResponse(
            "register.html", 
            {
                "request": request, 
                "error": f"Registration failed: {str(e)}",
                "device_id": device_id if 'device_id' in locals() else "",
                "device_type": device_type if 'device_type' in locals() else "edge",
                "location": location if 'location' in locals() else "",
                "description": description if 'description' in locals() else "",
                "capabilities": capabilities if 'capabilities' in locals() else ""
            }
        )


# =============================================================================
# SECURE REGISTRATION ENDPOINTS
# =============================================================================

# Initialize secure key exchange
secure_key_exchange = None

try:
    from secure_key_exchange import get_secure_key_exchange
    secure_key_exchange = get_secure_key_exchange()
    logger.info("✅ Secure key exchange initialized")
except Exception as e:
    logger.warning(f"⚠️ Secure key exchange not available: {e}")


@app.get("/secure-register", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def secure_register_form(request: Request):
    """Serve the secure registration page with MITM attack prevention."""
    return templates.TemplateResponse(
        "secure_register.html", 
        {"request": request, "timestamp": int(time.time())}
    )


@app.post("/api/secure_register")
@limiter.limit("5/minute")
async def process_secure_registration(request: Request):
    """Process secure registration request with chosen key exchange method."""
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
                    "error": "Secure key exchange service is not available. Please contact administrator.",
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
            success_message = f"""
            🔐 <strong>QR Code + OTP Method Initiated</strong><br>
            • QR Code file: {result['qr_code_file']}<br>
            • One-Time Password: <strong>{result['otp']}</strong><br>
            • Expires in: {result['expires_in']//60} minutes<br>
            <br>
            📱 <strong>Next Steps:</strong><br>
            1. Admin will provide QR code via secure channel<br>
            2. You will receive OTP via {user_request.get('secure_contact', 'secure channel')}<br>
            3. Scan QR code and enter OTP to decrypt your quantum keys
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
            • TOTP Secret: <strong>{result['totp_secret']}</strong><br>
            • Device ID: {result['device_id']}<br>
            <br>
            📱 <strong>Next Steps:</strong><br>
            1. Admin will provide TOTP secret via secure channel<br>
            2. Set up authenticator app with the secret<br>
            3. Use current TOTP code to authenticate and receive keys
            """
            
        elif key_exchange_method == "physical_token":
            result = secure_key_exchange.method_4_physical_token_exchange(user_request)
            success_message = f"""
            🔑 <strong>Physical Token Method Initiated</strong><br>
            • Token File: {result['token_file']}<br>
            • Token PIN: <strong>{result['token_pin']}</strong><br>
            • Delivery: {user_request.get('delivery_method', 'Secure delivery')}<br>
            <br>
            📦 <strong>Next Steps:</strong><br>
            1. Physical token will be delivered to your address<br>
            2. Enter the token PIN to decrypt your quantum keys<br>
            3. Install keys on your device
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
    """Serve verification page for completing key exchange."""
    if not secure_key_exchange:
        raise HTTPException(status_code=503, detail="Secure key exchange service not available")
    
    # Check if device has pending registration
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


@app.post("/api/secure_verify/{device_id}")
@limiter.limit("5/minute")
async def process_secure_verification(request: Request, device_id: str):
    """Process verification and deliver quantum keys."""
    try:
        if not secure_key_exchange:
            raise HTTPException(status_code=503, detail="Secure key exchange service not available")
        
        form = await request.form()
        verification_data = {}
        
        # Extract verification data based on method
        if "otp" in form:
            verification_data["otp"] = form.get("otp")
        if "totp_code" in form:
            verification_data["totp_code"] = form.get("totp_code")
        if "token_pin" in form:
            verification_data["token_pin"] = form.get("token_pin")
        
        # Verify and deliver keys
        quantum_keys = secure_key_exchange.verify_and_deliver_keys(device_id, verification_data)
        
        # Register device in QFLARE system
        device_info = {
            "device_id": device_id,
            "device_type": "secure_enrolled",
            "location": "Securely enrolled",
            "description": f"Device enrolled via secure key exchange",
            "capabilities": ["quantum_safe", "federated_learning"],
            "quantum_keys": quantum_keys
        }
        
        # Store in registry
        register_device(device_info)
        
        logger.info(f"✅ Device {device_id} successfully verified and enrolled with quantum keys")
        
        return templates.TemplateResponse(
            "secure_verify.html",
            {
                "request": request,
                "device_id": device_id,
                "success": "✅ Verification successful! Your quantum keys have been delivered and your device is now enrolled in the QFLARE network.",
                "quantum_keys_active": True,
                "timestamp": int(time.time())
            }
        )
        
    except ValueError as e:
        return templates.TemplateResponse(
            "secure_verify.html",
            {
                "request": request,
                "device_id": device_id,
                "error": str(e),
                "timestamp": int(time.time())
            }
        )
    except Exception as e:
        logger.error(f"Error in secure verification: {e}")
        return templates.TemplateResponse(
            "secure_verify.html",
            {
                "request": request,
                "device_id": device_id,
                "error": f"Verification failed: {str(e)}",
                "timestamp": int(time.time())
            }
        )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors."""
    return templates.TemplateResponse(
        "404.html", 
        {"request": request, "message": "Page not found"}, 
        status_code=404
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    """Handle 500 errors."""
    return templates.TemplateResponse(
        "500.html", 
        {"request": request, "message": "Internal server error"}, 
        status_code=500
    )


if __name__ == "__main__":
    import uvicorn
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get configuration
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    
    logger.info(f"Starting QFLARE server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )