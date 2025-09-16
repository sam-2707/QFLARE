#!/usr/bin/env python3
"""
Modern Web UI for QFLARE - Industry Standard Dashboard
Built with FastAPI + Modern Frontend Framework Integration
"""

import os
import sys
import json
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our enhanced components
from enhanced_device_registry import EnhancedDeviceRegistry, DeviceStatus, DeviceType
from server.auth.challenge_response import (
    get_challenge_manager, 
    ChallengeRequest, 
    ChallengeResponse,
    KeyType,
    create_challenge_request
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModernQFLAREApp:
    """Modern QFLARE web application with industry-standard UI"""
    
    def __init__(self):
        self.app = FastAPI(
            title="QFLARE - Federated Learning Platform",
            description="Production-grade federated learning coordination system",
            version="2.0.0",
            docs_url="/api/docs",
            redoc_url="/api/redoc"
        )
        
        # Initialize enhanced components
        self.device_registry = EnhancedDeviceRegistry()
        self.challenge_manager = get_challenge_manager()
        self.device_registry = EnhancedDeviceRegistry()
        
        # Configure middleware
        self.setup_middleware()
        
        # Setup routes
        self.setup_routes()
        
        # Setup static files and templates
        self.setup_static_files()
        
    def setup_middleware(self):
        """Configure security and CORS middleware"""
        
        # CORS middleware for API access
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "http://localhost:8080"],  # React/Vue dev servers
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Trusted host middleware for security
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1", "*.qflare.local"]
        )
    
    def setup_static_files(self):
        """Setup static file serving and templates"""
        
        # Create directories if they don't exist
        static_dir = project_root / "static" / "modern"
        templates_dir = project_root / "templates" / "modern"
        
        static_dir.mkdir(parents=True, exist_ok=True)
        templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        
        # Setup Jinja2 templates
        self.templates = Jinja2Templates(directory=str(templates_dir))
    
    def _get_device_type_enum(self, device_type_str: str) -> DeviceType:
        """Convert device type string to enum, handling various formats"""
        # Normalize the input
        device_type_str = device_type_str.lower().replace('-', '_')
        
        # Map common variations
        type_mapping = {
            'edge': DeviceType.EDGE,
            'edge_node': DeviceType.EDGE_NODE,
            'mobile': DeviceType.MOBILE,
            'iot': DeviceType.IOT,
            'server': DeviceType.SERVER,
            'gateway': DeviceType.GATEWAY
        }
        
        if device_type_str in type_mapping:
            return type_mapping[device_type_str]
        
        # Try direct enum lookup as fallback
        try:
            return DeviceType(device_type_str)
        except ValueError:
            # Default to EDGE if not found
            logger.warning(f"Unknown device type '{device_type_str}', defaulting to EDGE")
            return DeviceType.EDGE
    
    def setup_routes(self):
        """Setup API routes and web interface"""
        
        # Security dependency
        security = HTTPBearer(auto_error=False)
        
        # Root route - Modern dashboard
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            """Serve the modern dashboard"""
            return self.templates.TemplateResponse("dashboard.html", {
                "request": request,
                "title": "QFLARE - Federated Learning Platform",
                "version": "2.0.0"
            })
        
        # Device Management API
        @self.app.post("/api/v2/enrollment/tokens")
        async def create_enrollment_token(
            device_type: str,
            organization: str,
            max_uses: int = 1,
            validity_hours: int = 24
        ):
            """Create a new device enrollment token"""
            try:
                device_type_enum = self._get_device_type_enum(device_type)
                token = self.device_registry.generate_enrollment_token(
                    device_type=device_type_enum,
                    organization=organization,
                    max_uses=max_uses,
                    validity_hours=validity_hours
                )
                
                return {
                    "token": token.token,
                    "device_type": token.device_type.value,
                    "organization": token.organization,
                    "valid_until": token.valid_until.isoformat(),
                    "max_uses": token.max_uses
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/v2/enrollment/tokens")
        async def get_enrollment_tokens(
            status: Optional[str] = None,
            organization: Optional[str] = None,
            limit: int = 100
        ):
            """Get all enrollment tokens with filtering"""
            try:
                tokens = self.device_registry.list_enrollment_tokens(
                    organization=organization,
                    limit=limit
                )
                
                # Filter by status if provided
                if status:
                    tokens = [t for t in tokens if t.get('status') == status]
                
                return {
                    "success": True,
                    "tokens": tokens,
                    "count": len(tokens),
                    "filters": {
                        "status": status,
                        "organization": organization
                    }
                }
                
            except Exception as e:
                logger.error(f"Failed to get enrollment tokens: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v2/enrollment/generate-token")
        async def generate_enrollment_token(request: Dict[str, Any]):
            """Generate a new enrollment token with specified parameters"""
            try:
                device_type = request.get('device_type', 'edge-node')
                organization = request.get('organization', 'default')
                validity_hours = request.get('validity_hours', 24)
                max_uses = request.get('max_uses', 1)
                
                # Reuse the create_enrollment_token logic
                return await create_enrollment_token(
                    device_type=device_type,
                    organization=organization,
                    max_uses=max_uses,
                    validity_hours=validity_hours
                )
                
            except Exception as e:
                logger.error(f"Failed to generate enrollment token: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v2/devices/register")
        async def register_device_v2(
            request: Request,
            device_data: Dict[str, Any]
        ):
            """Enhanced device registration endpoint"""
            try:
                client_ip = request.client.host
                user_agent = request.headers.get("user-agent", "")
                
                result = self.device_registry.register_device(
                    device_id=device_data["device_id"],
                    public_key=device_data["public_key"],
                    device_type=device_data["device_type"],
                    capabilities=device_data["capabilities"],
                    enrollment_token=device_data["enrollment_token"],
                    location=device_data.get("location"),
                    metadata=device_data.get("metadata"),
                    ip_address=client_ip,
                    user_agent=user_agent
                )
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/v2/devices")
        async def list_devices_v2(
            status: Optional[str] = None,
            organization: Optional[str] = None,
            device_type: Optional[str] = None,
            limit: int = 100
        ):
            """List devices with enhanced filtering"""
            try:
                status_enum = DeviceStatus(status) if status else None
                type_enum = self._get_device_type_enum(device_type) if device_type else None
                
                devices = self.device_registry.list_devices(
                    status=status_enum,
                    organization=organization,
                    device_type=type_enum,
                    limit=limit
                )
                
                return {
                    "devices": devices,
                    "total": len(devices),
                    "filters": {
                        "status": status,
                        "organization": organization,
                        "device_type": device_type
                    }
                }
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/v2/devices/{device_id}")
        async def get_device_info_v2(device_id: str):
            """Get detailed device information"""
            device_info = self.device_registry.get_device_info(device_id)
            if not device_info:
                raise HTTPException(status_code=404, detail="Device not found")
            return device_info
        
        @self.app.put("/api/v2/devices/{device_id}/approve")
        async def approve_device_v2(device_id: str, approved_by: str = "admin"):
            """Approve a pending device"""
            success = self.device_registry.approve_device(device_id, approved_by)
            if not success:
                raise HTTPException(status_code=404, detail="Device not found or not pending")
            return {"status": "approved", "device_id": device_id}
        
        @self.app.put("/api/v2/devices/{device_id}/reject")
        async def reject_device_v2(device_id: str, reason: str = "Admin decision"):
            """Reject a pending device"""
            success = self.device_registry.reject_device(device_id, reason)
            if not success:
                raise HTTPException(status_code=404, detail="Device not found or not pending")
            return {"status": "rejected", "device_id": device_id, "reason": reason}
        
        @self.app.put("/api/v2/devices/{device_id}/revoke")
        async def revoke_device_v2(device_id: str, reason: str = "Security concern"):
            """Revoke device access"""
            success = self.device_registry.revoke_device(device_id, reason)
            if not success:
                raise HTTPException(status_code=404, detail="Device not found")
            return {"status": "revoked", "device_id": device_id, "reason": reason}
        
        # Challenge-Response Authentication API
        @self.app.post("/api/v2/auth/challenge")
        async def process_challenge_request(request: Dict[str, Any]):
            """Process timestamp-based challenge request"""
            try:
                challenge_request = ChallengeRequest(
                    device_id=request['device_id'],
                    timestamp=request['timestamp'],
                    nonce=request['nonce'],
                    signature=request.get('signature')
                )
                
                response = self.challenge_manager.process_challenge_request(challenge_request)
                
                return {
                    "success": response.status.value != "failed",
                    "challenge_id": response.challenge_id,
                    "encrypted_session_key": base64.b64encode(response.encrypted_session_key).decode() if response.encrypted_session_key else "",
                    "server_timestamp": response.server_timestamp,
                    "validity_duration": response.validity_duration,
                    "status": response.status.value
                }
                
            except Exception as e:
                logger.error(f"Challenge processing failed: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/api/v2/auth/register-key")
        async def register_device_key(request: Dict[str, Any]):
            """Register a device's public key for challenge-response"""
            try:
                device_id = request['device_id']
                public_key_b64 = request['public_key']
                key_type_str = request.get('key_type', 'kyber768')
                
                # Decode public key
                public_key = base64.b64decode(public_key_b64)
                key_type = KeyType(key_type_str.upper())
                
                success = self.challenge_manager.register_device_key(
                    device_id=device_id,
                    public_key=public_key,
                    key_type=key_type
                )
                
                return {
                    "success": success,
                    "device_id": device_id,
                    "key_type": key_type.value,
                    "message": "Public key registered successfully" if success else "Registration failed"
                }
                
            except Exception as e:
                logger.error(f"Key registration failed: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/v2/auth/session/{challenge_id}")
        async def validate_session(challenge_id: str):
            """Validate an active session"""
            try:
                session = self.challenge_manager.validate_session(challenge_id)
                
                if session:
                    return {
                        "valid": True,
                        "device_id": session.device_id,
                        "created_at": session.created_at.isoformat(),
                        "expires_at": session.expires_at.isoformat(),
                        "request_count": session.request_count
                    }
                else:
                    return {
                        "valid": False,
                        "message": "Session not found or expired"
                    }
                    
            except Exception as e:
                logger.error(f"Session validation failed: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.delete("/api/v2/auth/session/{challenge_id}")
        async def revoke_session(challenge_id: str):
            """Revoke an active session"""
            try:
                success = self.challenge_manager.revoke_session(challenge_id)
                
                return {
                    "success": success,
                    "challenge_id": challenge_id,
                    "message": "Session revoked successfully" if success else "Session not found"
                }
                
            except Exception as e:
                logger.error(f"Session revocation failed: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/v2/auth/status")
        async def get_auth_status():
            """Get authentication system status"""
            try:
                status = self.challenge_manager.get_system_status()
                
                return {
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "auth_system": status
                }
                
            except Exception as e:
                logger.error(f"Auth status failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # System Status and Metrics
        @self.app.get("/api/v2/system/status")
        async def system_status_v2():
            """Get comprehensive system status"""
            try:
                devices = self.device_registry.list_devices(limit=1000)
                
                status_counts = {}
                type_counts = {}
                org_counts = {}
                
                for device in devices:
                    status = device['status']
                    device_type = device['device_type']
                    org = device['organization']
                    
                    status_counts[status] = status_counts.get(status, 0) + 1
                    type_counts[device_type] = type_counts.get(device_type, 0) + 1
                    org_counts[org] = org_counts.get(org, 0) + 1
                
                return {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "statistics": {
                        "total_devices": len(devices),
                        "devices_by_status": status_counts,
                        "devices_by_type": type_counts,
                        "devices_by_organization": org_counts
                    },
                    "system_info": {
                        "version": "2.0.0",
                        "database": "sqlite",
                        "security_level": "enhanced"
                    }
                }
                
            except Exception as e:
                return {
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
        
        # Admin Interface Routes
        @self.app.get("/admin", response_class=HTMLResponse)
        async def admin_dashboard(request: Request):
            """Admin dashboard for device management"""
            return self.templates.TemplateResponse("admin.html", {
                "request": request,
                "title": "QFLARE Admin - Device Management"
            })
        
        @self.app.get("/devices", response_class=HTMLResponse)
        async def device_management(request: Request):
            """Device management interface"""
            return self.templates.TemplateResponse("devices.html", {
                "request": request,
                "title": "QFLARE - Device Management"
            })
        
        @self.app.get("/enrollment", response_class=HTMLResponse)
        async def enrollment_management(request: Request):
            """Enrollment token management interface"""
            return self.templates.TemplateResponse("enrollment.html", {
                "request": request,
                "title": "QFLARE - Enrollment Management"
            })
        
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            """Simple health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "2.0.0"
            }

def create_modern_templates():
    """Create modern HTML templates with industry-standard design"""
    
    templates_dir = project_root / "templates" / "modern"
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    # Main dashboard template
    dashboard_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    
    <!-- Tailwind CSS for modern styling -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Chart.js for data visualization -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <!-- Alpine.js for reactive components -->
    <script defer src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js"></script>
    
    <!-- Custom styles -->
    <style>
        [x-cloak] { display: none !important; }
        .fade-in { animation: fadeIn 0.5s ease-in; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body class="bg-gray-50 min-h-screen" x-data="qflareApp()" x-init="init()">
    
    <!-- Navigation Header -->
    <nav class="bg-white shadow-sm border-b border-gray-200">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between h-16">
                <div class="flex items-center">
                    <div class="flex-shrink-0">
                        <h1 class="text-2xl font-bold text-blue-600">QFLARE</h1>
                        <span class="text-xs text-gray-500">Federated Learning Platform v2.0</span>
                    </div>
                    
                    <nav class="hidden md:ml-8 md:flex md:space-x-8">
                        <a href="/" class="text-blue-600 border-b-2 border-blue-600 px-1 pt-1 text-sm font-medium">
                            Dashboard
                        </a>
                        <a href="/devices" class="text-gray-500 hover:text-gray-700 px-1 pt-1 text-sm font-medium">
                            Devices
                        </a>
                        <a href="/enrollment" class="text-gray-500 hover:text-gray-700 px-1 pt-1 text-sm font-medium">
                            Enrollment
                        </a>
                        <a href="/admin" class="text-gray-500 hover:text-gray-700 px-1 pt-1 text-sm font-medium">
                            Admin
                        </a>
                    </nav>
                </div>
                
                <div class="flex items-center space-x-4">
                    <div class="flex items-center space-x-2">
                        <div class="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                        <span class="text-sm text-gray-600">System Healthy</span>
                    </div>
                    
                    <button class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm font-medium transition duration-150">
                        Settings
                    </button>
                </div>
            </div>
        </div>
    </nav>
    
    <!-- Main Content -->
    <main class="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        
        <!-- Statistics Cards -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div class="bg-white overflow-hidden shadow rounded-lg fade-in">
                <div class="p-5">
                    <div class="flex items-center">
                        <div class="flex-shrink-0">
                            <div class="w-8 h-8 bg-blue-500 rounded-md flex items-center justify-center">
                                <svg class="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
                                    <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z"/>
                                </svg>
                            </div>
                        </div>
                        <div class="ml-5 w-0 flex-1">
                            <dl>
                                <dt class="text-sm font-medium text-gray-500 truncate">Total Devices</dt>
                                <dd class="text-lg font-medium text-gray-900" x-text="systemStats.total_devices || 0"></dd>
                            </dl>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="bg-white overflow-hidden shadow rounded-lg fade-in">
                <div class="p-5">
                    <div class="flex items-center">
                        <div class="flex-shrink-0">
                            <div class="w-8 h-8 bg-green-500 rounded-md flex items-center justify-center">
                                <svg class="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
                                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"/>
                                </svg>
                            </div>
                        </div>
                        <div class="ml-5 w-0 flex-1">
                            <dl>
                                <dt class="text-sm font-medium text-gray-500 truncate">Active Devices</dt>
                                <dd class="text-lg font-medium text-gray-900" x-text="systemStats.devices_by_status?.approved || 0"></dd>
                            </dl>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="bg-white overflow-hidden shadow rounded-lg fade-in">
                <div class="p-5">
                    <div class="flex items-center">
                        <div class="flex-shrink-0">
                            <div class="w-8 h-8 bg-yellow-500 rounded-md flex items-center justify-center">
                                <svg class="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
                                    <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"/>
                                </svg>
                            </div>
                        </div>
                        <div class="ml-5 w-0 flex-1">
                            <dl>
                                <dt class="text-sm font-medium text-gray-500 truncate">Pending Approval</dt>
                                <dd class="text-lg font-medium text-gray-900" x-text="systemStats.devices_by_status?.pending || 0"></dd>
                            </dl>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="bg-white overflow-hidden shadow rounded-lg fade-in">
                <div class="p-5">
                    <div class="flex items-center">
                        <div class="flex-shrink-0">
                            <div class="w-8 h-8 bg-purple-500 rounded-md flex items-center justify-center">
                                <svg class="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
                                    <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                                </svg>
                            </div>
                        </div>
                        <div class="ml-5 w-0 flex-1">
                            <dl>
                                <dt class="text-sm font-medium text-gray-500 truncate">Organizations</dt>
                                <dd class="text-lg font-medium text-gray-900" x-text="Object.keys(systemStats.devices_by_organization || {}).length"></dd>
                            </dl>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Device Status Chart -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div class="bg-white overflow-hidden shadow rounded-lg">
                <div class="px-4 py-5 sm:p-6">
                    <h3 class="text-lg leading-6 font-medium text-gray-900 mb-4">Device Status Distribution</h3>
                    <div class="h-64">
                        <canvas id="statusChart"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="bg-white overflow-hidden shadow rounded-lg">
                <div class="px-4 py-5 sm:p-6">
                    <h3 class="text-lg leading-6 font-medium text-gray-900 mb-4">Device Types</h3>
                    <div class="h-64">
                        <canvas id="typeChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Recent Devices Table -->
        <div class="bg-white shadow overflow-hidden sm:rounded-md">
            <div class="px-4 py-5 sm:px-6">
                <h3 class="text-lg leading-6 font-medium text-gray-900">Recent Device Registrations</h3>
                <p class="mt-1 max-w-2xl text-sm text-gray-500">Latest devices registered in the system</p>
            </div>
            
            <div class="border-t border-gray-200">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Device</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Organization</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Registered</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
                        <template x-for="device in recentDevices" :key="device.device_id">
                            <tr class="hover:bg-gray-50">
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <div>
                                        <div class="text-sm font-medium text-gray-900" x-text="device.device_id"></div>
                                        <div class="text-sm text-gray-500" x-text="device.location || 'Unknown Location'"></div>
                                    </div>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <span class="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-blue-100 text-blue-800" x-text="device.device_type"></span>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900" x-text="device.organization"></td>
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <span class="inline-flex px-2 py-1 text-xs font-semibold rounded-full" 
                                          :class="getStatusClass(device.status)" x-text="device.status"></span>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500" x-text="formatDate(device.created_at)"></td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium">
                                    <template x-if="device.status === 'pending'">
                                        <div class="flex space-x-2">
                                            <button @click="approveDevice(device.device_id)" 
                                                    class="text-green-600 hover:text-green-900">Approve</button>
                                            <button @click="rejectDevice(device.device_id)" 
                                                    class="text-red-600 hover:text-red-900">Reject</button>
                                        </div>
                                    </template>
                                    <template x-if="device.status !== 'pending'">
                                        <a :href="'/devices/' + device.device_id" class="text-blue-600 hover:text-blue-900">View Details</a>
                                    </template>
                                </td>
                            </tr>
                        </template>
                    </tbody>
                </table>
            </div>
        </div>
    </main>
    
    <!-- JavaScript -->
    <script>
        function qflareApp() {
            return {
                systemStats: {},
                recentDevices: [],
                loading: true,
                
                async init() {
                    await this.loadSystemStats();
                    await this.loadRecentDevices();
                    this.initCharts();
                    this.loading = false;
                },
                
                async loadSystemStats() {
                    try {
                        const response = await fetch('/api/v2/system/status');
                        const data = await response.json();
                        this.systemStats = data.statistics || {};
                    } catch (error) {
                        console.error('Failed to load system stats:', error);
                    }
                },
                
                async loadRecentDevices() {
                    try {
                        const response = await fetch('/api/v2/devices?limit=10');
                        const data = await response.json();
                        this.recentDevices = data.devices || [];
                    } catch (error) {
                        console.error('Failed to load recent devices:', error);
                    }
                },
                
                async approveDevice(deviceId) {
                    try {
                        const response = await fetch(`/api/v2/devices/${deviceId}/approve`, {
                            method: 'PUT'
                        });
                        if (response.ok) {
                            await this.loadRecentDevices();
                            await this.loadSystemStats();
                        }
                    } catch (error) {
                        console.error('Failed to approve device:', error);
                    }
                },
                
                async rejectDevice(deviceId) {
                    try {
                        const response = await fetch(`/api/v2/devices/${deviceId}/reject`, {
                            method: 'PUT'
                        });
                        if (response.ok) {
                            await this.loadRecentDevices();
                            await this.loadSystemStats();
                        }
                    } catch (error) {
                        console.error('Failed to reject device:', error);
                    }
                },
                
                getStatusClass(status) {
                    const classes = {
                        'approved': 'bg-green-100 text-green-800',
                        'pending': 'bg-yellow-100 text-yellow-800',
                        'rejected': 'bg-red-100 text-red-800',
                        'suspended': 'bg-gray-100 text-gray-800',
                        'revoked': 'bg-red-100 text-red-800'
                    };
                    return classes[status] || 'bg-gray-100 text-gray-800';
                },
                
                formatDate(dateString) {
                    return new Date(dateString).toLocaleDateString();
                },
                
                initCharts() {
                    // Status chart
                    const statusCtx = document.getElementById('statusChart').getContext('2d');
                    const statusData = this.systemStats.devices_by_status || {};
                    
                    new Chart(statusCtx, {
                        type: 'doughnut',
                        data: {
                            labels: Object.keys(statusData),
                            datasets: [{
                                data: Object.values(statusData),
                                backgroundColor: ['#10B981', '#F59E0B', '#EF4444', '#6B7280', '#8B5CF6']
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false
                        }
                    });
                    
                    // Type chart
                    const typeCtx = document.getElementById('typeChart').getContext('2d');
                    const typeData = this.systemStats.devices_by_type || {};
                    
                    new Chart(typeCtx, {
                        type: 'bar',
                        data: {
                            labels: Object.keys(typeData),
                            datasets: [{
                                label: 'Devices',
                                data: Object.values(typeData),
                                backgroundColor: '#3B82F6'
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            }
                        }
                    });
                }
            }
        }
    </script>
</body>
</html>"""
    
    with open(templates_dir / "dashboard.html", "w", encoding="utf-8") as f:
        f.write(dashboard_html)
    
    # Create additional templates (devices.html, admin.html, enrollment.html)
    # ... (additional templates would go here)
    
    logger.info("Modern templates created successfully")

def main():
    """Run the modern QFLARE application"""
    
    # Create templates if they don't exist
    create_modern_templates()
    
    # Initialize the application
    qflare_app = ModernQFLAREApp()
    
    # Run the server
    uvicorn.run(
        qflare_app.app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )

if __name__ == "__main__":
    main()