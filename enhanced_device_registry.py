#!/usr/bin/env python3
"""
Enhanced Device Registration System for QFLARE
Production-grade device authentication and enrollment
"""

import os
import sys
import json
import secrets
import hashlib
import time
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import sqlite3
from dataclasses import dataclass
from enum import Enum
import uuid
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeviceStatus(Enum):
    """Device registration status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUSPENDED = "suspended"
    REVOKED = "revoked"

class DeviceType(Enum):
    """Supported device types"""
    EDGE = "edge"
    EDGE_NODE = "edge-node"  # Add explicit edge-node support
    MOBILE = "mobile"
    IOT = "iot"
    SERVER = "server"
    GATEWAY = "gateway"

@dataclass
class DeviceCapabilities:
    """Device hardware and software capabilities"""
    cpu_cores: int
    memory_gb: float
    storage_gb: float
    network_bandwidth_mbps: float
    supported_algorithms: List[str]
    data_samples: int
    privacy_level: str = "standard"
    security_features: List[str] = None
    
    def __post_init__(self):
        if self.security_features is None:
            self.security_features = []

@dataclass
class DeviceCredentials:
    """Device authentication credentials"""
    device_id: str
    public_key: str
    certificate: Optional[str] = None
    certificate_chain: Optional[List[str]] = None
    key_algorithm: str = "Dilithium2"
    created_at: datetime = None
    expires_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.expires_at is None:
            self.expires_at = self.created_at + timedelta(days=365)

@dataclass
class EnrollmentToken:
    """Device enrollment token for secure registration"""
    token: str
    device_type: DeviceType
    organization: str
    valid_until: datetime
    max_uses: int = 1
    used_count: int = 0
    created_by: str = "admin"
    restrictions: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.restrictions is None:
            self.restrictions = {}

class EnhancedDeviceRegistry:
    """Production-grade device registration and management system"""
    
    def __init__(self, db_path: str = "data/device_registry.db"):
        self.db_path = db_path
        self.init_database()
        
        # Security configuration
        self.min_key_size = 2048
        self.max_devices_per_org = 10000
        self.enrollment_token_validity_hours = 24
        self.device_certificate_validity_days = 365
        
        # Rate limiting
        self.registration_rate_limit = 100  # per hour per IP
        self.registration_attempts = {}
        
    def init_database(self):
        """Initialize the device registry database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS devices (
                    device_id TEXT PRIMARY KEY,
                    device_type TEXT NOT NULL,
                    organization TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    public_key TEXT NOT NULL,
                    certificate TEXT,
                    certificate_chain TEXT,
                    capabilities TEXT NOT NULL,
                    location TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_seen DATETIME,
                    approved_at DATETIME,
                    approved_by TEXT,
                    registration_ip TEXT,
                    user_agent TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS enrollment_tokens (
                    token TEXT PRIMARY KEY,
                    device_type TEXT NOT NULL,
                    organization TEXT NOT NULL,
                    valid_until DATETIME NOT NULL,
                    max_uses INTEGER DEFAULT 1,
                    used_count INTEGER DEFAULT 0,
                    created_by TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    restrictions TEXT,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS registration_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    attempt_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN,
                    error_reason TEXT,
                    enrollment_token TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS device_sessions (
                    session_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    access_token TEXT NOT NULL,
                    refresh_token TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME NOT NULL,
                    last_used DATETIME,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (device_id) REFERENCES devices (device_id)
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_devices_status ON devices(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_devices_org ON devices(organization)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_devices_type ON devices(device_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tokens_valid ON enrollment_tokens(valid_until, is_active)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_ip_time ON registration_attempts(ip_address, attempt_time)")
            
    def generate_enrollment_token(self, device_type: DeviceType, organization: str, 
                                 created_by: str = "admin", max_uses: int = 1,
                                 validity_hours: int = None) -> EnrollmentToken:
        """Generate a secure enrollment token for device registration"""
        
        if validity_hours is None:
            validity_hours = self.enrollment_token_validity_hours
            
        token = secrets.token_urlsafe(32)
        valid_until = datetime.now() + timedelta(hours=validity_hours)
        
        enrollment_token = EnrollmentToken(
            token=token,
            device_type=device_type,
            organization=organization,
            valid_until=valid_until,
            max_uses=max_uses,
            created_by=created_by
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO enrollment_tokens 
                (token, device_type, organization, valid_until, max_uses, created_by, restrictions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                enrollment_token.token,
                enrollment_token.device_type.value,
                enrollment_token.organization,
                enrollment_token.valid_until,
                enrollment_token.max_uses,
                enrollment_token.created_by,
                json.dumps(enrollment_token.restrictions)
            ))
            
        logger.info(f"Generated enrollment token for {organization} ({device_type.value})")
        return enrollment_token
    
    def validate_enrollment_token(self, token: str) -> Optional[EnrollmentToken]:
        """Validate and consume an enrollment token"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT token, device_type, organization, valid_until, max_uses, 
                       used_count, created_by, restrictions, is_active
                FROM enrollment_tokens 
                WHERE token = ? AND is_active = 1
            """, (token,))
            
            row = cursor.fetchone()
            if not row:
                return None
                
            enrollment_token = EnrollmentToken(
                token=row[0],
                device_type=DeviceType(row[1]),
                organization=row[2],
                valid_until=datetime.fromisoformat(row[3]),
                max_uses=row[4],
                used_count=row[5],
                created_by=row[6],
                restrictions=json.loads(row[7]) if row[7] else {}
            )
            
            # Check if token is still valid
            if datetime.now() > enrollment_token.valid_until:
                logger.warning(f"Enrollment token {token[:8]}... has expired")
                return None
                
            # Check if token has remaining uses
            if enrollment_token.used_count >= enrollment_token.max_uses:
                logger.warning(f"Enrollment token {token[:8]}... has been exhausted")
                return None
                
            return enrollment_token
    
    def consume_enrollment_token(self, token: str) -> bool:
        """Mark an enrollment token as used"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE enrollment_tokens 
                SET used_count = used_count + 1,
                    is_active = CASE 
                        WHEN used_count + 1 >= max_uses THEN 0 
                        ELSE 1 
                    END
                WHERE token = ? AND is_active = 1
            """, (token,))
            
            return cursor.rowcount > 0
    
    def validate_device_id(self, device_id: str) -> bool:
        """Validate device ID format and uniqueness"""
        
        # Check format (alphanumeric, hyphens, underscores, 3-64 chars)
        if not re.match(r'^[a-zA-Z0-9_-]{3,64}$', device_id):
            return False
            
        # Check uniqueness
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT 1 FROM devices WHERE device_id = ?", (device_id,))
            return cursor.fetchone() is None
    
    def validate_capabilities(self, capabilities: Dict[str, Any]) -> DeviceCapabilities:
        """Validate and normalize device capabilities"""
        
        try:
            return DeviceCapabilities(
                cpu_cores=max(1, int(capabilities.get('cpu_cores', 1))),
                memory_gb=max(0.5, float(capabilities.get('memory_gb', 1.0))),
                storage_gb=max(1.0, float(capabilities.get('storage_gb', 10.0))),
                network_bandwidth_mbps=max(1.0, float(capabilities.get('network_bandwidth_mbps', 10.0))),
                supported_algorithms=capabilities.get('supported_algorithms', ['fedavg']),
                data_samples=max(0, int(capabilities.get('data_samples', 0))),
                privacy_level=capabilities.get('privacy_level', 'standard'),
                security_features=capabilities.get('security_features', [])
            )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid capabilities format: {e}")
    
    def validate_public_key(self, public_key: str, algorithm: str = "Dilithium2") -> bool:
        """Validate public key format and strength"""
        
        try:
            # Basic format validation
            if not public_key or len(public_key) < 32:
                return False
                
            # Algorithm-specific validation
            if algorithm == "Dilithium2":
                # Dilithium2 public keys should be base64 encoded and ~1312 bytes
                decoded = base64.b64decode(public_key)
                return 1000 <= len(decoded) <= 2000
            elif algorithm == "RSA":
                # RSA key size check
                decoded = base64.b64decode(public_key)
                return len(decoded) >= self.min_key_size // 8
            else:
                # Generic validation
                return len(public_key) >= 64
                
        except Exception:
            return False
    
    def check_rate_limit(self, ip_address: str) -> bool:
        """Check if IP address is within rate limits"""
        
        current_time = datetime.now()
        hour_ago = current_time - timedelta(hours=1)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM registration_attempts 
                WHERE ip_address = ? AND attempt_time > ?
            """, (ip_address, hour_ago))
            
            attempt_count = cursor.fetchone()[0]
            return attempt_count < self.registration_rate_limit
    
    def register_device(self, device_id: str, public_key: str, device_type: str,
                       capabilities: Dict[str, Any], enrollment_token: str,
                       location: str = None, metadata: Dict[str, Any] = None,
                       ip_address: str = None, user_agent: str = None) -> Dict[str, Any]:
        """Register a new device with enhanced security validation"""
        
        registration_time = datetime.now()
        
        try:
            # Rate limiting check
            if ip_address and not self.check_rate_limit(ip_address):
                raise ValueError("Rate limit exceeded for IP address")
            
            # Validate enrollment token
            token_obj = self.validate_enrollment_token(enrollment_token)
            if not token_obj:
                raise ValueError("Invalid or expired enrollment token")
            
            # Validate device ID
            if not self.validate_device_id(device_id):
                raise ValueError("Invalid device ID format or device already exists")
            
            # Validate device type
            try:
                device_type_enum = DeviceType(device_type)
            except ValueError:
                raise ValueError(f"Invalid device type: {device_type}")
            
            # Ensure device type matches enrollment token
            if device_type_enum != token_obj.device_type:
                raise ValueError("Device type doesn't match enrollment token")
            
            # Validate public key
            if not self.validate_public_key(public_key):
                raise ValueError("Invalid public key format or insufficient strength")
            
            # Validate capabilities
            device_capabilities = self.validate_capabilities(capabilities)
            
            # Generate device certificate (mock for now)
            device_cert = self.generate_device_certificate(device_id, public_key, token_obj.organization)
            
            # Create device credentials
            credentials = DeviceCredentials(
                device_id=device_id,
                public_key=public_key,
                certificate=device_cert,
                created_at=registration_time
            )
            
            # Store device in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO devices 
                    (device_id, device_type, organization, status, public_key, certificate,
                     capabilities, location, metadata, registration_ip, user_agent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    device_id,
                    device_type_enum.value,
                    token_obj.organization,
                    DeviceStatus.PENDING.value,
                    public_key,
                    device_cert,
                    json.dumps(device_capabilities.__dict__),
                    location,
                    json.dumps(metadata or {}),
                    ip_address,
                    user_agent
                ))
            
            # Consume enrollment token
            self.consume_enrollment_token(enrollment_token)
            
            # Log successful registration
            self.log_registration_attempt(
                device_id=device_id,
                ip_address=ip_address,
                user_agent=user_agent,
                success=True,
                enrollment_token=enrollment_token
            )
            
            # Generate access token
            access_token = self.generate_access_token(device_id)
            
            logger.info(f"Device {device_id} registered successfully for {token_obj.organization}")
            
            return {
                "status": "registered",
                "device_id": device_id,
                "access_token": access_token,
                "certificate": device_cert,
                "organization": token_obj.organization,
                "registration_status": DeviceStatus.PENDING.value,
                "message": "Device registered successfully. Pending admin approval."
            }
            
        except Exception as e:
            # Log failed registration
            self.log_registration_attempt(
                device_id=device_id,
                ip_address=ip_address,
                user_agent=user_agent,
                success=False,
                error_reason=str(e),
                enrollment_token=enrollment_token
            )
            
            logger.error(f"Device registration failed for {device_id}: {e}")
            raise
    
    def generate_device_certificate(self, device_id: str, public_key: str, organization: str) -> str:
        """Generate a mock device certificate (replace with real PKI)"""
        
        cert_data = {
            "device_id": device_id,
            "public_key": public_key,
            "organization": organization,
            "issued_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=self.device_certificate_validity_days)).isoformat(),
            "issuer": "QFLARE-CA",
            "serial_number": secrets.token_hex(16)
        }
        
        # In production, this would use proper X.509 certificate generation
        cert_string = base64.b64encode(json.dumps(cert_data).encode()).decode()
        return f"-----BEGIN CERTIFICATE-----\n{cert_string}\n-----END CERTIFICATE-----"
    
    def generate_access_token(self, device_id: str) -> str:
        """Generate a secure access token for the device"""
        
        session_id = str(uuid.uuid4())
        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=24)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO device_sessions 
                (session_id, device_id, access_token, refresh_token, expires_at)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, device_id, access_token, refresh_token, expires_at))
        
        return access_token
    
    def log_registration_attempt(self, device_id: str = None, ip_address: str = None,
                                user_agent: str = None, success: bool = False,
                                error_reason: str = None, enrollment_token: str = None):
        """Log device registration attempt for audit purposes"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO registration_attempts 
                (device_id, ip_address, user_agent, success, error_reason, enrollment_token)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (device_id, ip_address, user_agent, success, error_reason, enrollment_token))
    
    def approve_device(self, device_id: str, approved_by: str = "admin") -> bool:
        """Approve a pending device registration"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE devices 
                SET status = ?, approved_at = ?, approved_by = ?, updated_at = ?
                WHERE device_id = ? AND status = ?
            """, (
                DeviceStatus.APPROVED.value,
                datetime.now(),
                approved_by,
                datetime.now(),
                device_id,
                DeviceStatus.PENDING.value
            ))
            
            success = cursor.rowcount > 0
            if success:
                logger.info(f"Device {device_id} approved by {approved_by}")
            
            return success
    
    def reject_device(self, device_id: str, reason: str = "Admin decision") -> bool:
        """Reject a pending device registration"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE devices 
                SET status = ?, updated_at = ?, metadata = ?
                WHERE device_id = ? AND status = ?
            """, (
                DeviceStatus.REJECTED.value,
                datetime.now(),
                json.dumps({"rejection_reason": reason}),
                device_id,
                DeviceStatus.PENDING.value
            ))
            
            success = cursor.rowcount > 0
            if success:
                logger.info(f"Device {device_id} rejected: {reason}")
            
            return success
    
    def get_device_info(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a registered device"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM devices WHERE device_id = ?
            """, (device_id,))
            
            row = cursor.fetchone()
            if row:
                device_info = dict(row)
                device_info['capabilities'] = json.loads(device_info['capabilities'])
                device_info['metadata'] = json.loads(device_info['metadata'] or '{}')
                return device_info
            
            return None
    
    def list_devices(self, status: DeviceStatus = None, organization: str = None,
                    device_type: DeviceType = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List registered devices with optional filtering"""
        
        query = "SELECT * FROM devices WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
            
        if organization:
            query += " AND organization = ?"
            params.append(organization)
            
        if device_type:
            query += " AND device_type = ?"
            params.append(device_type.value)
            
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            devices = []
            for row in cursor.fetchall():
                device_info = dict(row)
                device_info['capabilities'] = json.loads(device_info['capabilities'])
                device_info['metadata'] = json.loads(device_info['metadata'] or '{}')
                devices.append(device_info)
                
            return devices
    
    def list_enrollment_tokens(self, organization: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List enrollment tokens with optional filtering"""
        
        query = """
            SELECT token, device_type, organization, valid_until, max_uses, 
                   used_count, created_by, restrictions, is_active, created_at
            FROM enrollment_tokens 
            WHERE 1=1
        """
        params = []
        
        if organization:
            query += " AND organization = ?"
            params.append(organization)
            
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            tokens = []
            for row in cursor.fetchall():
                token_info = dict(row)
                token_info['restrictions'] = json.loads(token_info['restrictions'] or '{}')
                
                # Determine status
                now = datetime.now()
                valid_until = datetime.fromisoformat(token_info['valid_until'])
                
                if not token_info['is_active']:
                    token_info['status'] = 'exhausted'
                elif now > valid_until:
                    token_info['status'] = 'expired'
                else:
                    token_info['status'] = 'active'
                
                tokens.append(token_info)
                
            return tokens
    
    def revoke_device(self, device_id: str, reason: str = "Security concern") -> bool:
        """Revoke device access and invalidate all sessions"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Update device status
            cursor = conn.execute("""
                UPDATE devices 
                SET status = ?, updated_at = ?, metadata = ?
                WHERE device_id = ?
            """, (
                DeviceStatus.REVOKED.value,
                datetime.now(),
                json.dumps({"revocation_reason": reason}),
                device_id
            ))
            
            # Invalidate all sessions
            conn.execute("""
                UPDATE device_sessions 
                SET is_active = 0 
                WHERE device_id = ?
            """, (device_id,))
            
            success = cursor.rowcount > 0
            if success:
                logger.info(f"Device {device_id} revoked: {reason}")
            
            return success

# Example usage and testing
if __name__ == "__main__":
    registry = EnhancedDeviceRegistry()
    
    # Generate enrollment token
    token = registry.generate_enrollment_token(
        device_type=DeviceType.EDGE,
        organization="AcmeCorp",
        created_by="admin",
        max_uses=5
    )
    
    print(f"Generated enrollment token: {token.token}")
    
    # Test device registration
    try:
        result = registry.register_device(
            device_id="edge-node-production-001",
            public_key=base64.b64encode(b"mock_dilithium2_public_key_data" * 50).decode(),
            device_type="edge",
            capabilities={
                "cpu_cores": 8,
                "memory_gb": 16.0,
                "storage_gb": 512.0,
                "network_bandwidth_mbps": 1000.0,
                "supported_algorithms": ["fedavg", "fedprox"],
                "data_samples": 10000,
                "security_features": ["tee", "secure_boot"]
            },
            enrollment_token=token.token,
            location="Production Data Center",
            ip_address="192.168.1.100"
        )
        
        print(f"Registration result: {result}")
        
        # Approve the device
        registry.approve_device("edge-node-production-001", "admin")
        
        # List devices
        devices = registry.list_devices()
        print(f"Registered devices: {len(devices)}")
        
    except Exception as e:
        print(f"Registration failed: {e}")