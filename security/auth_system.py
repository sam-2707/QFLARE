"""
OAuth2/OIDC Authentication and Authorization System for QFLARE
Includes RBAC, audit logging, and compliance features
"""

from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr
import secrets
import logging
import json
from enum import Enum
import asyncio
import aiofiles
from pathlib import Path

# Configure logging for security events
security_logger = logging.getLogger("qflare.security")
security_logger.setLevel(logging.INFO)
handler = logging.FileHandler("security_audit.log")
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
security_logger.addHandler(handler)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Role-Based Access Control
class UserRole(str, Enum):
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    OPERATOR = "operator"
    DEVICE_MANAGER = "device_manager"
    VIEWER = "viewer"
    DEVICE = "device"

class Permission(str, Enum):
    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_MONITOR = "system:monitor"
    
    # Device permissions
    DEVICE_REGISTER = "device:register"
    DEVICE_MANAGE = "device:manage"
    DEVICE_VIEW = "device:view"
    DEVICE_DELETE = "device:delete"
    
    # Security permissions
    SECURITY_AUDIT = "security:audit"
    SECURITY_CONFIG = "security:config"
    SECURITY_MONITOR = "security:monitor"
    
    # Federated Learning permissions
    FL_MANAGE = "fl:manage"
    FL_VIEW = "fl:view"
    FL_TRAIN = "fl:train"

# Role-Permission mapping
ROLE_PERMISSIONS = {
    UserRole.SUPER_ADMIN: [p for p in Permission],
    UserRole.ADMIN: [
        Permission.SYSTEM_CONFIG, Permission.SYSTEM_MONITOR,
        Permission.DEVICE_REGISTER, Permission.DEVICE_MANAGE, Permission.DEVICE_VIEW,
        Permission.SECURITY_AUDIT, Permission.SECURITY_MONITOR,
        Permission.FL_MANAGE, Permission.FL_VIEW
    ],
    UserRole.OPERATOR: [
        Permission.SYSTEM_MONITOR,
        Permission.DEVICE_REGISTER, Permission.DEVICE_MANAGE, Permission.DEVICE_VIEW,
        Permission.SECURITY_MONITOR,
        Permission.FL_VIEW, Permission.FL_TRAIN
    ],
    UserRole.DEVICE_MANAGER: [
        Permission.DEVICE_REGISTER, Permission.DEVICE_MANAGE, Permission.DEVICE_VIEW,
        Permission.FL_VIEW
    ],
    UserRole.VIEWER: [
        Permission.DEVICE_VIEW, Permission.FL_VIEW, Permission.SYSTEM_MONITOR
    ],
    UserRole.DEVICE: [Permission.FL_TRAIN]
}

# Pydantic Models
class User(BaseModel):
    username: str
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: UserRole
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user_info: User

class TokenData(BaseModel):
    username: Optional[str] = None
    permissions: List[str] = []

class AuditLog(BaseModel):
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: str
    user_agent: str
    result: str  # success, failure, unauthorized

class SecurityEvent(BaseModel):
    timestamp: datetime
    event_type: str
    severity: str
    description: str
    source_ip: str
    user_id: Optional[str] = None
    details: Dict[str, Any]

# Mock user database (in production, use real database)
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "QFLARE Administrator",
        "email": "admin@qflare.company.com",
        "hashed_password": pwd_context.hash("admin123"),
        "role": UserRole.ADMIN,
        "is_active": True,
        "created_at": datetime.utcnow(),
    },
    "operator": {
        "username": "operator",
        "full_name": "QFLARE Operator",
        "email": "operator@qflare.company.com",
        "hashed_password": pwd_context.hash("operator123"),
        "role": UserRole.OPERATOR,
        "is_active": True,
        "created_at": datetime.utcnow(),
    }
}

# Security functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def check_permission(required_permission: Permission):
    def permission_checker(current_user: User = Depends(get_current_active_user)):
        user_permissions = ROLE_PERMISSIONS.get(current_user.role, [])
        if required_permission not in user_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {required_permission}"
            )
        return current_user
    return permission_checker

# Audit logging functions
async def log_audit_event(
    user_id: str,
    action: str,
    resource: str,
    details: Dict[str, Any],
    ip_address: str,
    user_agent: str,
    result: str = "success"
):
    """Log audit event for compliance"""
    audit_entry = AuditLog(
        timestamp=datetime.utcnow(),
        user_id=user_id,
        action=action,
        resource=resource,
        details=details,
        ip_address=ip_address,
        user_agent=user_agent,
        result=result
    )
    
    # Write to audit log file
    audit_file = Path("audit_logs") / f"audit_{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"
    audit_file.parent.mkdir(exist_ok=True)
    
    async with aiofiles.open(audit_file, mode='a') as f:
        await f.write(audit_entry.json() + "\n")
    
    # Log to security logger
    security_logger.info(f"AUDIT: {user_id} {action} {resource} - {result}")

async def log_security_event(
    event_type: str,
    severity: str,
    description: str,
    source_ip: str,
    user_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
):
    """Log security event"""
    security_event = SecurityEvent(
        timestamp=datetime.utcnow(),
        event_type=event_type,
        severity=severity,
        description=description,
        source_ip=source_ip,
        user_id=user_id,
        details=details or {}
    )
    
    # Write to security events file
    security_file = Path("security_logs") / f"security_{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"
    security_file.parent.mkdir(exist_ok=True)
    
    async with aiofiles.open(security_file, mode='a') as f:
        await f.write(security_event.json() + "\n")
    
    # Log to security logger
    security_logger.warning(f"SECURITY: {event_type} - {description} from {source_ip}")

# Compliance functions
class ComplianceReport(BaseModel):
    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    total_users: int
    total_devices: int
    audit_events: int
    security_events: int
    compliance_status: str
    findings: List[Dict[str, Any]]

async def generate_compliance_report(
    start_date: datetime,
    end_date: datetime,
    current_user: User = Depends(check_permission(Permission.SECURITY_AUDIT))
) -> ComplianceReport:
    """Generate compliance report for audit purposes"""
    
    # Mock compliance data (in production, query actual database)
    findings = []
    
    # Check for security violations
    if await check_failed_login_attempts(start_date, end_date) > 100:
        findings.append({
            "type": "security",
            "severity": "medium",
            "description": "High number of failed login attempts detected",
            "recommendation": "Review authentication logs and consider implementing additional security measures"
        })
    
    # Check for unauthorized access attempts
    if await check_unauthorized_access_attempts(start_date, end_date) > 10:
        findings.append({
            "type": "security",
            "severity": "high",
            "description": "Unauthorized access attempts detected",
            "recommendation": "Investigate source IPs and consider IP whitelisting"
        })
    
    compliance_status = "COMPLIANT" if len(findings) == 0 else "ATTENTION_REQUIRED"
    
    report = ComplianceReport(
        report_id=f"COMP-{datetime.utcnow().strftime('%Y%m%d')}-{secrets.token_hex(4)}",
        generated_at=datetime.utcnow(),
        period_start=start_date,
        period_end=end_date,
        total_users=len(fake_users_db),
        total_devices=50,  # Mock data
        audit_events=1000,  # Mock data
        security_events=5,  # Mock data
        compliance_status=compliance_status,
        findings=findings
    )
    
    # Log compliance report generation
    await log_audit_event(
        user_id=current_user.username,
        action="generate_compliance_report",
        resource="compliance_reports",
        details={"report_id": report.report_id, "period": f"{start_date} to {end_date}"},
        ip_address="127.0.0.1",  # Should get real IP
        user_agent="API"
    )
    
    return report

async def check_failed_login_attempts(start_date: datetime, end_date: datetime) -> int:
    """Check failed login attempts in period"""
    # Mock implementation - in production, query audit logs
    return 45

async def check_unauthorized_access_attempts(start_date: datetime, end_date: datetime) -> int:
    """Check unauthorized access attempts in period"""
    # Mock implementation - in production, query security logs
    return 3

# Security scanning functions
class SecurityScanResult(BaseModel):
    scan_id: str
    scan_type: str
    started_at: datetime
    completed_at: datetime
    status: str
    vulnerabilities: List[Dict[str, Any]]
    recommendations: List[str]

async def run_security_scan(
    scan_type: str = "full",
    current_user: User = Depends(check_permission(Permission.SECURITY_CONFIG))
) -> SecurityScanResult:
    """Run automated security scan"""
    
    scan_id = f"SCAN-{datetime.utcnow().strftime('%Y%m%d')}-{secrets.token_hex(4)}"
    started_at = datetime.utcnow()
    
    # Mock security scan (in production, integrate with actual security tools)
    await asyncio.sleep(2)  # Simulate scan time
    
    vulnerabilities = [
        {
            "id": "QFLARE-001",
            "severity": "LOW",
            "title": "Default credentials detected",
            "description": "Some devices are using default credentials",
            "affected_components": ["device_auth"],
            "remediation": "Force password change on first login"
        }
    ]
    
    recommendations = [
        "Enable two-factor authentication for all admin accounts",
        "Implement IP whitelisting for administrative access",
        "Regular security awareness training for operators",
        "Update all dependencies to latest secure versions"
    ]
    
    result = SecurityScanResult(
        scan_id=scan_id,
        scan_type=scan_type,
        started_at=started_at,
        completed_at=datetime.utcnow(),
        status="COMPLETED",
        vulnerabilities=vulnerabilities,
        recommendations=recommendations
    )
    
    # Log security scan
    await log_audit_event(
        user_id=current_user.username,
        action="security_scan",
        resource="security_scans",
        details={"scan_id": scan_id, "scan_type": scan_type},
        ip_address="127.0.0.1",
        user_agent="API"
    )
    
    await log_security_event(
        event_type="security_scan_completed",
        severity="info",
        description=f"Security scan {scan_id} completed with {len(vulnerabilities)} vulnerabilities",
        source_ip="127.0.0.1",
        user_id=current_user.username,
        details={"scan_id": scan_id, "vulnerabilities_count": len(vulnerabilities)}
    )
    
    return result

# Security monitoring functions
class ThreatIntelligence(BaseModel):
    threat_id: str
    threat_type: str
    severity: str
    description: str
    indicators: List[str]
    first_seen: datetime
    last_seen: datetime
    active: bool

async def get_threat_intelligence() -> List[ThreatIntelligence]:
    """Get current threat intelligence"""
    # Mock threat intelligence (in production, integrate with threat feeds)
    threats = [
        ThreatIntelligence(
            threat_id="TI-001",
            threat_type="malware",
            severity="HIGH",
            description="New quantum-resistant crypto attack detected",
            indicators=["192.168.1.100", "malicious-domain.com"],
            first_seen=datetime.utcnow() - timedelta(hours=2),
            last_seen=datetime.utcnow() - timedelta(minutes=30),
            active=True
        )
    ]
    return threats

# Export functions for use in main API
__all__ = [
    'User', 'UserRole', 'Permission', 'Token', 'TokenData',
    'oauth2_scheme', 'get_current_user', 'get_current_active_user',
    'check_permission', 'authenticate_user', 'create_access_token',
    'log_audit_event', 'log_security_event', 'generate_compliance_report',
    'run_security_scan', 'get_threat_intelligence', 'ComplianceReport',
    'SecurityScanResult', 'ThreatIntelligence', 'ROLE_PERMISSIONS'
]