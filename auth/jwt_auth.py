"""
Enhanced JWT authentication and session management for QFLARE.

This module provides secure JWT token handling with refresh tokens,
multi-factor authentication, and comprehensive session management.
"""

import jwt
import secrets
import hashlib
import pyotp
import qrcode
import io
import base64
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import redis
import json
import uuid

logger = logging.getLogger(__name__)


class TokenType(Enum):
    """JWT token types."""
    ACCESS = "access"
    REFRESH = "refresh"
    RESET = "reset"
    VERIFICATION = "verification"


@dataclass
class TokenPair:
    """Access and refresh token pair."""
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "Bearer"


@dataclass
class TokenClaims:
    """JWT token claims."""
    user_id: str
    username: str
    email: str
    role: str
    token_type: TokenType
    issued_at: datetime
    expires_at: datetime
    session_id: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    mfa_verified: bool = False
    permissions: List[str] = None


class JWTManager:
    """Enhanced JWT token manager with security features."""
    
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 15,
        refresh_token_expire_days: int = 7,
        redis_client: Optional[redis.Redis] = None
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire = timedelta(minutes=access_token_expire_minutes)
        self.refresh_token_expire = timedelta(days=refresh_token_expire_days)
        self.redis_client = redis_client
        
        # Token blacklist for logout/revocation
        self._blacklisted_tokens = set()
    
    def generate_token_pair(
        self,
        user_id: str,
        username: str,
        email: str,
        role: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        mfa_verified: bool = False,
        permissions: Optional[List[str]] = None
    ) -> TokenPair:
        """Generate access and refresh token pair."""
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        # Access token claims
        access_claims = {
            "user_id": user_id,
            "username": username,
            "email": email,
            "role": role,
            "token_type": TokenType.ACCESS.value,
            "session_id": session_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "mfa_verified": mfa_verified,
            "permissions": permissions or [],
            "iat": now,
            "exp": now + self.access_token_expire,
            "jti": str(uuid.uuid4())  # JWT ID for revocation
        }
        
        # Refresh token claims
        refresh_claims = {
            "user_id": user_id,
            "username": username,
            "email": email,
            "role": role,
            "token_type": TokenType.REFRESH.value,
            "session_id": session_id,
            "iat": now,
            "exp": now + self.refresh_token_expire,
            "jti": str(uuid.uuid4())
        }
        
        access_token = jwt.encode(access_claims, self.secret_key, algorithm=self.algorithm)
        refresh_token = jwt.encode(refresh_claims, self.secret_key, algorithm=self.algorithm)
        
        # Store session in Redis if available
        if self.redis_client:
            session_data = {
                "user_id": user_id,
                "username": username,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "created_at": now.isoformat(),
                "last_activity": now.isoformat(),
                "is_active": True
            }
            
            session_key = f"session:{session_id}"
            self.redis_client.setex(
                session_key,
                int(self.refresh_token_expire.total_seconds()),
                json.dumps(session_data, default=str)
            )
        
        logger.info(f"Generated token pair for user {username} (session: {session_id})")
        
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=int(self.access_token_expire.total_seconds())
        )
    
    def verify_token(self, token: str, expected_type: TokenType = TokenType.ACCESS) -> Optional[TokenClaims]:
        """Verify and decode JWT token."""
        try:
            # Check if token is blacklisted
            if self._is_token_blacklisted(token):
                logger.warning("Attempt to use blacklisted token")
                return None
            
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Verify token type
            if payload.get("token_type") != expected_type.value:
                logger.warning(f"Token type mismatch. Expected: {expected_type.value}, Got: {payload.get('token_type')}")
                return None
            
            # Check session validity if Redis is available
            session_id = payload.get("session_id")
            if self.redis_client and session_id:
                session_key = f"session:{session_id}"
                session_data = self.redis_client.get(session_key)
                
                if not session_data:
                    logger.warning(f"Session {session_id} not found or expired")
                    return None
                
                session_info = json.loads(session_data)
                if not session_info.get("is_active", False):
                    logger.warning(f"Session {session_id} is inactive")
                    return None
                
                # Update last activity
                session_info["last_activity"] = datetime.now(timezone.utc).isoformat()
                self.redis_client.setex(
                    session_key,
                    int(self.refresh_token_expire.total_seconds()),
                    json.dumps(session_info, default=str)
                )
            
            return TokenClaims(
                user_id=payload["user_id"],
                username=payload["username"],
                email=payload["email"],
                role=payload["role"],
                token_type=TokenType(payload["token_type"]),
                issued_at=datetime.fromtimestamp(payload["iat"], timezone.utc),
                expires_at=datetime.fromtimestamp(payload["exp"], timezone.utc),
                session_id=session_id,
                ip_address=payload.get("ip_address"),
                user_agent=payload.get("user_agent"),
                mfa_verified=payload.get("mfa_verified", False),
                permissions=payload.get("permissions", [])
            )
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[TokenPair]:
        """Refresh access token using refresh token."""
        try:
            # Verify refresh token
            claims = self.verify_token(refresh_token, TokenType.REFRESH)
            if not claims:
                return None
            
            # Generate new token pair
            return self.generate_token_pair(
                user_id=claims.user_id,
                username=claims.username,
                email=claims.email,
                role=claims.role,
                ip_address=claims.ip_address,
                user_agent=claims.user_agent,
                mfa_verified=claims.mfa_verified,
                permissions=claims.permissions
            )
            
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a JWT token (add to blacklist)."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            jti = payload.get("jti")
            
            if jti:
                self._blacklisted_tokens.add(jti)
                
                # Store in Redis if available
                if self.redis_client:
                    exp_time = payload.get("exp")
                    if exp_time:
                        ttl = exp_time - datetime.now(timezone.utc).timestamp()
                        if ttl > 0:
                            self.redis_client.setex(f"blacklist:{jti}", int(ttl), "1")
                
                logger.info(f"Token revoked: {jti}")
                return True
            
        except Exception as e:
            logger.error(f"Token revocation error: {e}")
        
        return False
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke entire session."""
        try:
            if self.redis_client:
                session_key = f"session:{session_id}"
                session_data = self.redis_client.get(session_key)
                
                if session_data:
                    session_info = json.loads(session_data)
                    session_info["is_active"] = False
                    
                    self.redis_client.setex(
                        session_key,
                        int(self.refresh_token_expire.total_seconds()),
                        json.dumps(session_info, default=str)
                    )
                    
                    logger.info(f"Session revoked: {session_id}")
                    return True
            
        except Exception as e:
            logger.error(f"Session revocation error: {e}")
        
        return False
    
    def _is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm], options={"verify_exp": False})
            jti = payload.get("jti")
            
            if jti:
                # Check local blacklist
                if jti in self._blacklisted_tokens:
                    return True
                
                # Check Redis blacklist
                if self.redis_client and self.redis_client.exists(f"blacklist:{jti}"):
                    return True
            
        except Exception:
            pass
        
        return False
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all active sessions for a user."""
        sessions = []
        
        if self.redis_client:
            try:
                # Scan for user sessions
                pattern = "session:*"
                for key in self.redis_client.scan_iter(match=pattern):
                    session_data = self.redis_client.get(key)
                    if session_data:
                        session_info = json.loads(session_data)
                        if session_info.get("user_id") == user_id and session_info.get("is_active"):
                            session_info["session_id"] = key.decode().split(":")[1]
                            sessions.append(session_info)
                            
            except Exception as e:
                logger.error(f"Error retrieving user sessions: {e}")
        
        return sessions


class MFAManager:
    """Multi-Factor Authentication manager."""
    
    @staticmethod
    def generate_secret(length: int = 32) -> str:
        """Generate TOTP secret."""
        return pyotp.random_base32(length)
    
    @staticmethod
    def generate_qr_code(secret: str, user_email: str, issuer_name: str = "QFLARE") -> str:
        """Generate QR code for TOTP setup."""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email,
            issuer_name=issuer_name
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode()
    
    @staticmethod
    def verify_totp(secret: str, token: str, window: int = 1) -> bool:
        """Verify TOTP token."""
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=window)
        except Exception as e:
            logger.error(f"TOTP verification error: {e}")
            return False
    
    @staticmethod
    def generate_backup_codes(count: int = 10) -> List[str]:
        """Generate backup codes for MFA."""
        codes = []
        for _ in range(count):
            code = secrets.token_hex(4).upper()
            codes.append(f"{code[:4]}-{code[4:]}")
        return codes
    
    @staticmethod
    def hash_backup_code(code: str) -> str:
        """Hash backup code for storage."""
        return hashlib.sha256(code.encode()).hexdigest()
    
    @staticmethod
    def verify_backup_code(code: str, hashed_code: str) -> bool:
        """Verify backup code."""
        return hashlib.sha256(code.encode()).hexdigest() == hashed_code


class PasswordManager:
    """Advanced password hashing and validation."""
    
    @staticmethod
    def generate_salt() -> str:
        """Generate cryptographic salt."""
        return secrets.token_hex(16)
    
    @staticmethod
    def hash_password(password: str, salt: str) -> str:
        """Hash password with PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode(),
            iterations=100000,  # OWASP recommended minimum
        )
        return base64.b64encode(kdf.derive(password.encode())).decode()
    
    @staticmethod
    def verify_password(password: str, salt: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        try:
            return PasswordManager.hash_password(password, salt) == hashed_password
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    @staticmethod
    def check_password_strength(password: str) -> Dict[str, Any]:
        """Check password strength and provide feedback."""
        checks = {
            "length": len(password) >= 12,
            "uppercase": any(c.isupper() for c in password),
            "lowercase": any(c.islower() for c in password),
            "digit": any(c.isdigit() for c in password),
            "special": any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password),
            "no_common": password.lower() not in [
                "password", "123456", "qwerty", "admin", "letmein",
                "welcome", "monkey", "dragon", "password123"
            ]
        }
        
        strength_score = sum(checks.values())
        
        if strength_score >= 6:
            strength = "strong"
        elif strength_score >= 4:
            strength = "medium"
        else:
            strength = "weak"
        
        return {
            "strength": strength,
            "score": strength_score,
            "max_score": len(checks),
            "checks": checks,
            "suggestions": PasswordManager._get_password_suggestions(checks)
        }
    
    @staticmethod
    def _get_password_suggestions(checks: Dict[str, bool]) -> List[str]:
        """Get password improvement suggestions."""
        suggestions = []
        
        if not checks["length"]:
            suggestions.append("Use at least 12 characters")
        if not checks["uppercase"]:
            suggestions.append("Add uppercase letters")
        if not checks["lowercase"]:
            suggestions.append("Add lowercase letters")
        if not checks["digit"]:
            suggestions.append("Add numbers")
        if not checks["special"]:
            suggestions.append("Add special characters")
        if not checks["no_common"]:
            suggestions.append("Avoid common passwords")
        
        return suggestions


class SessionManager:
    """Advanced session management with Redis."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
    
    def create_session(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str,
        expire_seconds: int = 86400  # 24 hours
    ) -> str:
        """Create new user session."""
        session_id = str(uuid.uuid4())
        
        session_data = {
            "user_id": user_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_activity": datetime.now(timezone.utc).isoformat(),
            "is_active": True,
            "login_count": 1
        }
        
        if self.redis_client:
            session_key = f"session:{session_id}"
            self.redis_client.setex(
                session_key,
                expire_seconds,
                json.dumps(session_data, default=str)
            )
        
        logger.info(f"Session created: {session_id} for user {user_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        if not self.redis_client:
            return None
        
        try:
            session_key = f"session:{session_id}"
            session_data = self.redis_client.get(session_key)
            
            if session_data:
                return json.loads(session_data)
                
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
        
        return None
    
    def update_session_activity(self, session_id: str) -> bool:
        """Update session last activity."""
        if not self.redis_client:
            return False
        
        try:
            session_data = self.get_session(session_id)
            if session_data and session_data.get("is_active"):
                session_data["last_activity"] = datetime.now(timezone.utc).isoformat()
                
                session_key = f"session:{session_id}"
                ttl = self.redis_client.ttl(session_key)
                
                if ttl > 0:
                    self.redis_client.setex(
                        session_key,
                        ttl,
                        json.dumps(session_data, default=str)
                    )
                    return True
                    
        except Exception as e:
            logger.error(f"Error updating session activity: {e}")
        
        return False
    
    def terminate_session(self, session_id: str) -> bool:
        """Terminate a session."""
        if not self.redis_client:
            return False
        
        try:
            session_key = f"session:{session_id}"
            result = self.redis_client.delete(session_key)
            
            if result:
                logger.info(f"Session terminated: {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error terminating session {session_id}: {e}")
        
        return False
    
    def terminate_user_sessions(self, user_id: str, except_session: Optional[str] = None) -> int:
        """Terminate all sessions for a user."""
        if not self.redis_client:
            return 0
        
        terminated = 0
        
        try:
            pattern = "session:*"
            for key in self.redis_client.scan_iter(match=pattern):
                session_data = self.redis_client.get(key)
                if session_data:
                    session_info = json.loads(session_data)
                    session_id = key.decode().split(":")[1]
                    
                    if (session_info.get("user_id") == user_id and 
                        session_id != except_session):
                        
                        if self.redis_client.delete(key):
                            terminated += 1
                            
            logger.info(f"Terminated {terminated} sessions for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error terminating user sessions: {e}")
        
        return terminated