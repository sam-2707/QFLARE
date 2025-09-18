"""
JWT token management for QFLARE authentication system.
Provides secure token generation, validation, and role-based middleware.
"""

import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

from auth.user_models import User, UserRole, TokenData, get_user_by_username

logger = logging.getLogger(__name__)

# JWT Configuration
SECRET_KEY = "qflare-jwt-secret-key-2024-change-in-production"  # Change in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Security scheme for FastAPI
security = HTTPBearer()


def create_access_token(user: User) -> str:
    """
    Create JWT access token for authenticated user.
    
    Args:
        user: Authenticated user object
        
    Returns:
        JWT token string
    """
    now = datetime.utcnow()
    expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    payload = {
        "sub": user.username,  # Subject (username)
        "role": user.role.value,
        "iat": now,  # Issued at
        "exp": expire,  # Expiration
        "type": "access_token"
    }
    
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    logger.info(f"Created access token for user: {user.username}, role: {user.role}")
    
    return token


def verify_token(token: str) -> Optional[TokenData]:
    """
    Verify and decode JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        TokenData if valid, None if invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        username = payload.get("sub")
        role = payload.get("role")
        exp = payload.get("exp")
        iat = payload.get("iat")
        
        if not username or not role:
            return None
            
        # Convert timestamps
        exp_dt = datetime.fromtimestamp(exp)
        iat_dt = datetime.fromtimestamp(iat)
        
        # Check if token is expired
        if datetime.utcnow() > exp_dt:
            logger.warning(f"Token expired for user: {username}")
            return None
            
        return TokenData(
            username=username,
            role=UserRole(role),
            exp=exp_dt,
            iat=iat_dt
        )
        
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return None
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return None


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """
    FastAPI dependency to get current authenticated user from JWT token.
    
    Args:
        credentials: HTTP Bearer token credentials
        
    Returns:
        Current authenticated user
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    token = credentials.credentials
    
    # Verify token
    token_data = verify_token(token)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    user = get_user_by_username(token_data.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def get_current_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """
    FastAPI dependency to ensure current user has admin role.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Current user if admin
        
    Raises:
        HTTPException: If user doesn't have admin role
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return current_user


async def get_optional_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[User]:
    """
    FastAPI dependency to optionally get current user (for endpoints that work with or without auth).
    
    Args:
        credentials: Optional HTTP Bearer token credentials
        
    Returns:
        Current user if authenticated, None otherwise
    """
    if not credentials:
        return None
        
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def create_login_response(user: User, token: str) -> Dict[str, Any]:
    """
    Create standardized login response.
    
    Args:
        user: Authenticated user
        token: JWT access token
        
    Returns:
        Login response dictionary
    """
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "username": user.username,
            "role": user.role.value,
            "last_login": user.last_login.isoformat() if user.last_login else None,
            "is_active": user.is_active
        },
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60  # Convert to seconds
    }


def validate_token_format(authorization: str) -> Optional[str]:
    """
    Validate authorization header format and extract token.
    
    Args:
        authorization: Authorization header value
        
    Returns:
        Token string if valid format, None otherwise
    """
    if not authorization:
        return None
        
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
        
    return parts[1]