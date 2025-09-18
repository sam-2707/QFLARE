"""
User authentication models and utilities for QFLARE server.
Provides role-based access control with admin and user roles.
"""

from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import hashlib
import secrets
from pydantic import BaseModel, Field


class UserRole(str, Enum):
    """User role enumeration for role-based access control."""
    ADMIN = "admin"
    USER = "user"


class User(BaseModel):
    """User model for authentication and authorization."""
    username: str = Field(..., min_length=3, max_length=50)
    role: UserRole
    password_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    is_active: bool = True


class UserCredentials(BaseModel):
    """User credentials for login requests."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    role: Optional[UserRole] = None


class TokenData(BaseModel):
    """JWT token payload data."""
    username: str
    role: UserRole
    exp: datetime
    iat: datetime


class LoginResponse(BaseModel):
    """Response model for successful login."""
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]
    expires_in: int


class UserResponse(BaseModel):
    """Response model for user information."""
    username: str
    role: UserRole
    last_login: Optional[datetime]
    is_active: bool


def hash_password(password: str) -> str:
    """
    Hash password using SHA-256 with salt.
    In production, use bcrypt or similar.
    """
    salt = "qflare_salt_2024"  # In production, use random salt per user
    return hashlib.sha256((password + salt).encode()).hexdigest()


# In-memory user database (for demo purposes)
# In production, use a proper database with encrypted passwords
USER_DATABASE: Dict[str, User] = {
    "admin": User(
        username="admin",
        role=UserRole.ADMIN,
        password_hash=hash_password("admin123"),
        created_at=datetime.utcnow(),
        is_active=True
    ),
    "user": User(
        username="user", 
        role=UserRole.USER,
        password_hash=hash_password("user123"),
        created_at=datetime.utcnow(),
        is_active=True
    )
}


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against stored hash."""
    return hash_password(password) == password_hash


def authenticate_user(username: str, password: str, role: Optional[UserRole] = None) -> Optional[User]:
    """
    Authenticate user credentials and return user object if valid.
    
    Args:
        username: Username to authenticate
        password: Plain text password
        role: Optional role to validate against
        
    Returns:
        User object if authentication successful, None otherwise
    """
    user = USER_DATABASE.get(username.lower())
    
    if not user:
        return None
        
    if not user.is_active:
        return None
        
    if not verify_password(password, user.password_hash):
        return None
        
    # If role is specified, validate it matches
    if role and user.role != role:
        return None
        
    return user


def get_user_by_username(username: str) -> Optional[User]:
    """Get user by username."""
    return USER_DATABASE.get(username.lower())


def update_last_login(username: str) -> None:
    """Update user's last login timestamp."""
    user = USER_DATABASE.get(username.lower())
    if user:
        user.last_login = datetime.utcnow()


def create_user_response(user: User) -> Dict[str, Any]:
    """Create user response dict without sensitive data."""
    return {
        "username": user.username,
        "role": user.role.value,
        "last_login": user.last_login.isoformat() if user.last_login else None,
        "is_active": user.is_active
    }


def has_admin_role(user: User) -> bool:
    """Check if user has admin role."""
    return user.role == UserRole.ADMIN


def has_user_role(user: User) -> bool:
    """Check if user has user role."""
    return user.role == UserRole.USER


def can_access_admin_features(user: User) -> bool:
    """Check if user can access admin-only features."""
    return has_admin_role(user)


def can_access_user_features(user: User) -> bool:
    """Check if user can access user features."""
    return user.is_active  # Both admin and user can access user features