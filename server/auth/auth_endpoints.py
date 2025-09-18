"""
Authentication API endpoints for QFLARE server.
Provides login, logout, user info, and role validation endpoints.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any
import logging
from datetime import datetime

from auth.user_models import (
    UserCredentials, LoginResponse, UserResponse, User, UserRole,
    authenticate_user, update_last_login, create_user_response,
    has_admin_role, can_access_admin_features
)
from auth.jwt_utils import (
    create_access_token, get_current_user, get_current_admin_user,
    create_login_response
)

logger = logging.getLogger(__name__)
auth_router = APIRouter()


@auth_router.post("/login", response_model=Dict[str, Any])
async def login(credentials: UserCredentials):
    """
    Authenticate user and return JWT token.
    
    Supports both admin and user roles with role-specific validation.
    """
    try:
        # Authenticate user credentials
        user = authenticate_user(
            username=credentials.username,
            password=credentials.password,
            role=credentials.role  # Optional role validation
        )
        
        if not user:
            # Log failed login attempt
            logger.warning(f"Failed login attempt for username: {credentials.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials. Please check your username and password."
            )
        
        # Update last login timestamp
        update_last_login(user.username)
        
        # Generate JWT token
        access_token = create_access_token(user)
        
        # Create response
        response = create_login_response(user, access_token)
        
        logger.info(f"Successful login for user: {user.username}, role: {user.role}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during login"
        )


@auth_router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """
    Logout current user.
    
    Note: JWT tokens are stateless, so logout is mainly for client-side cleanup.
    In production, consider implementing token blacklisting.
    """
    try:
        logger.info(f"User logged out: {current_user.username}")
        
        return {
            "message": "Successfully logged out",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during logout"
        )


@auth_router.get("/me", response_model=Dict[str, Any])
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current authenticated user information.
    """
    try:
        user_info = create_user_response(current_user)
        
        # Add additional context
        user_info["is_admin"] = has_admin_role(current_user)
        user_info["can_access_admin_features"] = can_access_admin_features(current_user)
        user_info["permissions"] = get_user_permissions(current_user)
        
        return {
            "user": user_info,
            "status": "authenticated"
        }
        
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error getting user information"
        )


@auth_router.get("/validate-token")
async def validate_token(current_user: User = Depends(get_current_user)):
    """
    Validate JWT token and return user info.
    Used by frontend to check if token is still valid.
    """
    try:
        return {
            "valid": True,
            "user": create_user_response(current_user),
            "message": "Token is valid"
        }
        
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )


@auth_router.get("/admin/validate")
async def validate_admin_access(admin_user: User = Depends(get_current_admin_user)):
    """
    Validate admin access for admin-only endpoints.
    """
    try:
        return {
            "valid": True,
            "user": create_user_response(admin_user),
            "admin_access": True,
            "message": "Admin access validated"
        }
        
    except Exception as e:
        logger.error(f"Admin validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )


@auth_router.get("/permissions")
async def get_user_permissions_endpoint(current_user: User = Depends(get_current_user)):
    """
    Get detailed user permissions for frontend routing.
    """
    try:
        permissions = get_user_permissions(current_user)
        
        return {
            "permissions": permissions,
            "role": current_user.role.value,
            "username": current_user.username
        }
        
    except Exception as e:
        logger.error(f"Error getting permissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error getting permissions"
        )


def get_user_permissions(user: User) -> Dict[str, bool]:
    """
    Get detailed permissions for a user based on their role.
    
    Args:
        user: User object
        
    Returns:
        Dictionary of permission flags
    """
    if user.role == UserRole.ADMIN:
        return {
            "can_view_dashboard": True,
            "can_manage_devices": True,
            "can_register_devices": True,
            "can_view_security": True,
            "can_view_monitoring": True,
            "can_modify_settings": True,
            "can_manage_users": True,
            "can_access_admin_features": True,
            "can_participate_in_fl": True,
            "can_view_system_stats": True
        }
    elif user.role == UserRole.USER:
        return {
            "can_view_dashboard": True,
            "can_manage_devices": False,
            "can_register_devices": False,
            "can_view_security": False,
            "can_view_monitoring": False,
            "can_modify_settings": False,
            "can_manage_users": False,
            "can_access_admin_features": False,
            "can_participate_in_fl": True,
            "can_view_system_stats": False
        }
    else:
        # Default: no permissions
        return {
            "can_view_dashboard": False,
            "can_manage_devices": False,
            "can_register_devices": False,
            "can_view_security": False,
            "can_view_monitoring": False,
            "can_modify_settings": False,
            "can_manage_users": False,
            "can_access_admin_features": False,
            "can_participate_in_fl": False,
            "can_view_system_stats": False
        }


@auth_router.get("/health")
async def auth_health_check():
    """
    Health check endpoint for authentication service.
    """
    return {
        "status": "healthy",
        "service": "authentication",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }