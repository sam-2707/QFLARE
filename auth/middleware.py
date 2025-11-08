"""
Authentication middleware and utilities for QFLARE FastAPI backend.

This module provides middleware for JWT validation, session management,
and integration with the FastAPI application.
"""

import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone
from fastapi import Request, Response, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.base import BaseHTTPMiddleware
import redis
import json

from .jwt_auth import JWTManager, TokenClaims, TokenType
from .rbac import rbac_manager, Permission, Role, AccessControlContext
from database import get_db_session, UserService, AuditService

logger = logging.getLogger(__name__)

# Global instances
jwt_manager: Optional[JWTManager] = None
security = HTTPBearer()


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication and session management middleware."""
    
    def __init__(self, app, jwt_manager: JWTManager, skip_paths: Optional[list] = None):
        super().__init__(app)
        self.jwt_manager = jwt_manager
        self.skip_paths = skip_paths or [
            "/docs", "/redoc", "/openapi.json", "/health",
            "/auth/login", "/auth/register", "/auth/refresh"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Process request through authentication middleware."""
        start_time = datetime.now()
        
        # Skip authentication for certain paths
        if any(request.url.path.startswith(path) for path in self.skip_paths):
            response = await call_next(request)
            return response
        
        # Extract and validate JWT token
        auth_result = await self._authenticate_request(request)
        
        if auth_result:
            token_claims, session_data = auth_result
            
            # Add user context to request state
            request.state.user_id = token_claims.user_id
            request.state.username = token_claims.username
            request.state.user_role = token_claims.role
            request.state.session_id = token_claims.session_id
            request.state.token_claims = token_claims
            request.state.session_data = session_data
            
            # Create access control context
            request.state.access_context = AccessControlContext(
                token_claims.user_id, rbac_manager
            )
            
            # Process request
            response = await call_next(request)
            
            # Update session activity
            if hasattr(request.state, "session_id"):
                await self._update_session_activity(request.state.session_id)
            
            # Log successful request
            await self._log_request(request, response, token_claims, start_time, success=True)
            
        else:
            # Authentication failed
            response = Response(
                content=json.dumps({"detail": "Authentication required"}),
                status_code=401,
                media_type="application/json"
            )
            
            # Log failed authentication
            await self._log_request(request, response, None, start_time, success=False)
        
        return response
    
    async def _authenticate_request(self, request: Request) -> Optional[Tuple[TokenClaims, Dict[str, Any]]]:
        """Authenticate request and return token claims and session data."""
        try:
            # Extract Authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return None
            
            token = auth_header[7:]  # Remove "Bearer " prefix
            
            # Verify JWT token
            token_claims = self.jwt_manager.verify_token(token, TokenType.ACCESS)
            if not token_claims:
                return None
            
            # Validate session if available
            session_data = None
            if token_claims.session_id and self.jwt_manager.redis_client:
                session_key = f"session:{token_claims.session_id}"
                session_raw = self.jwt_manager.redis_client.get(session_key)
                
                if session_raw:
                    session_data = json.loads(session_raw)
                    
                    # Check if session is active
                    if not session_data.get("is_active", False):
                        logger.warning(f"Inactive session: {token_claims.session_id}")
                        return None
                    
                    # Verify IP address if configured for security
                    client_ip = self._get_client_ip(request)
                    stored_ip = session_data.get("ip_address")
                    
                    # Optional: Enable strict IP checking for high security
                    # if stored_ip and stored_ip != client_ip:
                    #     logger.warning(f"IP mismatch for session {token_claims.session_id}")
                    #     return None
            
            return token_claims, session_data or {}
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    async def _update_session_activity(self, session_id: str):
        """Update session last activity timestamp."""
        try:
            if self.jwt_manager.redis_client:
                session_key = f"session:{session_id}"
                session_data = self.jwt_manager.redis_client.get(session_key)
                
                if session_data:
                    session_info = json.loads(session_data)
                    session_info["last_activity"] = datetime.now(timezone.utc).isoformat()
                    
                    ttl = self.jwt_manager.redis_client.ttl(session_key)
                    if ttl > 0:
                        self.jwt_manager.redis_client.setex(
                            session_key,
                            ttl,
                            json.dumps(session_info, default=str)
                        )
                        
        except Exception as e:
            logger.error(f"Failed to update session activity: {e}")
    
    async def _log_request(
        self, 
        request: Request, 
        response: Response, 
        token_claims: Optional[TokenClaims], 
        start_time: datetime,
        success: bool
    ):
        """Log request for audit purposes."""
        try:
            duration = (datetime.now() - start_time).total_seconds()
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("User-Agent", "")
            
            # Log to audit system
            with get_db_session() as session:
                AuditService.log_event(
                    session=session,
                    event_type="api_request",
                    action=f"{request.method} {request.url.path}",
                    success=success,
                    user_id=token_claims.user_id if token_claims else None,
                    resource_type="api",
                    resource_id=request.url.path,
                    ip_address=client_ip,
                    user_agent=user_agent,
                    error_message=None if success else "Authentication failed",
                    details={
                        "method": request.method,
                        "path": request.url.path,
                        "query_params": str(request.query_params),
                        "status_code": response.status_code if hasattr(response, 'status_code') else None,
                        "duration_ms": duration * 1000,
                        "session_id": token_claims.session_id if token_claims else None
                    }
                )
                
        except Exception as e:
            logger.error(f"Failed to log request: {e}")
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (when behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        return request.client.host if request.client else "unknown"


# Dependency functions for FastAPI

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenClaims:
    """FastAPI dependency to get current authenticated user."""
    if not jwt_manager:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication system not initialized"
        )
    
    try:
        token_claims = jwt_manager.verify_token(credentials.credentials, TokenType.ACCESS)
        if not token_claims:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        return token_claims
        
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


async def get_current_active_user(current_user: TokenClaims = Depends(get_current_user)) -> TokenClaims:
    """FastAPI dependency to get current active user with session validation."""
    if not jwt_manager or not jwt_manager.redis_client:
        return current_user
    
    try:
        # Validate session
        session_key = f"session:{current_user.session_id}"
        session_data = jwt_manager.redis_client.get(session_key)
        
        if not session_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired"
            )
        
        session_info = json.loads(session_data)
        if not session_info.get("is_active", False):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session inactive"
            )
        
        return current_user
        
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session data"
        )


async def get_access_context(current_user: TokenClaims = Depends(get_current_active_user)) -> AccessControlContext:
    """FastAPI dependency to get access control context."""
    return AccessControlContext(current_user.user_id, rbac_manager)


def require_permission_dependency(permission: Permission):
    """Create FastAPI dependency that requires specific permission."""
    async def permission_dependency(access_context: AccessControlContext = Depends(get_access_context)):
        if not access_context.can(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission.value}"
            )
        return access_context
    
    return permission_dependency


def require_role_dependency(role: Role):
    """Create FastAPI dependency that requires specific role."""
    async def role_dependency(access_context: AccessControlContext = Depends(get_access_context)):
        if not access_context.has_role(role) and not access_context.is_admin():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: {role.value}"
            )
        return access_context
    
    return role_dependency


def admin_required(access_context: AccessControlContext = Depends(get_access_context)) -> AccessControlContext:
    """FastAPI dependency that requires admin role."""
    if not access_context.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator role required"
        )
    return access_context


# Rate limiting middleware

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using Redis."""
    
    def __init__(
        self, 
        app, 
        redis_client: Optional[redis.Redis] = None,
        default_limit: int = 100,  # requests per minute
        window_seconds: int = 60
    ):
        super().__init__(app)
        self.redis_client = redis_client
        self.default_limit = default_limit
        self.window_seconds = window_seconds
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting to requests."""
        if not self.redis_client:
            return await call_next(request)
        
        # Get client identifier (IP + user if authenticated)
        client_id = self._get_client_id(request)
        
        # Check rate limit
        if not await self._check_rate_limit(client_id):
            return Response(
                content=json.dumps({
                    "detail": "Rate limit exceeded",
                    "limit": self.default_limit,
                    "window": self.window_seconds
                }),
                status_code=429,
                media_type="application/json"
            )
        
        return await call_next(request)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        client_ip = request.client.host if request.client else "unknown"
        
        # Include user ID if authenticated
        if hasattr(request.state, "user_id"):
            return f"user:{request.state.user_id}"
        
        return f"ip:{client_ip}"
    
    async def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limit."""
        try:
            key = f"rate_limit:{client_id}"
            current_time = int(datetime.now().timestamp())
            window_start = current_time - self.window_seconds
            
            # Use Redis sorted set for sliding window
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current entries
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(key, self.window_seconds)
            
            results = pipe.execute()
            current_count = results[1]
            
            return current_count < self.default_limit
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Allow request on error
            return True


def initialize_auth(secret_key: str, redis_client: Optional[redis.Redis] = None):
    """Initialize authentication system."""
    global jwt_manager
    
    jwt_manager = JWTManager(
        secret_key=secret_key,
        redis_client=redis_client
    )
    
    logger.info("Authentication system initialized")