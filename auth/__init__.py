"""
Authentication initialization module for QFLARE.

This module provides convenience imports and initialization functions
for the authentication system.
"""

from .jwt_auth import (
    JWTManager, TokenType, TokenPair, TokenClaims,
    MFAManager, PasswordManager, SessionManager
)
from .rbac import (
    Permission, Role, RBACManager, rbac_manager,
    require_permission, require_role, AccessControlContext, get_access_context
)
from .middleware import (
    AuthMiddleware, RateLimitMiddleware, initialize_auth,
    get_current_user, get_current_active_user, get_access_context as get_access_context_dependency,
    require_permission_dependency, require_role_dependency, admin_required
)

__all__ = [
    # JWT Authentication
    'JWTManager', 'TokenType', 'TokenPair', 'TokenClaims',
    'MFAManager', 'PasswordManager', 'SessionManager',
    
    # RBAC
    'Permission', 'Role', 'RBACManager', 'rbac_manager',
    'require_permission', 'require_role', 'AccessControlContext', 'get_access_context',
    
    # Middleware
    'AuthMiddleware', 'RateLimitMiddleware', 'initialize_auth',
    'get_current_user', 'get_current_active_user', 'get_access_context_dependency',
    'require_permission_dependency', 'require_role_dependency', 'admin_required'
]