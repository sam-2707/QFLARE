"""
Role-Based Access Control (RBAC) system for QFLARE.

This module provides comprehensive permission management,
role definitions, and access control enforcement.
"""

from enum import Enum
from typing import Set, Dict, List, Optional, Any
from dataclasses import dataclass
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions."""
    
    # User Management
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    USER_LIST = "user:list"
    USER_MANAGE_ROLES = "user:manage_roles"
    USER_RESET_PASSWORD = "user:reset_password"
    
    # Project Management
    PROJECT_CREATE = "project:create"
    PROJECT_READ = "project:read"
    PROJECT_UPDATE = "project:update"
    PROJECT_DELETE = "project:delete"
    PROJECT_LIST = "project:list"
    PROJECT_MANAGE_ACCESS = "project:manage_access"
    
    # Training Management
    TRAINING_CREATE = "training:create"
    TRAINING_READ = "training:read"
    TRAINING_UPDATE = "training:update"
    TRAINING_DELETE = "training:delete"
    TRAINING_START = "training:start"
    TRAINING_STOP = "training:stop"
    TRAINING_MONITOR = "training:monitor"
    
    # Node Management
    NODE_REGISTER = "node:register"
    NODE_READ = "node:read"
    NODE_UPDATE = "node:update"
    NODE_DELETE = "node:delete"
    NODE_MANAGE = "node:manage"
    NODE_MONITOR = "node:monitor"
    
    # System Configuration
    CONFIG_READ = "config:read"
    CONFIG_UPDATE = "config:update"
    CONFIG_SYSTEM = "config:system"
    CONFIG_SECURITY = "config:security"
    
    # Monitoring and Metrics
    METRICS_READ = "metrics:read"
    METRICS_EXPORT = "metrics:export"
    LOGS_READ = "logs:read"
    LOGS_EXPORT = "logs:export"
    
    # Security and Audit
    AUDIT_READ = "audit:read"
    AUDIT_EXPORT = "audit:export"
    SECURITY_MANAGE = "security:manage"
    
    # API Access
    API_ADMIN = "api:admin"
    API_FEDERATED = "api:federated"
    API_MONITORING = "api:monitoring"
    
    # System Administration
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_MAINTENANCE = "system:maintenance"
    SYSTEM_BACKUP = "system:backup"


class Role(Enum):
    """System roles with hierarchical permissions."""
    
    ADMIN = "admin"
    OPERATOR = "operator"
    USER = "user"
    OBSERVER = "observer"


@dataclass
class RoleDefinition:
    """Role definition with permissions and metadata."""
    name: str
    description: str
    permissions: Set[Permission]
    is_system_role: bool = True
    parent_roles: Optional[List[Role]] = None


class RBACManager:
    """Role-Based Access Control manager."""
    
    def __init__(self):
        self._role_definitions = self._initialize_roles()
        self._user_roles: Dict[str, Set[Role]] = {}
        self._user_custom_permissions: Dict[str, Set[Permission]] = {}
        
    def _initialize_roles(self) -> Dict[Role, RoleDefinition]:
        """Initialize default role definitions."""
        return {
            Role.ADMIN: RoleDefinition(
                name="Administrator",
                description="Full system access with all permissions",
                permissions={
                    # All permissions
                    Permission.USER_CREATE, Permission.USER_READ, Permission.USER_UPDATE,
                    Permission.USER_DELETE, Permission.USER_LIST, Permission.USER_MANAGE_ROLES,
                    Permission.USER_RESET_PASSWORD,
                    
                    Permission.PROJECT_CREATE, Permission.PROJECT_READ, Permission.PROJECT_UPDATE,
                    Permission.PROJECT_DELETE, Permission.PROJECT_LIST, Permission.PROJECT_MANAGE_ACCESS,
                    
                    Permission.TRAINING_CREATE, Permission.TRAINING_READ, Permission.TRAINING_UPDATE,
                    Permission.TRAINING_DELETE, Permission.TRAINING_START, Permission.TRAINING_STOP,
                    Permission.TRAINING_MONITOR,
                    
                    Permission.NODE_REGISTER, Permission.NODE_READ, Permission.NODE_UPDATE,
                    Permission.NODE_DELETE, Permission.NODE_MANAGE, Permission.NODE_MONITOR,
                    
                    Permission.CONFIG_READ, Permission.CONFIG_UPDATE, Permission.CONFIG_SYSTEM,
                    Permission.CONFIG_SECURITY,
                    
                    Permission.METRICS_READ, Permission.METRICS_EXPORT, Permission.LOGS_READ,
                    Permission.LOGS_EXPORT,
                    
                    Permission.AUDIT_READ, Permission.AUDIT_EXPORT, Permission.SECURITY_MANAGE,
                    
                    Permission.API_ADMIN, Permission.API_FEDERATED, Permission.API_MONITORING,
                    
                    Permission.SYSTEM_ADMIN, Permission.SYSTEM_MAINTENANCE, Permission.SYSTEM_BACKUP
                }
            ),
            
            Role.OPERATOR: RoleDefinition(
                name="Operator",
                description="Project and training management with monitoring access",
                permissions={
                    Permission.USER_READ, Permission.USER_LIST,
                    
                    Permission.PROJECT_CREATE, Permission.PROJECT_READ, Permission.PROJECT_UPDATE,
                    Permission.PROJECT_LIST,
                    
                    Permission.TRAINING_CREATE, Permission.TRAINING_READ, Permission.TRAINING_UPDATE,
                    Permission.TRAINING_START, Permission.TRAINING_STOP, Permission.TRAINING_MONITOR,
                    
                    Permission.NODE_REGISTER, Permission.NODE_READ, Permission.NODE_UPDATE,
                    Permission.NODE_MANAGE, Permission.NODE_MONITOR,
                    
                    Permission.CONFIG_READ,
                    
                    Permission.METRICS_READ, Permission.METRICS_EXPORT, Permission.LOGS_READ,
                    
                    Permission.API_FEDERATED, Permission.API_MONITORING
                }
            ),
            
            Role.USER: RoleDefinition(
                name="User",
                description="Basic user with project creation and training access",
                permissions={
                    Permission.USER_READ,  # Own profile only
                    
                    Permission.PROJECT_CREATE, Permission.PROJECT_READ, Permission.PROJECT_UPDATE,
                    Permission.PROJECT_LIST,  # Own projects only
                    
                    Permission.TRAINING_CREATE, Permission.TRAINING_READ, Permission.TRAINING_UPDATE,
                    Permission.TRAINING_START, Permission.TRAINING_STOP, Permission.TRAINING_MONITOR,
                    
                    Permission.NODE_REGISTER, Permission.NODE_READ, Permission.NODE_MONITOR,
                    
                    Permission.CONFIG_READ,  # Limited config access
                    
                    Permission.METRICS_READ,
                    
                    Permission.API_FEDERATED
                }
            ),
            
            Role.OBSERVER: RoleDefinition(
                name="Observer",
                description="Read-only access for monitoring and auditing",
                permissions={
                    Permission.USER_READ, Permission.USER_LIST,
                    
                    Permission.PROJECT_READ, Permission.PROJECT_LIST,
                    
                    Permission.TRAINING_READ, Permission.TRAINING_MONITOR,
                    
                    Permission.NODE_READ, Permission.NODE_MONITOR,
                    
                    Permission.CONFIG_READ,
                    
                    Permission.METRICS_READ, Permission.LOGS_READ,
                    
                    Permission.AUDIT_READ,
                    
                    Permission.API_MONITORING
                }
            )
        }
    
    def assign_role(self, user_id: str, role: Role) -> bool:
        """Assign role to user."""
        try:
            if user_id not in self._user_roles:
                self._user_roles[user_id] = set()
            
            self._user_roles[user_id].add(role)
            logger.info(f"Role {role.value} assigned to user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to assign role {role.value} to user {user_id}: {e}")
            return False
    
    def remove_role(self, user_id: str, role: Role) -> bool:
        """Remove role from user."""
        try:
            if user_id in self._user_roles and role in self._user_roles[user_id]:
                self._user_roles[user_id].remove(role)
                logger.info(f"Role {role.value} removed from user {user_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove role {role.value} from user {user_id}: {e}")
            return False
    
    def grant_permission(self, user_id: str, permission: Permission) -> bool:
        """Grant custom permission to user."""
        try:
            if user_id not in self._user_custom_permissions:
                self._user_custom_permissions[user_id] = set()
            
            self._user_custom_permissions[user_id].add(permission)
            logger.info(f"Permission {permission.value} granted to user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to grant permission {permission.value} to user {user_id}: {e}")
            return False
    
    def revoke_permission(self, user_id: str, permission: Permission) -> bool:
        """Revoke custom permission from user."""
        try:
            if (user_id in self._user_custom_permissions and 
                permission in self._user_custom_permissions[user_id]):
                
                self._user_custom_permissions[user_id].remove(permission)
                logger.info(f"Permission {permission.value} revoked from user {user_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to revoke permission {permission.value} from user {user_id}: {e}")
            return False
    
    def has_permission(self, user_id: str, permission: Permission, resource_context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if user has specific permission."""
        try:
            # Check custom permissions first
            if (user_id in self._user_custom_permissions and 
                permission in self._user_custom_permissions[user_id]):
                return True
            
            # Check role-based permissions
            if user_id in self._user_roles:
                for role in self._user_roles[user_id]:
                    role_def = self._role_definitions.get(role)
                    if role_def and permission in role_def.permissions:
                        # Apply resource-based access control
                        return self._check_resource_access(user_id, role, permission, resource_context)
            
            return False
            
        except Exception as e:
            logger.error(f"Permission check error for user {user_id}, permission {permission.value}: {e}")
            return False
    
    def _check_resource_access(
        self, 
        user_id: str, 
        role: Role, 
        permission: Permission, 
        resource_context: Optional[Dict[str, Any]]
    ) -> bool:
        """Apply resource-based access control rules."""
        if not resource_context:
            return True
        
        # Admin has access to everything
        if role == Role.ADMIN:
            return True
        
        resource_type = resource_context.get("type")
        resource_owner = resource_context.get("owner_id")
        
        # Owner-based access control
        if resource_owner:
            # Users can access their own resources
            if user_id == resource_owner:
                return True
            
            # Operators can access resources in their managed projects
            if role == Role.OPERATOR:
                # Additional logic for project membership could be added here
                return True
        
        # Project-based access control
        if resource_type in ["project", "training", "node"] and role in [Role.USER, Role.OPERATOR]:
            project_id = resource_context.get("project_id")
            if project_id:
                # Check if user has access to this project
                # This would typically involve checking project membership
                return self._check_project_access(user_id, project_id, permission)
        
        # Default: allow read access for observers, deny others
        if role == Role.OBSERVER and permission.value.endswith(":read"):
            return True
        
        return False
    
    def _check_project_access(self, user_id: str, project_id: str, permission: Permission) -> bool:
        """Check project-specific access (placeholder for project membership logic)."""
        # This would integrate with the project service to check membership
        # For now, return True as a placeholder
        return True
    
    def get_user_roles(self, user_id: str) -> Set[Role]:
        """Get all roles assigned to user."""
        return self._user_roles.get(user_id, set())
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for user (roles + custom)."""
        permissions = set()
        
        # Add role-based permissions
        if user_id in self._user_roles:
            for role in self._user_roles[user_id]:
                role_def = self._role_definitions.get(role)
                if role_def:
                    permissions.update(role_def.permissions)
        
        # Add custom permissions
        if user_id in self._user_custom_permissions:
            permissions.update(self._user_custom_permissions[user_id])
        
        return permissions
    
    def get_role_permissions(self, role: Role) -> Set[Permission]:
        """Get permissions for a specific role."""
        role_def = self._role_definitions.get(role)
        return role_def.permissions if role_def else set()
    
    def create_custom_role(self, role_name: str, description: str, permissions: Set[Permission]) -> bool:
        """Create custom role (for future extensibility)."""
        # This would allow creating custom roles beyond the system defaults
        # Implementation would store custom roles in database
        logger.info(f"Custom role creation requested: {role_name}")
        return True


# Global RBAC manager instance
rbac_manager = RBACManager()


def require_permission(permission: Permission, resource_context_func=None):
    """Decorator to require specific permission for endpoint access."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This would typically get user_id from the current request context
            # For now, this is a placeholder for the decorator pattern
            user_id = kwargs.get("current_user_id") or getattr(func, "_current_user_id", None)
            
            if not user_id:
                raise PermissionError("Authentication required")
            
            # Get resource context if function provided
            resource_context = None
            if resource_context_func:
                resource_context = resource_context_func(*args, **kwargs)
            
            # Check permission
            if not rbac_manager.has_permission(user_id, permission, resource_context):
                raise PermissionError(f"Permission denied: {permission.value}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_role(role: Role):
    """Decorator to require specific role for endpoint access."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            user_id = kwargs.get("current_user_id") or getattr(func, "_current_user_id", None)
            
            if not user_id:
                raise PermissionError("Authentication required")
            
            user_roles = rbac_manager.get_user_roles(user_id)
            if role not in user_roles and Role.ADMIN not in user_roles:
                raise PermissionError(f"Role required: {role.value}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


class AccessControlContext:
    """Context manager for access control operations."""
    
    def __init__(self, user_id: str, rbac_manager: RBACManager):
        self.user_id = user_id
        self.rbac_manager = rbac_manager
    
    def can(self, permission: Permission, resource_context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if current user can perform action."""
        return self.rbac_manager.has_permission(self.user_id, permission, resource_context)
    
    def has_role(self, role: Role) -> bool:
        """Check if current user has role."""
        return role in self.rbac_manager.get_user_roles(self.user_id)
    
    def is_admin(self) -> bool:
        """Check if current user is admin."""
        return self.has_role(Role.ADMIN)
    
    def get_permissions(self) -> Set[Permission]:
        """Get all permissions for current user."""
        return self.rbac_manager.get_user_permissions(self.user_id)
    
    def ensure_permission(self, permission: Permission, resource_context: Optional[Dict[str, Any]] = None):
        """Ensure user has permission or raise exception."""
        if not self.can(permission, resource_context):
            raise PermissionError(f"Permission denied: {permission.value}")
    
    def ensure_role(self, role: Role):
        """Ensure user has role or raise exception."""
        if not self.has_role(role) and not self.is_admin():
            raise PermissionError(f"Role required: {role.value}")


def get_access_context(user_id: str) -> AccessControlContext:
    """Get access control context for user."""
    return AccessControlContext(user_id, rbac_manager)