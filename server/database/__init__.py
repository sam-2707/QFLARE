"""
Database Package for QFLARE

This package provides database abstraction layer with:
- SQLAlchemy models for all entities
- Connection management for SQLite/PostgreSQL
- High-level service layer for business operations
- Audit logging and security tracking
"""

from .connection import initialize_database, get_database, cleanup_database
from .models import Device, GlobalModel, ModelUpdate, TrainingSession, AuditLog, UserToken
from .services import DeviceService, ModelService, AuditService, TrainingService

__all__ = [
    # Connection management
    'initialize_database',
    'get_database', 
    'cleanup_database',
    
    # Models
    'Device',
    'GlobalModel',
    'ModelUpdate', 
    'TrainingSession',
    'AuditLog',
    'UserToken',
    
    # Services
    'DeviceService',
    'ModelService',
    'AuditService',
    'TrainingService'
]