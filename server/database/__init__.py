#!/usr/bin/env python3
"""
QFLARE Database Package
Production database layer for quantum key exchange system
"""

from .connection import (
    DatabaseManager,
    DatabaseConfig,
    db_manager,
    init_database,
    close_database,
    get_database_health,
    get_async_db_session,
    database_transaction
)

from .models import (
    Base,
    Device,
    KeyExchangeSession,
    AuditLog,
    SecurityEvent,
    PerformanceMetric,
    EnrollmentToken,
    SystemConfiguration,
    GlobalModel,
    ModelUpdate,
    TrainingSession,
    UserToken
)

from .repository import (
    DeviceRepository,
    KeyExchangeRepository,
    AuditRepository, 
    ConfigurationRepository,
    quick_device_lookup,
    quick_session_lookup,
    security_audit_log
)

from .services import (
    DeviceService,
    ModelService,
    AuditService,
    TrainingService
)

__all__ = [
    # Connection management
    'DatabaseManager',
    'DatabaseConfig', 
    'db_manager',
    'init_database',
    'close_database',
    'get_database_health',
    'get_async_db_session',
    'database_transaction',
    
    # Models
    'Base',
    'Device',
    'KeyExchangeSession', 
    'AuditLog',
    'SecurityEvent',
    'PerformanceMetric',
    'EnrollmentToken',
    'SystemConfiguration',
    'GlobalModel',
    'ModelUpdate', 
    'TrainingSession',
    'UserToken',
    
    # Repositories
    'DeviceRepository',
    'KeyExchangeRepository',
    'AuditRepository', 
    'ConfigurationRepository',
    
    # Services
    'DeviceService',
    'ModelService',
    'AuditService', 
    'TrainingService',
    
    # Convenience functions
    'quick_device_lookup',
    'quick_session_lookup',
    'security_audit_log'
]

# Version info
__version__ = '1.0.0'