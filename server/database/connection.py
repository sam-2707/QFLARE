"""
Database Configuration and Connection Management for QFLARE

This module provides database connection management, initialization,
and configuration for both SQLite (development) and PostgreSQL (production).
"""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.engine import Engine

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration management"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize database configuration.
        
        Args:
            config_dict: Configuration dictionary with database settings
        """
        self.config = config_dict or {}
        self._setup_default_config()
    
    def _setup_default_config(self):
        """Setup default database configuration"""
        defaults = {
            'database_type': os.getenv('DB_TYPE', 'sqlite'),
            'sqlite_path': os.getenv('SQLITE_PATH', 'qflare.db'),
            'postgres_host': os.getenv('POSTGRES_HOST', 'localhost'),
            'postgres_port': int(os.getenv('POSTGRES_PORT', '5432')),
            'postgres_db': os.getenv('POSTGRES_DB', 'qflare'),
            'postgres_user': os.getenv('POSTGRES_USER', 'qflare'),
            'postgres_password': os.getenv('POSTGRES_PASSWORD', 'qflare123'),
            'pool_size': int(os.getenv('DB_POOL_SIZE', '10')),
            'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', '20')),
            'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', '30')),
            'echo_sql': os.getenv('DB_ECHO', 'false').lower() == 'true'
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def get_database_url(self) -> str:
        """
        Get database URL based on configuration.
        
        Returns:
            Database URL string for SQLAlchemy
        """
        if self.config['database_type'] == 'sqlite':
            return f"sqlite:///{self.config['sqlite_path']}"
        elif self.config['database_type'] == 'postgresql':
            return (
                f"postgresql://{self.config['postgres_user']}:"
                f"{self.config['postgres_password']}@"
                f"{self.config['postgres_host']}:{self.config['postgres_port']}/"
                f"{self.config['postgres_db']}"
            )
        else:
            raise ValueError(f"Unsupported database type: {self.config['database_type']}")


class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize database manager.
        
        Args:
            config: Database configuration object
        """
        self.config = config
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self._setup_engine()
    
    def _setup_engine(self):
        """Setup SQLAlchemy engine with appropriate configuration"""
        database_url = self.config.get_database_url()
        
        engine_kwargs = {
            'echo': self.config.config['echo_sql'],
            'pool_pre_ping': True,  # Verify connections before use
        }
        
        if self.config.config['database_type'] == 'sqlite':
            # SQLite specific configuration
            engine_kwargs.update({
                'poolclass': StaticPool,
                'connect_args': {
                    'check_same_thread': False,
                    'timeout': 20
                }
            })
        else:
            # PostgreSQL specific configuration
            engine_kwargs.update({
                'pool_size': self.config.config['pool_size'],
                'max_overflow': self.config.config['max_overflow'],
                'pool_timeout': self.config.config['pool_timeout'],
                'pool_recycle': 3600,  # Recycle connections every hour
            })
        
        self.engine = create_engine(database_url, **engine_kwargs)
        
        # Enable WAL mode for SQLite for better concurrency
        if self.config.config['database_type'] == 'sqlite':
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.close()
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(f"Database engine initialized: {self.config.config['database_type']}")
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Error dropping database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """
        Get database session with automatic cleanup.
        
        Yields:
            SQLAlchemy session object
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_session_sync(self) -> Session:
        """
        Get database session for synchronous use.
        
        Returns:
            SQLAlchemy session object (caller must close)
        """
        return self.SessionLocal()
    
    def health_check(self) -> bool:
        """
        Check database connection health.
        
        Returns:
            True if database is accessible, False otherwise
        """
        try:
            with self.get_session() as session:
                from sqlalchemy import text
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get database connection information.
        
        Returns:
            Dictionary with connection details
        """
        return {
            'database_type': self.config.config['database_type'],
            'url': self.config.get_database_url(),
            'pool_size': getattr(self.engine.pool, 'size', None),
            'checked_out': getattr(self.engine.pool, 'checkedout', None),
            'overflow': getattr(self.engine.pool, 'overflow', None),
        }


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def initialize_database(config: Optional[Dict[str, Any]] = None) -> DatabaseManager:
    """
    Initialize global database manager.
    
    Args:
        config: Database configuration dictionary
        
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    
    db_config = DatabaseConfig(config)
    _db_manager = DatabaseManager(db_config)
    _db_manager.create_tables()
    
    logger.info("Database initialized successfully")
    return _db_manager


def get_database() -> DatabaseManager:
    """
    Get global database manager instance.
    
    Returns:
        DatabaseManager instance
        
    Raises:
        RuntimeError: If database not initialized
    """
    if _db_manager is None:
        raise RuntimeError("Database not initialized. Call initialize_database() first.")
    return _db_manager


def cleanup_database():
    """Cleanup database connections"""
    global _db_manager
    if _db_manager and _db_manager.engine:
        _db_manager.engine.dispose()
        logger.info("Database connections cleaned up")