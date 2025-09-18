#!/usr/bin/env python3
"""
QFLARE Database Connection Manager
Async database connection handling with SQLite/PostgreSQL support
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any
from urllib.parse import urlparse

from sqlalchemy.ext.asyncio import (
    create_async_engine, 
    AsyncSession, 
    AsyncEngine,
    async_sessionmaker
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy import text
import aiosqlite
import asyncpg

# Import our models
from .models import Base

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database configuration management"""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL', 'sqlite+aiosqlite:///./qflare.db')
        self.echo = os.getenv('DATABASE_ECHO', 'false').lower() == 'true'
        self.pool_size = int(os.getenv('DATABASE_POOL_SIZE', '5'))
        self.max_overflow = int(os.getenv('DATABASE_MAX_OVERFLOW', '10'))
        self.pool_timeout = int(os.getenv('DATABASE_POOL_TIMEOUT', '30'))
        self.pool_recycle = int(os.getenv('DATABASE_POOL_RECYCLE', '3600'))
        
    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith('sqlite')
    
    @property
    def is_postgresql(self) -> bool:
        return self.database_url.startswith('postgresql')
    
    def get_engine_kwargs(self) -> Dict[str, Any]:
        """Get engine configuration based on database type"""
        base_kwargs = {
            'echo': self.echo,
            'future': True,
        }
        
        if self.is_sqlite:
            # SQLite-specific configuration
            base_kwargs.update({
                'poolclass': StaticPool,
                'connect_args': {
                    'check_same_thread': False,
                    'timeout': 20,
                    'isolation_level': None,  # Use autocommit mode
                }
            })
        elif self.is_postgresql:
            # PostgreSQL-specific configuration
            base_kwargs.update({
                'pool_size': self.pool_size,
                'max_overflow': self.max_overflow,
                'pool_timeout': self.pool_timeout,
                'pool_recycle': self.pool_recycle,
                'pool_pre_ping': True,
                'connect_args': {
                    'server_settings': {
                        'application_name': 'qflare_quantum_system',
                        'jit': 'off',  # Disable JIT for predictable performance
                    }
                }
            })
        
        return base_kwargs

class DatabaseManager:
    """Async database connection manager"""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize database engine and session factory"""
        if self._initialized:
            return
            
        try:
            # Create async engine
            engine_kwargs = self.config.get_engine_kwargs()
            self.engine = create_async_engine(
                self.config.database_url,
                **engine_kwargs
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False
            )
            
            # Test connection
            await self._test_connection()
            
            # Create tables if they don't exist
            await self.create_tables()
            
            self._initialized = True
            logger.info(f"Database initialized successfully: {self.config.database_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def _test_connection(self) -> None:
        """Test database connection"""
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise
    
    async def create_tables(self) -> None:
        """Create database tables"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup"""
        if not self._initialized:
            await self.initialize()
        
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    async def execute_raw_sql(self, sql: str, parameters: Optional[Dict] = None) -> Any:
        """Execute raw SQL with parameters"""
        if not self._initialized:
            await self.initialize()
        
        async with self.engine.begin() as conn:
            result = await conn.execute(text(sql), parameters or {})
            return result
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            async with self.get_session() as session:
                # Simple query to test responsiveness
                result = await session.execute(text("SELECT 1 as health_check"))
                row = result.fetchone()
                
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return {
                'status': 'healthy',
                'response_time_ms': round(response_time, 2),
                'database_url': self.config.database_url.split('@')[-1] if '@' in self.config.database_url else self.config.database_url,
                'engine_pool_size': getattr(self.engine.pool, 'size', 'N/A'),
                'engine_pool_checked_out': getattr(self.engine.pool, 'checkedout', 'N/A'),
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'database_url': self.config.database_url.split('@')[-1] if '@' in self.config.database_url else self.config.database_url,
            }
    
    async def get_connection_info(self) -> Dict[str, Any]:
        """Get detailed connection information"""
        if not self._initialized:
            return {'status': 'not_initialized'}
        
        try:
            # Get basic info
            info = {
                'database_type': 'sqlite' if self.config.is_sqlite else 'postgresql',
                'database_url': self.config.database_url.split('@')[-1] if '@' in self.config.database_url else self.config.database_url,
                'echo_enabled': self.config.echo,
                'initialized': self._initialized,
            }
            
            # Add pool information if available
            if hasattr(self.engine, 'pool'):
                pool = self.engine.pool
                info.update({
                    'pool_size': getattr(pool, 'size', None),
                    'pool_checked_out': getattr(pool, 'checkedout', None),
                    'pool_overflow': getattr(pool, 'overflow', None),
                    'pool_checked_in': getattr(pool, 'checkedin', None),
                })
            
            # Add database-specific info
            if self.config.is_postgresql:
                info.update({
                    'pool_timeout': self.config.pool_timeout,
                    'pool_recycle': self.config.pool_recycle,
                    'max_overflow': self.config.max_overflow,
                })
            
            return info
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def close(self) -> None:
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")
        self._initialized = False

# Global database manager instance
db_manager = DatabaseManager()

# Convenience functions for common operations
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session - convenience function"""
    async with db_manager.get_session() as session:
        yield session

async def init_database(config: Optional[DatabaseConfig] = None) -> None:
    """Initialize database - convenience function"""
    global db_manager
    if config:
        db_manager = DatabaseManager(config)
    await db_manager.initialize()

async def close_database() -> None:
    """Close database connections - convenience function"""
    await db_manager.close()

async def get_database_health() -> Dict[str, Any]:
    """Get database health status - convenience function"""
    return await db_manager.health_check()

# Database session dependency for FastAPI
async def get_async_db_session():
    """FastAPI dependency for database sessions"""
    async with db_manager.get_session() as session:
        yield session

# Context manager for database operations
@asynccontextmanager
async def database_transaction():
    """Context manager for database transactions"""
    async with db_manager.get_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise