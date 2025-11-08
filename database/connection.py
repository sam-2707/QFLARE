"""
Database connection and session management for QFLARE.

This module handles database connections, session lifecycle,
connection pooling, and Redis integration.
"""

import os
import logging
import redis
from contextlib import contextmanager
from typing import Generator, Optional, Dict, Any, Union
from sqlalchemy import create_engine, event, pool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
import json
import pickle
from datetime import datetime, timedelta
from .models import Base

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration with environment variable support."""
    
    def __init__(self):
        # PostgreSQL configuration
        self.db_host = os.getenv("QFLARE_DB_HOST", "localhost")
        self.db_port = int(os.getenv("QFLARE_DB_PORT", "5432"))
        self.db_name = os.getenv("QFLARE_DB_NAME", "qflare")
        self.db_user = os.getenv("QFLARE_DB_USER", "qflare")
        self.db_password = os.getenv("QFLARE_DB_PASSWORD", "qflare_password")
        
        # Connection pool settings
        self.pool_size = int(os.getenv("QFLARE_DB_POOL_SIZE", "10"))
        self.max_overflow = int(os.getenv("QFLARE_DB_MAX_OVERFLOW", "20"))
        self.pool_timeout = int(os.getenv("QFLARE_DB_POOL_TIMEOUT", "30"))
        self.pool_recycle = int(os.getenv("QFLARE_DB_POOL_RECYCLE", "3600"))
        
        # Redis configuration
        self.redis_host = os.getenv("QFLARE_REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("QFLARE_REDIS_PORT", "6379"))
        self.redis_db = int(os.getenv("QFLARE_REDIS_DB", "0"))
        self.redis_password = os.getenv("QFLARE_REDIS_PASSWORD", None)
        self.redis_ssl = os.getenv("QFLARE_REDIS_SSL", "false").lower() == "true"
        
        # Cache settings
        self.cache_ttl_default = int(os.getenv("QFLARE_CACHE_TTL_DEFAULT", "3600"))  # 1 hour
        self.cache_ttl_sessions = int(os.getenv("QFLARE_CACHE_TTL_SESSIONS", "86400"))  # 24 hours
        self.cache_ttl_metrics = int(os.getenv("QFLARE_CACHE_TTL_METRICS", "300"))  # 5 minutes
    
    @property
    def database_url(self) -> str:
        """Get SQLAlchemy database URL."""
        return f"postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        scheme = "rediss" if self.redis_ssl else "redis"
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"{scheme}://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"


class DatabaseManager:
    """Centralized database and cache management."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._redis_client: Optional[redis.Redis] = None
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure database logging."""
        logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
        
    def initialize(self):
        """Initialize database connections and create tables if needed."""
        try:
            self._create_engine()
            self._create_session_factory()
            self._setup_redis()
            self._create_tables()
            logger.info("Database manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            raise
    
    def _create_engine(self):
        """Create SQLAlchemy engine with connection pooling."""
        engine_kwargs = {
            'pool_size': self.config.pool_size,
            'max_overflow': self.config.max_overflow,
            'pool_timeout': self.config.pool_timeout,
            'pool_recycle': self.config.pool_recycle,
            'pool_pre_ping': True,  # Enable connection health checks
            'echo': False,  # Set to True for SQL debugging
        }
        
        self._engine = create_engine(self.config.database_url, **engine_kwargs)
        
        # Add connection event listeners
        @event.listens_for(self._engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            # This is for PostgreSQL optimizations
            pass
        
        @event.listens_for(self._engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            logger.debug("Connection checked out from pool")
        
        logger.info(f"Database engine created: {self.config.db_host}:{self.config.db_port}")
    
    def _create_session_factory(self):
        """Create session factory."""
        self._session_factory = sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False
        )
    
    def _setup_redis(self):
        """Setup Redis connection with retry logic."""
        try:
            self._redis_client = redis.Redis.from_url(
                self.config.redis_url,
                decode_responses=False,  # Keep binary for pickle support
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                health_check_interval=30
            )
            
            # Test connection
            self._redis_client.ping()
            logger.info(f"Redis connected: {self.config.redis_host}:{self.config.redis_port}")
            
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection failed: {e}. Continuing without cache.")
            self._redis_client = None
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        try:
            Base.metadata.create_all(self._engine)
            logger.info("Database tables created/verified")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup."""
        if not self._session_factory:
            raise RuntimeError("Database not initialized")
        
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_redis(self) -> Optional[redis.Redis]:
        """Get Redis client."""
        return self._redis_client
    
    def close(self):
        """Close all connections."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database engine disposed")
        
        if self._redis_client:
            self._redis_client.close()
            logger.info("Redis connection closed")


class CacheManager:
    """Redis-based caching with fallback to database."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.redis_client = db_manager.get_redis()
        self.config = db_manager.config
    
    def get(self, key: str, default=None) -> Any:
        """Get value from cache with database fallback."""
        if not self.redis_client:
            return self._get_from_db(key, default)
        
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return pickle.loads(cached_data)
        except (redis.RedisError, pickle.PickleError) as e:
            logger.warning(f"Redis get error for key {key}: {e}")
        
        # Fallback to database
        return self._get_from_db(key, default)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with database backup."""
        ttl = ttl or self.config.cache_ttl_default
        
        # Try Redis first
        if self.redis_client:
            try:
                serialized_value = pickle.dumps(value)
                self.redis_client.setex(key, ttl, serialized_value)
                logger.debug(f"Cached key {key} in Redis with TTL {ttl}")
                return True
            except (redis.RedisError, pickle.PickleError) as e:
                logger.warning(f"Redis set error for key {key}: {e}")
        
        # Fallback to database
        return self._set_to_db(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache and database."""
        deleted = False
        
        # Delete from Redis
        if self.redis_client:
            try:
                self.redis_client.delete(key)
                deleted = True
            except redis.RedisError as e:
                logger.warning(f"Redis delete error for key {key}: {e}")
        
        # Delete from database
        self._delete_from_db(key)
        
        return deleted
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if self.redis_client:
            try:
                return bool(self.redis_client.exists(key))
            except redis.RedisError:
                pass
        
        return self._exists_in_db(key)
    
    def flush_expired(self):
        """Remove expired entries from database cache."""
        try:
            with self.db_manager.get_session() as session:
                from .models import CacheEntry
                expired_count = session.query(CacheEntry).filter(
                    CacheEntry.expires_at < datetime.utcnow()
                ).delete()
                
                if expired_count > 0:
                    logger.info(f"Removed {expired_count} expired cache entries")
                    
        except SQLAlchemyError as e:
            logger.error(f"Failed to flush expired cache entries: {e}")
    
    def _get_from_db(self, key: str, default=None) -> Any:
        """Get value from database cache."""
        try:
            with self.db_manager.get_session() as session:
                from .models import CacheEntry
                
                entry = session.query(CacheEntry).filter(
                    CacheEntry.cache_key == key
                ).first()
                
                if not entry:
                    return default
                
                # Check expiration
                if entry.expires_at and entry.expires_at < datetime.utcnow():
                    session.delete(entry)
                    return default
                
                return entry.cache_value
                
        except SQLAlchemyError as e:
            logger.error(f"Database cache get error for key {key}: {e}")
            return default
    
    def _set_to_db(self, key: str, value: Any, ttl: int) -> bool:
        """Set value in database cache."""
        try:
            with self.db_manager.get_session() as session:
                from .models import CacheEntry
                
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)
                
                # Upsert cache entry
                entry = session.query(CacheEntry).filter(
                    CacheEntry.cache_key == key
                ).first()
                
                if entry:
                    entry.cache_value = value
                    entry.expires_at = expires_at
                else:
                    entry = CacheEntry(
                        cache_key=key,
                        cache_value=value,
                        expires_at=expires_at
                    )
                    session.add(entry)
                
                logger.debug(f"Cached key {key} in database with TTL {ttl}")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Database cache set error for key {key}: {e}")
            return False
    
    def _delete_from_db(self, key: str):
        """Delete key from database cache."""
        try:
            with self.db_manager.get_session() as session:
                from .models import CacheEntry
                
                session.query(CacheEntry).filter(
                    CacheEntry.cache_key == key
                ).delete()
                
        except SQLAlchemyError as e:
            logger.error(f"Database cache delete error for key {key}: {e}")
    
    def _exists_in_db(self, key: str) -> bool:
        """Check if key exists in database cache."""
        try:
            with self.db_manager.get_session() as session:
                from .models import CacheEntry
                
                entry = session.query(CacheEntry).filter(
                    CacheEntry.cache_key == key
                ).first()
                
                if not entry:
                    return False
                
                # Check expiration
                if entry.expires_at and entry.expires_at < datetime.utcnow():
                    session.delete(entry)
                    return False
                
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Database cache exists error for key {key}: {e}")
            return False


# Global database manager instance
db_manager: Optional[DatabaseManager] = None
cache_manager: Optional[CacheManager] = None


def initialize_database(config: Optional[DatabaseConfig] = None):
    """Initialize global database manager."""
    global db_manager, cache_manager
    
    db_manager = DatabaseManager(config)
    db_manager.initialize()
    
    cache_manager = CacheManager(db_manager)
    
    logger.info("Global database manager initialized")


def get_db_session() -> Generator[Session, None, None]:
    """Get database session (for dependency injection)."""
    if not db_manager:
        raise RuntimeError("Database not initialized. Call initialize_database() first.")
    
    with db_manager.get_session() as session:
        yield session


def get_cache() -> CacheManager:
    """Get cache manager."""
    if not cache_manager:
        raise RuntimeError("Database not initialized. Call initialize_database() first.")
    
    return cache_manager


def close_database():
    """Close database connections."""
    global db_manager, cache_manager
    
    if db_manager:
        db_manager.close()
        db_manager = None
    
    cache_manager = None
    logger.info("Database connections closed")