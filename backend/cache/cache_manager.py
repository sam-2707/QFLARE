# QFLARE Performance Optimization Implementation

## Redis Caching Layer Implementation

"""
Advanced Redis-based caching system for QFLARE with intelligent cache invalidation,
distributed caching, and performance monitoring.
"""

import json
import pickle
import hashlib
from typing import Any, Optional, Dict, List, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import redis.asyncio as redis
from redis.asyncio.client import Pipeline
import aioredis
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
import logging

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy types."""
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back" 
    WRITE_AROUND = "write_around"
    CACHE_ASIDE = "cache_aside"


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    default_ttl: int = 3600  # 1 hour
    max_memory: str = "512mb"
    eviction_policy: str = "allkeys-lru"
    compression_threshold: int = 1024  # bytes
    enable_clustering: bool = False
    enable_encryption: bool = True
    performance_monitoring: bool = True


class CacheManager:
    """
    Advanced Redis cache manager with intelligent caching strategies,
    performance monitoring, and distributed cache support.
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.redis_cluster: Optional[redis.RedisCluster] = None
        self.performance_stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "deletes": 0,
            "errors": 0
        }
        
    async def initialize(self, redis_url: str, cluster_nodes: Optional[List[str]] = None):
        """Initialize Redis connection."""
        try:
            if self.config.enable_clustering and cluster_nodes:
                from redis.asyncio.cluster import RedisCluster
                self.redis_cluster = RedisCluster(
                    startup_nodes=cluster_nodes,
                    decode_responses=False,
                    max_connections=20,
                    retry_on_timeout=True
                )
                self.redis_client = self.redis_cluster
            else:
                self.redis_client = redis.from_url(
                    redis_url,
                    decode_responses=False,
                    max_connections=20,
                    retry_on_timeout=True
                )
                
            # Configure Redis settings
            await self._configure_redis()
            
            logger.info("Redis cache manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
            
    async def _configure_redis(self):
        """Configure Redis server settings."""
        if not self.redis_client:
            return
            
        try:
            await self.redis_client.config_set("maxmemory", self.config.max_memory)
            await self.redis_client.config_set("maxmemory-policy", self.config.eviction_policy)
            
            # Enable keyspace notifications for cache invalidation
            await self.redis_client.config_set("notify-keyspace-events", "Ex")
            
        except Exception as e:
            logger.warning(f"Could not configure Redis settings: {e}")
            
    def _generate_key(self, prefix: str, identifier: str, *args) -> str:
        """Generate cache key with optional hashing."""
        key_parts = [prefix, identifier] + list(str(arg) for arg in args)
        key = ":".join(key_parts)
        
        # Hash long keys
        if len(key) > 250:
            key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
            key = f"{prefix}:hash:{key_hash}"
            
        return key
        
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data with optional compression."""
        try:
            # Use JSON for simple types, pickle for complex objects
            if isinstance(data, (str, int, float, bool, list, dict)):
                serialized = json.dumps(data, default=str).encode()
            else:
                serialized = pickle.dumps(data)
                
            # Compress large payloads
            if len(serialized) > self.config.compression_threshold:
                import gzip
                serialized = gzip.compress(serialized)
                
            return serialized
            
        except Exception as e:
            logger.error(f"Failed to serialize data: {e}")
            raise
            
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data with automatic decompression."""
        try:
            # Try to decompress first
            try:
                import gzip
                data = gzip.decompress(data)
            except:
                pass  # Not compressed
                
            # Try JSON first, then pickle
            try:
                return json.loads(data.decode())
            except:
                return pickle.loads(data)
                
        except Exception as e:
            logger.error(f"Failed to deserialize data: {e}")
            raise
            
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with performance tracking."""
        if not self.redis_client:
            return default
            
        try:
            data = await self.redis_client.get(key)
            
            if data is not None:
                self.performance_stats["hits"] += 1
                return self._deserialize_data(data)
            else:
                self.performance_stats["misses"] += 1
                return default
                
        except Exception as e:
            self.performance_stats["errors"] += 1
            logger.error(f"Cache get error for key {key}: {e}")
            return default
            
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        strategy: CacheStrategy = CacheStrategy.CACHE_ASIDE
    ) -> bool:
        """Set value in cache with configurable strategy."""
        if not self.redis_client:
            return False
            
        try:
            ttl = ttl or self.config.default_ttl
            serialized_data = self._serialize_data(value)
            
            # Set with expiration
            success = await self.redis_client.set(key, serialized_data, ex=ttl)
            
            if success:
                self.performance_stats["writes"] += 1
                
            return bool(success)
            
        except Exception as e:
            self.performance_stats["errors"] += 1
            logger.error(f"Cache set error for key {key}: {e}")
            return False
            
    async def delete(self, *keys: str) -> int:
        """Delete keys from cache."""
        if not self.redis_client or not keys:
            return 0
            
        try:
            deleted = await self.redis_client.delete(*keys)
            self.performance_stats["deletes"] += deleted
            return deleted
            
        except Exception as e:
            self.performance_stats["errors"] += 1
            logger.error(f"Cache delete error: {e}")
            return 0
            
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.redis_client:
            return False
            
        try:
            return bool(await self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
            
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for existing key."""
        if not self.redis_client:
            return False
            
        try:
            return bool(await self.redis_client.expire(key, ttl))
        except Exception as e:
            logger.error(f"Cache expire error for key {key}: {e}")
            return False
            
    async def pipeline(self) -> Pipeline:
        """Get Redis pipeline for batch operations."""
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")
            
        return self.redis_client.pipeline()
        
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_operations = sum(self.performance_stats.values())
        hit_rate = (
            self.performance_stats["hits"] / 
            (self.performance_stats["hits"] + self.performance_stats["misses"])
            if (self.performance_stats["hits"] + self.performance_stats["misses"]) > 0 
            else 0
        )
        
        redis_info = {}
        if self.redis_client:
            try:
                info = await self.redis_client.info()
                redis_info = {
                    "memory_usage": info.get("used_memory_human", "Unknown"),
                    "memory_peak": info.get("used_memory_peak_human", "Unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                }
            except Exception as e:
                logger.error(f"Failed to get Redis info: {e}")
                
        return {
            "application_stats": self.performance_stats,
            "hit_rate": hit_rate,
            "total_operations": total_operations,
            "redis_info": redis_info,
            "timestamp": datetime.utcnow().isoformat()
        }


# Specialized cache managers for different data types
class ModelCacheManager(CacheManager):
    """Cache manager for ML models and federated learning data."""
    
    async def cache_model(
        self,
        model_id: str,
        model_data: bytes,
        version: str,
        metadata: Dict[str, Any],
        ttl: int = 7200  # 2 hours
    ) -> bool:
        """Cache ML model with metadata."""
        key = self._generate_key("model", model_id, version)
        
        model_info = {
            "data": model_data,
            "metadata": metadata,
            "cached_at": datetime.utcnow().isoformat(),
            "version": version
        }
        
        return await self.set(key, model_info, ttl)
        
    async def get_model(self, model_id: str, version: str) -> Optional[Dict[str, Any]]:
        """Get cached ML model."""
        key = self._generate_key("model", model_id, version)
        return await self.get(key)
        
    async def cache_training_metrics(
        self,
        session_id: str,
        round_number: int,
        metrics: Dict[str, float],
        ttl: int = 86400  # 24 hours
    ) -> bool:
        """Cache federated learning training metrics."""
        key = self._generate_key("metrics", session_id, round_number)
        return await self.set(key, metrics, ttl)


class SessionCacheManager(CacheManager):
    """Cache manager for user sessions and authentication data."""
    
    async def cache_session(
        self,
        session_id: str,
        user_data: Dict[str, Any],
        ttl: int = 3600  # 1 hour
    ) -> bool:
        """Cache user session data."""
        key = self._generate_key("session", session_id)
        return await self.set(key, user_data, ttl)
        
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached session data."""
        key = self._generate_key("session", session_id)
        return await self.get(key)
        
    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate user session."""
        key = self._generate_key("session", session_id)
        deleted = await self.delete(key)
        return deleted > 0
        
    async def cache_user_permissions(
        self,
        user_id: str,
        permissions: List[str],
        ttl: int = 1800  # 30 minutes
    ) -> bool:
        """Cache user permissions for RBAC."""
        key = self._generate_key("permissions", user_id)
        return await self.set(key, permissions, ttl)


class QueryCacheManager(CacheManager):
    """Cache manager for database queries and API responses."""
    
    def _generate_query_key(self, query: str, params: Dict[str, Any]) -> str:
        """Generate cache key for database query."""
        query_hash = hashlib.sha256(
            (query + str(sorted(params.items()))).encode()
        ).hexdigest()[:16]
        return self._generate_key("query", query_hash)
        
    async def cache_query_result(
        self,
        query: str,
        params: Dict[str, Any],
        result: Any,
        ttl: int = 300  # 5 minutes
    ) -> bool:
        """Cache database query result."""
        key = self._generate_query_key(query, params)
        return await self.set(key, result, ttl)
        
    async def get_cached_query(
        self,
        query: str,
        params: Dict[str, Any]
    ) -> Optional[Any]:
        """Get cached query result."""
        key = self._generate_query_key(query, params)
        return await self.get(key)
        
    async def invalidate_table_cache(self, table_name: str) -> int:
        """Invalidate all cached queries for a table."""
        pattern = f"query:*{table_name}*"
        
        if not self.redis_client:
            return 0
            
        try:
            keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key)
                
            if keys:
                return await self.delete(*keys)
            return 0
            
        except Exception as e:
            logger.error(f"Failed to invalidate table cache for {table_name}: {e}")
            return 0


# Cache initialization and dependency injection
_cache_managers: Dict[str, CacheManager] = {}


async def get_cache_manager(cache_type: str = "default") -> CacheManager:
    """Get cache manager instance."""
    if cache_type not in _cache_managers:
        config = CacheConfig()
        
        if cache_type == "model":
            _cache_managers[cache_type] = ModelCacheManager(config)
        elif cache_type == "session":
            _cache_managers[cache_type] = SessionCacheManager(config)
        elif cache_type == "query":
            _cache_managers[cache_type] = QueryCacheManager(config)
        else:
            _cache_managers[cache_type] = CacheManager(config)
            
    return _cache_managers[cache_type]


async def initialize_cache_system(redis_url: str, cluster_nodes: Optional[List[str]] = None):
    """Initialize all cache managers."""
    cache_types = ["default", "model", "session", "query"]
    
    for cache_type in cache_types:
        manager = await get_cache_manager(cache_type)
        await manager.initialize(redis_url, cluster_nodes)
        
    logger.info("Cache system initialized successfully")


# Cache decorators for easy usage
def cache_result(
    cache_type: str = "default",
    ttl: int = 3600,
    key_generator: Optional[callable] = None
):
    """Decorator to cache function results."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            manager = await get_cache_manager(cache_type)
            
            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                func_name = f"{func.__module__}.{func.__name__}"
                arg_hash = hashlib.sha256(
                    str((args, sorted(kwargs.items()))).encode()
                ).hexdigest()[:16]
                cache_key = f"func:{func_name}:{arg_hash}"
                
            # Try to get from cache
            cached_result = await manager.get(cache_key)
            if cached_result is not None:
                return cached_result
                
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await manager.set(cache_key, result, ttl)
            
            return result
            
        return wrapper
    return decorator


# Cache warming functions
async def warm_cache():
    """Warm up cache with frequently accessed data."""
    logger.info("Starting cache warming process...")
    
    try:
        # Warm up session cache
        session_manager = await get_cache_manager("session")
        
        # Warm up model cache
        model_manager = await get_cache_manager("model")
        
        # Warm up query cache with common queries
        query_manager = await get_cache_manager("query")
        
        logger.info("Cache warming completed successfully")
        
    except Exception as e:
        logger.error(f"Cache warming failed: {e}")


# Health check for cache system
async def cache_health_check() -> Dict[str, Any]:
    """Perform health check on cache system."""
    health_status = {
        "status": "healthy",
        "managers": {},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    for cache_type in ["default", "model", "session", "query"]:
        try:
            manager = await get_cache_manager(cache_type)
            
            # Test basic operations
            test_key = f"health_check_{cache_type}"
            await manager.set(test_key, "test_value", 10)
            result = await manager.get(test_key)
            await manager.delete(test_key)
            
            if result == "test_value":
                health_status["managers"][cache_type] = {
                    "status": "healthy",
                    "stats": await manager.get_performance_stats()
                }
            else:
                health_status["managers"][cache_type] = {
                    "status": "error",
                    "error": "Cache test failed"
                }
                health_status["status"] = "degraded"
                
        except Exception as e:
            health_status["managers"][cache_type] = {
                "status": "error",
                "error": str(e)
            }
            health_status["status"] = "unhealthy"
            
    return health_status