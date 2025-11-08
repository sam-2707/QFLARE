# QFLARE Performance Integration Module

"""
Integration module that brings together all performance optimizations
and provides a unified interface for the QFLARE system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import os

from fastapi import FastAPI
import redis.asyncio as redis

# Import performance modules
from backend.cache.cache_manager import (
    CacheManager, CacheConfig, initialize_cache_system,
    get_cache_manager, cache_health_check
)
from backend.database.optimizer import (
    OptimizedDatabaseManager, DatabaseConfig, initialize_database_manager
)
from backend.api.performance import (
    PerformanceMiddleware, RateLimitConfig, CompressionConfig,
    CacheConfig as APICacheConfig, setup_performance_middleware
)
from backend.api.load_balancer import (
    LoadBalancer, CDNManager, ServerInfo, LoadBalanceStrategy,
    CDNConfig, setup_load_balancing
)

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """
    Unified performance optimizer that coordinates all optimization
    components for the QFLARE system.
    """
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.database_manager: Optional[OptimizedDatabaseManager] = None
        self.cache_managers: Dict[str, CacheManager] = {}
        self.load_balancer: Optional[LoadBalancer] = None
        self.cdn_manager: Optional[CDNManager] = None
        self.initialized = False
        
    async def initialize(
        self,
        app: FastAPI,
        redis_url: str = None,
        database_url: str = None,
        backend_servers: List[Dict[str, Any]] = None,
        **configs
    ):
        """
        Initialize all performance optimization components.
        
        Args:
            app: FastAPI application instance
            redis_url: Redis connection URL
            database_url: Database connection URL
            backend_servers: List of backend server configurations
            **configs: Additional configuration overrides
        """
        logger.info("Initializing QFLARE performance optimizations...")
        
        try:
            # Initialize Redis connection
            await self._initialize_redis(redis_url or self._get_redis_url())
            
            # Initialize database optimization
            await self._initialize_database(database_url or self._get_database_url())
            
            # Initialize cache system
            await self._initialize_cache_system()
            
            # Setup API performance middleware
            await self._setup_api_performance(app, configs)
            
            # Setup load balancing and CDN
            if backend_servers:
                await self._setup_load_balancing(app, backend_servers, configs)
                
            # Setup monitoring endpoints
            self._setup_monitoring_endpoints(app)
            
            self.initialized = True
            logger.info("Performance optimization system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize performance optimizer: {e}")
            raise
            
    async def _initialize_redis(self, redis_url: str):
        """Initialize Redis connection."""
        self.redis_client = redis.from_url(
            redis_url,
            decode_responses=False,
            max_connections=20,
            retry_on_timeout=True
        )
        
        # Test connection
        await self.redis_client.ping()
        logger.info("Redis connection established")
        
    async def _initialize_database(self, database_url: str):
        """Initialize optimized database manager."""
        config = DatabaseConfig(
            url=database_url,
            min_size=5,
            max_size=20,
            enable_performance_monitoring=True,
            slow_query_threshold=1.0
        )
        
        await initialize_database_manager(config)
        self.database_manager = await get_database_manager()
        logger.info("Optimized database manager initialized")
        
    async def _initialize_cache_system(self):
        """Initialize comprehensive cache system."""
        await initialize_cache_system(
            self._get_redis_url(),
            cluster_nodes=None  # Set if using Redis cluster
        )
        
        # Get cache managers
        self.cache_managers = {
            "default": await get_cache_manager("default"),
            "model": await get_cache_manager("model"),
            "session": await get_cache_manager("session"),
            "query": await get_cache_manager("query")
        }
        
        logger.info("Cache system initialized")
        
    async def _setup_api_performance(self, app: FastAPI, configs: Dict[str, Any]):
        """Setup API performance middleware."""
        # Rate limiting configuration
        rate_limit_config = RateLimitConfig(
            requests_per_minute=configs.get("rate_limit_rpm", 60),
            requests_per_hour=configs.get("rate_limit_rph", 1000),
            requests_per_day=configs.get("rate_limit_rpd", 10000),
            strategy=configs.get("rate_limit_strategy", "sliding_window")
        )
        
        # API caching configuration
        api_cache_config = APICacheConfig(
            enable_response_cache=configs.get("enable_api_cache", True),
            default_ttl=configs.get("api_cache_ttl", 300),
            cache_private_responses=False
        )
        
        # Compression configuration
        compression_config = CompressionConfig(
            enable_gzip=True,
            enable_brotli=True,
            min_size=1024
        )
        
        # Setup middleware
        setup_performance_middleware(
            app,
            self.redis_client,
            rate_limit_config,
            api_cache_config,
            compression_config
        )
        
        logger.info("API performance middleware configured")
        
    async def _setup_load_balancing(
        self,
        app: FastAPI,
        backend_servers: List[Dict[str, Any]],
        configs: Dict[str, Any]
    ):
        """Setup load balancing and CDN integration."""
        # Convert server configs to ServerInfo objects
        servers = []
        for server_config in backend_servers:
            server = ServerInfo(
                host=server_config["host"],
                port=server_config["port"],
                weight=server_config.get("weight", 1),
                max_connections=server_config.get("max_connections", 100),
                location=server_config.get("location")
            )
            servers.append(server)
            
        # Load balancing strategy
        strategy = LoadBalanceStrategy(
            configs.get("load_balance_strategy", "health_based")
        )
        
        # CDN configuration
        cdn_config = CDNConfig(
            enable_cdn=configs.get("enable_cdn", True),
            cdn_endpoints=configs.get("cdn_endpoints", []),
            cache_ttl=configs.get("cdn_cache_ttl", 3600),
            edge_locations=configs.get("edge_locations", {})
        )
        
        # Setup load balancing
        setup_load_balancing(
            app, servers, strategy, cdn_config, self.redis_client
        )
        
        logger.info("Load balancing and CDN configured")
        
    def _setup_monitoring_endpoints(self, app: FastAPI):
        """Setup performance monitoring endpoints."""
        
        @app.get("/monitoring/performance/summary")
        async def get_performance_summary():
            """Get comprehensive performance summary."""
            return await self.get_performance_summary()
            
        @app.get("/monitoring/performance/cache")
        async def get_cache_performance():
            """Get cache performance metrics."""
            return await cache_health_check()
            
        @app.get("/monitoring/performance/database")
        async def get_database_performance():
            """Get database performance metrics."""
            if self.database_manager:
                return await self.database_manager.get_performance_metrics()
            return {"error": "Database manager not initialized"}
            
        @app.get("/monitoring/performance/suggestions")
        async def get_optimization_suggestions():
            """Get performance optimization suggestions."""
            return await self.get_optimization_suggestions()
            
        @app.post("/monitoring/performance/cache/warm")
        async def warm_cache():
            """Warm up cache with frequently accessed data."""
            try:
                from backend.cache.cache_manager import warm_cache
                await warm_cache()
                return {"success": True, "message": "Cache warming initiated"}
            except Exception as e:
                return {"success": False, "error": str(e)}
                
        @app.post("/monitoring/performance/database/vacuum")
        async def vacuum_database():
            """Perform database vacuum and analyze."""
            if self.database_manager:
                try:
                    await self.database_manager.vacuum_analyze_tables()
                    return {"success": True, "message": "Database vacuum completed"}
                except Exception as e:
                    return {"success": False, "error": str(e)}
            return {"success": False, "error": "Database manager not initialized"}
            
        logger.info("Performance monitoring endpoints configured")
        
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_status": "healthy",
            "components": {}
        }
        
        try:
            # Cache performance
            cache_health = await cache_health_check()
            summary["components"]["cache"] = {
                "status": cache_health.get("status", "unknown"),
                "managers": len(cache_health.get("managers", {})),
                "details": cache_health
            }
            
            # Database performance
            if self.database_manager:
                db_health = await self.database_manager.health_check()
                db_metrics = await self.database_manager.get_performance_metrics()
                
                summary["components"]["database"] = {
                    "status": db_health.get("status", "unknown"),
                    "connection_time": db_health.get("checks", {}).get("connectivity", {}).get("response_time", 0),
                    "active_connections": db_health.get("checks", {}).get("connections", {}).get("active_connections", 0),
                    "query_metrics": {
                        "total_queries": db_metrics.get("summary", {}).get("total_queries", 0),
                        "avg_query_time": db_metrics.get("summary", {}).get("avg_query_time", 0),
                        "error_rate": db_metrics.get("summary", {}).get("error_rate", 0)
                    }
                }
            
            # Redis health
            if self.redis_client:
                try:
                    redis_info = await self.redis_client.info()
                    summary["components"]["redis"] = {
                        "status": "healthy",
                        "memory_usage": redis_info.get("used_memory_human", "Unknown"),
                        "connected_clients": redis_info.get("connected_clients", 0),
                        "ops_per_sec": redis_info.get("instantaneous_ops_per_sec", 0)
                    }
                except Exception as e:
                    summary["components"]["redis"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    
            # Overall system status
            unhealthy_components = [
                name for name, component in summary["components"].items()
                if component.get("status") != "healthy"
            ]
            
            if unhealthy_components:
                summary["system_status"] = "degraded"
                summary["unhealthy_components"] = unhealthy_components
                
        except Exception as e:
            summary["system_status"] = "error"
            summary["error"] = str(e)
            
        return summary
        
    async def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Get performance optimization suggestions."""
        suggestions = []
        
        try:
            # Database optimization suggestions
            if self.database_manager:
                db_suggestions = await self.database_manager.get_optimization_suggestions()
                suggestions.extend(db_suggestions)
                
            # Cache optimization suggestions
            cache_health = await cache_health_check()
            for manager_name, manager_status in cache_health.get("managers", {}).items():
                if manager_status.get("status") != "healthy":
                    suggestions.append({
                        "type": "cache_health",
                        "description": f"Cache manager '{manager_name}' is unhealthy",
                        "recommendation": "Check Redis connectivity and configuration"
                    })
                    
            # Memory usage suggestions
            if self.redis_client:
                try:
                    redis_info = await self.redis_client.info()
                    memory_usage_mb = redis_info.get("used_memory", 0) / (1024 * 1024)
                    
                    if memory_usage_mb > 400:  # > 400MB
                        suggestions.append({
                            "type": "redis_memory",
                            "description": f"Redis memory usage is high: {memory_usage_mb:.1f}MB",
                            "recommendation": "Consider increasing memory or implementing cache eviction policies"
                        })
                except Exception:
                    pass
                    
        except Exception as e:
            suggestions.append({
                "type": "system_error",
                "description": f"Error generating suggestions: {e}",
                "recommendation": "Check system logs for detailed error information"
            })
            
        return suggestions
        
    def _get_redis_url(self) -> str:
        """Get Redis URL from environment or default."""
        return os.getenv(
            "REDIS_URL",
            "redis://localhost:6379/0"
        )
        
    def _get_database_url(self) -> str:
        """Get database URL from environment or default."""
        return os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://qflare_user:password@localhost:5432/qflare"
        )
        
    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.redis_client:
                await self.redis_client.close()
                
            if self.database_manager:
                await self.database_manager.close()
                
            logger.info("Performance optimizer cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global performance optimizer instance
_performance_optimizer: Optional[PerformanceOptimizer] = None


async def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


async def initialize_performance_system(
    app: FastAPI,
    **config_overrides
) -> PerformanceOptimizer:
    """
    Initialize the complete QFLARE performance optimization system.
    
    Args:
        app: FastAPI application instance
        **config_overrides: Configuration overrides
        
    Returns:
        Initialized PerformanceOptimizer instance
    """
    optimizer = await get_performance_optimizer()
    
    # Default configuration
    default_config = {
        # Rate limiting
        "rate_limit_rpm": 100,
        "rate_limit_rph": 2000,
        "rate_limit_rpd": 20000,
        "rate_limit_strategy": "sliding_window",
        
        # Caching
        "enable_api_cache": True,
        "api_cache_ttl": 300,
        
        # Load balancing
        "load_balance_strategy": "health_based",
        
        # CDN
        "enable_cdn": True,
        "cdn_cache_ttl": 3600,
        
        # Backend servers (example configuration)
        "backend_servers": [
            {
                "host": "localhost",
                "port": 8001,
                "weight": 1,
                "max_connections": 100,
                "location": "us-east"
            }
        ]
    }
    
    # Merge configurations
    config = {**default_config, **config_overrides}
    
    # Initialize optimizer
    await optimizer.initialize(
        app=app,
        redis_url=config.get("redis_url"),
        database_url=config.get("database_url"),
        backend_servers=config.get("backend_servers", []),
        **config
    )
    
    return optimizer


# Performance testing utilities
class PerformanceTester:
    """Utilities for performance testing and benchmarking."""
    
    def __init__(self, optimizer: PerformanceOptimizer):
        self.optimizer = optimizer
        
    async def run_cache_benchmark(self, iterations: int = 1000) -> Dict[str, Any]:
        """Run cache performance benchmark."""
        import time
        
        cache_manager = await get_cache_manager("default")
        
        # Write benchmark
        start_time = time.time()
        for i in range(iterations):
            await cache_manager.set(f"bench_key_{i}", f"value_{i}", 300)
        write_time = time.time() - start_time
        
        # Read benchmark
        start_time = time.time()
        for i in range(iterations):
            await cache_manager.get(f"bench_key_{i}")
        read_time = time.time() - start_time
        
        # Cleanup
        keys_to_delete = [f"bench_key_{i}" for i in range(iterations)]
        await cache_manager.delete(*keys_to_delete)
        
        return {
            "iterations": iterations,
            "write_time": write_time,
            "read_time": read_time,
            "writes_per_second": iterations / write_time,
            "reads_per_second": iterations / read_time
        }
        
    async def run_database_benchmark(
        self,
        query: str = "SELECT 1",
        iterations: int = 100
    ) -> Dict[str, Any]:
        """Run database performance benchmark."""
        import time
        
        if not self.optimizer.database_manager:
            return {"error": "Database manager not initialized"}
            
        start_time = time.time()
        for _ in range(iterations):
            await self.optimizer.database_manager.execute_optimized_query(query)
        total_time = time.time() - start_time
        
        return {
            "iterations": iterations,
            "total_time": total_time,
            "queries_per_second": iterations / total_time,
            "avg_query_time": total_time / iterations
        }
        
    async def generate_load_test_data(self) -> Dict[str, Any]:
        """Generate synthetic load test data."""
        return {
            "cache_benchmark": await self.run_cache_benchmark(),
            "database_benchmark": await self.run_database_benchmark(),
            "timestamp": datetime.utcnow().isoformat()
        }