# QFLARE Database Performance Optimization

"""
Advanced database optimization module with query optimization,
connection pooling, and performance monitoring.
"""

import time
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import text
from sqlalchemy import event
from sqlalchemy.engine.events import PoolEvents
import asyncpg

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration with performance tuning parameters."""
    url: str
    min_size: int = 5
    max_size: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    pool_pre_ping: bool = True
    statement_cache_size: int = 100
    query_timeout: int = 30
    slow_query_threshold: float = 1.0
    enable_query_logging: bool = False
    enable_performance_monitoring: bool = True


@dataclass
class QueryMetrics:
    """Query performance metrics."""
    query_hash: str
    query_text: str
    execution_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_executed: Optional[datetime] = None
    errors: int = 0


class QueryOptimizer:
    """Advanced query optimization and performance monitoring."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.query_metrics: Dict[str, QueryMetrics] = {}
        self.slow_queries: List[Dict[str, Any]] = []
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "pool_hits": 0,
            "pool_misses": 0,
            "connection_errors": 0
        }
        
    def _get_query_hash(self, query: str) -> str:
        """Generate hash for query normalization."""
        import hashlib
        # Normalize query by removing extra whitespace and parameters
        normalized = ' '.join(query.strip().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
        
    def record_query_execution(
        self,
        query: str,
        execution_time: float,
        success: bool = True
    ):
        """Record query execution metrics."""
        query_hash = self._get_query_hash(query)
        
        if query_hash not in self.query_metrics:
            self.query_metrics[query_hash] = QueryMetrics(
                query_hash=query_hash,
                query_text=query[:200] + "..." if len(query) > 200 else query
            )
            
        metrics = self.query_metrics[query_hash]
        
        if success:
            metrics.execution_count += 1
            metrics.total_time += execution_time
            metrics.min_time = min(metrics.min_time, execution_time)
            metrics.max_time = max(metrics.max_time, execution_time)
            metrics.avg_time = metrics.total_time / metrics.execution_count
            metrics.last_executed = datetime.utcnow()
            
            # Record slow queries
            if execution_time > self.config.slow_query_threshold:
                slow_query = {
                    "query": query[:500],
                    "execution_time": execution_time,
                    "timestamp": datetime.utcnow().isoformat(),
                    "query_hash": query_hash
                }
                self.slow_queries.append(slow_query)
                
                # Keep only last 100 slow queries
                if len(self.slow_queries) > 100:
                    self.slow_queries = self.slow_queries[-100:]
                    
        else:
            metrics.errors += 1
            
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        total_queries = sum(m.execution_count for m in self.query_metrics.values())
        total_errors = sum(m.errors for m in self.query_metrics.values())
        
        # Top slow queries
        slow_query_metrics = [
            {
                "query_hash": m.query_hash,
                "query": m.query_text,
                "avg_time": m.avg_time,
                "max_time": m.max_time,
                "execution_count": m.execution_count,
                "total_time": m.total_time
            }
            for m in sorted(
                self.query_metrics.values(),
                key=lambda x: x.avg_time,
                reverse=True
            )[:10]
        ]
        
        # Most frequent queries
        frequent_queries = [
            {
                "query_hash": m.query_hash,
                "query": m.query_text,
                "execution_count": m.execution_count,
                "avg_time": m.avg_time,
                "total_time": m.total_time
            }
            for m in sorted(
                self.query_metrics.values(),
                key=lambda x: x.execution_count,
                reverse=True
            )[:10]
        ]
        
        return {
            "summary": {
                "total_queries": total_queries,
                "unique_queries": len(self.query_metrics),
                "total_errors": total_errors,
                "error_rate": total_errors / max(total_queries, 1),
                "slow_queries_count": len(self.slow_queries),
                "avg_query_time": sum(m.avg_time for m in self.query_metrics.values()) / max(len(self.query_metrics), 1)
            },
            "connection_stats": self.connection_stats,
            "slowest_queries": slow_query_metrics,
            "most_frequent_queries": frequent_queries,
            "recent_slow_queries": self.slow_queries[-10:],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    def suggest_optimizations(self) -> List[Dict[str, str]]:
        """Analyze metrics and suggest optimizations."""
        suggestions = []
        
        # Check for slow queries
        slow_queries = [m for m in self.query_metrics.values() if m.avg_time > self.config.slow_query_threshold]
        if slow_queries:
            suggestions.append({
                "type": "slow_queries",
                "description": f"Found {len(slow_queries)} queries with average execution time > {self.config.slow_query_threshold}s",
                "recommendation": "Consider adding indexes, optimizing WHERE clauses, or using query caching"
            })
            
        # Check for high error rates
        error_rate = sum(m.errors for m in self.query_metrics.values()) / max(sum(m.execution_count for m in self.query_metrics.values()), 1)
        if error_rate > 0.01:  # > 1%
            suggestions.append({
                "type": "high_error_rate",
                "description": f"Query error rate is {error_rate:.2%}",
                "recommendation": "Review failed queries for syntax errors, constraint violations, or connection issues"
            })
            
        # Check for connection pool issues
        pool_hit_rate = self.connection_stats["pool_hits"] / max(
            self.connection_stats["pool_hits"] + self.connection_stats["pool_misses"], 1
        )
        if pool_hit_rate < 0.9:
            suggestions.append({
                "type": "connection_pool",
                "description": f"Connection pool hit rate is {pool_hit_rate:.2%}",
                "recommendation": "Consider increasing pool size or connection timeout"
            })
            
        # Check for frequently executed queries
        frequent_queries = [m for m in self.query_metrics.values() if m.execution_count > 100]
        if frequent_queries:
            suggestions.append({
                "type": "cache_opportunities",
                "description": f"Found {len(frequent_queries)} frequently executed queries",
                "recommendation": "Consider implementing query result caching for these queries"
            })
            
        return suggestions


class OptimizedDatabaseManager:
    """Database manager with advanced performance optimization."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[sessionmaker] = None
        self.optimizer = QueryOptimizer(config)
        self._connection_pool_stats = {}
        
    async def initialize(self):
        """Initialize database with optimized settings."""
        # Create engine with performance optimizations
        self.engine = create_async_engine(
            self.config.url,
            poolclass=QueuePool,
            pool_size=self.config.min_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            pool_pre_ping=self.config.pool_pre_ping,
            connect_args={
                "statement_cache_size": self.config.statement_cache_size,
                "server_settings": {
                    "application_name": "qflare_optimized",
                    "jit": "off",  # Disable JIT for smaller queries
                }
            },
            echo=self.config.enable_query_logging,
            future=True
        )
        
        # Create session factory
        self.session_factory = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Setup event listeners for performance monitoring
        self._setup_event_listeners()
        
        # Apply database optimizations
        await self._apply_database_optimizations()
        
        logger.info("Optimized database manager initialized")
        
    def _setup_event_listeners(self):
        """Setup SQLAlchemy event listeners for monitoring."""
        if not self.config.enable_performance_monitoring:
            return
            
        @event.listens_for(self.engine.sync_engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()
            
        @event.listens_for(self.engine.sync_engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            execution_time = time.time() - context._query_start_time
            self.optimizer.record_query_execution(statement, execution_time, True)
            
        @event.listens_for(self.engine.sync_engine, "dbapi_error")
        def dbapi_error(exception_context):
            if hasattr(exception_context, 'statement'):
                self.optimizer.record_query_execution(
                    exception_context.statement, 0, False
                )
                
        @event.listens_for(self.engine.sync_engine.pool, "connect")
        def on_connect(dbapi_conn, connection_record):
            self.optimizer.connection_stats["total_connections"] += 1
            self.optimizer.connection_stats["active_connections"] += 1
            
        @event.listens_for(self.engine.sync_engine.pool, "checkout")
        def on_checkout(dbapi_conn, connection_record, connection_proxy):
            self.optimizer.connection_stats["pool_hits"] += 1
            
        @event.listens_for(self.engine.sync_engine.pool, "invalidate")
        def on_invalidate(dbapi_conn, connection_record, exception):
            self.optimizer.connection_stats["connection_errors"] += 1
            
    async def _apply_database_optimizations(self):
        """Apply PostgreSQL-specific optimizations."""
        if not self.engine:
            return
            
        optimizations = [
            # Connection-level optimizations
            "SET statement_timeout = '30s'",
            "SET lock_timeout = '10s'", 
            "SET idle_in_transaction_session_timeout = '5min'",
            
            # Query planner optimizations
            "SET random_page_cost = 1.1",  # SSD-optimized
            "SET effective_cache_size = '1GB'",
            "SET shared_buffers = '256MB'",
            "SET work_mem = '32MB'",
            
            # Performance optimizations
            "SET enable_seqscan = on",
            "SET enable_indexscan = on",
            "SET enable_bitmapscan = on",
            "SET enable_hashjoin = on",
            "SET enable_mergejoin = on",
            "SET enable_nestloop = on",
            
            # Logging optimizations
            "SET log_min_duration_statement = 1000",  # Log slow queries
        ]
        
        try:
            async with self.session_factory() as session:
                for optimization in optimizations:
                    try:
                        await session.execute(text(optimization))
                    except Exception as e:
                        logger.warning(f"Could not apply optimization '{optimization}': {e}")
                        
        except Exception as e:
            logger.error(f"Failed to apply database optimizations: {e}")
            
    @asynccontextmanager
    async def get_session(self):
        """Get database session with automatic cleanup."""
        if not self.session_factory:
            raise RuntimeError("Database not initialized")
            
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
            
    async def execute_optimized_query(
        self,
        query: str,
        params: Dict[str, Any] = None,
        use_cache: bool = False,
        cache_ttl: int = 300
    ) -> List[Dict[str, Any]]:
        """Execute query with optimization and optional caching."""
        start_time = time.time()
        
        try:
            # Check cache if enabled
            if use_cache:
                from cache.cache_manager import get_cache_manager
                cache_manager = await get_cache_manager("query")
                
                cached_result = await cache_manager.get_cached_query(query, params or {})
                if cached_result is not None:
                    return cached_result
                    
            # Execute query
            async with self.get_session() as session:
                result = await session.execute(text(query), params or {})
                rows = result.fetchall()
                
                # Convert to list of dicts
                columns = result.keys()
                data = [dict(zip(columns, row)) for row in rows]
                
                # Cache result if enabled
                if use_cache:
                    await cache_manager.cache_query_result(
                        query, params or {}, data, cache_ttl
                    )
                    
                return data
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.optimizer.record_query_execution(query, execution_time, False)
            raise
            
    async def bulk_insert_optimized(
        self,
        table_name: str,
        data: List[Dict[str, Any]],
        batch_size: int = 1000,
        on_conflict: str = "IGNORE"
    ) -> int:
        """Optimized bulk insert using PostgreSQL COPY."""
        if not data:
            return 0
            
        inserted_count = 0
        
        try:
            async with self.get_session() as session:
                # Process in batches for memory efficiency
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    
                    # Build bulk insert query
                    columns = list(batch[0].keys())
                    placeholders = ", ".join(f":{col}" for col in columns)
                    
                    if on_conflict == "IGNORE":
                        query = f"""
                        INSERT INTO {table_name} ({', '.join(columns)})
                        VALUES ({placeholders})
                        ON CONFLICT DO NOTHING
                        """
                    elif on_conflict == "UPDATE":
                        update_clause = ", ".join(f"{col} = EXCLUDED.{col}" for col in columns[1:])
                        query = f"""
                        INSERT INTO {table_name} ({', '.join(columns)})
                        VALUES ({placeholders})
                        ON CONFLICT ({columns[0]}) DO UPDATE SET {update_clause}
                        """
                    else:
                        query = f"""
                        INSERT INTO {table_name} ({', '.join(columns)})
                        VALUES ({placeholders})
                        """
                        
                    # Execute batch insert
                    await session.execute(text(query), batch)
                    inserted_count += len(batch)
                    
                await session.commit()
                
        except Exception as e:
            logger.error(f"Bulk insert failed: {e}")
            raise
            
        return inserted_count
        
    async def vacuum_analyze_tables(self, table_names: Optional[List[str]] = None):
        """Perform vacuum and analyze on tables for optimization."""
        try:
            async with self.get_session() as session:
                if table_names:
                    for table in table_names:
                        await session.execute(text(f"VACUUM ANALYZE {table}"))
                else:
                    await session.execute(text("VACUUM ANALYZE"))
                    
                await session.commit()
                logger.info("Database vacuum and analyze completed")
                
        except Exception as e:
            logger.error(f"Vacuum analyze failed: {e}")
            
    async def get_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """Get detailed statistics for a table."""
        stats_query = f"""
        SELECT 
            schemaname,
            tablename,
            attname as column_name,
            n_distinct,
            correlation,
            most_common_vals,
            most_common_freqs,
            histogram_bounds
        FROM pg_stats 
        WHERE tablename = '{table_name}'
        """
        
        size_query = f"""
        SELECT 
            pg_size_pretty(pg_total_relation_size('{table_name}')) as total_size,
            pg_size_pretty(pg_relation_size('{table_name}')) as table_size,
            pg_size_pretty(pg_total_relation_size('{table_name}') - pg_relation_size('{table_name}')) as index_size
        """
        
        try:
            stats = await self.execute_optimized_query(stats_query)
            size_info = await self.execute_optimized_query(size_query)
            
            return {
                "column_statistics": stats,
                "size_information": size_info[0] if size_info else {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get table statistics for {table_name}: {e}")
            return {}
            
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive database performance metrics."""
        return self.optimizer.get_performance_report()
        
    async def get_optimization_suggestions(self) -> List[Dict[str, str]]:
        """Get optimization suggestions based on current metrics."""
        return self.optimizer.suggest_optimizations()
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        try:
            # Test basic connectivity
            start_time = time.time()
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
                
            connection_time = time.time() - start_time
            health_status["checks"]["connectivity"] = {
                "status": "healthy",
                "response_time": connection_time
            }
            
            # Check active connections
            connection_query = """
            SELECT count(*) as active_connections
            FROM pg_stat_activity
            WHERE state = 'active'
            """
            
            result = await self.execute_optimized_query(connection_query)
            active_connections = result[0]["active_connections"] if result else 0
            
            health_status["checks"]["connections"] = {
                "status": "healthy" if active_connections < self.config.max_size * 0.8 else "warning",
                "active_connections": active_connections,
                "max_connections": self.config.max_size
            }
            
            # Check for long-running queries
            long_queries_query = """
            SELECT count(*) as long_running_queries
            FROM pg_stat_activity
            WHERE state = 'active' AND query_start < now() - interval '5 minutes'
            """
            
            result = await self.execute_optimized_query(long_queries_query)
            long_queries = result[0]["long_running_queries"] if result else 0
            
            health_status["checks"]["long_queries"] = {
                "status": "healthy" if long_queries == 0 else "warning",
                "count": long_queries
            }
            
            # Overall health determination
            warning_checks = [
                check for check in health_status["checks"].values()
                if check["status"] == "warning"
            ]
            
            if warning_checks:
                health_status["status"] = "warning"
                
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
            
        return health_status
        
    async def close(self):
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
_db_manager: Optional[OptimizedDatabaseManager] = None


async def get_database_manager() -> OptimizedDatabaseManager:
    """Get database manager instance."""
    global _db_manager
    if _db_manager is None:
        raise RuntimeError("Database manager not initialized")
    return _db_manager


async def initialize_database_manager(config: DatabaseConfig):
    """Initialize global database manager."""
    global _db_manager
    _db_manager = OptimizedDatabaseManager(config)
    await _db_manager.initialize()
    logger.info("Global database manager initialized")