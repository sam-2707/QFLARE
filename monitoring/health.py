"""
Health check system for QFLARE with comprehensive monitoring endpoints.

This module provides health check endpoints for various system components
including database connectivity, Redis, external services, and application state.
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass, asdict
import psutil
import aioredis
import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from monitoring.logging import get_logger, log_context
from monitoring.metrics import qflare_metrics
from database.connection import get_async_session


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    response_time_ms: float
    timestamp: str
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["status"] = self.status.value
        return result


class HealthChecker:
    """Base health checker class."""
    
    def __init__(self, name: str, timeout: float = 5.0):
        self.name = name
        self.timeout = timeout
        self.logger = get_logger(f"health.{name}")
    
    async def check(self) -> HealthCheckResult:
        """Perform health check."""
        start_time = time.time()
        
        try:
            with log_context(component="health_check", operation=self.name):
                result = await asyncio.wait_for(
                    self._perform_check(),
                    timeout=self.timeout
                )
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=result.get("status", HealthStatus.HEALTHY),
                message=result.get("message", "OK"),
                response_time_ms=response_time,
                timestamp=datetime.now(timezone.utc).isoformat(),
                details=result.get("details")
            )
            
        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timeout after {self.timeout}s",
                response_time_ms=response_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.logger.error(f"Health check failed: {str(e)}")
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check error: {str(e)}",
                response_time_ms=response_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Override this method to implement specific health check logic."""
        raise NotImplementedError


class DatabaseHealthChecker(HealthChecker):
    """Database connectivity health checker."""
    
    def __init__(self):
        super().__init__("database", timeout=10.0)
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            async with get_async_session() as session:
                # Test basic connectivity
                start_time = time.time()
                result = await session.execute(text("SELECT 1"))
                query_time = (time.time() - start_time) * 1000
                
                # Get database stats
                db_stats = await session.execute(text("""
                    SELECT 
                        count(*) as active_connections
                    FROM pg_stat_activity 
                    WHERE state = 'active'
                """))
                active_connections = db_stats.scalar()
                
                # Check for slow queries
                slow_queries = await session.execute(text("""
                    SELECT count(*) 
                    FROM pg_stat_activity 
                    WHERE state = 'active' AND query_start < NOW() - INTERVAL '30 seconds'
                """))
                slow_query_count = slow_queries.scalar()
                
                status = HealthStatus.HEALTHY
                message = "Database is healthy"
                
                if query_time > 1000:  # > 1 second
                    status = HealthStatus.DEGRADED
                    message = f"Database response slow: {query_time:.1f}ms"
                elif slow_query_count > 5:
                    status = HealthStatus.DEGRADED
                    message = f"High number of slow queries: {slow_query_count}"
                
                return {
                    "status": status,
                    "message": message,
                    "details": {
                        "query_time_ms": query_time,
                        "active_connections": active_connections,
                        "slow_queries": slow_query_count,
                        "connection_successful": True
                    }
                }
                
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Database connection failed: {str(e)}",
                "details": {
                    "connection_successful": False,
                    "error": str(e)
                }
            }


class RedisHealthChecker(HealthChecker):
    """Redis connectivity and performance health checker."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        super().__init__("redis", timeout=5.0)
        self.redis_url = redis_url
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance."""
        redis_client = None
        try:
            redis_client = aioredis.from_url(self.redis_url)
            
            # Test basic connectivity with ping
            start_time = time.time()
            await redis_client.ping()
            ping_time = (time.time() - start_time) * 1000
            
            # Get Redis info
            info = await redis_client.info()
            
            # Test read/write operations
            test_key = f"health_check_{int(time.time())}"
            start_time = time.time()
            await redis_client.set(test_key, "test", ex=60)
            value = await redis_client.get(test_key)
            await redis_client.delete(test_key)
            rw_time = (time.time() - start_time) * 1000
            
            status = HealthStatus.HEALTHY
            message = "Redis is healthy"
            
            if ping_time > 100:  # > 100ms
                status = HealthStatus.DEGRADED
                message = f"Redis response slow: ping {ping_time:.1f}ms"
            elif info.get('used_memory_rss', 0) > info.get('maxmemory', float('inf')) * 0.9:
                status = HealthStatus.DEGRADED
                message = "Redis memory usage high"
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "ping_time_ms": ping_time,
                    "rw_time_ms": rw_time,
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_rss": info.get("used_memory_rss", 0),
                    "maxmemory": info.get("maxmemory", 0),
                    "redis_version": info.get("redis_version", "unknown")
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Redis connection failed: {str(e)}",
                "details": {
                    "connection_successful": False,
                    "error": str(e)
                }
            }
        finally:
            if redis_client:
                await redis_client.close()


class SystemResourcesHealthChecker(HealthChecker):
    """System resources health checker."""
    
    def __init__(self, 
                 cpu_threshold: float = 80.0,
                 memory_threshold: float = 85.0,
                 disk_threshold: float = 90.0):
        super().__init__("system_resources")
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Check system resource utilization."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Load average (Unix/Linux only)
            load_avg = None
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                # Windows doesn't have load average
                pass
            
            # Network stats
            network = psutil.net_io_counters()
            
            # Determine status
            issues = []
            status = HealthStatus.HEALTHY
            
            if cpu_percent > self.cpu_threshold:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
                status = HealthStatus.DEGRADED
            
            if memory_percent > self.memory_threshold:
                issues.append(f"High memory usage: {memory_percent:.1f}%")
                status = HealthStatus.DEGRADED
            
            if disk_percent > self.disk_threshold:
                issues.append(f"High disk usage: {disk_percent:.1f}%")
                status = HealthStatus.UNHEALTHY
            
            message = "System resources healthy" if not issues else "; ".join(issues)
            
            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk_percent,
                "disk_total_gb": disk.total / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv,
            }
            
            if load_avg:
                details.update({
                    "load_avg_1min": load_avg[0],
                    "load_avg_5min": load_avg[1],
                    "load_avg_15min": load_avg[2]
                })
            
            return {
                "status": status,
                "message": message,
                "details": details
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"System resource check failed: {str(e)}",
                "details": {"error": str(e)}
            }


class ExternalServiceHealthChecker(HealthChecker):
    """External service health checker."""
    
    def __init__(self, service_name: str, url: str, expected_status: int = 200):
        super().__init__(f"external_service_{service_name}")
        self.service_name = service_name
        self.url = url
        self.expected_status = expected_status
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Check external service availability."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                start_time = time.time()
                response = await client.get(self.url)
                response_time = (time.time() - start_time) * 1000
                
                status = HealthStatus.HEALTHY
                message = f"{self.service_name} is healthy"
                
                if response.status_code != self.expected_status:
                    status = HealthStatus.UNHEALTHY
                    message = f"{self.service_name} returned status {response.status_code}"
                elif response_time > 5000:  # > 5 seconds
                    status = HealthStatus.DEGRADED
                    message = f"{self.service_name} response slow: {response_time:.1f}ms"
                
                return {
                    "status": status,
                    "message": message,
                    "details": {
                        "response_time_ms": response_time,
                        "status_code": response.status_code,
                        "url": self.url
                    }
                }
                
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"{self.service_name} check failed: {str(e)}",
                "details": {
                    "url": self.url,
                    "error": str(e)
                }
            }


class ApplicationHealthChecker(HealthChecker):
    """Application-specific health checker."""
    
    def __init__(self):
        super().__init__("application")
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Check application-specific health indicators."""
        try:
            # Check if critical services are running
            issues = []
            
            # Check metrics collection
            if not hasattr(qflare_metrics, '_initialized') or not qflare_metrics._initialized:
                issues.append("Metrics collection not initialized")
            
            # Check background tasks (if any)
            # This would check task queues, background processes, etc.
            
            # Check recent errors
            from monitoring.logging import error_tracker
            error_summary = error_tracker.get_error_summary()
            
            if error_summary["total_errors"] > 100:  # High error rate
                issues.append(f"High error rate: {error_summary['total_errors']} errors")
            
            status = HealthStatus.HEALTHY if not issues else HealthStatus.DEGRADED
            message = "Application healthy" if not issues else "; ".join(issues)
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "error_summary": error_summary,
                    "metrics_initialized": hasattr(qflare_metrics, '_initialized') and qflare_metrics._initialized,
                    "uptime_seconds": time.time() - getattr(self, '_start_time', time.time())
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Application health check failed: {str(e)}",
                "details": {"error": str(e)}
            }


class HealthCheckManager:
    """Manages all health checks and provides aggregated results."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.checkers: List[HealthChecker] = []
        self.last_check_time: Optional[datetime] = None
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.check_interval = timedelta(minutes=1)  # Default check interval
    
    def add_checker(self, checker: HealthChecker):
        """Add a health checker."""
        self.checkers.append(checker)
        self.logger.info(f"Added health checker: {checker.name}")
    
    def setup_default_checkers(self, redis_url: str = "redis://localhost:6379"):
        """Setup default health checkers."""
        self.add_checker(DatabaseHealthChecker())
        self.add_checker(RedisHealthChecker(redis_url))
        self.add_checker(SystemResourcesHealthChecker())
        self.add_checker(ApplicationHealthChecker())
    
    async def check_all(self) -> Dict[str, Any]:
        """Run all health checks and return aggregated results."""
        start_time = time.time()
        
        with log_context(component="health_check", operation="check_all"):
            self.logger.info("Starting health check cycle")
            
            # Run all checks concurrently
            tasks = [checker.check() for checker in self.checkers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            check_results = {}
            overall_status = HealthStatus.HEALTHY
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    checker_name = self.checkers[i].name
                    result = HealthCheckResult(
                        name=checker_name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check exception: {str(result)}",
                        response_time_ms=0,
                        timestamp=datetime.now(timezone.utc).isoformat()
                    )
                
                check_results[result.name] = result.to_dict()
                
                # Update overall status
                if result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
            
            # Store results
            self.last_results = {name: HealthCheckResult(**data) for name, data in check_results.items()}
            self.last_check_time = datetime.now(timezone.utc)
            
            total_time = (time.time() - start_time) * 1000
            
            # Log summary
            unhealthy_count = sum(1 for r in self.last_results.values() if r.status == HealthStatus.UNHEALTHY)
            degraded_count = sum(1 for r in self.last_results.values() if r.status == HealthStatus.DEGRADED)
            
            self.logger.info(
                f"Health check completed: {overall_status.value} "
                f"({unhealthy_count} unhealthy, {degraded_count} degraded)",
                extra={
                    "health_check_summary": {
                        "overall_status": overall_status.value,
                        "total_checks": len(check_results),
                        "unhealthy_count": unhealthy_count,
                        "degraded_count": degraded_count,
                        "total_time_ms": total_time
                    }
                }
            )
            
            # Update metrics
            qflare_metrics.health_check_duration.observe(total_time / 1000)
            qflare_metrics.health_check_status.labels(
                status=overall_status.value
            ).set(1)
            
            return {
                "status": overall_status.value,
                "timestamp": self.last_check_time.isoformat(),
                "checks": check_results,
                "summary": {
                    "total_checks": len(check_results),
                    "healthy_count": len(check_results) - unhealthy_count - degraded_count,
                    "degraded_count": degraded_count,
                    "unhealthy_count": unhealthy_count,
                    "total_time_ms": total_time
                }
            }
    
    async def check_single(self, checker_name: str) -> Optional[Dict[str, Any]]:
        """Run a single health check by name."""
        checker = next((c for c in self.checkers if c.name == checker_name), None)
        if not checker:
            return None
        
        result = await checker.check()
        return result.to_dict()
    
    def get_last_results(self) -> Dict[str, Any]:
        """Get the last health check results."""
        if not self.last_check_time:
            return {
                "status": "unknown",
                "message": "No health checks performed yet",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "checks": {}
            }
        
        # Check if results are stale
        if datetime.now(timezone.utc) - self.last_check_time > self.check_interval * 2:
            overall_status = HealthStatus.UNKNOWN
        else:
            unhealthy_count = sum(1 for r in self.last_results.values() if r.status == HealthStatus.UNHEALTHY)
            degraded_count = sum(1 for r in self.last_results.values() if r.status == HealthStatus.DEGRADED)
            
            if unhealthy_count > 0:
                overall_status = HealthStatus.UNHEALTHY
            elif degraded_count > 0:
                overall_status = HealthStatus.DEGRADED
            else:
                overall_status = HealthStatus.HEALTHY
        
        return {
            "status": overall_status.value,
            "timestamp": self.last_check_time.isoformat(),
            "checks": {name: result.to_dict() for name, result in self.last_results.items()},
            "summary": {
                "total_checks": len(self.last_results),
                "healthy_count": len(self.last_results) - sum(1 for r in self.last_results.values() 
                                                             if r.status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]),
                "degraded_count": sum(1 for r in self.last_results.values() if r.status == HealthStatus.DEGRADED),
                "unhealthy_count": sum(1 for r in self.last_results.values() if r.status == HealthStatus.UNHEALTHY)
            }
        }


# Global health check manager
health_manager = HealthCheckManager()


def initialize_health_checks(redis_url: str = "redis://localhost:6379"):
    """Initialize the health check system."""
    health_manager.setup_default_checkers(redis_url)
    logger = get_logger(__name__)
    logger.info("Health check system initialized")


# Background health check task
async def run_background_health_checks():
    """Run health checks in the background periodically."""
    logger = get_logger(__name__)
    
    while True:
        try:
            await health_manager.check_all()
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"Background health check failed: {str(e)}")
            await asyncio.sleep(30)  # Retry after 30 seconds on error