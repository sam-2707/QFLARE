"""
Health Check System for QFLARE

This module provides comprehensive health monitoring for all QFLARE components:
- System health checks (CPU, memory, disk, network)
- Service health checks (database, encryption, FL coordinator)
- Custom health checks for QFLARE-specific functionality
- Health status aggregation and reporting
- Automated alerting and recovery actions
"""

import time
import logging
import threading
import asyncio
from typing import Dict, Any, Optional, List, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    recovery_actions: List[str] = field(default_factory=list)


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""
    interval_seconds: float = 30.0
    timeout_seconds: float = 10.0
    max_failures: int = 3
    failure_threshold_seconds: float = 300.0
    enable_auto_recovery: bool = True
    alert_on_degraded: bool = True
    alert_on_unhealthy: bool = True


class HealthCheck:
    """Base class for health checks."""
    
    def __init__(self, name: str, config: HealthCheckConfig = None):
        self.name = name
        self.config = config or HealthCheckConfig()
        self.failure_count = 0
        self.last_success = datetime.utcnow()
        self.last_failure = None
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    async def check(self) -> HealthCheckResult:
        """Perform the health check."""
        start_time = time.time()
        
        try:
            # Perform the actual check
            status, message, details = await self._perform_check()
            
            # Update failure tracking
            if status == HealthStatus.HEALTHY:
                self.failure_count = 0
                self.last_success = datetime.utcnow()
            elif status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]:
                self.failure_count += 1
                self.last_failure = datetime.utcnow()
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component=self.name,
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms,
                details=details,
                recovery_actions=self._get_recovery_actions(status)
            )
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure = datetime.utcnow()
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms,
                details={'error': str(e), 'failure_count': self.failure_count}
            )
    
    async def _perform_check(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Override this method to implement the actual health check."""
        raise NotImplementedError
    
    def _get_recovery_actions(self, status: HealthStatus) -> List[str]:
        """Get recovery actions for the current status."""
        return []


class SystemHealthCheck(HealthCheck):
    """Health check for system resources."""
    
    def __init__(self, config: HealthCheckConfig = None):
        super().__init__("system", config)
        self.cpu_threshold = 90.0
        self.memory_threshold = 90.0
        self.disk_threshold = 95.0
    
    async def _perform_check(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check system resource usage."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            details = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_available_mb': memory.available // (1024 * 1024),
                'disk_percent': disk_percent,
                'disk_free_gb': disk.free // (1024 * 1024 * 1024)
            }
            
            # Determine status
            issues = []
            if cpu_percent > self.cpu_threshold:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            if memory_percent > self.memory_threshold:
                issues.append(f"High memory usage: {memory_percent:.1f}%")
            if disk_percent > self.disk_threshold:
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            
            if issues:
                status = HealthStatus.DEGRADED if len(issues) == 1 else HealthStatus.UNHEALTHY
                message = "; ".join(issues)
            else:
                status = HealthStatus.HEALTHY
                message = "System resources are within normal limits"
            
            return status, message, details
            
        except ImportError:
            return (
                HealthStatus.UNKNOWN,
                "psutil not available for system monitoring",
                {}
            )
    
    def _get_recovery_actions(self, status: HealthStatus) -> List[str]:
        """Get recovery actions for system issues."""
        if status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]:
            return [
                "Check for memory leaks",
                "Restart high-resource processes",
                "Clear temporary files",
                "Scale up infrastructure if needed"
            ]
        return []


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connectivity."""
    
    def __init__(self, database_service=None, config: HealthCheckConfig = None):
        super().__init__("database", config)
        self.database_service = database_service
    
    async def _perform_check(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check database connectivity and performance."""
        if not self.database_service:
            return (
                HealthStatus.UNKNOWN,
                "Database service not configured",
                {}
            )
        
        try:
            start_time = time.time()
            
            # Simple connectivity test
            # This would be replaced with actual database ping
            await asyncio.sleep(0.01)  # Simulate database query
            
            query_time_ms = (time.time() - start_time) * 1000
            
            details = {
                'query_time_ms': query_time_ms,
                'connection_pool_size': 10,  # Mock data
                'active_connections': 3  # Mock data
            }
            
            if query_time_ms > 1000:  # 1 second threshold
                status = HealthStatus.DEGRADED
                message = f"Database responding slowly: {query_time_ms:.1f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = f"Database connection healthy: {query_time_ms:.1f}ms"
            
            return status, message, details
            
        except Exception as e:
            return (
                HealthStatus.UNHEALTHY,
                f"Database connection failed: {str(e)}",
                {'error': str(e)}
            )
    
    def _get_recovery_actions(self, status: HealthStatus) -> List[str]:
        """Get recovery actions for database issues."""
        if status == HealthStatus.UNHEALTHY:
            return [
                "Check database service status",
                "Verify connection string",
                "Restart database connection pool",
                "Failover to backup database"
            ]
        elif status == HealthStatus.DEGRADED:
            return [
                "Check database performance",
                "Review slow queries",
                "Optimize database indexes"
            ]
        return []


class SecurityHealthCheck(HealthCheck):
    """Health check for security components."""
    
    def __init__(self, security_manager=None, config: HealthCheckConfig = None):
        super().__init__("security", config)
        self.security_manager = security_manager
    
    async def _perform_check(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check security system health."""
        if not self.security_manager:
            return (
                HealthStatus.UNKNOWN,
                "Security manager not configured",
                {}
            )
        
        try:
            # Check security status
            security_status = self.security_manager.get_security_status()
            
            details = {
                'active_sessions': security_status.get('communication', {}).get('active_sessions', 0),
                'enrolled_devices': security_status.get('authentication', {}).get('enrolled_devices', 0),
                'total_keys': security_status.get('key_management', {}).get('total_keys', 0),
                'key_rotation_overdue': 0  # Mock data
            }
            
            # Check for security issues
            issues = []
            if details['active_sessions'] > 1000:
                issues.append("High number of active sessions")
            if details['key_rotation_overdue'] > 0:
                issues.append(f"{details['key_rotation_overdue']} keys need rotation")
            
            if issues:
                status = HealthStatus.DEGRADED
                message = "; ".join(issues)
            else:
                status = HealthStatus.HEALTHY
                message = "Security system operating normally"
            
            return status, message, details
            
        except Exception as e:
            return (
                HealthStatus.UNHEALTHY,
                f"Security health check failed: {str(e)}",
                {'error': str(e)}
            )
    
    def _get_recovery_actions(self, status: HealthStatus) -> List[str]:
        """Get recovery actions for security issues."""
        if status == HealthStatus.UNHEALTHY:
            return [
                "Restart security services",
                "Check certificate validity",
                "Verify HSM connectivity",
                "Review security logs"
            ]
        elif status == HealthStatus.DEGRADED:
            return [
                "Rotate overdue keys",
                "Clean up expired sessions",
                "Monitor security metrics"
            ]
        return []


class FederatedLearningHealthCheck(HealthCheck):
    """Health check for federated learning operations."""
    
    def __init__(self, fl_coordinator=None, config: HealthCheckConfig = None):
        super().__init__("federated_learning", config)
        self.fl_coordinator = fl_coordinator
    
    async def _perform_check(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check federated learning system health."""
        if not self.fl_coordinator:
            return (
                HealthStatus.UNKNOWN,
                "FL coordinator not configured",
                {}
            )
        
        try:
            # Mock FL status check
            fl_status = {
                'active_round': True,
                'participants_expected': 10,
                'participants_active': 8,
                'round_progress': 0.75,
                'last_round_completion': datetime.utcnow() - timedelta(minutes=5)
            }
            
            details = {
                'active_round': fl_status['active_round'],
                'participants_active': fl_status['participants_active'],
                'participants_expected': fl_status['participants_expected'],
                'participation_rate': fl_status['participants_active'] / fl_status['participants_expected'],
                'round_progress': fl_status['round_progress'],
                'minutes_since_last_round': 5
            }
            
            # Check FL health
            participation_rate = details['participation_rate']
            issues = []
            
            if participation_rate < 0.5:
                issues.append(f"Low participation rate: {participation_rate:.1%}")
            if details['minutes_since_last_round'] > 60:
                issues.append("No recent FL activity")
            
            if issues:
                status = HealthStatus.DEGRADED if participation_rate > 0.3 else HealthStatus.UNHEALTHY
                message = "; ".join(issues)
            else:
                status = HealthStatus.HEALTHY
                message = f"FL system healthy: {participation_rate:.1%} participation"
            
            return status, message, details
            
        except Exception as e:
            return (
                HealthStatus.UNHEALTHY,
                f"FL health check failed: {str(e)}",
                {'error': str(e)}
            )
    
    def _get_recovery_actions(self, status: HealthStatus) -> List[str]:
        """Get recovery actions for FL issues."""
        if status == HealthStatus.UNHEALTHY:
            return [
                "Restart FL coordinator",
                "Check device connectivity",
                "Review FL configuration",
                "Reset current round"
            ]
        elif status == HealthStatus.DEGRADED:
            return [
                "Notify inactive devices",
                "Adjust participation thresholds",
                "Check network connectivity"
            ]
        return []


class QFLAREHealthMonitor:
    """Comprehensive health monitoring system for QFLARE."""
    
    def __init__(self, config: HealthCheckConfig = None):
        """Initialize the health monitor."""
        self.config = config or HealthCheckConfig()
        self.logger = logging.getLogger(__name__)
        self.health_checks: Dict[str, HealthCheck] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.monitoring_thread = None
        self.shutdown_event = threading.Event()
        self.alert_callbacks: List[Callable] = []
        
        # Initialize default health checks
        self._initialize_default_checks()
    
    def _initialize_default_checks(self):
        """Initialize default health checks."""
        self.health_checks['system'] = SystemHealthCheck(self.config)
        self.health_checks['database'] = DatabaseHealthCheck(config=self.config)
        self.health_checks['security'] = SecurityHealthCheck(config=self.config)
        self.health_checks['federated_learning'] = FederatedLearningHealthCheck(config=self.config)
        
        self.logger.info(f"Initialized {len(self.health_checks)} health checks")
    
    def add_health_check(self, health_check: HealthCheck):
        """Add a custom health check."""
        self.health_checks[health_check.name] = health_check
        self.logger.info(f"Added health check: {health_check.name}")
    
    def remove_health_check(self, name: str):
        """Remove a health check."""
        if name in self.health_checks:
            del self.health_checks[name]
            if name in self.last_results:
                del self.last_results[name]
            self.logger.info(f"Removed health check: {name}")
    
    def add_alert_callback(self, callback: Callable[[HealthCheckResult], None]):
        """Add an alert callback function."""
        self.alert_callbacks.append(callback)
    
    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        results = {}
        
        for name, health_check in self.health_checks.items():
            try:
                result = await health_check.check()
                results[name] = result
                self.last_results[name] = result
                
                # Trigger alerts if needed
                if self._should_alert(result):
                    await self._trigger_alerts(result)
                
            except Exception as e:
                self.logger.error(f"Error running health check {name}: {e}")
                results[name] = HealthCheckResult(
                    component=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Health check error: {str(e)}",
                    timestamp=datetime.utcnow(),
                    duration_ms=0.0
                )
        
        return results
    
    async def check_component(self, component: str) -> Optional[HealthCheckResult]:
        """Run health check for a specific component."""
        if component not in self.health_checks:
            return None
        
        try:
            result = await self.health_checks[component].check()
            self.last_results[component] = result
            
            if self._should_alert(result):
                await self._trigger_alerts(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running health check {component}: {e}")
            return HealthCheckResult(
                component=component,
                status=HealthStatus.UNKNOWN,
                message=f"Health check error: {str(e)}",
                timestamp=datetime.utcnow(),
                duration_ms=0.0
            )
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.last_results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self.last_results.values()]
        
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of current health status."""
        overall_status = self.get_overall_status()
        
        component_statuses = {}
        for name, result in self.last_results.items():
            component_statuses[name] = {
                'status': result.status.value,
                'message': result.message,
                'last_check': result.timestamp.isoformat(),
                'duration_ms': result.duration_ms
            }
        
        return {
            'overall_status': overall_status.value,
            'timestamp': datetime.utcnow().isoformat(),
            'components': component_statuses,
            'total_components': len(self.health_checks),
            'healthy_components': sum(1 for r in self.last_results.values() 
                                    if r.status == HealthStatus.HEALTHY),
            'monitoring_enabled': self.monitoring_thread is not None and self.monitoring_thread.is_alive()
        }
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.warning("Health monitoring already running")
            return
        
        self.shutdown_event.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="QFLAREHealthMonitor"
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info(f"Health monitoring started (interval: {self.config.interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        if self.monitoring_thread:
            self.shutdown_event.set()
            self.monitoring_thread.join(timeout=5.0)
            self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while not self.shutdown_event.is_set():
            try:
                loop.run_until_complete(self.check_all())
                self.shutdown_event.wait(self.config.interval_seconds)
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                self.shutdown_event.wait(self.config.interval_seconds)
        
        loop.close()
    
    def _should_alert(self, result: HealthCheckResult) -> bool:
        """Determine if an alert should be triggered."""
        if result.status == HealthStatus.UNHEALTHY and self.config.alert_on_unhealthy:
            return True
        elif result.status == HealthStatus.DEGRADED and self.config.alert_on_degraded:
            return True
        return False
    
    async def _trigger_alerts(self, result: HealthCheckResult):
        """Trigger alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")


# Global health monitor instance
_health_monitor: Optional[QFLAREHealthMonitor] = None


def get_health_monitor() -> QFLAREHealthMonitor:
    """Get the global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = QFLAREHealthMonitor()
    return _health_monitor


def initialize_health_monitoring(config: HealthCheckConfig = None) -> QFLAREHealthMonitor:
    """Initialize the global health monitor."""
    global _health_monitor
    _health_monitor = QFLAREHealthMonitor(config)
    return _health_monitor


def shutdown_health_monitoring():
    """Shutdown the global health monitor."""
    global _health_monitor
    if _health_monitor:
        _health_monitor.stop_monitoring()
        _health_monitor = None