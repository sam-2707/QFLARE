"""
Prometheus metrics collection and export for QFLARE.

This module provides comprehensive metrics collection for system monitoring,
performance tracking, and observability.
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional, List, Callable
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info, Enum,
    CollectorRegistry, multiprocess, generate_latest,
    CONTENT_TYPE_LATEST, REGISTRY
)
from prometheus_client.exposition import MetricsHandler
from functools import wraps
import threading
from datetime import datetime, timezone
import uuid

logger = logging.getLogger(__name__)


class QFLAREMetrics:
    """Centralized metrics collection for QFLARE system."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY
        self._initialize_metrics()
        self._system_monitor_thread = None
        self._monitoring_active = False
    
    def _initialize_metrics(self):
        """Initialize all Prometheus metrics."""
        
        # System Metrics
        self.system_cpu_usage = Gauge(
            'qflare_system_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'qflare_system_memory_usage_bytes',
            'Memory usage in bytes',
            ['type'],  # total, available, used
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'qflare_system_disk_usage_bytes',
            'Disk usage in bytes',
            ['type', 'device'],  # total, used, free
            registry=self.registry
        )
        
        self.system_network_bytes = Counter(
            'qflare_system_network_bytes_total',
            'Network bytes transferred',
            ['direction'],  # sent, received
            registry=self.registry
        )
        
        # Application Metrics
        self.http_requests_total = Counter(
            'qflare_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'qflare_http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        self.websocket_connections = Gauge(
            'qflare_websocket_connections_active',
            'Active WebSocket connections',
            registry=self.registry
        )
        
        # Database Metrics
        self.db_connections = Gauge(
            'qflare_db_connections_active',
            'Active database connections',
            ['pool'],
            registry=self.registry
        )
        
        self.db_query_duration = Histogram(
            'qflare_db_query_duration_seconds',
            'Database query duration in seconds',
            ['operation'],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry
        )
        
        self.db_query_errors = Counter(
            'qflare_db_query_errors_total',
            'Database query errors',
            ['operation', 'error_type'],
            registry=self.registry
        )
        
        # Cache Metrics
        self.cache_operations = Counter(
            'qflare_cache_operations_total',
            'Cache operations',
            ['operation', 'result'],  # get/set/delete, hit/miss/success/error
            registry=self.registry
        )
        
        self.cache_size = Gauge(
            'qflare_cache_size_bytes',
            'Cache size in bytes',
            ['cache_type'],
            registry=self.registry
        )
        
        # Federated Learning Metrics
        self.fl_nodes_total = Gauge(
            'qflare_fl_nodes_total',
            'Total federated learning nodes',
            ['status'],  # online, offline, training, idle, error
            registry=self.registry
        )
        
        self.fl_training_sessions = Gauge(
            'qflare_fl_training_sessions_active',
            'Active training sessions',
            registry=self.registry
        )
        
        self.fl_training_rounds = Counter(
            'qflare_fl_training_rounds_total',
            'Total training rounds completed',
            ['project_id'],
            registry=self.registry
        )
        
        self.fl_model_accuracy = Gauge(
            'qflare_fl_model_accuracy',
            'Model accuracy score',
            ['project_id', 'session_id'],
            registry=self.registry
        )
        
        self.fl_model_loss = Gauge(
            'qflare_fl_model_loss',
            'Model loss value',
            ['project_id', 'session_id'],
            registry=self.registry
        )
        
        self.fl_client_participation = Counter(
            'qflare_fl_client_participation_total',
            'Client participation in training rounds',
            ['client_id', 'project_id'],
            registry=self.registry
        )
        
        # Cryptography Metrics
        self.crypto_operations = Counter(
            'qflare_crypto_operations_total',
            'Cryptographic operations',
            ['operation', 'algorithm'],  # encrypt/decrypt/sign/verify, kyber/dilithium
            registry=self.registry
        )
        
        self.crypto_operation_duration = Histogram(
            'qflare_crypto_operation_duration_seconds',
            'Cryptographic operation duration',
            ['operation', 'algorithm'],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0],
            registry=self.registry
        )
        
        # Privacy Metrics
        self.privacy_budget_used = Gauge(
            'qflare_privacy_budget_used',
            'Privacy budget consumed',
            ['project_id', 'type'],  # epsilon, delta
            registry=self.registry
        )
        
        self.noise_scale = Gauge(
            'qflare_noise_scale',
            'Differential privacy noise scale',
            ['project_id'],
            registry=self.registry
        )
        
        # Security Metrics
        self.auth_attempts = Counter(
            'qflare_auth_attempts_total',
            'Authentication attempts',
            ['result'],  # success, failure, locked
            registry=self.registry
        )
        
        self.security_events = Counter(
            'qflare_security_events_total',
            'Security events',
            ['event_type', 'severity'],  # intrusion_detected/byzantine_client/etc, low/medium/high
            registry=self.registry
        )
        
        self.byzantine_clients_detected = Counter(
            'qflare_byzantine_clients_detected_total',
            'Byzantine clients detected',
            ['project_id', 'detection_method'],
            registry=self.registry
        )
        
        # Error Metrics
        self.application_errors = Counter(
            'qflare_application_errors_total',
            'Application errors',
            ['component', 'error_type'],
            registry=self.registry
        )
        
        # Performance Metrics
        self.response_time_percentiles = Summary(
            'qflare_response_time_seconds',
            'Response time percentiles',
            ['endpoint'],
            registry=self.registry
        )
        
        self.throughput = Gauge(
            'qflare_throughput_operations_per_second',
            'Operations throughput',
            ['component'],
            registry=self.registry
        )
        
        # Business Metrics
        self.active_users = Gauge(
            'qflare_active_users',
            'Active users',
            ['period'],  # 1h, 24h, 7d
            registry=self.registry
        )
        
        self.projects_created = Counter(
            'qflare_projects_created_total',
            'Total projects created',
            registry=self.registry
        )
        
        # Custom Application Info
        self.app_info = Info(
            'qflare_app_info',
            'Application information',
            registry=self.registry
        )
    
    def start_system_monitoring(self, interval: int = 30):
        """Start background system metrics collection."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._system_monitor_thread = threading.Thread(
            target=self._monitor_system_metrics,
            args=(interval,),
            daemon=True
        )
        self._system_monitor_thread.start()
        logger.info(f"System monitoring started with {interval}s interval")
    
    def stop_system_monitoring(self):
        """Stop background system metrics collection."""
        self._monitoring_active = False
        if self._system_monitor_thread:
            self._system_monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _monitor_system_metrics(self, interval: int):
        """Background thread for system metrics collection."""
        while self._monitoring_active:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.system_cpu_usage.set(cpu_percent)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.system_memory_usage.labels(type='total').set(memory.total)
                self.system_memory_usage.labels(type='available').set(memory.available)
                self.system_memory_usage.labels(type='used').set(memory.used)
                
                # Disk metrics
                for partition in psutil.disk_partitions():
                    try:
                        disk_usage = psutil.disk_usage(partition.mountpoint)
                        device = partition.device.replace(':', '_')  # Prometheus label safety
                        self.system_disk_usage.labels(type='total', device=device).set(disk_usage.total)
                        self.system_disk_usage.labels(type='used', device=device).set(disk_usage.used)
                        self.system_disk_usage.labels(type='free', device=device).set(disk_usage.free)
                    except (PermissionError, FileNotFoundError):
                        continue
                
                # Network metrics
                network_io = psutil.net_io_counters()
                if network_io:
                    self.system_network_bytes.labels(direction='sent')._value._value = network_io.bytes_sent
                    self.system_network_bytes.labels(direction='received')._value._value = network_io.bytes_recv
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(interval)
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.http_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_db_query(self, operation: str, duration: float, error: Optional[str] = None):
        """Record database query metrics."""
        self.db_query_duration.labels(operation=operation).observe(duration)
        
        if error:
            self.db_query_errors.labels(
                operation=operation,
                error_type=error
            ).inc()
    
    def record_cache_operation(self, operation: str, result: str):
        """Record cache operation metrics."""
        self.cache_operations.labels(
            operation=operation,
            result=result
        ).inc()
    
    def record_crypto_operation(self, operation: str, algorithm: str, duration: float):
        """Record cryptographic operation metrics."""
        self.crypto_operations.labels(
            operation=operation,
            algorithm=algorithm
        ).inc()
        
        self.crypto_operation_duration.labels(
            operation=operation,
            algorithm=algorithm
        ).observe(duration)
    
    def record_fl_training_round(self, project_id: str, accuracy: float, loss: float, session_id: str):
        """Record federated learning training metrics."""
        self.fl_training_rounds.labels(project_id=project_id).inc()
        self.fl_model_accuracy.labels(project_id=project_id, session_id=session_id).set(accuracy)
        self.fl_model_loss.labels(project_id=project_id, session_id=session_id).set(loss)
    
    def record_client_participation(self, client_id: str, project_id: str):
        """Record client participation in training."""
        self.fl_client_participation.labels(
            client_id=client_id,
            project_id=project_id
        ).inc()
    
    def record_auth_attempt(self, result: str):
        """Record authentication attempt."""
        self.auth_attempts.labels(result=result).inc()
    
    def record_security_event(self, event_type: str, severity: str):
        """Record security event."""
        self.security_events.labels(
            event_type=event_type,
            severity=severity
        ).inc()
    
    def record_byzantine_detection(self, project_id: str, detection_method: str):
        """Record Byzantine client detection."""
        self.byzantine_clients_detected.labels(
            project_id=project_id,
            detection_method=detection_method
        ).inc()
    
    def record_error(self, component: str, error_type: str):
        """Record application error."""
        self.application_errors.labels(
            component=component,
            error_type=error_type
        ).inc()
    
    def set_node_count(self, status: str, count: int):
        """Set federated learning node count by status."""
        self.fl_nodes_total.labels(status=status).set(count)
    
    def set_active_training_sessions(self, count: int):
        """Set active training sessions count."""
        self.fl_training_sessions.set(count)
    
    def set_websocket_connections(self, count: int):
        """Set active WebSocket connections."""
        self.websocket_connections.set(count)
    
    def set_db_connections(self, pool: str, count: int):
        """Set database connection pool size."""
        self.db_connections.labels(pool=pool).set(count)
    
    def set_privacy_budget(self, project_id: str, budget_type: str, value: float):
        """Set privacy budget usage."""
        self.privacy_budget_used.labels(
            project_id=project_id,
            type=budget_type
        ).set(value)
    
    def set_app_info(self, version: str, environment: str, build_time: str):
        """Set application information."""
        self.app_info.info({
            'version': version,
            'environment': environment,
            'build_time': build_time,
            'component': 'qflare'
        })
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return generate_latest(self.registry)


# Decorators for automatic metrics collection

def monitor_http_requests(metrics: QFLAREMetrics):
    """Decorator to monitor HTTP request metrics."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Extract request info (this would be adapted based on framework)
                method = getattr(func, '_method', 'unknown')
                endpoint = getattr(func, '_endpoint', func.__name__)
                status_code = getattr(result, 'status_code', 200)
                
                metrics.record_http_request(method, endpoint, status_code, duration)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                method = getattr(func, '_method', 'unknown')
                endpoint = getattr(func, '_endpoint', func.__name__)
                
                metrics.record_http_request(method, endpoint, 500, duration)
                metrics.record_error('api', type(e).__name__)
                raise
        
        return wrapper
    return decorator


def monitor_database_queries(metrics: QFLAREMetrics):
    """Decorator to monitor database query metrics."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            operation = func.__name__
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.record_db_query(operation, duration)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_db_query(operation, duration, type(e).__name__)
                raise
        
        return wrapper
    return decorator


def monitor_crypto_operations(metrics: QFLAREMetrics, algorithm: str):
    """Decorator to monitor cryptographic operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            operation = func.__name__
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.record_crypto_operation(operation, algorithm, duration)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_crypto_operation(operation, algorithm, duration)
                metrics.record_error('crypto', type(e).__name__)
                raise
        
        return wrapper
    return decorator


# Global metrics instance
qflare_metrics = QFLAREMetrics()


def get_metrics() -> QFLAREMetrics:
    """Get global metrics instance."""
    return qflare_metrics


def initialize_metrics(app_version: str, environment: str):
    """Initialize metrics collection."""
    qflare_metrics.set_app_info(
        version=app_version,
        environment=environment,
        build_time=datetime.now(timezone.utc).isoformat()
    )
    qflare_metrics.start_system_monitoring()
    logger.info("Metrics collection initialized")