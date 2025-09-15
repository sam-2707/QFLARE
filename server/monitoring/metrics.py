"""
Prometheus Metrics Collection for QFLARE

This module provides comprehensive metrics collection for federated learning operations:
- Training metrics (accuracy, loss, convergence)
- System performance metrics (CPU, memory, network)
- Security metrics (authentication, key operations)
- FL-specific metrics (round completion, participant tracking)
- Custom business metrics for QFLARE operations
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
import json

# Prometheus client imports
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info, Enum,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server, push_to_gateway
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client not available, using mock implementations")

logger = logging.getLogger(__name__)


@dataclass
class MetricConfig:
    """Configuration for metric collection."""
    enabled: bool = True
    collection_interval: float = 10.0  # seconds
    retention_period: int = 7200  # seconds (2 hours)
    export_port: int = 8000
    pushgateway_url: Optional[str] = None
    custom_labels: Dict[str, str] = None

    def __post_init__(self):
        if self.custom_labels is None:
            self.custom_labels = {}


class QFLAREMetricsCollector:
    """Comprehensive metrics collector for QFLARE federated learning system."""
    
    def __init__(self, config: MetricConfig = None):
        """Initialize the metrics collector."""
        self.config = config or MetricConfig()
        self.logger = logging.getLogger(__name__)
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self._metrics = {}
        self._custom_metrics = {}
        self._collection_thread = None
        self._shutdown_event = threading.Event()
        
        # Initialize metrics
        self._initialize_metrics()
        
        # Start collection if enabled
        if self.config.enabled:
            self.start_collection()
    
    def _initialize_metrics(self):
        """Initialize all QFLARE metrics."""
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("Prometheus not available, metrics will be logged only")
            return
        
        # Federated Learning Metrics
        self._metrics['fl_round_counter'] = Counter(
            'qflare_fl_round_total',
            'Total number of FL rounds completed',
            ['status', 'algorithm'],
            registry=self.registry
        )
        
        self._metrics['fl_round_duration'] = Histogram(
            'qflare_fl_round_duration_seconds',
            'Duration of FL rounds in seconds',
            ['algorithm', 'participants'],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600],
            registry=self.registry
        )
        
        self._metrics['fl_participant_count'] = Gauge(
            'qflare_fl_participants_current',
            'Current number of active FL participants',
            ['round_id'],
            registry=self.registry
        )
        
        self._metrics['fl_model_accuracy'] = Gauge(
            'qflare_fl_model_accuracy',
            'Current federated model accuracy',
            ['dataset', 'metric_type'],
            registry=self.registry
        )
        
        self._metrics['fl_model_loss'] = Gauge(
            'qflare_fl_model_loss',
            'Current federated model loss',
            ['dataset', 'loss_type'],
            registry=self.registry
        )
        
        # Training Metrics
        self._metrics['training_iterations'] = Counter(
            'qflare_training_iterations_total',
            'Total number of training iterations',
            ['device_id', 'model_type'],
            registry=self.registry
        )
        
        self._metrics['training_duration'] = Histogram(
            'qflare_training_duration_seconds',
            'Duration of local training in seconds',
            ['device_id', 'model_type'],
            buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 120],
            registry=self.registry
        )
        
        self._metrics['model_size'] = Histogram(
            'qflare_model_size_bytes',
            'Size of model updates in bytes',
            ['compression', 'device_type'],
            buckets=[1024, 10240, 102400, 1048576, 10485760, 104857600],
            registry=self.registry
        )
        
        # Security Metrics
        self._metrics['auth_attempts'] = Counter(
            'qflare_auth_attempts_total',
            'Total authentication attempts',
            ['method', 'status', 'device_type'],
            registry=self.registry
        )
        
        self._metrics['key_operations'] = Counter(
            'qflare_key_operations_total',
            'Total cryptographic key operations',
            ['operation', 'key_type', 'status'],
            registry=self.registry
        )
        
        self._metrics['session_duration'] = Histogram(
            'qflare_session_duration_seconds',
            'Duration of secure sessions',
            ['session_type'],
            buckets=[60, 300, 600, 1800, 3600, 7200, 14400],
            registry=self.registry
        )
        
        # System Performance Metrics
        self._metrics['cpu_usage'] = Gauge(
            'qflare_cpu_usage_percent',
            'CPU usage percentage',
            ['core', 'process'],
            registry=self.registry
        )
        
        self._metrics['memory_usage'] = Gauge(
            'qflare_memory_usage_bytes',
            'Memory usage in bytes',
            ['type', 'process'],
            registry=self.registry
        )
        
        self._metrics['network_bytes'] = Counter(
            'qflare_network_bytes_total',
            'Total network bytes transferred',
            ['direction', 'protocol', 'device_id'],
            registry=self.registry
        )
        
        self._metrics['disk_usage'] = Gauge(
            'qflare_disk_usage_bytes',
            'Disk usage in bytes',
            ['path', 'type'],
            registry=self.registry
        )
        
        # API and Communication Metrics
        self._metrics['api_requests'] = Counter(
            'qflare_api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self._metrics['api_duration'] = Histogram(
            'qflare_api_duration_seconds',
            'API request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
            registry=self.registry
        )
        
        self._metrics['websocket_connections'] = Gauge(
            'qflare_websocket_connections_current',
            'Current WebSocket connections',
            ['type'],
            registry=self.registry
        )
        
        # Error and Health Metrics
        self._metrics['errors'] = Counter(
            'qflare_errors_total',
            'Total errors by category',
            ['component', 'error_type', 'severity'],
            registry=self.registry
        )
        
        self._metrics['health_status'] = Enum(
            'qflare_health_status',
            'Health status of QFLARE components',
            ['component'],
            states=['healthy', 'degraded', 'unhealthy'],
            registry=self.registry
        )
        
        # Business Metrics
        self._metrics['devices_enrolled'] = Gauge(
            'qflare_devices_enrolled_total',
            'Total number of enrolled devices',
            ['device_type', 'status'],
            registry=self.registry
        )
        
        self._metrics['data_quality_score'] = Gauge(
            'qflare_data_quality_score',
            'Data quality score for training data',
            ['device_id', 'dataset'],
            registry=self.registry
        )
        
        self.logger.info(f"Initialized {len(self._metrics)} metric collectors")
    
    def record_fl_round(self, round_id: str, duration: float, participants: int, 
                       status: str = 'completed', algorithm: str = 'fedavg'):
        """Record federated learning round metrics."""
        if PROMETHEUS_AVAILABLE and 'fl_round_counter' in self._metrics:
            self._metrics['fl_round_counter'].labels(
                status=status, algorithm=algorithm
            ).inc()
            
            self._metrics['fl_round_duration'].labels(
                algorithm=algorithm, participants=str(participants)
            ).observe(duration)
            
            self._metrics['fl_participant_count'].labels(
                round_id=round_id
            ).set(participants)
        
        # Log for non-Prometheus environments
        self.logger.info(f"FL Round {round_id}: {duration:.2f}s, {participants} participants, {status}")
    
    def record_model_metrics(self, accuracy: float, loss: float, 
                           dataset: str = 'global', metric_type: str = 'test'):
        """Record model performance metrics."""
        if PROMETHEUS_AVAILABLE and 'fl_model_accuracy' in self._metrics:
            self._metrics['fl_model_accuracy'].labels(
                dataset=dataset, metric_type=metric_type
            ).set(accuracy)
            
            self._metrics['fl_model_loss'].labels(
                dataset=dataset, loss_type=metric_type
            ).set(loss)
        
        self.logger.info(f"Model {metric_type} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
    
    def record_training_metrics(self, device_id: str, duration: float, 
                              iterations: int, model_type: str = 'cnn'):
        """Record local training metrics."""
        if PROMETHEUS_AVAILABLE and 'training_iterations' in self._metrics:
            self._metrics['training_iterations'].labels(
                device_id=device_id, model_type=model_type
            ).inc(iterations)
            
            self._metrics['training_duration'].labels(
                device_id=device_id, model_type=model_type
            ).observe(duration)
        
        self.logger.info(f"Training {device_id}: {iterations} iterations in {duration:.2f}s")
    
    def record_auth_attempt(self, method: str, status: str, device_type: str = 'edge'):
        """Record authentication attempt."""
        if PROMETHEUS_AVAILABLE and 'auth_attempts' in self._metrics:
            self._metrics['auth_attempts'].labels(
                method=method, status=status, device_type=device_type
            ).inc()
        
        self.logger.info(f"Auth attempt: {method} - {status}")
    
    def record_key_operation(self, operation: str, key_type: str, status: str = 'success'):
        """Record cryptographic key operation."""
        if PROMETHEUS_AVAILABLE and 'key_operations' in self._metrics:
            self._metrics['key_operations'].labels(
                operation=operation, key_type=key_type, status=status
            ).inc()
        
        self.logger.debug(f"Key operation: {operation} ({key_type}) - {status}")
    
    def record_api_request(self, method: str, endpoint: str, 
                          duration: float, status_code: int):
        """Record API request metrics."""
        if PROMETHEUS_AVAILABLE and 'api_requests' in self._metrics:
            self._metrics['api_requests'].labels(
                method=method, endpoint=endpoint, status_code=str(status_code)
            ).inc()
            
            self._metrics['api_duration'].labels(
                method=method, endpoint=endpoint
            ).observe(duration)
        
        self.logger.debug(f"API {method} {endpoint}: {status_code} in {duration:.3f}s")
    
    def record_error(self, component: str, error_type: str, 
                    severity: str = 'error', details: str = ""):
        """Record error occurrence."""
        if PROMETHEUS_AVAILABLE and 'errors' in self._metrics:
            self._metrics['errors'].labels(
                component=component, error_type=error_type, severity=severity
            ).inc()
        
        self.logger.error(f"Error in {component}: {error_type} ({severity}) - {details}")
    
    def set_health_status(self, component: str, status: str):
        """Set health status for a component."""
        if PROMETHEUS_AVAILABLE and 'health_status' in self._metrics:
            self._metrics['health_status'].labels(component=component).state(status)
        
        self.logger.info(f"Health status {component}: {status}")
    
    def update_system_metrics(self, cpu_percent: float, memory_bytes: int, 
                            disk_bytes: int, network_in: int, network_out: int):
        """Update system performance metrics."""
        if PROMETHEUS_AVAILABLE:
            if 'cpu_usage' in self._metrics:
                self._metrics['cpu_usage'].labels(core='total', process='qflare').set(cpu_percent)
            
            if 'memory_usage' in self._metrics:
                self._metrics['memory_usage'].labels(type='rss', process='qflare').set(memory_bytes)
            
            if 'disk_usage' in self._metrics:
                self._metrics['disk_usage'].labels(path='/data', type='used').set(disk_bytes)
            
            if 'network_bytes' in self._metrics:
                self._metrics['network_bytes'].labels(
                    direction='in', protocol='tcp', device_id='server'
                ).inc(network_in)
                self._metrics['network_bytes'].labels(
                    direction='out', protocol='tcp', device_id='server'
                ).inc(network_out)
    
    def add_custom_metric(self, name: str, metric_type: str, description: str, 
                         labels: List[str] = None):
        """Add a custom metric."""
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning(f"Cannot add custom metric {name}: Prometheus not available")
            return None
        
        labels = labels or []
        
        if metric_type == 'counter':
            metric = Counter(name, description, labels, registry=self.registry)
        elif metric_type == 'gauge':
            metric = Gauge(name, description, labels, registry=self.registry)
        elif metric_type == 'histogram':
            metric = Histogram(name, description, labels, registry=self.registry)
        elif metric_type == 'summary':
            metric = Summary(name, description, labels, registry=self.registry)
        else:
            self.logger.error(f"Unknown metric type: {metric_type}")
            return None
        
        self._custom_metrics[name] = metric
        self.logger.info(f"Added custom metric: {name} ({metric_type})")
        return metric
    
    def get_custom_metric(self, name: str):
        """Get a custom metric by name."""
        return self._custom_metrics.get(name)
    
    def start_collection(self):
        """Start metrics collection thread."""
        if self._collection_thread and self._collection_thread.is_alive():
            self.logger.warning("Metrics collection already running")
            return
        
        self._shutdown_event.clear()
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            name="QFLAREMetricsCollector"
        )
        self._collection_thread.daemon = True
        self._collection_thread.start()
        
        # Start HTTP server for Prometheus scraping
        if PROMETHEUS_AVAILABLE and self.config.export_port:
            try:
                start_http_server(self.config.export_port, registry=self.registry)
                self.logger.info(f"Metrics server started on port {self.config.export_port}")
            except Exception as e:
                self.logger.error(f"Failed to start metrics server: {e}")
        
        self.logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection."""
        if self._collection_thread:
            self._shutdown_event.set()
            self._collection_thread.join(timeout=5.0)
            self.logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main collection loop."""
        while not self._shutdown_event.is_set():
            try:
                self._collect_system_metrics()
                time.sleep(self.config.collection_interval)
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.config.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics (delta from last collection)
            network = psutil.net_io_counters()
            
            self.update_system_metrics(
                cpu_percent=cpu_percent,
                memory_bytes=memory.used,
                disk_bytes=disk.used,
                network_in=network.bytes_recv,
                network_out=network.bytes_sent
            )
            
        except ImportError:
            # psutil not available, skip system metrics
            pass
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def get_metrics_export(self) -> str:
        """Get metrics in Prometheus format."""
        if PROMETHEUS_AVAILABLE and self.registry:
            return generate_latest(self.registry).decode('utf-8')
        else:
            return "# Prometheus not available\n"
    
    def push_to_gateway(self, job_name: str = 'qflare'):
        """Push metrics to Prometheus pushgateway."""
        if not PROMETHEUS_AVAILABLE or not self.config.pushgateway_url:
            self.logger.warning("Cannot push to gateway: Prometheus or gateway URL not available")
            return
        
        try:
            push_to_gateway(
                self.config.pushgateway_url,
                job=job_name,
                registry=self.registry
            )
            self.logger.debug(f"Metrics pushed to gateway: {job_name}")
        except Exception as e:
            self.logger.error(f"Failed to push metrics to gateway: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics_count': len(self._metrics) + len(self._custom_metrics),
            'collection_enabled': self.config.enabled,
            'prometheus_available': PROMETHEUS_AVAILABLE
        }
        
        if PROMETHEUS_AVAILABLE:
            summary['export_port'] = self.config.export_port
            summary['pushgateway_url'] = self.config.pushgateway_url
        
        return summary


# Global metrics collector instance
_metrics_collector: Optional[QFLAREMetricsCollector] = None


def get_metrics_collector() -> QFLAREMetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = QFLAREMetricsCollector()
    return _metrics_collector


def initialize_metrics(config: MetricConfig = None) -> QFLAREMetricsCollector:
    """Initialize the global metrics collector."""
    global _metrics_collector
    _metrics_collector = QFLAREMetricsCollector(config)
    return _metrics_collector


def shutdown_metrics():
    """Shutdown the global metrics collector."""
    global _metrics_collector
    if _metrics_collector:
        _metrics_collector.stop_collection()
        _metrics_collector = None