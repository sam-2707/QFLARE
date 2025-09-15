"""
Integrated Monitoring and Observability System for QFLARE

This module provides a unified interface for all monitoring capabilities:
- Metrics collection and export
- Health monitoring and checks
- Distributed tracing
- Alerting and notifications
- Dashboard and reporting
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

from .metrics import (
    QFLAREMetricsCollector, MetricConfig, 
    get_metrics_collector, initialize_metrics
)
from .health import (
    QFLAREHealthMonitor, HealthCheckConfig, HealthStatus,
    get_health_monitor, initialize_health_monitoring
)
from .tracing import (
    QFLARETracer, FederatedLearningTracer,
    get_tracer, get_fl_tracer, initialize_tracing
)
from .alerting import (
    QFLAREAlertManager, Alert, AlertSeverity,
    get_alert_manager, initialize_alerting
)

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for the monitoring system."""
    # Metrics configuration
    metrics_enabled: bool = True
    metrics_port: int = 8000
    metrics_interval: float = 10.0
    
    # Health monitoring configuration
    health_enabled: bool = True
    health_interval: float = 30.0
    health_timeout: float = 10.0
    
    # Tracing configuration
    tracing_enabled: bool = True
    tracing_service_name: str = "qflare"
    tracing_sampling_rate: float = 1.0
    
    # Alerting configuration
    alerting_enabled: bool = True
    alert_evaluation_interval: float = 30.0
    notification_interval: int = 5  # minutes
    
    # Integration settings
    enable_prometheus: bool = True
    enable_jaeger: bool = False
    enable_grafana: bool = False


class QFLAREMonitoringSystem:
    """Integrated monitoring and observability system for QFLARE."""
    
    def __init__(self, config: MonitoringConfig = None):
        """Initialize the monitoring system."""
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.metrics_collector: Optional[QFLAREMetricsCollector] = None
        self.health_monitor: Optional[QFLAREHealthMonitor] = None
        self.tracer: Optional[QFLARETracer] = None
        self.fl_tracer: Optional[FederatedLearningTracer] = None
        self.alert_manager: Optional[QFLAREAlertManager] = None
        
        # State
        self.initialized = False
        self.started = False
    
    def initialize(self, 
                  security_manager=None,
                  database_service=None,
                  fl_coordinator=None):
        """Initialize all monitoring components."""
        if self.initialized:
            self.logger.warning("Monitoring system already initialized")
            return
        
        try:
            # Initialize metrics collection
            if self.config.metrics_enabled:
                metrics_config = MetricConfig(
                    enabled=True,
                    collection_interval=self.config.metrics_interval,
                    export_port=self.config.metrics_port
                )
                self.metrics_collector = initialize_metrics(metrics_config)
                self.logger.info("Metrics collection initialized")
            
            # Initialize health monitoring
            if self.config.health_enabled:
                health_config = HealthCheckConfig(
                    interval_seconds=self.config.health_interval,
                    timeout_seconds=self.config.health_timeout
                )
                self.health_monitor = initialize_health_monitoring(health_config)
                
                # Configure health checks with actual services
                if security_manager:
                    from .health import SecurityHealthCheck
                    security_check = SecurityHealthCheck(security_manager, health_config)
                    self.health_monitor.add_health_check(security_check)
                
                if database_service:
                    from .health import DatabaseHealthCheck
                    db_check = DatabaseHealthCheck(database_service, health_config)
                    self.health_monitor.add_health_check(db_check)
                
                if fl_coordinator:
                    from .health import FederatedLearningHealthCheck
                    fl_check = FederatedLearningHealthCheck(fl_coordinator, health_config)
                    self.health_monitor.add_health_check(fl_check)
                
                self.logger.info("Health monitoring initialized")
            
            # Initialize distributed tracing
            if self.config.tracing_enabled:
                self.tracer = initialize_tracing(self.config.tracing_service_name)
                self.fl_tracer = get_fl_tracer()
                self.tracer.sampling_rate = self.config.tracing_sampling_rate
                self.logger.info("Distributed tracing initialized")
            
            # Initialize alerting
            if self.config.alerting_enabled:
                self.alert_manager = initialize_alerting()
                
                # Add health check alert integration
                if self.health_monitor:
                    self.health_monitor.add_alert_callback(self._health_alert_callback)
                
                self.logger.info("Alerting system initialized")
            
            self.initialized = True
            self.logger.info("QFLARE monitoring system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring system: {e}")
            raise
    
    def start(self):
        """Start all monitoring components."""
        if not self.initialized:
            raise RuntimeError("Monitoring system not initialized")
        
        if self.started:
            self.logger.warning("Monitoring system already started")
            return
        
        try:
            # Start metrics collection
            if self.metrics_collector:
                self.metrics_collector.start_collection()
            
            # Start health monitoring
            if self.health_monitor:
                self.health_monitor.start_monitoring()
            
            # Start alerting
            if self.alert_manager:
                self.alert_manager.start_monitoring()
            
            self.started = True
            self.logger.info("QFLARE monitoring system started")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring system: {e}")
            raise
    
    def stop(self):
        """Stop all monitoring components."""
        if not self.started:
            return
        
        try:
            # Stop alerting
            if self.alert_manager:
                self.alert_manager.stop_monitoring()
            
            # Stop health monitoring
            if self.health_monitor:
                self.health_monitor.stop_monitoring()
            
            # Stop metrics collection
            if self.metrics_collector:
                self.metrics_collector.stop_collection()
            
            self.started = False
            self.logger.info("QFLARE monitoring system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring system: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall monitoring system status."""
        status = {
            'timestamp': datetime.utcnow().isoformat(),
            'initialized': self.initialized,
            'started': self.started,
            'components': {}
        }
        
        # Metrics status
        if self.metrics_collector:
            status['components']['metrics'] = self.metrics_collector.get_metrics_summary()
        
        # Health status
        if self.health_monitor:
            status['components']['health'] = self.health_monitor.get_health_summary()
        
        # Tracing status
        if self.tracer:
            status['components']['tracing'] = self.tracer.get_tracing_statistics()
        
        # Alerting status
        if self.alert_manager:
            status['components']['alerting'] = self.alert_manager.get_alert_statistics()
        
        return status
    
    def record_fl_training_metrics(self, device_id: str, duration: float, 
                                 accuracy: float, loss: float, 
                                 model_size: int = 0, iterations: int = 1):
        """Record federated learning training metrics."""
        if self.metrics_collector:
            self.metrics_collector.record_training_metrics(
                device_id=device_id,
                duration=duration,
                iterations=iterations
            )
            self.metrics_collector.record_model_metrics(
                accuracy=accuracy,
                loss=loss,
                dataset=f"device_{device_id}"
            )
            
            if model_size > 0:
                # Record model size (would need to add this metric)
                pass
    
    def record_fl_round_metrics(self, round_id: str, duration: float, 
                              participants: int, global_accuracy: float,
                              algorithm: str = "fedavg"):
        """Record federated learning round metrics."""
        if self.metrics_collector:
            self.metrics_collector.record_fl_round(
                round_id=round_id,
                duration=duration,
                participants=participants,
                algorithm=algorithm
            )
            self.metrics_collector.record_model_metrics(
                accuracy=global_accuracy,
                loss=0.0,  # Would need actual loss value
                dataset="global",
                metric_type="validation"
            )
    
    def record_security_metrics(self, operation: str, success: bool, 
                              device_id: str = "", method: str = ""):
        """Record security operation metrics."""
        if self.metrics_collector:
            if operation in ['login', 'authenticate']:
                self.metrics_collector.record_auth_attempt(
                    method=method or 'jwt',
                    status='success' if success else 'failure'
                )
            elif operation in ['key_generation', 'key_rotation', 'encryption']:
                self.metrics_collector.record_key_operation(
                    operation=operation,
                    key_type='post_quantum',
                    status='success' if success else 'failure'
                )
    
    def create_performance_alert(self, component: str, metric: str, 
                               threshold: float, current_value: float):
        """Create a performance-related alert."""
        if not self.alert_manager:
            return
        
        severity = AlertSeverity.WARNING
        if current_value > threshold * 1.5:
            severity = AlertSeverity.ERROR
        elif current_value > threshold * 2:
            severity = AlertSeverity.CRITICAL
        
        alert = Alert(
            id=f"perf_{component}_{metric}_{int(datetime.utcnow().timestamp())}",
            title=f"Performance Alert: {component} {metric}",
            description=f"{metric} is {current_value:.2f}, exceeding threshold of {threshold:.2f}",
            severity=severity,
            source=f"monitoring:performance",
            timestamp=datetime.utcnow(),
            labels={
                'component': component,
                'metric': metric,
                'type': 'performance'
            },
            annotations={
                'threshold': str(threshold),
                'current_value': str(current_value),
                'percentage_over': f"{((current_value / threshold) - 1) * 100:.1f}%"
            }
        )
        
        self.alert_manager.fire_alert(alert)
    
    def create_fl_alert(self, issue_type: str, description: str, 
                       severity: AlertSeverity = AlertSeverity.WARNING,
                       round_id: str = "", device_id: str = ""):
        """Create a federated learning related alert."""
        if not self.alert_manager:
            return
        
        alert = Alert(
            id=f"fl_{issue_type}_{int(datetime.utcnow().timestamp())}",
            title=f"FL Alert: {issue_type.replace('_', ' ').title()}",
            description=description,
            severity=severity,
            source="monitoring:federated_learning",
            timestamp=datetime.utcnow(),
            labels={
                'component': 'federated_learning',
                'issue_type': issue_type,
                'round_id': round_id,
                'device_id': device_id
            }
        )
        
        self.alert_manager.fire_alert(alert)
    
    async def run_health_check(self, component: str = None) -> Dict[str, Any]:
        """Run health checks and return results."""
        if not self.health_monitor:
            return {'error': 'Health monitoring not initialized'}
        
        if component:
            result = await self.health_monitor.check_component(component)
            return {
                'component': component,
                'status': result.status.value if result else 'unknown',
                'message': result.message if result else 'Component not found',
                'timestamp': result.timestamp.isoformat() if result else datetime.utcnow().isoformat()
            }
        else:
            results = await self.health_monitor.check_all()
            return {
                'overall_status': self.health_monitor.get_overall_status().value,
                'components': {
                    name: {
                        'status': result.status.value,
                        'message': result.message,
                        'duration_ms': result.duration_ms
                    }
                    for name, result in results.items()
                },
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_metrics_export(self) -> str:
        """Get metrics in Prometheus format."""
        if self.metrics_collector:
            return self.metrics_collector.get_metrics_export()
        return "# Metrics not available\n"
    
    def configure_email_alerts(self, smtp_host: str, smtp_port: int,
                             username: str, password: str,
                             from_email: str, to_emails: List[str]):
        """Configure email alerting."""
        if self.alert_manager:
            from .alerting import EmailNotificationChannel
            
            email_channel = EmailNotificationChannel(
                smtp_host=smtp_host,
                smtp_port=smtp_port,
                username=username,
                password=password,
                from_email=from_email,
                to_emails=to_emails
            )
            
            self.alert_manager.add_notification_channel(email_channel)
            self.logger.info("Email alerting configured")
    
    def configure_webhook_alerts(self, webhook_url: str, 
                               headers: Dict[str, str] = None):
        """Configure webhook alerting."""
        if self.alert_manager:
            from .alerting import WebhookNotificationChannel
            
            webhook_channel = WebhookNotificationChannel(
                webhook_url=webhook_url,
                headers=headers
            )
            
            self.alert_manager.add_notification_channel(webhook_channel)
            self.logger.info("Webhook alerting configured")
    
    def configure_slack_alerts(self, webhook_url: str, channel: str = "#alerts"):
        """Configure Slack alerting."""
        if self.alert_manager:
            from .alerting import SlackNotificationChannel
            
            slack_channel = SlackNotificationChannel(
                webhook_url=webhook_url,
                channel=channel
            )
            
            self.alert_manager.add_notification_channel(slack_channel)
            self.logger.info("Slack alerting configured")
    
    def _health_alert_callback(self, health_result):
        """Callback for health check alerts."""
        if not self.alert_manager:
            return
        
        if health_result.status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]:
            severity = AlertSeverity.WARNING if health_result.status == HealthStatus.DEGRADED else AlertSeverity.ERROR
            
            alert = Alert(
                id=f"health_{health_result.component}_{int(datetime.utcnow().timestamp())}",
                title=f"Health Check Alert: {health_result.component}",
                description=health_result.message,
                severity=severity,
                source=f"health_check:{health_result.component}",
                timestamp=health_result.timestamp,
                labels={
                    'component': health_result.component,
                    'type': 'health_check',
                    'status': health_result.status.value
                },
                annotations={
                    'duration_ms': str(health_result.duration_ms),
                    'details': str(health_result.details)
                }
            )
            
            self.alert_manager.fire_alert(alert)


# Global monitoring system instance
_monitoring_system: Optional[QFLAREMonitoringSystem] = None


def get_monitoring_system() -> QFLAREMonitoringSystem:
    """Get the global monitoring system instance."""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = QFLAREMonitoringSystem()
    return _monitoring_system


def initialize_monitoring(config: MonitoringConfig = None,
                        security_manager=None,
                        database_service=None,
                        fl_coordinator=None) -> QFLAREMonitoringSystem:
    """Initialize the global monitoring system."""
    global _monitoring_system
    _monitoring_system = QFLAREMonitoringSystem(config)
    _monitoring_system.initialize(
        security_manager=security_manager,
        database_service=database_service,
        fl_coordinator=fl_coordinator
    )
    return _monitoring_system


def start_monitoring():
    """Start the global monitoring system."""
    monitoring_system = get_monitoring_system()
    monitoring_system.start()


def stop_monitoring():
    """Stop the global monitoring system."""
    global _monitoring_system
    if _monitoring_system:
        _monitoring_system.stop()


def shutdown_monitoring():
    """Shutdown the global monitoring system."""
    global _monitoring_system
    if _monitoring_system:
        _monitoring_system.stop()
        _monitoring_system = None


# Export key classes and functions
__all__ = [
    'QFLAREMonitoringSystem',
    'MonitoringConfig',
    'get_monitoring_system',
    'initialize_monitoring',
    'start_monitoring',
    'stop_monitoring',
    'shutdown_monitoring'
]