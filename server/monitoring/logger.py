"""
Comprehensive Monitoring and Logging System for QFLARE.

This module provides structured logging, security event monitoring,
performance metrics, and audit trails for the QFLARE system.
"""

import logging
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import hashlib
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventType(Enum):
    """Types of security events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ENROLLMENT = "enrollment"
    MODEL_UPDATE = "model_update"
    POISONING_DETECTED = "poisoning_detected"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SYSTEM_ERROR = "system_error"
    PERFORMANCE = "performance"
    AUDIT = "audit"


@dataclass
class SecurityEvent:
    """Security event record."""
    event_id: str
    timestamp: datetime
    event_type: EventType
    security_level: SecurityLevel
    source_ip: str
    user_agent: str
    device_id: Optional[str]
    details: Dict[str, Any]
    session_id: Optional[str]
    request_id: Optional[str]
    response_time: Optional[float]
    success: bool
    error_message: Optional[str]


@dataclass
class PerformanceMetric:
    """Performance metric record."""
    metric_id: str
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    tags: Dict[str, str]
    source: str


class SecurityMonitor:
    """Security monitoring and alerting system."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize security monitor.
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Event storage
        self.security_events: List[SecurityEvent] = []
        self.performance_metrics: List[PerformanceMetric] = []
        
        # Alert thresholds
        self.alert_thresholds = {
            "failed_auth_attempts": 5,
            "suspicious_activity_count": 10,
            "poisoning_detections": 3,
            "rate_limit_violations": 20,
            "response_time_threshold": 5.0  # seconds
        }
        
        # Alert state
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        
        # Threading
        self.lock = threading.Lock()
        self.event_queue = queue.Queue()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Set up loggers
        self._setup_loggers()
    
    def _setup_loggers(self):
        """Set up specialized loggers."""
        # Security events logger
        security_handler = logging.FileHandler(self.log_dir / "security.log")
        security_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.security_logger = logging.getLogger("security")
        self.security_logger.addHandler(security_handler)
        self.security_logger.setLevel(logging.INFO)
        
        # Performance logger
        perf_handler = logging.FileHandler(self.log_dir / "performance.log")
        perf_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.perf_logger = logging.getLogger("performance")
        self.perf_logger.addHandler(perf_handler)
        self.perf_logger.setLevel(logging.INFO)
        
        # Audit logger
        audit_handler = logging.FileHandler(self.log_dir / "audit.log")
        audit_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.audit_logger = logging.getLogger("audit")
        self.audit_logger.addHandler(audit_handler)
        self.audit_logger.setLevel(logging.INFO)
    
    def log_security_event(
        self,
        event_type: EventType,
        security_level: SecurityLevel,
        source_ip: str,
        user_agent: str,
        device_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        response_time: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Log a security event.
        
        Args:
            event_type: Type of security event
            security_level: Security level of the event
            source_ip: Source IP address
            user_agent: User agent string
            device_id: Device identifier
            details: Additional event details
            session_id: Session identifier
            request_id: Request identifier
            response_time: Response time in seconds
            success: Whether the operation was successful
            error_message: Error message if any
        """
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            event_type=event_type,
            security_level=security_level,
            source_ip=source_ip,
            user_agent=user_agent,
            device_id=device_id,
            details=details or {},
            session_id=session_id,
            request_id=request_id,
            response_time=response_time,
            success=success,
            error_message=error_message
        )
        
        # Add to queue for processing
        self.event_queue.put(event)
        
        # Log to file
        log_entry = {
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type.value,
            "security_level": event.security_level.value,
            "source_ip": event.source_ip,
            "device_id": event.device_id,
            "success": event.success,
            "response_time": event.response_time,
            "error_message": event.error_message,
            "details": event.details
        }
        
        if event.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            self.security_logger.warning(json.dumps(log_entry))
        else:
            self.security_logger.info(json.dumps(log_entry))
    
    def log_performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str,
        tags: Optional[Dict[str, str]] = None,
        source: str = "qflare"
    ):
        """Log a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            tags: Additional tags
            source: Source of the metric
        """
        metric = PerformanceMetric(
            metric_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            unit=unit,
            tags=tags or {},
            source=source
        )
        
        with self.lock:
            self.performance_metrics.append(metric)
        
        # Log to file
        log_entry = {
            "metric_id": metric.metric_id,
            "timestamp": metric.timestamp.isoformat(),
            "metric_name": metric.metric_name,
            "value": metric.value,
            "unit": metric.unit,
            "tags": metric.tags,
            "source": metric.source
        }
        
        self.perf_logger.info(json.dumps(log_entry))
    
    def log_audit_event(
        self,
        action: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log an audit event.
        
        Args:
            action: Action performed
            user_id: User identifier
            resource: Resource accessed
            details: Additional details
        """
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "user_id": user_id,
            "resource": resource,
            "details": details or {}
        }
        
        self.audit_logger.info(json.dumps(audit_entry))
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                # Process events from queue
                while not self.event_queue.empty():
                    event = self.event_queue.get_nowait()
                    self._process_security_event(event)
                
                # Check for alerts
                self._check_alerts()
                
                # Clean up old data
                self._cleanup_old_data()
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _process_security_event(self, event: SecurityEvent):
        """Process a security event.
        
        Args:
            event: Security event to process
        """
        with self.lock:
            self.security_events.append(event)
        
        # Check for immediate alerts
        if event.security_level == SecurityLevel.CRITICAL:
            self._trigger_alert("critical_security_event", {
                "event_type": event.event_type.value,
                "source_ip": event.source_ip,
                "device_id": event.device_id,
                "details": event.details
            })
        
        # Check for suspicious patterns
        self._check_suspicious_patterns(event)
    
    def _check_suspicious_patterns(self, event: SecurityEvent):
        """Check for suspicious activity patterns.
        
        Args:
            event: Security event to analyze
        """
        with self.lock:
            # Count failed authentication attempts
            failed_auth_events = [
                e for e in self.security_events
                if (e.event_type == EventType.AUTHENTICATION and 
                    not e.success and 
                    e.source_ip == event.source_ip and
                    e.timestamp > datetime.now() - timedelta(minutes=5))
            ]
            
            if len(failed_auth_events) >= self.alert_thresholds["failed_auth_attempts"]:
                self._trigger_alert("multiple_failed_auth", {
                    "source_ip": event.source_ip,
                    "failed_attempts": len(failed_auth_events),
                    "time_window": "5 minutes"
                })
            
            # Check for poisoning detections
            poisoning_events = [
                e for e in self.security_events
                if (e.event_type == EventType.POISONING_DETECTED and
                    e.timestamp > datetime.now() - timedelta(hours=1))
            ]
            
            if len(poisoning_events) >= self.alert_thresholds["poisoning_detections"]:
                self._trigger_alert("multiple_poisoning_detections", {
                    "detection_count": len(poisoning_events),
                    "time_window": "1 hour"
                })
    
    def _check_alerts(self):
        """Check for alert conditions."""
        with self.lock:
            # Check for high response times
            recent_events = [
                e for e in self.security_events
                if (e.response_time and 
                    e.timestamp > datetime.now() - timedelta(minutes=5))
            ]
            
            if recent_events:
                avg_response_time = sum(e.response_time for e in recent_events) / len(recent_events)
                
                if avg_response_time > self.alert_thresholds["response_time_threshold"]:
                    self._trigger_alert("high_response_time", {
                        "average_response_time": avg_response_time,
                        "threshold": self.alert_thresholds["response_time_threshold"]
                    })
    
    def _trigger_alert(self, alert_type: str, details: Dict[str, Any]):
        """Trigger a security alert.
        
        Args:
            alert_type: Type of alert
            details: Alert details
        """
        alert_id = f"{alert_type}_{int(time.time())}"
        
        alert = {
            "alert_id": alert_id,
            "alert_type": alert_type,
            "timestamp": datetime.now().isoformat(),
            "details": details,
            "status": "active"
        }
        
        with self.lock:
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
        
        # Log alert
        logger.warning(f"SECURITY ALERT: {alert_type} - {json.dumps(details)}")
        
        # In production, this would send notifications (email, Slack, etc.)
        self._send_alert_notification(alert)
    
    def _send_alert_notification(self, alert: Dict[str, Any]):
        """Send alert notification (placeholder for production).
        
        Args:
            alert: Alert to send
        """
        # In production, implement actual notification sending
        # For now, just log the alert
        logger.critical(f"ALERT NOTIFICATION: {alert['alert_type']} - {alert['details']}")
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        cutoff_time = datetime.now() - timedelta(days=7)
        
        with self.lock:
            # Clean up old security events
            self.security_events = [
                e for e in self.security_events
                if e.timestamp > cutoff_time
            ]
            
            # Clean up old performance metrics
            self.performance_metrics = [
                m for m in self.performance_metrics
                if m.timestamp > cutoff_time
            ]
            
            # Clean up old alerts
            self.alert_history = [
                a for a in self.alert_history
                if datetime.fromisoformat(a["timestamp"]) > cutoff_time
            ]
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics.
        
        Returns:
            Dictionary with security statistics
        """
        with self.lock:
            now = datetime.now()
            last_24h = now - timedelta(hours=24)
            
            recent_events = [
                e for e in self.security_events
                if e.timestamp > last_24h
            ]
            
            stats = {
                "total_events_24h": len(recent_events),
                "events_by_type": {},
                "events_by_security_level": {},
                "active_alerts": len(self.active_alerts),
                "total_alerts_24h": len([
                    a for a in self.alert_history
                    if datetime.fromisoformat(a["timestamp"]) > last_24h
                ])
            }
            
            # Count by event type
            for event in recent_events:
                event_type = event.event_type.value
                stats["events_by_type"][event_type] = stats["events_by_type"].get(event_type, 0) + 1
            
            # Count by security level
            for event in recent_events:
                level = event.security_level.value
                stats["events_by_security_level"][level] = stats["events_by_security_level"].get(level, 0) + 1
            
            return stats
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        with self.lock:
            now = datetime.now()
            last_hour = now - timedelta(hours=1)
            
            recent_metrics = [
                m for m in self.performance_metrics
                if m.timestamp > last_hour
            ]
            
            stats = {
                "total_metrics_1h": len(recent_metrics),
                "metrics_by_name": {},
                "average_response_time": None
            }
            
            # Group metrics by name
            for metric in recent_metrics:
                name = metric.metric_name
                if name not in stats["metrics_by_name"]:
                    stats["metrics_by_name"][name] = []
                stats["metrics_by_name"][name].append(metric.value)
            
            # Calculate averages
            for name, values in stats["metrics_by_name"].items():
                stats["metrics_by_name"][name] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
            
            # Calculate average response time
            response_times = [
                e.response_time for e in self.security_events
                if e.response_time and e.timestamp > last_hour
            ]
            
            if response_times:
                stats["average_response_time"] = sum(response_times) / len(response_times)
            
            return stats


# Global security monitor instance
security_monitor = SecurityMonitor()


def log_security_event(
    event_type: EventType,
    security_level: SecurityLevel,
    source_ip: str,
    user_agent: str,
    **kwargs
):
    """Log a security event (convenience function)."""
    security_monitor.log_security_event(
        event_type=event_type,
        security_level=security_level,
        source_ip=source_ip,
        user_agent=user_agent,
        **kwargs
    )


def log_performance_metric(metric_name: str, value: float, unit: str, **kwargs):
    """Log a performance metric (convenience function)."""
    security_monitor.log_performance_metric(
        metric_name=metric_name,
        value=value,
        unit=unit,
        **kwargs
    )


def log_audit_event(action: str, **kwargs):
    """Log an audit event (convenience function)."""
    security_monitor.log_audit_event(action=action, **kwargs)


def get_monitoring_stats() -> Dict[str, Any]:
    """Get comprehensive monitoring statistics."""
    return {
        "security": security_monitor.get_security_stats(),
        "performance": security_monitor.get_performance_stats()
    } 