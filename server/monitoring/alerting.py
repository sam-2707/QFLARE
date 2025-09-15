"""
Alerting System for QFLARE

This module provides comprehensive alerting capabilities:
- Rule-based alerting for metrics and health checks
- Multiple notification channels (email, webhook, Slack)
- Alert aggregation and deduplication
- Escalation policies and alert routing
- Integration with monitoring and health systems
"""

import time
import logging
import threading
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import urllib.request
import urllib.parse

try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    logging.warning("Email modules not available, email notifications disabled")

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Represents an alert."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str
    timestamp: datetime
    status: AlertStatus = AlertStatus.OPEN
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    escalation_level: int = 0
    last_notification: Optional[datetime] = None
    notification_count: int = 0
    
    def acknowledge(self, acknowledged_by: str = "system"):
        """Acknowledge the alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.utcnow()
        self.acknowledged_by = acknowledged_by
    
    def resolve(self):
        """Resolve the alert."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.utcnow()
    
    def suppress(self):
        """Suppress the alert."""
        self.status = AlertStatus.SUPPRESSED
    
    def escalate(self):
        """Escalate the alert."""
        self.escalation_level += 1
    
    def should_notify(self, interval_minutes: int = 5) -> bool:
        """Check if alert should send notification."""
        if self.status in [AlertStatus.RESOLVED, AlertStatus.SUPPRESSED]:
            return False
        
        if not self.last_notification:
            return True
        
        time_since_last = datetime.utcnow() - self.last_notification
        return time_since_last >= timedelta(minutes=interval_minutes)
    
    def record_notification(self):
        """Record that a notification was sent."""
        self.last_notification = datetime.utcnow()
        self.notification_count += 1


@dataclass
class AlertRule:
    """Defines conditions for triggering alerts."""
    name: str
    description: str
    condition: Callable[[], bool]
    severity: AlertSeverity
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    evaluation_interval: float = 60.0  # seconds
    for_duration: float = 0.0  # seconds (alert only after condition is true for this long)
    last_evaluation: Optional[datetime] = None
    condition_met_since: Optional[datetime] = None
    
    def should_evaluate(self) -> bool:
        """Check if rule should be evaluated."""
        if not self.enabled:
            return False
        
        if not self.last_evaluation:
            return True
        
        time_since_last = datetime.utcnow() - self.last_evaluation
        return time_since_last.total_seconds() >= self.evaluation_interval
    
    def evaluate(self) -> Optional[Alert]:
        """Evaluate the rule and return alert if conditions are met."""
        self.last_evaluation = datetime.utcnow()
        
        try:
            condition_result = self.condition()
            
            if condition_result:
                if not self.condition_met_since:
                    self.condition_met_since = datetime.utcnow()
                
                # Check if condition has been met for required duration
                if self.for_duration > 0:
                    duration_met = (datetime.utcnow() - self.condition_met_since).total_seconds()
                    if duration_met < self.for_duration:
                        return None
                
                # Create alert
                alert_id = f"{self.name}_{int(time.time())}"
                return Alert(
                    id=alert_id,
                    title=f"Alert: {self.name}",
                    description=self.description,
                    severity=self.severity,
                    source=f"rule:{self.name}",
                    timestamp=datetime.utcnow(),
                    labels=self.labels.copy(),
                    annotations=self.annotations.copy()
                )
            else:
                # Condition not met, reset timer
                self.condition_met_since = None
                return None
                
        except Exception as e:
            logger.error(f"Error evaluating alert rule {self.name}: {e}")
            return None


class NotificationChannel:
    """Base class for notification channels."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification for an alert."""
        if not self.enabled:
            return False
        
        try:
            return await self._send_notification(alert)
        except Exception as e:
            self.logger.error(f"Failed to send notification via {self.name}: {e}")
            return False
    
    async def _send_notification(self, alert: Alert) -> bool:
        """Override this method to implement actual notification sending."""
        raise NotImplementedError


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""
    
    def __init__(self, name: str = "email", smtp_host: str = "localhost",
                 smtp_port: int = 587, username: str = "", password: str = "",
                 from_email: str = "", to_emails: List[str] = None,
                 use_tls: bool = True, enabled: bool = True):
        super().__init__(name, enabled)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails or []
        self.use_tls = use_tls
    
    async def _send_notification(self, alert: Alert) -> bool:
        """Send email notification."""
        if not EMAIL_AVAILABLE:
            self.logger.warning("Email modules not available, cannot send email notification")
            return False
            
        if not self.to_emails:
            self.logger.warning("No recipient emails configured")
            return False
        
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] QFLARE Alert: {alert.title}"
            
            # Create email body
            body = self._format_email_body(alert)
            msg.attach(MimeText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            if self.use_tls:
                server.starttls()
            if self.username and self.password:
                server.login(self.username, self.password)
            
            text = msg.as_string()
            server.sendmail(self.from_email, self.to_emails, text)
            server.quit()
            
            self.logger.info(f"Email notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            return False
    
    def _format_email_body(self, alert: Alert) -> str:
        """Format email body for alert."""
        severity_color = {
            AlertSeverity.INFO: "#17a2b8",
            AlertSeverity.WARNING: "#ffc107", 
            AlertSeverity.ERROR: "#dc3545",
            AlertSeverity.CRITICAL: "#dc3545"
        }
        
        color = severity_color.get(alert.severity, "#6c757d")
        
        body = f"""
        <html>
        <body>
            <h2 style="color: {color};">QFLARE Alert: {alert.title}</h2>
            
            <table style="border-collapse: collapse; width: 100%;">
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Severity</td>
                    <td style="border: 1px solid #ddd; padding: 8px; color: {color};">{alert.severity.value.upper()}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Source</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{alert.source}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Timestamp</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{alert.timestamp.isoformat()}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Description</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{alert.description}</td>
                </tr>
            </table>
            
            <h3>Labels</h3>
            <ul>
        """
        
        for key, value in alert.labels.items():
            body += f"<li><strong>{key}:</strong> {value}</li>"
        
        body += """
            </ul>
            
            <h3>Annotations</h3>
            <ul>
        """
        
        for key, value in alert.annotations.items():
            body += f"<li><strong>{key}:</strong> {value}</li>"
        
        body += """
            </ul>
            
            <p><em>This alert was generated by QFLARE monitoring system.</em></p>
        </body>
        </html>
        """
        
        return body


class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel."""
    
    def __init__(self, name: str = "webhook", webhook_url: str = "",
                 headers: Dict[str, str] = None, enabled: bool = True):
        super().__init__(name, enabled)
        self.webhook_url = webhook_url
        self.headers = headers or {'Content-Type': 'application/json'}
    
    async def _send_notification(self, alert: Alert) -> bool:
        """Send webhook notification."""
        if not self.webhook_url:
            self.logger.warning("No webhook URL configured")
            return False
        
        try:
            # Prepare payload
            payload = {
                'alert_id': alert.id,
                'title': alert.title,
                'description': alert.description,
                'severity': alert.severity.value,
                'source': alert.source,
                'timestamp': alert.timestamp.isoformat(),
                'status': alert.status.value,
                'labels': alert.labels,
                'annotations': alert.annotations,
                'escalation_level': alert.escalation_level
            }
            
            # Send webhook
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers=self.headers,
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                if 200 <= response.status < 300:
                    self.logger.info(f"Webhook notification sent for alert {alert.id}")
                    return True
                else:
                    self.logger.error(f"Webhook returned status {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
            return False


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel using webhooks."""
    
    def __init__(self, name: str = "slack", webhook_url: str = "",
                 channel: str = "#alerts", username: str = "QFLARE",
                 enabled: bool = True):
        super().__init__(name, enabled)
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
    
    async def _send_notification(self, alert: Alert) -> bool:
        """Send Slack notification."""
        if not self.webhook_url:
            self.logger.warning("No Slack webhook URL configured")
            return False
        
        try:
            # Format Slack message
            color = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "danger",
                AlertSeverity.CRITICAL: "danger"
            }.get(alert.severity, "danger")
            
            fields = [
                {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                {"title": "Source", "value": alert.source, "short": True},
                {"title": "Timestamp", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"), "short": True}
            ]
            
            # Add labels as fields
            for key, value in alert.labels.items():
                fields.append({"title": key, "value": value, "short": True})
            
            payload = {
                "channel": self.channel,
                "username": self.username,
                "text": f"QFLARE Alert: {alert.title}",
                "attachments": [
                    {
                        "color": color,
                        "title": alert.title,
                        "text": alert.description,
                        "fields": fields,
                        "footer": "QFLARE Monitoring",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            # Send to Slack
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    self.logger.info(f"Slack notification sent for alert {alert.id}")
                    return True
                else:
                    self.logger.error(f"Slack webhook returned status {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
            return False


class QFLAREAlertManager:
    """Central alert management system for QFLARE."""
    
    def __init__(self):
        """Initialize the alert manager."""
        self.logger = logging.getLogger(__name__)
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self.suppression_rules: List[Callable[[Alert], bool]] = []
        
        # Configuration
        self.evaluation_interval = 30.0  # seconds
        self.notification_interval = 5  # minutes
        self.max_history_size = 10000
        self.auto_resolve_timeout = timedelta(hours=24)
        
        # Monitoring thread
        self.monitoring_thread = None
        self.shutdown_event = threading.Event()
        
        # Initialize default rules
        self._initialize_default_rules()
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            self.logger.info(f"Removed alert rule: {rule_name}")
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add a notification channel."""
        self.notification_channels[channel.name] = channel
        self.logger.info(f"Added notification channel: {channel.name}")
    
    def add_suppression_rule(self, rule: Callable[[Alert], bool]):
        """Add a suppression rule."""
        self.suppression_rules.append(rule)
    
    def fire_alert(self, alert: Alert):
        """Manually fire an alert."""
        # Check suppression rules
        for suppression_rule in self.suppression_rules:
            try:
                if suppression_rule(alert):
                    alert.suppress()
                    self.logger.info(f"Alert {alert.id} suppressed by rule")
                    break
            except Exception as e:
                self.logger.error(f"Error in suppression rule: {e}")
        
        # Store alert
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Limit history size
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]
        
        self.logger.info(f"Alert fired: {alert.title} (severity: {alert.severity.value})")
        
        # Send notifications (handle cases with no event loop)
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self._send_notifications(alert))
        except RuntimeError:
            # No event loop running, schedule for later or use thread
            import threading
            thread = threading.Thread(
                target=lambda: asyncio.run(self._send_notifications(alert)),
                daemon=True
            )
            thread.start()
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "user"):
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledge(acknowledged_by)
            self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolve()
            del self.active_alerts[alert_id]
            self.logger.info(f"Alert {alert_id} resolved")
    
    def get_active_alerts(self, severity: AlertSeverity = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = list(self.active_alerts.values())
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        active_by_severity = {}
        for severity in AlertSeverity:
            active_by_severity[severity.value] = len([
                alert for alert in self.active_alerts.values()
                if alert.severity == severity
            ])
        
        return {
            'active_alerts': len(self.active_alerts),
            'total_rules': len(self.alert_rules),
            'enabled_rules': len([rule for rule in self.alert_rules.values() if rule.enabled]),
            'notification_channels': len(self.notification_channels),
            'enabled_channels': len([ch for ch in self.notification_channels.values() if ch.enabled]),
            'active_by_severity': active_by_severity,
            'total_history': len(self.alert_history)
        }
    
    def start_monitoring(self):
        """Start alert monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.warning("Alert monitoring already running")
            return
        
        self.shutdown_event.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="QFLAREAlertManager"
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop alert monitoring."""
        if self.monitoring_thread:
            self.shutdown_event.set()
            self.monitoring_thread.join(timeout=5.0)
            self.logger.info("Alert monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while not self.shutdown_event.is_set():
            try:
                loop.run_until_complete(self._evaluate_rules())
                loop.run_until_complete(self._process_notifications())
                self._cleanup_old_alerts()
                
                self.shutdown_event.wait(self.evaluation_interval)
            except Exception as e:
                self.logger.error(f"Error in alert monitoring loop: {e}")
                self.shutdown_event.wait(self.evaluation_interval)
        
        loop.close()
    
    async def _evaluate_rules(self):
        """Evaluate all alert rules."""
        for rule in self.alert_rules.values():
            if rule.should_evaluate():
                alert = rule.evaluate()
                if alert:
                    self.fire_alert(alert)
    
    async def _process_notifications(self):
        """Process pending notifications."""
        for alert in self.active_alerts.values():
            if alert.should_notify(self.notification_interval):
                await self._send_notifications(alert)
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        if alert.status == AlertStatus.SUPPRESSED:
            return
        
        notification_tasks = []
        for channel in self.notification_channels.values():
            if channel.enabled:
                task = channel.send_notification(alert)
                notification_tasks.append(task)
        
        if notification_tasks:
            results = await asyncio.gather(*notification_tasks, return_exceptions=True)
            success_count = sum(1 for result in results if result is True)
            
            if success_count > 0:
                alert.record_notification()
                self.logger.info(f"Sent {success_count}/{len(notification_tasks)} notifications for alert {alert.id}")
    
    def _cleanup_old_alerts(self):
        """Clean up old resolved alerts."""
        cutoff_time = datetime.utcnow() - self.auto_resolve_timeout
        
        alerts_to_resolve = []
        for alert_id, alert in self.active_alerts.items():
            if alert.timestamp < cutoff_time and alert.status == AlertStatus.OPEN:
                alerts_to_resolve.append(alert_id)
        
        for alert_id in alerts_to_resolve:
            self.resolve_alert(alert_id)
            self.logger.info(f"Auto-resolved old alert: {alert_id}")
    
    def _initialize_default_rules(self):
        """Initialize default alert rules."""
        # Example: High CPU usage alert
        def high_cpu_condition():
            # This would integrate with metrics system
            return False  # Placeholder
        
        self.add_alert_rule(AlertRule(
            name="high_cpu_usage",
            description="CPU usage is above 90%",
            condition=high_cpu_condition,
            severity=AlertSeverity.WARNING,
            labels={"component": "system", "resource": "cpu"},
            evaluation_interval=60.0,
            for_duration=300.0  # Alert after 5 minutes
        ))
        
        # Example: FL round failure alert
        def fl_round_failure_condition():
            # This would check FL coordinator status
            return False  # Placeholder
        
        self.add_alert_rule(AlertRule(
            name="fl_round_failure",
            description="Federated learning round failed",
            condition=fl_round_failure_condition,
            severity=AlertSeverity.ERROR,
            labels={"component": "federated_learning", "type": "round_failure"},
            evaluation_interval=30.0
        ))


# Global alert manager instance
_alert_manager: Optional[QFLAREAlertManager] = None


def get_alert_manager() -> QFLAREAlertManager:
    """Get the global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = QFLAREAlertManager()
    return _alert_manager


def initialize_alerting() -> QFLAREAlertManager:
    """Initialize the global alert manager."""
    global _alert_manager
    _alert_manager = QFLAREAlertManager()
    return _alert_manager


def shutdown_alerting():
    """Shutdown the global alert manager."""
    global _alert_manager
    if _alert_manager:
        _alert_manager.stop_monitoring()
        _alert_manager = None