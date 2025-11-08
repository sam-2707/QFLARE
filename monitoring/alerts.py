"""
Alert and notification system for QFLARE monitoring.

This module provides configurable alerting based on metrics thresholds,
health check failures, and critical system events.
"""

import asyncio
import smtplib
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, asdict
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import aiohttp
import aiofiles

from monitoring.logging import get_logger, log_context
from monitoring.health import HealthStatus, health_manager


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert definition."""
    id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    source: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None
    resolved_at: Optional[datetime] = None
    suppressed_until: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["severity"] = self.severity.value
        result["status"] = self.status.value
        result["timestamp"] = self.timestamp.isoformat()
        if self.resolved_at:
            result["resolved_at"] = self.resolved_at.isoformat()
        if self.suppressed_until:
            result["suppressed_until"] = self.suppressed_until.isoformat()
        return result


class AlertRule:
    """Base alert rule class."""
    
    def __init__(self, 
                 name: str, 
                 severity: AlertSeverity,
                 description: str = "",
                 cooldown_minutes: int = 5):
        self.name = name
        self.severity = severity
        self.description = description
        self.cooldown_minutes = cooldown_minutes
        self.last_triggered: Optional[datetime] = None
        self.logger = get_logger(f"alerts.{name}")
    
    def should_trigger(self) -> bool:
        """Check if rule should trigger (considering cooldown)."""
        if not self.last_triggered:
            return True
        
        cooldown_period = timedelta(minutes=self.cooldown_minutes)
        return datetime.now(timezone.utc) - self.last_triggered > cooldown_period
    
    async def evaluate(self) -> Optional[Alert]:
        """Evaluate the rule and return alert if triggered."""
        if not self.should_trigger():
            return None
        
        try:
            condition_met, message, details = await self._check_condition()
            
            if condition_met:
                self.last_triggered = datetime.now(timezone.utc)
                
                alert = Alert(
                    id=f"{self.name}_{int(self.last_triggered.timestamp())}",
                    name=self.name,
                    severity=self.severity,
                    status=AlertStatus.ACTIVE,
                    message=message,
                    source="alert_rules",
                    timestamp=self.last_triggered,
                    details=details
                )
                
                self.logger.warning(f"Alert triggered: {message}", extra={"alert": alert.to_dict()})
                return alert
        
        except Exception as e:
            self.logger.error(f"Error evaluating alert rule {self.name}: {str(e)}")
        
        return None
    
    async def _check_condition(self) -> tuple[bool, str, Optional[Dict[str, Any]]]:
        """Override this method to implement specific alert logic."""
        raise NotImplementedError


class HealthCheckAlertRule(AlertRule):
    """Alert rule for health check failures."""
    
    def __init__(self, 
                 checker_name: str,
                 severity: AlertSeverity = AlertSeverity.ERROR,
                 consecutive_failures: int = 2):
        super().__init__(
            name=f"health_check_{checker_name}",
            severity=severity,
            description=f"Health check failure for {checker_name}",
            cooldown_minutes=5
        )
        self.checker_name = checker_name
        self.consecutive_failures = consecutive_failures
        self.failure_count = 0
    
    async def _check_condition(self) -> tuple[bool, str, Optional[Dict[str, Any]]]:
        """Check if health check is failing."""
        result = await health_manager.check_single(self.checker_name)
        
        if not result:
            return False, "", None
        
        if result["status"] == HealthStatus.UNHEALTHY.value:
            self.failure_count += 1
            
            if self.failure_count >= self.consecutive_failures:
                return (
                    True,
                    f"Health check '{self.checker_name}' failed {self.failure_count} consecutive times: {result['message']}",
                    {
                        "checker_name": self.checker_name,
                        "failure_count": self.failure_count,
                        "response_time_ms": result["response_time_ms"],
                        "details": result.get("details", {})
                    }
                )
        else:
            self.failure_count = 0
        
        return False, "", None


class MetricThresholdAlertRule(AlertRule):
    """Alert rule for metric threshold violations."""
    
    def __init__(self,
                 metric_name: str,
                 threshold: float,
                 operator: str = "greater_than",
                 severity: AlertSeverity = AlertSeverity.WARNING,
                 evaluation_period_minutes: int = 5):
        super().__init__(
            name=f"metric_threshold_{metric_name}",
            severity=severity,
            description=f"Metric threshold alert for {metric_name}",
            cooldown_minutes=10
        )
        self.metric_name = metric_name
        self.threshold = threshold
        self.operator = operator  # greater_than, less_than, equal_to
        self.evaluation_period_minutes = evaluation_period_minutes
    
    async def _check_condition(self) -> tuple[bool, str, Optional[Dict[str, Any]]]:
        """Check if metric exceeds threshold."""
        from monitoring.metrics import qflare_metrics
        
        # This is a simplified implementation
        # In a real system, you'd query your metrics backend (Prometheus)
        try:
            # Get current metric value (this would be implemented based on your metrics storage)
            # For now, we'll simulate with system metrics
            import psutil
            
            current_value = None
            
            if "cpu" in self.metric_name.lower():
                current_value = psutil.cpu_percent()
            elif "memory" in self.metric_name.lower():
                current_value = psutil.virtual_memory().percent
            elif "disk" in self.metric_name.lower():
                current_value = psutil.disk_usage('/').percent if hasattr(psutil.disk_usage('/'), 'percent') else 0
            
            if current_value is not None:
                threshold_met = False
                
                if self.operator == "greater_than":
                    threshold_met = current_value > self.threshold
                elif self.operator == "less_than":
                    threshold_met = current_value < self.threshold
                elif self.operator == "equal_to":
                    threshold_met = abs(current_value - self.threshold) < 0.1
                
                if threshold_met:
                    return (
                        True,
                        f"Metric '{self.metric_name}' {self.operator} threshold: {current_value:.2f} {self.operator.replace('_', ' ')} {self.threshold}",
                        {
                            "metric_name": self.metric_name,
                            "current_value": current_value,
                            "threshold": self.threshold,
                            "operator": self.operator
                        }
                    )
        
        except Exception as e:
            self.logger.error(f"Error checking metric threshold: {str(e)}")
        
        return False, "", None


class ErrorRateAlertRule(AlertRule):
    """Alert rule for high error rates."""
    
    def __init__(self,
                 error_threshold_per_minute: int = 10,
                 severity: AlertSeverity = AlertSeverity.WARNING):
        super().__init__(
            name="high_error_rate",
            severity=severity,
            description="High error rate detected",
            cooldown_minutes=10
        )
        self.error_threshold_per_minute = error_threshold_per_minute
    
    async def _check_condition(self) -> tuple[bool, str, Optional[Dict[str, Any]]]:
        """Check if error rate is too high."""
        from monitoring.logging import error_tracker
        
        error_summary = error_tracker.get_error_summary()
        
        # Calculate recent error rate (simplified)
        recent_errors = len([
            error for error in error_tracker.recent_errors
            if datetime.fromisoformat(error["timestamp"].replace('Z', '+00:00')) > 
               datetime.now(timezone.utc) - timedelta(minutes=1)
        ])
        
        if recent_errors > self.error_threshold_per_minute:
            return (
                True,
                f"High error rate detected: {recent_errors} errors in the last minute",
                {
                    "errors_per_minute": recent_errors,
                    "threshold": self.error_threshold_per_minute,
                    "total_errors": error_summary["total_errors"],
                    "error_types": error_summary["total_error_types"]
                }
            )
        
        return False, "", None


class NotificationChannel:
    """Base notification channel class."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"notifications.{name}")
    
    async def send(self, alert: Alert) -> bool:
        """Send alert notification."""
        try:
            with log_context(component="notifications", operation=f"send_{self.name}"):
                success = await self._send_notification(alert)
                
                if success:
                    self.logger.info(f"Alert notification sent: {alert.name}")
                else:
                    self.logger.error(f"Failed to send alert notification: {alert.name}")
                
                return success
        
        except Exception as e:
            self.logger.error(f"Error sending notification: {str(e)}")
            return False
    
    async def _send_notification(self, alert: Alert) -> bool:
        """Override this method to implement specific notification logic."""
        raise NotImplementedError


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""
    
    def __init__(self,
                 smtp_server: str,
                 smtp_port: int,
                 username: str,
                 password: str,
                 from_email: str,
                 to_emails: List[str],
                 use_tls: bool = True):
        super().__init__("email")
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        self.use_tls = use_tls
    
    async def _send_notification(self, alert: Alert) -> bool:
        """Send email notification."""
        try:
            # Create email message
            msg = MimeMultipart()
            msg['From'] = self.from_email
            msg['To'] = ", ".join(self.to_emails)
            msg['Subject'] = f"[QFLARE Alert] {alert.severity.value.upper()}: {alert.name}"
            
            # Create email body
            body = f"""
QFLARE Alert Notification

Alert: {alert.name}
Severity: {alert.severity.value.upper()}
Status: {alert.status.value}
Time: {alert.timestamp.isoformat()}
Source: {alert.source}

Message: {alert.message}

Details:
{json.dumps(alert.details, indent=2) if alert.details else 'No additional details'}

---
This is an automated alert from QFLARE monitoring system.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            
            if self.use_tls:
                server.starttls()
            
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to send email: {str(e)}")
            return False


class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel (Slack, Teams, etc.)."""
    
    def __init__(self, webhook_url: str, webhook_type: str = "generic"):
        super().__init__(f"webhook_{webhook_type}")
        self.webhook_url = webhook_url
        self.webhook_type = webhook_type
    
    async def _send_notification(self, alert: Alert) -> bool:
        """Send webhook notification."""
        try:
            payload = self._format_payload(alert)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status < 400
        
        except Exception as e:
            self.logger.error(f"Failed to send webhook: {str(e)}")
            return False
    
    def _format_payload(self, alert: Alert) -> Dict[str, Any]:
        """Format payload for specific webhook type."""
        if self.webhook_type == "slack":
            return {
                "text": f"QFLARE Alert: {alert.name}",
                "attachments": [
                    {
                        "color": self._get_slack_color(alert.severity),
                        "fields": [
                            {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                            {"title": "Status", "value": alert.status.value, "short": True},
                            {"title": "Source", "value": alert.source, "short": True},
                            {"title": "Time", "value": alert.timestamp.isoformat(), "short": True},
                            {"title": "Message", "value": alert.message, "short": False}
                        ]
                    }
                ]
            }
        
        # Generic webhook format
        return alert.to_dict()
    
    def _get_slack_color(self, severity: AlertSeverity) -> str:
        """Get Slack color for severity level."""
        colors = {
            AlertSeverity.INFO: "good",
            AlertSeverity.WARNING: "warning", 
            AlertSeverity.ERROR: "danger",
            AlertSeverity.CRITICAL: "#ff0000"
        }
        return colors.get(severity, "warning")


class FileNotificationChannel(NotificationChannel):
    """File-based notification channel for testing/debugging."""
    
    def __init__(self, file_path: str):
        super().__init__("file")
        self.file_path = file_path
    
    async def _send_notification(self, alert: Alert) -> bool:
        """Write alert to file."""
        try:
            alert_data = alert.to_dict()
            alert_line = json.dumps(alert_data) + "\n"
            
            async with aiofiles.open(self.file_path, "a", encoding="utf-8") as f:
                await f.write(alert_line)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to write to file: {str(e)}")
            return False


class AlertManager:
    """Manages alert rules, notifications, and alert lifecycle."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.rules: List[AlertRule] = []
        self.channels: List[NotificationChannel] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.max_history_size = 10000
        self.running = False
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules.append(rule)
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def add_channel(self, channel: NotificationChannel):
        """Add a notification channel."""
        self.channels.append(channel)
        self.logger.info(f"Added notification channel: {channel.name}")
    
    def setup_default_rules(self):
        """Setup default alert rules."""
        # Health check alerts
        self.add_rule(HealthCheckAlertRule("database", AlertSeverity.CRITICAL))
        self.add_rule(HealthCheckAlertRule("redis", AlertSeverity.ERROR))
        self.add_rule(HealthCheckAlertRule("system_resources", AlertSeverity.WARNING))
        
        # Metric threshold alerts
        self.add_rule(MetricThresholdAlertRule("cpu_usage", 90.0, "greater_than", AlertSeverity.WARNING))
        self.add_rule(MetricThresholdAlertRule("memory_usage", 95.0, "greater_than", AlertSeverity.ERROR))
        self.add_rule(MetricThresholdAlertRule("disk_usage", 95.0, "greater_than", AlertSeverity.CRITICAL))
        
        # Error rate alert
        self.add_rule(ErrorRateAlertRule(error_threshold_per_minute=20, severity=AlertSeverity.ERROR))
    
    async def evaluate_rules(self) -> List[Alert]:
        """Evaluate all rules and return new alerts."""
        new_alerts = []
        
        with log_context(component="alert_manager", operation="evaluate_rules"):
            self.logger.debug("Evaluating alert rules")
            
            for rule in self.rules:
                try:
                    alert = await rule.evaluate()
                    if alert:
                        new_alerts.append(alert)
                        self.active_alerts[alert.id] = alert
                        self.alert_history.append(alert)
                        
                        # Trim history if needed
                        if len(self.alert_history) > self.max_history_size:
                            self.alert_history = self.alert_history[-self.max_history_size:]
                
                except Exception as e:
                    self.logger.error(f"Error evaluating rule {rule.name}: {str(e)}")
        
        return new_alerts
    
    async def send_notifications(self, alerts: List[Alert]):
        """Send notifications for alerts."""
        if not alerts or not self.channels:
            return
        
        with log_context(component="alert_manager", operation="send_notifications"):
            self.logger.info(f"Sending notifications for {len(alerts)} alerts")
            
            for alert in alerts:
                # Send to all channels
                tasks = [channel.send(alert) for channel in self.channels]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                success_count = sum(1 for result in results if result is True)
                self.logger.info(f"Alert {alert.id} sent to {success_count}/{len(self.channels)} channels")
    
    def resolve_alert(self, alert_id: str, message: str = "Manually resolved"):
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now(timezone.utc)
            
            self.logger.info(f"Alert resolved: {alert_id} - {message}")
            del self.active_alerts[alert_id]
    
    def suppress_alert(self, alert_id: str, duration_minutes: int = 60):
        """Suppress an active alert for a period of time."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            alert.suppressed_until = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
            
            self.logger.info(f"Alert suppressed: {alert_id} for {duration_minutes} minutes")
    
    def get_active_alerts(self) -> Dict[str, Dict[str, Any]]:
        """Get all active alerts."""
        now = datetime.now(timezone.utc)
        
        # Check for suppressed alerts that should be reactivated
        for alert in list(self.active_alerts.values()):
            if (alert.status == AlertStatus.SUPPRESSED and 
                alert.suppressed_until and 
                now > alert.suppressed_until):
                alert.status = AlertStatus.ACTIVE
                alert.suppressed_until = None
        
        return {alert_id: alert.to_dict() for alert_id, alert in self.active_alerts.items()}
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history."""
        return [alert.to_dict() for alert in self.alert_history[-limit:]]
    
    async def start_monitoring(self, check_interval_seconds: int = 60):
        """Start the alert monitoring loop."""
        self.running = True
        self.logger.info("Alert monitoring started")
        
        while self.running:
            try:
                # Evaluate rules and send notifications
                new_alerts = await self.evaluate_rules()
                if new_alerts:
                    await self.send_notifications(new_alerts)
                
                await asyncio.sleep(check_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in alert monitoring loop: {str(e)}")
                await asyncio.sleep(30)  # Retry after 30 seconds on error
    
    def stop_monitoring(self):
        """Stop the alert monitoring loop."""
        self.running = False
        self.logger.info("Alert monitoring stopped")


# Global alert manager
alert_manager = AlertManager()


def initialize_alerting(
    email_config: Optional[Dict[str, Any]] = None,
    webhook_urls: Optional[List[str]] = None,
    file_path: Optional[str] = None
):
    """Initialize the alerting system."""
    
    # Setup default rules
    alert_manager.setup_default_rules()
    
    # Setup notification channels
    if email_config:
        channel = EmailNotificationChannel(**email_config)
        alert_manager.add_channel(channel)
    
    if webhook_urls:
        for url in webhook_urls:
            channel = WebhookNotificationChannel(url, "slack")
            alert_manager.add_channel(channel)
    
    if file_path:
        channel = FileNotificationChannel(file_path)
        alert_manager.add_channel(channel)
    
    logger = get_logger(__name__)
    logger.info("Alert system initialized")