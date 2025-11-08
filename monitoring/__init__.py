"""
Monitoring package initialization and main interface.

This module provides the main entry point for the QFLARE monitoring system,
integrating metrics collection, structured logging, health checks, and alerting.
"""

import asyncio
import os
from typing import Optional, Dict, Any, List
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from monitoring.metrics import qflare_metrics, initialize_metrics
from monitoring.logging import initialize_logging, get_logger, log_context
from monitoring.health import health_manager, initialize_health_checks, run_background_health_checks
from monitoring.alerts import alert_manager, initialize_alerting


# Initialize logging first
log_level = os.getenv("LOG_LEVEL", "INFO")
log_file = os.getenv("LOG_FILE")

initialize_logging(
    service_name="qflare",
    version=os.getenv("QFLARE_VERSION", "1.0.0"),
    log_level=log_level,
    log_file=log_file
)

logger = get_logger(__name__)


class MonitoringSystem:
    """Main monitoring system coordinator."""
    
    def __init__(self):
        self.initialized = False
        self.background_tasks = []
        
    async def initialize(self, 
                        redis_url: str = "redis://localhost:6379",
                        enable_metrics: bool = True,
                        enable_health_checks: bool = True,
                        enable_alerting: bool = True,
                        alert_config: Optional[Dict[str, Any]] = None):
        """Initialize all monitoring components."""
        
        with log_context(component="monitoring", operation="initialize"):
            logger.info("Initializing monitoring system")
            
            try:
                # Initialize metrics collection
                if enable_metrics:
                    initialize_metrics()
                    logger.info("Metrics collection initialized")
                
                # Initialize health checks
                if enable_health_checks:
                    initialize_health_checks(redis_url)
                    logger.info("Health check system initialized")
                
                # Initialize alerting
                if enable_alerting:
                    alert_config = alert_config or {}
                    initialize_alerting(**alert_config)
                    logger.info("Alert system initialized")
                
                self.initialized = True
                logger.info("Monitoring system initialization completed")
                
            except Exception as e:
                logger.error(f"Failed to initialize monitoring system: {str(e)}")
                raise
    
    async def start_background_tasks(self):
        """Start background monitoring tasks."""
        if not self.initialized:
            raise RuntimeError("Monitoring system not initialized")
        
        logger.info("Starting background monitoring tasks")
        
        # Start health check monitoring
        health_task = asyncio.create_task(run_background_health_checks())
        self.background_tasks.append(health_task)
        
        # Start alert monitoring
        alert_task = asyncio.create_task(alert_manager.start_monitoring())
        self.background_tasks.append(alert_task)
        
        logger.info("Background monitoring tasks started")
    
    async def shutdown(self):
        """Shutdown monitoring system and clean up resources."""
        logger.info("Shutting down monitoring system")
        
        # Stop alert monitoring
        alert_manager.stop_monitoring()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("Monitoring system shutdown completed")


# Global monitoring system instance
monitoring_system = MonitoringSystem()


# FastAPI monitoring endpoints

def create_monitoring_app() -> FastAPI:
    """Create FastAPI app with monitoring endpoints."""
    
    app = FastAPI(
        title="QFLARE Monitoring API",
        description="Monitoring and observability endpoints for QFLARE",
        version="1.0.0"
    )
    
    @app.get("/health")
    async def health_check():
        """Get comprehensive health check status."""
        try:
            results = await health_manager.check_all()
            return JSONResponse(
                content=results,
                status_code=200 if results["status"] == "healthy" else 503
            )
        except Exception as e:
            logger.error(f"Health check endpoint error: {str(e)}")
            return JSONResponse(
                content={
                    "status": "unhealthy",
                    "message": f"Health check error: {str(e)}",
                    "timestamp": "unknown"
                },
                status_code=503
            )
    
    @app.get("/health/{checker_name}")
    async def single_health_check(checker_name: str):
        """Get status of a specific health checker."""
        result = await health_manager.check_single(checker_name)
        if not result:
            raise HTTPException(status_code=404, detail="Health checker not found")
        
        return JSONResponse(
            content=result,
            status_code=200 if result["status"] == "healthy" else 503
        )
    
    @app.get("/health/quick")
    async def quick_health_check():
        """Get cached health check results for quick status."""
        results = health_manager.get_last_results()
        return JSONResponse(
            content=results,
            status_code=200 if results["status"] == "healthy" else 503
        )
    
    @app.get("/metrics")
    async def get_metrics():
        """Get Prometheus metrics."""
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        
        metrics_data = generate_latest()
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )
    
    @app.get("/metrics/summary")
    async def get_metrics_summary():
        """Get metrics summary in JSON format."""
        try:
            # Get system metrics
            import psutil
            
            summary = {
                "system": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100,
                    "load_avg": getattr(psutil, 'getloadavg', lambda: [0, 0, 0])()
                },
                "application": {
                    "metrics_initialized": hasattr(qflare_metrics, '_initialized') and qflare_metrics._initialized,
                    "health_checks_enabled": len(health_manager.checkers) > 0,
                    "alert_rules_count": len(alert_manager.rules),
                    "active_alerts_count": len(alert_manager.active_alerts)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Metrics summary error: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to get metrics summary")
    
    @app.get("/alerts")
    async def get_alerts():
        """Get all active alerts."""
        return alert_manager.get_active_alerts()
    
    @app.get("/alerts/history")
    async def get_alert_history(limit: int = 100):
        """Get alert history."""
        return alert_manager.get_alert_history(limit)
    
    @app.post("/alerts/{alert_id}/resolve")
    async def resolve_alert(alert_id: str, message: str = "Manually resolved via API"):
        """Resolve an active alert."""
        if alert_id not in alert_manager.active_alerts:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert_manager.resolve_alert(alert_id, message)
        return {"message": f"Alert {alert_id} resolved"}
    
    @app.post("/alerts/{alert_id}/suppress")
    async def suppress_alert(alert_id: str, duration_minutes: int = 60):
        """Suppress an active alert."""
        if alert_id not in alert_manager.active_alerts:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert_manager.suppress_alert(alert_id, duration_minutes)
        return {"message": f"Alert {alert_id} suppressed for {duration_minutes} minutes"}
    
    @app.get("/status")
    async def get_system_status():
        """Get comprehensive system status."""
        try:
            # Get health status
            health_results = health_manager.get_last_results()
            
            # Get active alerts
            active_alerts = alert_manager.get_active_alerts()
            
            # Get system info
            import psutil
            
            system_info = {
                "cpu_percent": psutil.cpu_percent(),
                "memory": {
                    "percent": psutil.virtual_memory().percent,
                    "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                    "available_gb": round(psutil.virtual_memory().available / (1024**3), 2)
                },
                "disk": {
                    "percent": round((psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100, 2),
                    "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
                    "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2)
                }
            }
            
            # Determine overall status
            overall_status = "healthy"
            if health_results["status"] != "healthy" or active_alerts:
                overall_status = "degraded"
            
            critical_alerts = [
                alert for alert in active_alerts.values() 
                if alert["severity"] == "critical"
            ]
            if critical_alerts:
                overall_status = "critical"
            
            return {
                "status": overall_status,
                "timestamp": health_results["timestamp"],
                "health": health_results,
                "alerts": {
                    "active_count": len(active_alerts),
                    "critical_count": len(critical_alerts),
                    "active_alerts": active_alerts
                },
                "system": system_info,
                "monitoring": {
                    "metrics_enabled": hasattr(qflare_metrics, '_initialized') and qflare_metrics._initialized,
                    "health_checks_enabled": len(health_manager.checkers) > 0,
                    "alerting_enabled": len(alert_manager.rules) > 0
                }
            }
            
        except Exception as e:
            logger.error(f"System status error: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to get system status")
    
    return app


# Convenience functions for external use

async def initialize_monitoring(config: Dict[str, Any] = None):
    """Initialize the monitoring system with configuration."""
    config = config or {}
    
    await monitoring_system.initialize(
        redis_url=config.get("redis_url", "redis://localhost:6379"),
        enable_metrics=config.get("enable_metrics", True),
        enable_health_checks=config.get("enable_health_checks", True),
        enable_alerting=config.get("enable_alerting", True),
        alert_config=config.get("alert_config", {})
    )


async def start_monitoring():
    """Start background monitoring tasks."""
    await monitoring_system.start_background_tasks()


async def shutdown_monitoring():
    """Shutdown monitoring system."""
    await monitoring_system.shutdown()


# Context managers for monitoring

class MonitoringContext:
    """Context manager for monitoring operations."""
    
    def __init__(self, operation_name: str, component: str = "app"):
        self.operation_name = operation_name
        self.component = component
        self.logger = get_logger(component)
    
    async def __aenter__(self):
        """Start monitoring context."""
        self.logger.info(f"Starting operation: {self.operation_name}")
        qflare_metrics.operations_total.labels(
            operation=self.operation_name,
            component=self.component
        ).inc()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """End monitoring context."""
        if exc_type:
            self.logger.error(f"Operation failed: {self.operation_name} - {str(exc_val)}")
            qflare_metrics.operations_failed.labels(
                operation=self.operation_name,
                component=self.component,
                error_type=exc_type.__name__
            ).inc()
        else:
            self.logger.info(f"Operation completed: {self.operation_name}")


def monitor_operation(operation_name: str, component: str = "app"):
    """Decorator for monitoring operations."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                async with MonitoringContext(operation_name, component):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                # For sync functions, use synchronous monitoring
                logger = get_logger(component)
                logger.info(f"Starting operation: {operation_name}")
                
                try:
                    result = func(*args, **kwargs)
                    logger.info(f"Operation completed: {operation_name}")
                    return result
                except Exception as e:
                    logger.error(f"Operation failed: {operation_name} - {str(e)}")
                    raise
            
            return sync_wrapper
    
    return decorator


# Export main components
__all__ = [
    'monitoring_system',
    'create_monitoring_app', 
    'initialize_monitoring',
    'start_monitoring',
    'shutdown_monitoring',
    'MonitoringContext',
    'monitor_operation',
    'qflare_metrics',
    'health_manager',
    'alert_manager',
    'get_logger',
    'log_context'
]