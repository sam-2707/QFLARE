"""
Structured logging system for QFLARE with correlation IDs and comprehensive tracing.

This module provides centralized logging configuration with JSON formatting,
correlation tracking, and integration with monitoring systems.
"""

import logging
import logging.config
import json
import sys
import os
import uuid
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from functools import wraps
from contextlib import contextmanager
import traceback
from dataclasses import dataclass, asdict


@dataclass
class LogContext:
    """Log context with correlation and tracing information."""
    correlation_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


class CorrelationContext:
    """Thread-local correlation context manager."""
    
    def __init__(self):
        self._local = threading.local()
    
    def set_context(self, context: LogContext):
        """Set correlation context for current thread."""
        self._local.context = context
    
    def get_context(self) -> Optional[LogContext]:
        """Get correlation context for current thread."""
        return getattr(self._local, 'context', None)
    
    def clear_context(self):
        """Clear correlation context for current thread."""
        if hasattr(self._local, 'context'):
            delattr(self._local, 'context')
    
    def get_correlation_id(self) -> Optional[str]:
        """Get correlation ID from current context."""
        context = self.get_context()
        return context.correlation_id if context else None
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID in current context."""
        context = self.get_context()
        if context:
            context.correlation_id = correlation_id
        else:
            self.set_context(LogContext(correlation_id=correlation_id))


# Global correlation context
correlation_context = CorrelationContext()


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging with correlation support."""
    
    def __init__(self, service_name: str = "qflare", version: str = "unknown"):
        super().__init__()
        self.service_name = service_name
        self.version = version
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with structured fields."""
        
        # Base log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "version": self.version,
            "thread": record.thread,
            "thread_name": record.threadName,
            "process": record.process,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add correlation context if available
        context = correlation_context.get_context()
        if context:
            log_entry.update({
                "correlation_id": context.correlation_id,
                "user_id": context.user_id,
                "session_id": context.session_id,
                "request_id": context.request_id,
                "component": context.component,
                "operation": context.operation,
                "trace_id": context.trace_id,
                "span_id": context.span_id,
            })
        
        # Add exception information
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in [
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                'thread', 'threadName', 'processName', 'process', 'message', 'exc_info',
                'exc_text', 'stack_info', 'getMessage'
            ]:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        # Performance metrics if available
        if hasattr(record, 'duration'):
            log_entry["performance"] = {
                "duration_ms": record.duration * 1000
            }
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class QFLARELogger:
    """Enhanced logger with correlation support and structured logging."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.name = name
    
    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """Log message with automatic context injection."""
        # Extract extra fields for structured logging
        extra = kwargs.pop('extra', {})
        
        # Add correlation context
        context = correlation_context.get_context()
        if context:
            extra.update(asdict(context))
        
        # Add performance data if provided
        if 'duration' in kwargs:
            extra['duration'] = kwargs.pop('duration')
        
        # Add component/operation if not in context
        if 'component' in kwargs:
            extra['component'] = kwargs.pop('component')
        if 'operation' in kwargs:
            extra['operation'] = kwargs.pop('operation')
        
        kwargs['extra'] = extra
        self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        self._log_with_context(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message."""
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        """Log exception with traceback."""
        kwargs['exc_info'] = True
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
    
    def audit(self, event_type: str, success: bool, **kwargs):
        """Log audit event with standardized format."""
        self.info(
            f"Audit event: {event_type}",
            extra={
                'audit_event': {
                    'type': event_type,
                    'success': success,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    **kwargs
                }
            }
        )
    
    def performance(self, operation: str, duration: float, **kwargs):
        """Log performance metric."""
        self.info(
            f"Performance: {operation} completed in {duration:.3f}s",
            duration=duration,
            operation=operation,
            extra=kwargs
        )
    
    def security(self, event_type: str, severity: str, details: Dict[str, Any]):
        """Log security event."""
        self.warning(
            f"Security event: {event_type}",
            extra={
                'security_event': {
                    'type': event_type,
                    'severity': severity,
                    'details': details,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            }
        )


def get_logger(name: str) -> QFLARELogger:
    """Get enhanced logger instance."""
    return QFLARELogger(name)


@contextmanager
def log_context(
    correlation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    component: Optional[str] = None,
    operation: Optional[str] = None
):
    """Context manager for setting log correlation context."""
    if not correlation_id:
        correlation_id = str(uuid.uuid4())
    
    context = LogContext(
        correlation_id=correlation_id,
        user_id=user_id,
        session_id=session_id,
        request_id=request_id,
        component=component,
        operation=operation
    )
    
    # Store previous context
    previous_context = correlation_context.get_context()
    
    try:
        correlation_context.set_context(context)
        yield correlation_id
    finally:
        if previous_context:
            correlation_context.set_context(previous_context)
        else:
            correlation_context.clear_context()


def log_function_calls(component: str = None):
    """Decorator to log function entry/exit with performance timing."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            operation = func.__name__
            
            with log_context(component=component, operation=operation):
                start_time = datetime.now()
                
                logger.debug(
                    f"Function {operation} started",
                    extra={
                        'function_call': {
                            'function': operation,
                            'args_count': len(args),
                            'kwargs_keys': list(kwargs.keys()),
                            'start_time': start_time.isoformat()
                        }
                    }
                )
                
                try:
                    result = func(*args, **kwargs)
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    logger.debug(
                        f"Function {operation} completed successfully",
                        duration=duration,
                        extra={
                            'function_call': {
                                'function': operation,
                                'success': True,
                                'duration_ms': duration * 1000
                            }
                        }
                    )
                    
                    return result
                    
                except Exception as e:
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    logger.error(
                        f"Function {operation} failed: {str(e)}",
                        duration=duration,
                        extra={
                            'function_call': {
                                'function': operation,
                                'success': False,
                                'error_type': type(e).__name__,
                                'error_message': str(e),
                                'duration_ms': duration * 1000
                            }
                        }
                    )
                    raise
        
        return wrapper
    return decorator


def log_async_function_calls(component: str = None):
    """Decorator to log async function entry/exit with performance timing."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            operation = func.__name__
            
            with log_context(component=component, operation=operation):
                start_time = datetime.now()
                
                logger.debug(
                    f"Async function {operation} started",
                    extra={
                        'function_call': {
                            'function': operation,
                            'async': True,
                            'args_count': len(args),
                            'kwargs_keys': list(kwargs.keys()),
                            'start_time': start_time.isoformat()
                        }
                    }
                )
                
                try:
                    result = await func(*args, **kwargs)
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    logger.debug(
                        f"Async function {operation} completed successfully",
                        duration=duration,
                        extra={
                            'function_call': {
                                'function': operation,
                                'async': True,
                                'success': True,
                                'duration_ms': duration * 1000
                            }
                        }
                    )
                    
                    return result
                    
                except Exception as e:
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    logger.error(
                        f"Async function {operation} failed: {str(e)}",
                        duration=duration,
                        extra={
                            'function_call': {
                                'function': operation,
                                'async': True,
                                'success': False,
                                'error_type': type(e).__name__,
                                'error_message': str(e),
                                'duration_ms': duration * 1000
                            }
                        }
                    )
                    raise
        
        return wrapper
    return decorator


class LoggingConfig:
    """Centralized logging configuration."""
    
    @staticmethod
    def setup_logging(
        service_name: str = "qflare",
        version: str = "unknown",
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 5
    ):
        """Setup centralized logging configuration."""
        
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "structured": {
                    "()": StructuredFormatter,
                    "service_name": service_name,
                    "version": version
                },
                "simple": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": sys.stdout,
                    "formatter": "structured",
                    "level": log_level
                }
            },
            "loggers": {
                "": {  # root logger
                    "handlers": ["console"],
                    "level": log_level,
                    "propagate": False
                },
                "qflare": {
                    "handlers": ["console"],
                    "level": log_level,
                    "propagate": False
                }
            }
        }
        
        # Add file handler if specified
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            config["handlers"]["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": log_file,
                "maxBytes": max_file_size,
                "backupCount": backup_count,
                "formatter": "structured",
                "level": log_level
            }
            
            # Add file handler to loggers
            config["loggers"][""]["handlers"].append("file")
            config["loggers"]["qflare"]["handlers"].append("file")
        
        # Configure third-party loggers
        config["loggers"].update({
            "uvicorn": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False
            },
            "fastapi": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False
            },
            "sqlalchemy.engine": {
                "handlers": ["console"],
                "level": "WARNING",
                "propagate": False
            },
            "redis": {
                "handlers": ["console"],
                "level": "WARNING",
                "propagate": False
            }
        })
        
        logging.config.dictConfig(config)
        
        # Log configuration success
        logger = get_logger(__name__)
        logger.info(
            "Logging system initialized",
            extra={
                "config": {
                    "service_name": service_name,
                    "version": version,
                    "log_level": log_level,
                    "log_file": log_file,
                    "structured_logging": True
                }
            }
        )


# Error tracking and aggregation

class ErrorTracker:
    """Track and aggregate application errors."""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.recent_errors: List[Dict[str, Any]] = []
        self.max_recent_errors = 1000
    
    def track_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """Track an error occurrence."""
        error_key = f"{error_type}:{hash(error_message) % 10000}"
        
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        error_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": error_type,
            "message": error_message,
            "count": self.error_counts[error_key],
            "context": context or {}
        }
        
        self.recent_errors.append(error_record)
        
        # Keep only recent errors
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors = self.recent_errors[-self.max_recent_errors:]
        
        # Log high-frequency errors
        if self.error_counts[error_key] % 10 == 0:
            logger = get_logger(__name__)
            logger.warning(
                f"Recurring error detected: {error_type} occurred {self.error_counts[error_key]} times",
                extra={"error_tracking": error_record}
            )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics."""
        return {
            "total_error_types": len(self.error_counts),
            "total_errors": sum(self.error_counts.values()),
            "top_errors": sorted(
                [(k, v) for k, v in self.error_counts.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "recent_errors_count": len(self.recent_errors)
        }


# Global error tracker
error_tracker = ErrorTracker()


def initialize_logging(
    service_name: str = "qflare",
    version: str = "unknown",
    log_level: str = "INFO",
    log_file: Optional[str] = None
):
    """Initialize the logging system."""
    LoggingConfig.setup_logging(
        service_name=service_name,
        version=version,
        log_level=log_level,
        log_file=log_file
    )