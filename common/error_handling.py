"""
Enhanced error handling and retry logic for QFLARE components.
"""

import logging
import time
import asyncio
from typing import Any, Callable, Optional, Dict
from functools import wraps
import traceback

logger = logging.getLogger(__name__)

class QFlareError(Exception):
    """Base exception for QFLARE specific errors."""
    pass

class NetworkError(QFlareError):
    """Network-related errors."""
    pass

class AuthenticationError(QFlareError):
    """Authentication-related errors."""
    pass

class TrainingError(QFlareError):
    """Training-related errors."""
    pass

class AggregationError(QFlareError):
    """Model aggregation errors."""
    pass

class RetryConfig:
    """Configuration for retry logic."""
    
    def __init__(self, 
                 max_retries: int = 3,
                 initial_delay: float = 1.0,
                 backoff_factor: float = 2.0,
                 max_delay: float = 60.0,
                 retriable_exceptions: tuple = None):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.retriable_exceptions = retriable_exceptions or (Exception,)

def retry_on_failure(config: RetryConfig = None):
    """
    Decorator to add retry logic to functions.
    
    Args:
        config: RetryConfig instance with retry parameters
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = config.initial_delay
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retriable_exceptions as e:
                    if attempt == config.max_retries:
                        logger.error(f"Function {func.__name__} failed after {config.max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"Attempt {attempt + 1}/{config.max_retries + 1} failed for {func.__name__}: {e}")
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    
                    time.sleep(delay)
                    delay = min(delay * config.backoff_factor, config.max_delay)
                except Exception as e:
                    # Non-retriable exception
                    logger.error(f"Non-retriable error in {func.__name__}: {e}")
                    raise
        
        return wrapper
    return decorator

def async_retry_on_failure(config: RetryConfig = None):
    """
    Async version of retry decorator.
    
    Args:
        config: RetryConfig instance with retry parameters
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = config.initial_delay
            
            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except config.retriable_exceptions as e:
                    if attempt == config.max_retries:
                        logger.error(f"Async function {func.__name__} failed after {config.max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"Attempt {attempt + 1}/{config.max_retries + 1} failed for {func.__name__}: {e}")
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    
                    await asyncio.sleep(delay)
                    delay = min(delay * config.backoff_factor, config.max_delay)
                except Exception as e:
                    # Non-retriable exception
                    logger.error(f"Non-retriable error in {func.__name__}: {e}")
                    raise
        
        return wrapper
    return decorator

class SafeExecutor:
    """Safely execute functions with error handling and logging."""
    
    @staticmethod
    def execute_with_fallback(primary_func: Callable, 
                            fallback_func: Callable = None,
                            error_message: str = None,
                            *args, **kwargs) -> Any:
        """
        Execute primary function with optional fallback.
        
        Args:
            primary_func: Primary function to execute
            fallback_func: Fallback function if primary fails
            error_message: Custom error message
            *args, **kwargs: Arguments for primary function
            
        Returns:
            Result from primary or fallback function
        """
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            error_msg = error_message or f"Primary function {primary_func.__name__} failed"
            logger.error(f"{error_msg}: {e}")
            
            if fallback_func:
                try:
                    logger.info(f"Attempting fallback function {fallback_func.__name__}")
                    return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback function {fallback_func.__name__} also failed: {fallback_error}")
                    raise QFlareError(f"Both primary and fallback functions failed") from e
            else:
                raise
    
    @staticmethod
    async def async_execute_with_fallback(primary_func: Callable,
                                        fallback_func: Callable = None,
                                        error_message: str = None,
                                        *args, **kwargs) -> Any:
        """Async version of execute_with_fallback."""
        try:
            if asyncio.iscoroutinefunction(primary_func):
                return await primary_func(*args, **kwargs)
            else:
                return primary_func(*args, **kwargs)
        except Exception as e:
            error_msg = error_message or f"Primary function {primary_func.__name__} failed"
            logger.error(f"{error_msg}: {e}")
            
            if fallback_func:
                try:
                    logger.info(f"Attempting fallback function {fallback_func.__name__}")
                    if asyncio.iscoroutinefunction(fallback_func):
                        return await fallback_func(*args, **kwargs)
                    else:
                        return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback function {fallback_func.__name__} also failed: {fallback_error}")
                    raise QFlareError(f"Both primary and fallback functions failed") from e
            else:
                raise

class HealthChecker:
    """Component health checking utilities."""
    
    @staticmethod
    def check_component_health(component_name: str, 
                             health_check_func: Callable,
                             timeout: float = 30.0) -> Dict[str, Any]:
        """
        Check health of a component.
        
        Args:
            component_name: Name of component to check
            health_check_func: Function that returns health status
            timeout: Timeout for health check
            
        Returns:
            Dictionary with health status
        """
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(health_check_func):
                # Handle async functions
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        asyncio.wait_for(health_check_func(), timeout=timeout)
                    )
                finally:
                    loop.close()
            else:
                # Handle sync functions
                result = health_check_func()
            
            elapsed = time.time() - start_time
            
            return {
                "component": component_name,
                "status": "healthy",
                "response_time": elapsed,
                "details": result if isinstance(result, dict) else {"result": result},
                "timestamp": time.time()
            }
            
        except asyncio.TimeoutError:
            return {
                "component": component_name,
                "status": "timeout", 
                "response_time": timeout,
                "error": f"Health check timed out after {timeout}s",
                "timestamp": time.time()
            }
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "component": component_name,
                "status": "unhealthy",
                "response_time": elapsed,
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": time.time()
            }

class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self,
                 failure_threshold: int = 5,
                 timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def _can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        elif self.state == "HALF_OPEN":
            return True
        return False
    
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if not self._can_execute():
            raise QFlareError(f"Circuit breaker is OPEN, cannot execute {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
        except Exception as e:
            # Unexpected exception, don't count as failure
            logger.warning(f"Unexpected exception in circuit breaker: {e}")
            raise

def validate_input(validation_func: Callable) -> Callable:
    """
    Decorator to validate function inputs.
    
    Args:
        validation_func: Function that validates inputs and returns True/False
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                if not validation_func(*args, **kwargs):
                    raise ValueError(f"Input validation failed for {func.__name__}")
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Input validation error in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator

def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    return wrapper

def catch_and_log_exceptions(logger_instance: logging.Logger = None):
    """Decorator to catch and log exceptions."""
    if logger_instance is None:
        logger_instance = logger
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger_instance.error(f"Exception in {func.__name__}: {e}")
                logger_instance.debug(f"Full traceback: {traceback.format_exc()}")
                raise
        return wrapper
    return decorator