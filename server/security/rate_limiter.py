"""
Enhanced Rate Limiting System for QFLARE.

This module provides advanced rate limiting with security features including
IP-based limiting, device-based limiting, and adaptive rate limiting.
"""

import time
import logging
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import redis
from fastapi import Request, HTTPException, status

logger = logging.getLogger(__name__)

# Rate limiting configuration
RATE_LIMIT_CONFIG = {
    "enrollment": {"requests": 5, "window": 300},  # 5 requests per 5 minutes
    "challenge": {"requests": 30, "window": 60},   # 30 requests per minute
    "model_submit": {"requests": 10, "window": 300},  # 10 requests per 5 minutes
    "model_download": {"requests": 60, "window": 60},  # 60 requests per minute
    "health_check": {"requests": 120, "window": 60},  # 120 requests per minute
    "default": {"requests": 100, "window": 60}  # 100 requests per minute
}

# Security thresholds
SECURITY_THRESHOLDS = {
    "max_failed_attempts": 5,
    "lockout_duration": 900,  # 15 minutes
    "suspicious_activity_threshold": 10,
    "ip_whitelist": [],  # Add trusted IPs here
    "ip_blacklist": []   # Add blocked IPs here
}


@dataclass
class RateLimitRecord:
    """Record for tracking rate limit data."""
    requests: deque = field(default_factory=lambda: deque())
    failed_attempts: int = 0
    last_failed_attempt: float = 0
    is_locked: bool = False
    lockout_until: float = 0
    suspicious_activity_count: int = 0


class EnhancedRateLimiter:
    """Enhanced rate limiter with security features."""
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize the rate limiter.
        
        Args:
            redis_url: Redis URL for distributed rate limiting
        """
        self.redis_client = None
        self.use_redis = False
        
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                self.use_redis = True
                logger.info("Using Redis for distributed rate limiting")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, using in-memory storage")
        
        # In-memory storage for rate limiting
        self.rate_limit_data: Dict[str, RateLimitRecord] = defaultdict(RateLimitRecord)
        self.lock = threading.Lock()
        
        # Security monitoring
        self.suspicious_ips: Dict[str, int] = defaultdict(int)
        self.blocked_ips: Dict[str, float] = {}
        
        # Cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_records, daemon=True)
        self.cleanup_thread.start()
    
    def _get_client_identifier(self, request: Request) -> str:
        """Get a unique identifier for the client.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Client identifier string
        """
        # Try to get real IP (considering proxies)
        client_ip = request.headers.get("X-Forwarded-For", request.client.host)
        if client_ip and "," in client_ip:
            client_ip = client_ip.split(",")[0].strip()
        
        # Add user agent for additional uniqueness
        user_agent = request.headers.get("User-Agent", "")
        
        # Create identifier
        identifier = f"{client_ip}:{hashlib.md5(user_agent.encode()).hexdigest()[:8]}"
        return identifier
    
    def _get_rate_limit_key(self, endpoint: str, client_id: str) -> str:
        """Get Redis key for rate limiting.
        
        Args:
            endpoint: API endpoint
            client_id: Client identifier
            
        Returns:
            Redis key string
        """
        return f"rate_limit:{endpoint}:{client_id}"
    
    def _check_redis_rate_limit(self, endpoint: str, client_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using Redis.
        
        Args:
            endpoint: API endpoint
            client_id: Client identifier
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        try:
            key = self._get_rate_limit_key(endpoint, client_id)
            config = RATE_LIMIT_CONFIG.get(endpoint, RATE_LIMIT_CONFIG["default"])
            
            current_time = time.time()
            window_start = current_time - config["window"]
            
            # Get current requests from Redis
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zadd(key, {str(current_time): current_time})
            pipe.zcard(key)
            pipe.expire(key, config["window"])
            results = pipe.execute()
            
            request_count = results[2]
            
            # Check if limit exceeded
            if request_count > config["requests"]:
                return False, {
                    "limit": config["requests"],
                    "window": config["window"],
                    "remaining": 0,
                    "reset_time": window_start + config["window"]
                }
            
            return True, {
                "limit": config["requests"],
                "window": config["window"],
                "remaining": config["requests"] - request_count,
                "reset_time": window_start + config["window"]
            }
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            return True, {}  # Allow request if Redis fails
    
    def _check_memory_rate_limit(self, endpoint: str, client_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using in-memory storage.
        
        Args:
            endpoint: API endpoint
            client_id: Client identifier
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        with self.lock:
            record = self.rate_limit_data[client_id]
            config = RATE_LIMIT_CONFIG.get(endpoint, RATE_LIMIT_CONFIG["default"])
            
            current_time = time.time()
            window_start = current_time - config["window"]
            
            # Remove old requests
            while record.requests and record.requests[0] < window_start:
                record.requests.popleft()
            
            # Check if limit exceeded
            if len(record.requests) >= config["requests"]:
                return False, {
                    "limit": config["requests"],
                    "window": config["window"],
                    "remaining": 0,
                    "reset_time": window_start + config["window"]
                }
            
            # Add current request
            record.requests.append(current_time)
            
            return True, {
                "limit": config["requests"],
                "window": config["window"],
                "remaining": config["requests"] - len(record.requests),
                "reset_time": window_start + config["window"]
            }
    
    def check_rate_limit(self, request: Request, endpoint: str) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit for a request.
        
        Args:
            request: FastAPI request object
            endpoint: API endpoint
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        client_id = self._get_client_identifier(request)
        
        # Check IP blacklist
        client_ip = request.headers.get("X-Forwarded-For", request.client.host)
        if client_ip in SECURITY_THRESHOLDS["ip_blacklist"]:
            logger.warning(f"Blocked request from blacklisted IP: {client_ip}")
            return False, {"error": "IP address is blacklisted"}
        
        # Check if IP is temporarily blocked
        if client_ip in self.blocked_ips:
            if time.time() < self.blocked_ips[client_ip]:
                logger.warning(f"Blocked request from temporarily blocked IP: {client_ip}")
                return False, {"error": "IP address is temporarily blocked"}
            else:
                del self.blocked_ips[client_ip]
        
        # Check rate limit
        if self.use_redis:
            allowed, rate_limit_info = self._check_redis_rate_limit(endpoint, client_id)
        else:
            allowed, rate_limit_info = self._check_memory_rate_limit(endpoint, client_id)
        
        # Log suspicious activity
        if not allowed:
            self._record_failed_attempt(client_ip, endpoint)
        
        return allowed, rate_limit_info
    
    def _record_failed_attempt(self, client_ip: str, endpoint: str):
        """Record a failed attempt for security monitoring.
        
        Args:
            client_ip: Client IP address
            endpoint: API endpoint
        """
        current_time = time.time()
        
        # Increment suspicious activity counter
        self.suspicious_ips[client_ip] += 1
        
        # Check if IP should be blocked
        if self.suspicious_ips[client_ip] >= SECURITY_THRESHOLDS["suspicious_activity_threshold"]:
            self.blocked_ips[client_ip] = current_time + SECURITY_THRESHOLDS["lockout_duration"]
            logger.warning(f"IP {client_ip} blocked due to suspicious activity")
    
    def _cleanup_old_records(self):
        """Clean up old rate limit records."""
        while True:
            try:
                time.sleep(300)  # Clean up every 5 minutes
                
                current_time = time.time()
                
                with self.lock:
                    # Clean up old records
                    keys_to_remove = []
                    for client_id, record in self.rate_limit_data.items():
                        # Remove old requests
                        while record.requests and record.requests[0] < current_time - 3600:
                            record.requests.popleft()
                        
                        # Remove empty records
                        if not record.requests and not record.is_locked:
                            keys_to_remove.append(client_id)
                    
                    for key in keys_to_remove:
                        del self.rate_limit_data[key]
                    
                    # Clean up old blocked IPs
                    blocked_to_remove = []
                    for ip, block_until in self.blocked_ips.items():
                        if current_time > block_until:
                            blocked_to_remove.append(ip)
                    
                    for ip in blocked_to_remove:
                        del self.blocked_ips[ip]
                        if ip in self.suspicious_ips:
                            del self.suspicious_ips[ip]
                
                logger.debug("Cleaned up old rate limit records")
                
            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}")
    
    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics.
        
        Returns:
            Dictionary with rate limiting statistics
        """
        with self.lock:
            total_records = len(self.rate_limit_data)
            total_blocked_ips = len(self.blocked_ips)
            total_suspicious_ips = len(self.suspicious_ips)
            
            return {
                "total_records": total_records,
                "blocked_ips": total_blocked_ips,
                "suspicious_ips": total_suspicious_ips,
                "blocked_ips_list": list(self.blocked_ips.keys()),
                "suspicious_ips_list": list(self.suspicious_ips.keys())
            }
    
    def reset_rate_limits(self, client_id: Optional[str] = None):
        """Reset rate limits for a client or all clients.
        
        Args:
            client_id: Client identifier to reset, or None for all clients
        """
        with self.lock:
            if client_id:
                if client_id in self.rate_limit_data:
                    del self.rate_limit_data[client_id]
                    logger.info(f"Reset rate limits for client: {client_id}")
            else:
                self.rate_limit_data.clear()
                logger.info("Reset all rate limits")


# Global rate limiter instance
rate_limiter = EnhancedRateLimiter()


def rate_limit_middleware(endpoint: str):
    """Decorator for rate limiting endpoints.
    
    Args:
        endpoint: Endpoint name for rate limiting configuration
    """
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            allowed, rate_limit_info = rate_limiter.check_rate_limit(request, endpoint)
            
            if not allowed:
                error_detail = rate_limit_info.get("error", "Rate limit exceeded")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": error_detail,
                        "rate_limit_info": rate_limit_info
                    }
                )
            
            # Add rate limit headers
            response = await func(request, *args, **kwargs)
            
            # Add rate limit headers to response
            if hasattr(response, 'headers'):
                response.headers["X-RateLimit-Limit"] = str(rate_limit_info.get("limit", 0))
                response.headers["X-RateLimit-Remaining"] = str(rate_limit_info.get("remaining", 0))
                response.headers["X-RateLimit-Reset"] = str(int(rate_limit_info.get("reset_time", 0)))
            
            return response
        
        return wrapper
    return decorator


def get_rate_limit_stats() -> Dict[str, Any]:
    """Get rate limiting statistics."""
    return rate_limiter.get_rate_limit_stats()


def reset_rate_limits(client_id: Optional[str] = None):
    """Reset rate limits."""
    rate_limiter.reset_rate_limits(client_id) 