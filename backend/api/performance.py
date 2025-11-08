# QFLARE API Performance Optimization

"""
Comprehensive API performance optimization with rate limiting,
request/response optimization, and intelligent caching.
"""

import time
import asyncio
import json
import gzip
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import hashlib
import logging

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.compression import CompressionMiddleware
from starlette.middleware.cors import CORSMiddleware
import redis.asyncio as redis
from pydantic import BaseModel
import httpx

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    ADAPTIVE = "adaptive"


@dataclass 
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    enable_adaptive: bool = True
    whitelist_ips: List[str] = None
    premium_multiplier: float = 2.0


@dataclass
class CompressionConfig:
    """Response compression configuration."""
    enable_gzip: bool = True
    enable_brotli: bool = True
    min_size: int = 1024
    compression_level: int = 6
    exclude_media_types: List[str] = None


@dataclass
class CacheConfig:
    """API response caching configuration."""
    enable_response_cache: bool = True
    default_ttl: int = 300  # 5 minutes
    max_cache_size: str = "100MB"
    vary_headers: List[str] = None
    cache_private_responses: bool = False


class APIMetrics:
    """API performance metrics collector."""
    
    def __init__(self):
        self.request_count = 0
        self.total_response_time = 0.0
        self.min_response_time = float('inf')
        self.max_response_time = 0.0
        self.error_count = 0
        self.endpoint_metrics: Dict[str, Dict[str, Any]] = {}
        self.status_code_counts: Dict[int, int] = {}
        
    def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        response_time: float,
        request_size: int = 0,
        response_size: int = 0
    ):
        """Record API request metrics."""
        self.request_count += 1
        self.total_response_time += response_time
        self.min_response_time = min(self.min_response_time, response_time)
        self.max_response_time = max(self.max_response_time, response_time)
        
        if status_code >= 400:
            self.error_count += 1
            
        # Track status codes
        self.status_code_counts[status_code] = self.status_code_counts.get(status_code, 0) + 1
        
        # Track endpoint-specific metrics
        endpoint_key = f"{method} {path}"
        if endpoint_key not in self.endpoint_metrics:
            self.endpoint_metrics[endpoint_key] = {
                "count": 0,
                "total_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0,
                "errors": 0,
                "total_request_size": 0,
                "total_response_size": 0
            }
            
        endpoint_stats = self.endpoint_metrics[endpoint_key]
        endpoint_stats["count"] += 1
        endpoint_stats["total_time"] += response_time
        endpoint_stats["min_time"] = min(endpoint_stats["min_time"], response_time)
        endpoint_stats["max_time"] = max(endpoint_stats["max_time"], response_time)
        endpoint_stats["total_request_size"] += request_size
        endpoint_stats["total_response_size"] += response_size
        
        if status_code >= 400:
            endpoint_stats["errors"] += 1
            
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        avg_response_time = (
            self.total_response_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        error_rate = (
            self.error_count / self.request_count 
            if self.request_count > 0 else 0
        )
        
        # Calculate endpoint averages
        endpoint_summary = {}
        for endpoint, stats in self.endpoint_metrics.items():
            if stats["count"] > 0:
                endpoint_summary[endpoint] = {
                    "count": stats["count"],
                    "avg_response_time": stats["total_time"] / stats["count"],
                    "min_response_time": stats["min_time"],
                    "max_response_time": stats["max_time"],
                    "error_rate": stats["errors"] / stats["count"],
                    "avg_request_size": stats["total_request_size"] / stats["count"],
                    "avg_response_size": stats["total_response_size"] / stats["count"]
                }
                
        return {
            "total_requests": self.request_count,
            "avg_response_time": avg_response_time,
            "min_response_time": self.min_response_time if self.min_response_time != float('inf') else 0,
            "max_response_time": self.max_response_time,
            "error_rate": error_rate,
            "error_count": self.error_count,
            "status_codes": self.status_code_counts,
            "endpoints": endpoint_summary,
            "timestamp": datetime.utcnow().isoformat()
        }


class RateLimiter:
    """Advanced rate limiter with multiple strategies."""
    
    def __init__(self, config: RateLimitConfig, redis_client: redis.Redis):
        self.config = config
        self.redis = redis_client
        
    async def is_allowed(
        self,
        identifier: str,
        endpoint: str = "default",
        user_tier: str = "basic"
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed based on rate limits."""
        try:
            if self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                return await self._sliding_window_check(identifier, endpoint, user_tier)
            elif self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return await self._token_bucket_check(identifier, endpoint, user_tier)
            elif self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
                return await self._fixed_window_check(identifier, endpoint, user_tier)
            else:
                return await self._adaptive_check(identifier, endpoint, user_tier)
                
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Fail open - allow request if rate limiter fails
            return True, {"error": "rate_limiter_error"}
            
    async def _sliding_window_check(
        self,
        identifier: str,
        endpoint: str,
        user_tier: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Sliding window rate limiting."""
        now = time.time()
        window_size = 60  # 1 minute
        
        # Get rate limits based on user tier
        limit = self._get_rate_limit(user_tier, "minute")
        
        key = f"rate_limit:sliding:{identifier}:{endpoint}"
        
        # Use Redis sorted set to track requests in time windows
        pipe = self.redis.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, now - window_size)
        
        # Count current requests in window
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(now): now})
        
        # Set expiration
        pipe.expire(key, int(window_size) + 1)
        
        results = await pipe.execute()
        current_count = results[1]
        
        info = {
            "limit": limit,
            "remaining": max(0, limit - current_count - 1),
            "reset_at": int(now + window_size),
            "strategy": "sliding_window"
        }
        
        return current_count < limit, info
        
    async def _token_bucket_check(
        self,
        identifier: str,
        endpoint: str,
        user_tier: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Token bucket rate limiting."""
        now = time.time()
        
        # Get rate limits
        rate = self._get_rate_limit(user_tier, "minute") / 60.0  # tokens per second
        capacity = self.config.burst_size
        
        key = f"rate_limit:bucket:{identifier}:{endpoint}"
        
        # Get current bucket state
        bucket_data = await self.redis.get(key)
        
        if bucket_data:
            bucket_info = json.loads(bucket_data)
            tokens = bucket_info["tokens"]
            last_update = bucket_info["last_update"]
        else:
            tokens = capacity
            last_update = now
            
        # Add tokens based on elapsed time
        elapsed = now - last_update
        tokens = min(capacity, tokens + elapsed * rate)
        
        # Check if request can be served
        if tokens >= 1:
            tokens -= 1
            allowed = True
        else:
            allowed = False
            
        # Save bucket state
        bucket_info = {
            "tokens": tokens,
            "last_update": now
        }
        await self.redis.set(key, json.dumps(bucket_info), ex=3600)  # 1 hour TTL
        
        info = {
            "tokens_remaining": int(tokens),
            "capacity": capacity,
            "refill_rate": rate,
            "strategy": "token_bucket"
        }
        
        return allowed, info
        
    async def _fixed_window_check(
        self,
        identifier: str,
        endpoint: str,
        user_tier: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Fixed window rate limiting."""
        now = time.time()
        window_start = int(now // 60) * 60  # 1-minute windows
        
        limit = self._get_rate_limit(user_tier, "minute")
        key = f"rate_limit:fixed:{identifier}:{endpoint}:{window_start}"
        
        # Increment counter
        current_count = await self.redis.incr(key)
        
        # Set expiration on first request
        if current_count == 1:
            await self.redis.expire(key, 60)
            
        info = {
            "limit": limit,
            "remaining": max(0, limit - current_count),
            "reset_at": window_start + 60,
            "strategy": "fixed_window"
        }
        
        return current_count <= limit, info
        
    async def _adaptive_check(
        self,
        identifier: str,
        endpoint: str,
        user_tier: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Adaptive rate limiting based on system load."""
        # Get base rate limit
        base_allowed, base_info = await self._sliding_window_check(identifier, endpoint, user_tier)
        
        if not base_allowed:
            return False, base_info
            
        # Check system load and adjust limits
        system_load = await self._get_system_load()
        
        if system_load > 0.8:  # High load
            # Reduce limits by 50%
            adjusted_limit = int(base_info["limit"] * 0.5)
            if base_info["limit"] - base_info["remaining"] > adjusted_limit:
                base_info["adaptive_limit"] = adjusted_limit
                base_info["load_factor"] = system_load
                return False, base_info
                
        base_info["adaptive_limit"] = base_info["limit"]
        base_info["load_factor"] = system_load
        base_info["strategy"] = "adaptive"
        
        return True, base_info
        
    def _get_rate_limit(self, user_tier: str, period: str) -> int:
        """Get rate limit for user tier and period."""
        multiplier = 1.0
        if user_tier == "premium":
            multiplier = self.config.premium_multiplier
        elif user_tier == "enterprise":
            multiplier = self.config.premium_multiplier * 2
            
        if period == "minute":
            return int(self.config.requests_per_minute * multiplier)
        elif period == "hour":
            return int(self.config.requests_per_hour * multiplier)
        elif period == "day":
            return int(self.config.requests_per_day * multiplier)
            
        return self.config.requests_per_minute
        
    async def _get_system_load(self) -> float:
        """Get current system load factor."""
        try:
            # This would integrate with your monitoring system
            # For now, return a mock value
            return 0.5
        except Exception:
            return 0.5


class ResponseCache:
    """Intelligent API response caching."""
    
    def __init__(self, config: CacheConfig, redis_client: redis.Redis):
        self.config = config
        self.redis = redis_client
        
    def _generate_cache_key(
        self,
        method: str,
        path: str,
        query_params: Dict[str, Any],
        headers: Dict[str, str]
    ) -> str:
        """Generate cache key for request."""
        # Include relevant headers in cache key
        vary_headers = self.config.vary_headers or []
        header_values = {k: headers.get(k, "") for k in vary_headers}
        
        key_data = {
            "method": method,
            "path": path,
            "params": sorted(query_params.items()) if query_params else [],
            "headers": sorted(header_values.items())
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        return f"api_cache:{key_hash}"
        
    async def get_cached_response(
        self,
        method: str,
        path: str,
        query_params: Dict[str, Any],
        headers: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Get cached API response."""
        if not self.config.enable_response_cache:
            return None
            
        cache_key = self._generate_cache_key(method, path, query_params, headers)
        
        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            
        return None
        
    async def cache_response(
        self,
        method: str,
        path: str,
        query_params: Dict[str, Any],
        headers: Dict[str, str],
        response_data: Any,
        status_code: int,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache API response."""
        if not self.config.enable_response_cache:
            return False
            
        # Don't cache error responses or private data
        if status_code >= 400:
            return False
            
        if not self.config.cache_private_responses and "private" in headers.get("cache-control", "").lower():
            return False
            
        cache_key = self._generate_cache_key(method, path, query_params, headers)
        ttl = ttl or self.config.default_ttl
        
        try:
            cache_data = {
                "data": response_data,
                "status_code": status_code,
                "cached_at": datetime.utcnow().isoformat(),
                "ttl": ttl
            }
            
            await self.redis.set(
                cache_key,
                json.dumps(cache_data, default=str),
                ex=ttl
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
            return False
            
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cached responses matching pattern."""
        try:
            keys = []
            async for key in self.redis.scan_iter(match=f"api_cache:*{pattern}*"):
                keys.append(key)
                
            if keys:
                return await self.redis.delete(*keys)
                
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            
        return 0


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Comprehensive performance middleware."""
    
    def __init__(
        self,
        app: FastAPI,
        redis_client: redis.Redis,
        rate_limit_config: RateLimitConfig,
        cache_config: CacheConfig
    ):
        super().__init__(app)
        self.metrics = APIMetrics()
        self.rate_limiter = RateLimiter(rate_limit_config, redis_client)
        self.response_cache = ResponseCache(cache_config, redis_client)
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with performance optimizations."""
        start_time = time.time()
        
        # Extract request information
        method = request.method
        path = request.url.path
        query_params = dict(request.query_params)
        headers = dict(request.headers)
        
        # Get client identifier for rate limiting
        client_ip = request.client.host if request.client else "unknown"
        client_id = headers.get("x-client-id", client_ip)
        
        # Check rate limits
        allowed, rate_info = await self.rate_limiter.is_allowed(
            client_id, path, headers.get("x-user-tier", "basic")
        )
        
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "rate_limit_info": rate_info
                },
                headers={
                    "X-RateLimit-Limit": str(rate_info.get("limit", 0)),
                    "X-RateLimit-Remaining": str(rate_info.get("remaining", 0)),
                    "X-RateLimit-Reset": str(rate_info.get("reset_at", 0))
                }
            )
            
        # Check for cached response (GET requests only)
        cached_response = None
        if method == "GET":
            cached_response = await self.response_cache.get_cached_response(
                method, path, query_params, headers
            )
            
        if cached_response:
            response = JSONResponse(
                content=cached_response["data"],
                status_code=cached_response["status_code"],
                headers={"X-Cache": "HIT"}
            )
            
            # Record metrics for cached response
            response_time = time.time() - start_time
            self.metrics.record_request(method, path, cached_response["status_code"], response_time)
            
            return response
            
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate response time and size
            response_time = time.time() - start_time
            request_size = int(headers.get("content-length", 0))
            
            # Get response size
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
                
            response_size = len(response_body)
            
            # Record metrics
            self.metrics.record_request(
                method, path, response.status_code, response_time, request_size, response_size
            )
            
            # Cache GET responses with successful status codes
            if method == "GET" and 200 <= response.status_code < 300:
                try:
                    response_data = json.loads(response_body.decode())
                    await self.response_cache.cache_response(
                        method, path, query_params, headers, response_data, response.status_code
                    )
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass  # Skip caching for non-JSON responses
                    
            # Add performance headers
            response.headers["X-Response-Time"] = f"{response_time:.3f}s"
            response.headers["X-Cache"] = "MISS"
            response.headers["X-RateLimit-Limit"] = str(rate_info.get("limit", 0))
            response.headers["X-RateLimit-Remaining"] = str(rate_info.get("remaining", 0))
            
            # Create new response with the body
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            self.metrics.record_request(method, path, 500, response_time)
            
            logger.error(f"Request processing error: {e}")
            
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
                headers={"X-Response-Time": f"{response_time:.3f}s"}
            )
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.metrics.get_summary()


class ConnectionPoolOptimizer:
    """Optimize HTTP client connection pooling."""
    
    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None
        
    async def initialize(self):
        """Initialize optimized HTTP client."""
        limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0
        )
        
        timeout = httpx.Timeout(
            connect=5.0,
            read=30.0,
            write=10.0,
            pool=5.0
        )
        
        self.client = httpx.AsyncClient(
            limits=limits,
            timeout=timeout,
            http2=True  # Enable HTTP/2
        )
        
    async def close(self):
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
            
    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make optimized HTTP request."""
        if not self.client:
            await self.initialize()
            
        return await self.client.request(method, url, **kwargs)


def setup_performance_middleware(
    app: FastAPI,
    redis_client: redis.Redis,
    rate_limit_config: Optional[RateLimitConfig] = None,
    cache_config: Optional[CacheConfig] = None,
    compression_config: Optional[CompressionConfig] = None
):
    """Setup all performance middleware."""
    
    # Default configurations
    if not rate_limit_config:
        rate_limit_config = RateLimitConfig()
        
    if not cache_config:
        cache_config = CacheConfig()
        
    if not compression_config:
        compression_config = CompressionConfig()
        
    # Add compression middleware
    if compression_config.enable_gzip:
        app.add_middleware(
            CompressionMiddleware,
            minimum_size=compression_config.min_size
        )
        
    # Add CORS middleware with optimizations
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=86400  # Cache preflight responses for 24 hours
    )
    
    # Add performance middleware
    app.add_middleware(
        PerformanceMiddleware,
        redis_client=redis_client,
        rate_limit_config=rate_limit_config,
        cache_config=cache_config
    )
    
    logger.info("Performance middleware configured successfully")