# QFLARE Load Balancing and CDN Integration

"""
Advanced load balancing strategies and CDN integration for
optimal performance and availability.
"""

import asyncio
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import json

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, RedirectResponse
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class LoadBalanceStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    HASH_BASED = "hash_based"
    GEOGRAPHIC = "geographic"
    HEALTH_BASED = "health_based"


@dataclass
class ServerInfo:
    """Backend server information."""
    host: str
    port: int
    weight: int = 1
    max_connections: int = 100
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_health_check: Optional[float] = None
    is_healthy: bool = True
    location: Optional[str] = None  # Geographic location
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
        
    @property
    def health_score(self) -> float:
        """Calculate server health score (0-1, higher is better)."""
        if not self.is_healthy:
            return 0.0
            
        # Factor in response time, error rate, and load
        error_rate = (
            self.failed_requests / max(self.total_requests, 1)
        )
        load_factor = self.current_connections / self.max_connections
        response_time_factor = min(1.0, 1.0 / max(self.avg_response_time, 0.1))
        
        score = (
            (1.0 - error_rate) * 0.4 +  # 40% error rate
            (1.0 - load_factor) * 0.3 +  # 30% load
            response_time_factor * 0.3    # 30% response time
        )
        
        return max(0.0, min(1.0, score))


@dataclass
class CDNConfig:
    """CDN configuration."""
    enable_cdn: bool = True
    cdn_endpoints: List[str] = None
    cache_ttl: int = 3600  # 1 hour
    static_file_patterns: List[str] = None
    purge_on_update: bool = True
    edge_locations: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.static_file_patterns is None:
            self.static_file_patterns = [
                r".*\.(js|css|png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf)$",
                r"^/static/.*",
                r"^/assets/.*"
            ]
            
        if self.cdn_endpoints is None:
            self.cdn_endpoints = []
            
        if self.edge_locations is None:
            self.edge_locations = {
                "us-east": ["cdn1.example.com", "cdn2.example.com"],
                "us-west": ["cdn3.example.com", "cdn4.example.com"],
                "eu": ["cdn5.example.com", "cdn6.example.com"],
                "asia": ["cdn7.example.com", "cdn8.example.com"]
            }


class LoadBalancer:
    """Advanced load balancer with multiple strategies."""
    
    def __init__(
        self,
        servers: List[ServerInfo],
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.HEALTH_BASED,
        redis_client: Optional[redis.Redis] = None
    ):
        self.servers = {server.url: server for server in servers}
        self.strategy = strategy
        self.redis = redis_client
        self.current_index = 0
        self.client_mappings: Dict[str, str] = {}  # For hash-based routing
        
    async def get_server(
        self,
        client_id: Optional[str] = None,
        request_path: Optional[str] = None,
        client_location: Optional[str] = None
    ) -> Optional[ServerInfo]:
        """Get the best server based on strategy."""
        healthy_servers = [
            server for server in self.servers.values()
            if server.is_healthy
        ]
        
        if not healthy_servers:
            logger.error("No healthy servers available")
            return None
            
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin(healthy_servers)
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(healthy_servers)
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._least_connections(healthy_servers)
        elif self.strategy == LoadBalanceStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time(healthy_servers)
        elif self.strategy == LoadBalanceStrategy.HASH_BASED:
            return self._hash_based(healthy_servers, client_id or "default")
        elif self.strategy == LoadBalanceStrategy.GEOGRAPHIC:
            return self._geographic(healthy_servers, client_location)
        else:  # HEALTH_BASED
            return self._health_based(healthy_servers)
            
    def _round_robin(self, servers: List[ServerInfo]) -> ServerInfo:
        """Simple round-robin selection."""
        server = servers[self.current_index % len(servers)]
        self.current_index += 1
        return server
        
    def _weighted_round_robin(self, servers: List[ServerInfo]) -> ServerInfo:
        """Weighted round-robin based on server weights."""
        total_weight = sum(server.weight for server in servers)
        
        if total_weight == 0:
            return self._round_robin(servers)
            
        # Create weighted list
        weighted_servers = []
        for server in servers:
            weighted_servers.extend([server] * server.weight)
            
        return self._round_robin(weighted_servers)
        
    def _least_connections(self, servers: List[ServerInfo]) -> ServerInfo:
        """Select server with least active connections."""
        return min(servers, key=lambda s: s.current_connections)
        
    def _least_response_time(self, servers: List[ServerInfo]) -> ServerInfo:
        """Select server with lowest average response time."""
        return min(servers, key=lambda s: s.avg_response_time or float('inf'))
        
    def _hash_based(self, servers: List[ServerInfo], client_id: str) -> ServerInfo:
        """Consistent hash-based selection for session affinity."""
        # Use consistent hashing
        hash_value = int(hashlib.sha256(client_id.encode()).hexdigest()[:8], 16)
        server_index = hash_value % len(servers)
        return servers[server_index]
        
    def _geographic(
        self,
        servers: List[ServerInfo],
        client_location: Optional[str]
    ) -> ServerInfo:
        """Geographic-based selection."""
        if not client_location:
            return self._health_based(servers)
            
        # Find servers in the same location
        local_servers = [
            server for server in servers
            if server.location == client_location
        ]
        
        if local_servers:
            return self._health_based(local_servers)
        else:
            return self._health_based(servers)
            
    def _health_based(self, servers: List[ServerInfo]) -> ServerInfo:
        """Select server based on health score."""
        # Weight servers by health score
        health_scores = [server.health_score for server in servers]
        
        if all(score == 0 for score in health_scores):
            # All servers unhealthy, use round-robin
            return self._round_robin(servers)
            
        # Weighted random selection based on health scores
        import random
        total_score = sum(health_scores)
        
        if total_score == 0:
            return self._round_robin(servers)
            
        rand_value = random.uniform(0, total_score)
        cumulative = 0
        
        for i, server in enumerate(servers):
            cumulative += health_scores[i]
            if rand_value <= cumulative:
                return server
                
        return servers[-1]  # Fallback
        
    async def record_request(
        self,
        server_url: str,
        response_time: float,
        success: bool = True
    ):
        """Record request statistics for a server."""
        if server_url in self.servers:
            server = self.servers[server_url]
            
            if success:
                # Update average response time using exponential moving average
                if server.avg_response_time == 0:
                    server.avg_response_time = response_time
                else:
                    server.avg_response_time = (
                        0.9 * server.avg_response_time + 0.1 * response_time
                    )
                    
                server.total_requests += 1
            else:
                server.failed_requests += 1
                
    async def health_check_all(self):
        """Perform health checks on all servers."""
        async def check_server(server: ServerInfo):
            try:
                start_time = time.time()
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{server.url}/health",
                        timeout=5.0
                    )
                    
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    server.is_healthy = True
                    server.avg_response_time = (
                        0.9 * (server.avg_response_time or 0) + 0.1 * response_time
                    )
                else:
                    server.is_healthy = False
                    
            except Exception as e:
                logger.warning(f"Health check failed for {server.url}: {e}")
                server.is_healthy = False
                
            server.last_health_check = time.time()
            
        # Check all servers concurrently
        await asyncio.gather(
            *[check_server(server) for server in self.servers.values()],
            return_exceptions=True
        )
        
    def get_server_stats(self) -> Dict[str, Any]:
        """Get statistics for all servers."""
        return {
            "servers": {
                url: {
                    "host": server.host,
                    "port": server.port,
                    "weight": server.weight,
                    "current_connections": server.current_connections,
                    "total_requests": server.total_requests,
                    "failed_requests": server.failed_requests,
                    "avg_response_time": server.avg_response_time,
                    "health_score": server.health_score,
                    "is_healthy": server.is_healthy,
                    "location": server.location
                }
                for url, server in self.servers.items()
            },
            "strategy": self.strategy.value,
            "total_servers": len(self.servers),
            "healthy_servers": len([s for s in self.servers.values() if s.is_healthy]),
            "timestamp": time.time()
        }


class CDNManager:
    """CDN integration and management."""
    
    def __init__(self, config: CDNConfig):
        self.config = config
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "purges": 0,
            "errors": 0
        }
        
    def should_use_cdn(self, request_path: str) -> bool:
        """Determine if request should be served from CDN."""
        if not self.config.enable_cdn:
            return False
            
        import re
        
        for pattern in self.config.static_file_patterns:
            if re.match(pattern, request_path):
                return True
                
        return False
        
    def get_cdn_url(
        self,
        request_path: str,
        client_location: Optional[str] = None
    ) -> Optional[str]:
        """Get appropriate CDN URL for request."""
        if not self.should_use_cdn(request_path):
            return None
            
        # Select CDN endpoint based on client location
        if client_location and client_location in self.config.edge_locations:
            endpoints = self.config.edge_locations[client_location]
            if endpoints:
                # Simple round-robin for CDN endpoints
                endpoint = endpoints[hash(request_path) % len(endpoints)]
                return f"https://{endpoint}{request_path}"
                
        # Fallback to default CDN endpoint
        if self.config.cdn_endpoints:
            endpoint = self.config.cdn_endpoints[0]
            return f"https://{endpoint}{request_path}"
            
        return None
        
    async def purge_cache(
        self,
        paths: List[str],
        tags: Optional[List[str]] = None
    ) -> bool:
        """Purge CDN cache for specified paths or tags."""
        if not self.config.enable_cdn:
            return True
            
        try:
            # This would integrate with your CDN provider's API
            # Example for common CDN providers:
            
            purge_requests = []
            for endpoint in self.config.cdn_endpoints:
                # Cloudflare-style purge
                purge_data = {"files": [f"https://{endpoint}{path}" for path in paths]}
                if tags:
                    purge_data["tags"] = tags
                    
                purge_requests.append(
                    self._make_purge_request(endpoint, purge_data)
                )
                
            # Execute purge requests concurrently
            results = await asyncio.gather(*purge_requests, return_exceptions=True)
            
            success_count = sum(1 for r in results if r is True)
            self.cache_stats["purges"] += len(paths)
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"CDN cache purge failed: {e}")
            self.cache_stats["errors"] += 1
            return False
            
    async def _make_purge_request(self, endpoint: str, purge_data: Dict) -> bool:
        """Make purge request to CDN endpoint."""
        try:
            # This would use your CDN provider's API
            # Mock implementation
            await asyncio.sleep(0.1)  # Simulate API call
            return True
            
        except Exception as e:
            logger.error(f"Purge request failed for {endpoint}: {e}")
            return False
            
    def get_cache_headers(self, request_path: str) -> Dict[str, str]:
        """Get appropriate cache headers for response."""
        headers = {}
        
        if self.should_use_cdn(request_path):
            # Static files - long cache TTL
            headers.update({
                "Cache-Control": f"public, max-age={self.config.cache_ttl}, immutable",
                "Expires": self._get_expires_header(self.config.cache_ttl),
                "ETag": self._generate_etag(request_path),
                "Vary": "Accept-Encoding"
            })
        else:
            # Dynamic content - shorter cache TTL
            headers.update({
                "Cache-Control": "public, max-age=300, must-revalidate",
                "Expires": self._get_expires_header(300),
                "Vary": "Accept-Encoding, Authorization"
            })
            
        return headers
        
    def _get_expires_header(self, ttl: int) -> str:
        """Generate Expires header value."""
        from datetime import datetime, timedelta
        expires_time = datetime.utcnow() + timedelta(seconds=ttl)
        return expires_time.strftime("%a, %d %b %Y %H:%M:%S GMT")
        
    def _generate_etag(self, content: str) -> str:
        """Generate ETag for content."""
        return f'"{hashlib.sha256(content.encode()).hexdigest()[:16]}"'
        
    def get_stats(self) -> Dict[str, Any]:
        """Get CDN statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (
            self.cache_stats["hits"] / total_requests
            if total_requests > 0 else 0
        )
        
        return {
            "enabled": self.config.enable_cdn,
            "endpoints": len(self.config.cdn_endpoints),
            "edge_locations": list(self.config.edge_locations.keys()),
            "cache_stats": self.cache_stats,
            "hit_rate": hit_rate,
            "timestamp": time.time()
        }


class LoadBalancingMiddleware:
    """Middleware for load balancing and CDN integration."""
    
    def __init__(
        self,
        load_balancer: LoadBalancer,
        cdn_manager: CDNManager
    ):
        self.load_balancer = load_balancer
        self.cdn_manager = cdn_manager
        
    async def __call__(self, request: Request, call_next):
        """Process request with load balancing and CDN."""
        request_path = request.url.path
        
        # Check if request should be served from CDN
        cdn_url = self.cdn_manager.get_cdn_url(request_path)
        if cdn_url:
            # Redirect to CDN
            self.cdn_manager.cache_stats["hits"] += 1
            return RedirectResponse(url=cdn_url, status_code=302)
            
        # Get client information for load balancing
        client_ip = request.client.host if request.client else "unknown"
        client_location = request.headers.get("cf-ipcountry")  # Cloudflare country header
        
        # Select backend server
        server = await self.load_balancer.get_server(
            client_id=client_ip,
            request_path=request_path,
            client_location=client_location
        )
        
        if not server:
            raise HTTPException(status_code=503, detail="No healthy servers available")
            
        # Proxy request to selected server
        start_time = time.time()
        
        try:
            # Update connection count
            server.current_connections += 1
            
            # Forward request to backend server
            async with httpx.AsyncClient() as client:
                # Build target URL
                target_url = f"{server.url}{request_path}"
                if request.url.query:
                    target_url += f"?{request.url.query}"
                    
                # Forward headers (excluding hop-by-hop headers)
                forward_headers = {
                    k: v for k, v in request.headers.items()
                    if k.lower() not in [
                        "host", "connection", "upgrade", "proxy-authenticate",
                        "proxy-authorization", "te", "trailers", "transfer-encoding"
                    ]
                }
                
                # Add load balancer headers
                forward_headers["X-Forwarded-For"] = client_ip
                forward_headers["X-Load-Balancer"] = "qflare-lb"
                forward_headers["X-Backend-Server"] = server.url
                
                # Forward request
                response = await client.request(
                    method=request.method,
                    url=target_url,
                    headers=forward_headers,
                    content=await request.body(),
                    timeout=30.0
                )
                
            # Record successful request
            response_time = time.time() - start_time
            await self.load_balancer.record_request(server.url, response_time, True)
            
            # Add cache headers for CDN
            cache_headers = self.cdn_manager.get_cache_headers(request_path)
            
            # Create response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers={**dict(response.headers), **cache_headers},
                media_type=response.headers.get("content-type")
            )
            
        except Exception as e:
            # Record failed request
            response_time = time.time() - start_time
            await self.load_balancer.record_request(server.url, response_time, False)
            
            logger.error(f"Request forwarding failed: {e}")
            raise HTTPException(status_code=502, detail="Backend server error")
            
        finally:
            # Update connection count
            server.current_connections = max(0, server.current_connections - 1)


def setup_load_balancing(
    app: FastAPI,
    servers: List[ServerInfo],
    strategy: LoadBalanceStrategy = LoadBalanceStrategy.HEALTH_BASED,
    cdn_config: Optional[CDNConfig] = None,
    redis_client: Optional[redis.Redis] = None
):
    """Setup load balancing and CDN integration."""
    
    # Initialize load balancer
    load_balancer = LoadBalancer(servers, strategy, redis_client)
    
    # Initialize CDN manager
    if not cdn_config:
        cdn_config = CDNConfig()
    cdn_manager = CDNManager(cdn_config)
    
    # Add middleware
    middleware = LoadBalancingMiddleware(load_balancer, cdn_manager)
    app.middleware("http")(middleware)
    
    # Start health check task
    async def health_check_task():
        while True:
            try:
                await load_balancer.health_check_all()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Health check task error: {e}")
                await asyncio.sleep(10)  # Retry after 10 seconds
                
    asyncio.create_task(health_check_task())
    
    # Add monitoring endpoints
    @app.get("/monitoring/load-balancer/stats")
    async def get_load_balancer_stats():
        return load_balancer.get_server_stats()
        
    @app.get("/monitoring/cdn/stats")
    async def get_cdn_stats():
        return cdn_manager.get_stats()
        
    @app.post("/monitoring/cdn/purge")
    async def purge_cdn_cache(paths: List[str], tags: Optional[List[str]] = None):
        success = await cdn_manager.purge_cache(paths, tags)
        return {"success": success, "paths": paths, "tags": tags or []}
        
    logger.info("Load balancing and CDN integration configured successfully")