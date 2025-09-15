#!/usr/bin/env python3
"""
QFLARE Server Startup Script
Simple development server launcher
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set default environment variables
os.environ.setdefault('DATABASE_URL', 'sqlite:///data/qflare_dev.db')
os.environ.setdefault('REDIS_URL', 'redis://localhost:6379/0')
os.environ.setdefault('QFLARE_JWT_SECRET', 'dev-secret-key-change-in-production')
os.environ.setdefault('QFLARE_SGX_MODE', 'SIM')
os.environ.setdefault('QFLARE_LOG_LEVEL', 'INFO')

def create_basic_server():
    """Create a basic QFLARE server for development"""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
        import uvicorn
        import redis
        import sqlite3
        from datetime import datetime
        
        # Create FastAPI app
        app = FastAPI(
            title="QFLARE Development Server",
            description="Quantum-Safe Federated Learning Platform",
            version="1.0.0"
        )
        
        # Test database connection
        def test_database():
            try:
                db_url = os.environ['DATABASE_URL']
                if db_url.startswith('sqlite'):
                    db_path = db_url.replace('sqlite:///', '')
                    os.makedirs(os.path.dirname(db_path), exist_ok=True)
                    conn = sqlite3.connect(db_path)
                    conn.execute("CREATE TABLE IF NOT EXISTS health_check (id INTEGER PRIMARY KEY, timestamp TEXT)")
                    conn.execute("INSERT INTO health_check (timestamp) VALUES (?)", (datetime.now().isoformat(),))
                    conn.commit()
                    conn.close()
                    return True
            except Exception as e:
                logger.error(f"Database test failed: {e}")
                return False
        
        # Test Redis connection
        def test_redis():
            try:
                r = redis.Redis.from_url(os.environ['REDIS_URL'])
                r.ping()
                r.set('qflare:health_check', datetime.now().isoformat())
                return True
            except Exception as e:
                logger.error(f"Redis test failed: {e}")
                return False
        
        @app.get("/")
        async def root():
            return {
                "message": "Welcome to QFLARE - Quantum-Safe Federated Learning Platform",
                "status": "running",
                "version": "1.0.0-dev",
                "timestamp": datetime.now().isoformat()
            }
        
        @app.get("/health")
        async def health_check():
            db_status = test_database()
            redis_status = test_redis()
            
            return {
                "status": "healthy" if db_status and redis_status else "unhealthy",
                "components": {
                    "database": "ok" if db_status else "error",
                    "redis": "ok" if redis_status else "error"
                },
                "timestamp": datetime.now().isoformat(),
                "environment": {
                    "database_url": os.environ.get('DATABASE_URL', 'not_set'),
                    "redis_url": os.environ.get('REDIS_URL', 'not_set'),
                    "sgx_mode": os.environ.get('QFLARE_SGX_MODE', 'not_set'),
                    "log_level": os.environ.get('QFLARE_LOG_LEVEL', 'not_set')
                }
            }
        
        @app.get("/api/v1/status")
        async def api_status():
            return {
                "api_version": "v1",
                "status": "active",
                "features": [
                    "federated_learning",
                    "post_quantum_crypto",
                    "tee_integration",
                    "monitoring"
                ],
                "algorithms": [
                    "fedavg",
                    "fedprox", 
                    "fedbn",
                    "per_fedavg"
                ]
            }
        
        @app.post("/api/v1/fl/register")
        async def register_device(device_data: dict):
            """Register a new edge device"""
            device_id = device_data.get('device_id', 'unknown')
            logger.info(f"Device registration request: {device_id}")
            
            return {
                "status": "registered",
                "device_id": device_id,
                "token": f"dev_token_{device_id}",
                "server_info": {
                    "fl_server_url": "http://localhost:8000/api/v1/fl",
                    "metrics_url": "http://localhost:8000/metrics"
                }
            }
        
        @app.get("/metrics")
        async def metrics():
            """Prometheus-style metrics endpoint"""
            return {
                "qflare_server_status": 1,
                "qflare_registered_devices": 0,
                "qflare_active_training_rounds": 0,
                "qflare_completed_rounds": 0
            }
        
        return app
        
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.info("Please install: pip install fastapi uvicorn redis")
        sys.exit(1)

def main():
    """Main entry point"""
    logger.info("🚀 Starting QFLARE Development Server...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/logs', exist_ok=True)
    
    # Test dependencies
    logger.info("Testing Redis connection...")
    try:
        import redis
        r = redis.Redis.from_url(os.environ['REDIS_URL'])
        r.ping()
        logger.info("✅ Redis connection successful")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        logger.info("Please ensure Redis is running: docker run -d -p 6379:6379 redis:7-alpine")
        sys.exit(1)
    
    # Create and run server
    logger.info("Creating QFLARE server...")
    app = create_basic_server()
    
    logger.info("🎯 QFLARE Server starting on http://localhost:8000")
    logger.info("📊 Health check: http://localhost:8000/health")
    logger.info("📚 API status: http://localhost:8000/api/v1/status")
    logger.info("📈 Metrics: http://localhost:8000/metrics")
    
    # Start the server
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )

if __name__ == "__main__":
    main()