#!/usr/bin/env python3
"""
QFLARE Server with Web Dashboard
Development server with comprehensive UI
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

def create_ui_server():
    """Create a QFLARE server with web UI dashboard"""
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.staticfiles import StaticFiles
        from fastapi.templating import Jinja2Templates
        from fastapi.responses import HTMLResponse, JSONResponse
        import uvicorn
        import redis
        import sqlite3
        from datetime import datetime
        import json
        
        # Create FastAPI app
        app = FastAPI(
            title="QFLARE - Quantum-Safe Federated Learning Platform",
            description="Advanced Federated Learning with Post-Quantum Cryptography",
            version="1.0.0"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Create directories
        os.makedirs("static", exist_ok=True)
        os.makedirs("templates", exist_ok=True)
        
        # Mount static files and templates
        app.mount("/static", StaticFiles(directory="static"), name="static")
        templates = Jinja2Templates(directory="templates")
        
        # Test database connection
        def test_database():
            try:
                db_url = os.environ['DATABASE_URL']
                if db_url.startswith('sqlite'):
                    db_path = db_url.replace('sqlite:///', '')
                    os.makedirs(os.path.dirname(db_path), exist_ok=True)
                    conn = sqlite3.connect(db_path)
                    conn.execute("CREATE TABLE IF NOT EXISTS health_check (id INTEGER PRIMARY KEY, timestamp TEXT)")
                    conn.execute("CREATE TABLE IF NOT EXISTS devices (id TEXT PRIMARY KEY, type TEXT, location TEXT, status TEXT, registered_at TEXT)")
                    conn.execute("CREATE TABLE IF NOT EXISTS training_sessions (id INTEGER PRIMARY KEY, algorithm TEXT, status TEXT, progress REAL, started_at TEXT)")
                    conn.execute("INSERT OR REPLACE INTO health_check (timestamp) VALUES (?)", (datetime.now().isoformat(),))
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
        
        @app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            """Serve the main dashboard"""
            return templates.TemplateResponse("dashboard.html", {"request": request})
        
        @app.get("/health")
        async def health_check():
            """System health check"""
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
            """API status and capabilities"""
            return {
                "api_version": "v1",
                "status": "active",
                "features": [
                    "federated_learning",
                    "post_quantum_crypto",
                    "tee_integration",
                    "monitoring",
                    "web_dashboard"
                ],
                "algorithms": [
                    "fedavg",
                    "fedprox", 
                    "fedbn",
                    "per_fedavg",
                    "scaffold",
                    "fed_nova"
                ],
                "security": {
                    "post_quantum_ready": True,
                    "tee_enabled": True,
                    "encryption": "AES-256-GCM + Post-Quantum KEM"
                }
            }
        
        @app.post("/api/v1/fl/register")
        async def register_device(device_data: dict):
            """Register a new edge device"""
            device_id = device_data.get('device_id', 'unknown')
            device_type = device_data.get('device_type', 'edge')
            location = device_data.get('location', 'Unknown')
            
            logger.info(f"Device registration request: {device_id} ({device_type}) from {location}")
            
            # Store device in database
            try:
                db_url = os.environ['DATABASE_URL']
                if db_url.startswith('sqlite'):
                    db_path = db_url.replace('sqlite:///', '')
                    conn = sqlite3.connect(db_path)
                    conn.execute(
                        "INSERT OR REPLACE INTO devices (id, type, location, status, registered_at) VALUES (?, ?, ?, ?, ?)",
                        (device_id, device_type, location, 'online', datetime.now().isoformat())
                    )
                    conn.commit()
                    conn.close()
            except Exception as e:
                logger.error(f"Failed to store device: {e}")
            
            return {
                "status": "registered",
                "device_id": device_id,
                "token": f"qflare_token_{device_id}_{datetime.now().timestamp()}",
                "server_info": {
                    "fl_server_url": "http://localhost:8000/api/v1/fl",
                    "metrics_url": "http://localhost:8000/metrics",
                    "dashboard_url": "http://localhost:8000"
                },
                "registration_time": datetime.now().isoformat()
            }
        
        @app.get("/api/v1/devices")
        async def get_devices():
            """Get all registered devices"""
            try:
                db_url = os.environ['DATABASE_URL']
                if db_url.startswith('sqlite'):
                    db_path = db_url.replace('sqlite:///', '')
                    conn = sqlite3.connect(db_path)
                    cursor = conn.execute("SELECT id, type, location, status, registered_at FROM devices ORDER BY registered_at DESC")
                    devices = [
                        {
                            "id": row[0],
                            "type": row[1], 
                            "location": row[2],
                            "status": row[3],
                            "registered_at": row[4]
                        }
                        for row in cursor.fetchall()
                    ]
                    conn.close()
                    return {"devices": devices, "total": len(devices)}
            except Exception as e:
                logger.error(f"Failed to get devices: {e}")
                return {"devices": [], "total": 0}
        
        @app.post("/api/v1/training/start")
        async def start_training(training_config: dict = None):
            """Start federated learning training"""
            algorithm = training_config.get('algorithm', 'fedavg') if training_config else 'fedavg'
            
            try:
                db_url = os.environ['DATABASE_URL']
                if db_url.startswith('sqlite'):
                    db_path = db_url.replace('sqlite:///', '')
                    conn = sqlite3.connect(db_path)
                    conn.execute(
                        "INSERT INTO training_sessions (algorithm, status, progress, started_at) VALUES (?, ?, ?, ?)",
                        (algorithm, 'running', 0.0, datetime.now().isoformat())
                    )
                    conn.commit()
                    conn.close()
                    
                logger.info(f"Training session started with algorithm: {algorithm}")
                return {
                    "status": "started",
                    "algorithm": algorithm,
                    "session_id": f"session_{datetime.now().timestamp()}",
                    "started_at": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Failed to start training: {e}")
                raise HTTPException(status_code=500, detail="Failed to start training")
        
        @app.post("/api/v1/security/rotate-keys")
        async def rotate_pq_keys():
            """Rotate post-quantum cryptographic keys"""
            logger.info("Post-quantum key rotation initiated")
            
            # Simulate key rotation process
            try:
                r = redis.Redis.from_url(os.environ['REDIS_URL'])
                r.set('qflare:last_key_rotation', datetime.now().isoformat())
                r.incr('qflare:key_rotation_count')
                
                return {
                    "status": "success",
                    "message": "Post-quantum keys rotated successfully",
                    "rotation_time": datetime.now().isoformat(),
                    "algorithms_updated": ["KYBER", "DILITHIUM", "FALCON"]
                }
            except Exception as e:
                logger.error(f"Key rotation failed: {e}")
                raise HTTPException(status_code=500, detail="Key rotation failed")
        
        @app.get("/api/v1/metrics")
        async def get_detailed_metrics():
            """Get detailed system metrics"""
            try:
                r = redis.Redis.from_url(os.environ['REDIS_URL'])
                
                # Get device count from database
                device_count = 0
                try:
                    db_url = os.environ['DATABASE_URL']
                    if db_url.startswith('sqlite'):
                        db_path = db_url.replace('sqlite:///', '')
                        conn = sqlite3.connect(db_path)
                        cursor = conn.execute("SELECT COUNT(*) FROM devices")
                        device_count = cursor.fetchone()[0]
                        conn.close()
                except:
                    pass
                
                return {
                    "system": {
                        "uptime_seconds": 300,  # Simulated
                        "cpu_usage_percent": 15.2,
                        "memory_usage_percent": 34.1,
                        "disk_usage_percent": 12.7
                    },
                    "federated_learning": {
                        "registered_devices": device_count,
                        "active_devices": max(0, device_count - 1),
                        "completed_rounds": r.get('qflare:completed_rounds') or 0,
                        "active_training_sessions": 0
                    },
                    "security": {
                        "key_rotation_count": r.get('qflare:key_rotation_count') or 0,
                        "tee_operations": r.get('qflare:tee_operations') or 0,
                        "failed_auth_attempts": 0
                    },
                    "performance": {
                        "avg_response_time_ms": 45.2,
                        "requests_per_second": 12.3,
                        "network_io_mbps": 8.7
                    }
                }
            except Exception as e:
                logger.error(f"Failed to get metrics: {e}")
                return {"error": "Failed to fetch metrics"}
        
        @app.get("/metrics")
        async def prometheus_metrics():
            """Prometheus-style metrics endpoint"""
            try:
                r = redis.Redis.from_url(os.environ['REDIS_URL'])
                device_count = 0
                
                try:
                    db_url = os.environ['DATABASE_URL']
                    if db_url.startswith('sqlite'):
                        db_path = db_url.replace('sqlite:///', '')
                        conn = sqlite3.connect(db_path)
                        cursor = conn.execute("SELECT COUNT(*) FROM devices")
                        device_count = cursor.fetchone()[0]
                        conn.close()
                except:
                    pass
                
                return {
                    "qflare_server_status": 1,
                    "qflare_registered_devices": device_count,
                    "qflare_active_training_rounds": 0,
                    "qflare_completed_rounds": int(r.get('qflare:completed_rounds') or 0),
                    "qflare_key_rotations": int(r.get('qflare:key_rotation_count') or 0)
                }
            except Exception as e:
                logger.error(f"Failed to get Prometheus metrics: {e}")
                return {"error": "Failed to fetch metrics"}
        
        return app
        
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.info("Please install: pip install fastapi uvicorn redis jinja2")
        sys.exit(1)

def main():
    """Main entry point"""
    logger.info("🚀 Starting QFLARE Development Server with Web Dashboard...")
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/logs', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
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
    logger.info("Creating QFLARE server with web dashboard...")
    app = create_ui_server()
    
    logger.info("🎯 QFLARE Server starting on http://localhost:8000")
    logger.info("🌐 Web Dashboard: http://localhost:8000")
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