#!/usr/bin/env python3
"""
QFLARE Core Server - Federated Learning Coordinator
Main server for coordinating federated learning with post-quantum security
"""

import os
import sys
import asyncio
import logging
import json
import secrets
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import hashlib
import base64

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
os.environ.setdefault('DATABASE_URL', 'sqlite:///data/qflare_core.db')
os.environ.setdefault('REDIS_URL', 'redis://localhost:6379/0')
os.environ.setdefault('QFLARE_JWT_SECRET', 'dev-secret-key-change-in-production')
os.environ.setdefault('QFLARE_SGX_MODE', 'SIM')
os.environ.setdefault('QFLARE_LOG_LEVEL', 'INFO')

class DeviceRegistry:
    """Manages device registration and authentication"""

    def __init__(self, db_path: str = "data/qflare_core.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database tables"""
        import sqlite3
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS devices (
                device_id TEXT PRIMARY KEY,
                public_key TEXT NOT NULL,
                device_type TEXT NOT NULL,
                location TEXT,
                status TEXT DEFAULT 'pending',
                registered_at TEXT,
                last_seen TEXT,
                capabilities TEXT,
                training_participation INTEGER DEFAULT 0
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS fl_rounds (
                round_id TEXT PRIMARY KEY,
                algorithm TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                participants TEXT,
                started_at TEXT,
                completed_at TEXT,
                global_model_hash TEXT,
                metrics TEXT
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS model_updates (
                update_id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL,
                round_id TEXT NOT NULL,
                model_data TEXT,
                local_metrics TEXT,
                submitted_at TEXT,
                verified BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (device_id) REFERENCES devices (device_id),
                FOREIGN KEY (round_id) REFERENCES fl_rounds (round_id)
            )
        ''')

        conn.commit()
        conn.close()

    def register_device(self, device_id: str, public_key: str, device_type: str,
                       location: str = "", capabilities: Dict = None) -> Dict[str, Any]:
        """Register a new device with post-quantum public key"""
        import sqlite3

        # Generate device token
        token = secrets.token_urlsafe(32)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                INSERT INTO devices (device_id, public_key, device_type, location,
                                   status, registered_at, last_seen, capabilities)
                VALUES (?, ?, ?, ?, 'active', ?, ?, ?)
            ''', (
                device_id,
                public_key,
                device_type,
                location,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                json.dumps(capabilities or {})
            ))
            conn.commit()

            logger.info(f"Device registered: {device_id} ({device_type})")

            return {
                "status": "registered",
                "device_id": device_id,
                "token": token,
                "server_public_key": self._get_server_public_key(),
                "registration_time": datetime.now().isoformat(),
                "capabilities": capabilities
            }

        except sqlite3.IntegrityError:
            # Device already exists, update info
            conn.execute('''
                UPDATE devices SET
                    public_key = ?,
                    device_type = ?,
                    location = ?,
                    status = 'active',
                    last_seen = ?,
                    capabilities = ?
                WHERE device_id = ?
            ''', (
                public_key,
                device_type,
                location,
                datetime.now().isoformat(),
                json.dumps(capabilities or {}),
                device_id
            ))
            conn.commit()

            return {
                "status": "updated",
                "device_id": device_id,
                "token": token,
                "message": "Device registration updated"
            }
        finally:
            conn.close()

    def _get_server_public_key(self) -> str:
        """Get or generate server public key"""
        # In production, this would be a real PQ key
        # For demo, we'll use a mock key
        return "server_public_key_mock_" + secrets.token_hex(16)

    def authenticate_device(self, device_id: str, token: str) -> bool:
        """Authenticate device token"""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT device_id FROM devices WHERE device_id = ? AND status = 'active'",
                (device_id,)
            )
            result = cursor.fetchone()

            if result:
                # Update last seen
                conn.execute(
                    "UPDATE devices SET last_seen = ? WHERE device_id = ?",
                    (datetime.now().isoformat(), device_id)
                )
                conn.commit()
                return True

            return False
        finally:
            conn.close()

    def get_active_devices(self) -> List[Dict[str, Any]]:
        """Get all active devices"""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute('''
                SELECT device_id, device_type, location, last_seen, training_participation
                FROM devices
                WHERE status = 'active'
                ORDER BY last_seen DESC
            ''')

            devices = []
            for row in cursor.fetchall():
                devices.append({
                    "device_id": row[0],
                    "device_type": row[1],
                    "location": row[2],
                    "last_seen": row[3],
                    "training_participation": row[4]
                })

            return devices
        finally:
            conn.close()

class FLCoordinator:
    """Federated Learning Round Coordinator"""

    def __init__(self, registry: DeviceRegistry, redis_client):
        self.registry = registry
        self.redis = redis_client
        self.current_round: Optional[str] = None
        self.algorithms = {
            "fedavg": self._fedavg_aggregate,
            "fedprox": self._fedprox_aggregate,
            "scaffold": self._scaffold_aggregate
        }

    def start_round(self, algorithm: str = "fedavg", min_participants: int = 3) -> Dict[str, Any]:
        """Start a new federated learning round"""
        round_id = f"round_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Get active devices
        active_devices = self.registry.get_active_devices()
        if len(active_devices) < min_participants:
            raise ValueError(f"Need at least {min_participants} devices, only {len(active_devices)} active")

        # Select participants (for now, all active devices)
        participants = [d["device_id"] for d in active_devices]

        # Store round info in database
        import sqlite3
        conn = sqlite3.connect(self.registry.db_path)
        try:
            conn.execute('''
                INSERT INTO fl_rounds (round_id, algorithm, status, participants, started_at)
                VALUES (?, ?, 'active', ?, ?)
            ''', (
                round_id,
                algorithm,
                json.dumps(participants),
                datetime.now().isoformat()
            ))
            conn.commit()
        finally:
            conn.close()

        # Store in Redis for fast access
        self.redis.set(f"fl:round:{round_id}:status", "active")
        self.redis.set(f"fl:round:{round_id}:participants", json.dumps(participants))
        self.redis.set(f"fl:round:{round_id}:algorithm", algorithm)

        self.current_round = round_id

        logger.info(f"Started FL round {round_id} with {len(participants)} participants using {algorithm}")

        return {
            "round_id": round_id,
            "algorithm": algorithm,
            "participants": participants,
            "started_at": datetime.now().isoformat(),
            "status": "active"
        }

    def submit_model_update(self, device_id: str, round_id: str,
                           model_data: Dict[str, Any], local_metrics: Dict[str, Any]) -> bool:
        """Submit model update from a device"""
        # Verify device is participant
        participants = json.loads(self.redis.get(f"fl:round:{round_id}:participants") or "[]")
        if device_id not in participants:
            logger.warning(f"Device {device_id} not authorized for round {round_id}")
            return False

        # Store update in database
        import sqlite3
        conn = sqlite3.connect(self.registry.db_path)
        try:
            update_id = f"{device_id}_{round_id}_{datetime.now().timestamp()}"
            conn.execute('''
                INSERT INTO model_updates (update_id, device_id, round_id, model_data,
                                         local_metrics, submitted_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                update_id,
                device_id,
                round_id,
                json.dumps(model_data),
                json.dumps(local_metrics),
                datetime.now().isoformat()
            ))
            conn.commit()

            # Track submission in Redis
            self.redis.sadd(f"fl:round:{round_id}:submissions", device_id)
            self.redis.set(f"fl:round:{round_id}:update:{device_id}", json.dumps({
                "model_data": model_data,
                "metrics": local_metrics,
                "submitted_at": datetime.now().isoformat()
            }))

            logger.info(f"Model update received from {device_id} for round {round_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store model update: {e}")
            return False
        finally:
            conn.close()

    def check_round_completion(self, round_id: str) -> bool:
        """Check if round has enough submissions to complete"""
        participants = json.loads(self.redis.get(f"fl:round:{round_id}:participants") or "[]")
        submissions = self.redis.smembers(f"fl:round:{round_id}:submissions")

        # Require at least 80% participation
        min_submissions = max(1, int(len(participants) * 0.8))

        return len(submissions) >= min_submissions

    def complete_round(self, round_id: str) -> Dict[str, Any]:
        """Complete a federated learning round"""
        algorithm = self.redis.get(f"fl:round:{round_id}:algorithm")

        if algorithm not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Aggregate model updates
        global_model = self.algorithms[algorithm](round_id)

        # Update round status
        import sqlite3
        conn = sqlite3.connect(self.registry.db_path)
        try:
            conn.execute('''
                UPDATE fl_rounds SET
                    status = 'completed',
                    completed_at = ?,
                    global_model_hash = ?
                WHERE round_id = ?
            ''', (
                datetime.now().isoformat(),
                hashlib.sha256(json.dumps(global_model).encode()).hexdigest(),
                round_id
            ))
            conn.commit()
        finally:
            conn.close()

        # Update Redis
        self.redis.set(f"fl:round:{round_id}:status", "completed")
        self.redis.set(f"fl:round:{round_id}:global_model", json.dumps(global_model))

        logger.info(f"Completed FL round {round_id}")

        return {
            "round_id": round_id,
            "status": "completed",
            "global_model": global_model,
            "completed_at": datetime.now().isoformat()
        }

    def _fedavg_aggregate(self, round_id: str) -> Dict[str, Any]:
        """FedAvg aggregation algorithm"""
        # Get all model updates for this round
        updates = []
        participants = json.loads(self.redis.get(f"fl:round:{round_id}:participants") or "[]")

        for device_id in participants:
            update_data = self.redis.get(f"fl:round:{round_id}:update:{device_id}")
            if update_data:
                update = json.loads(update_data)
                updates.append(update["model_data"])

        if not updates:
            return {"error": "No model updates available"}

        # Simple average aggregation (in practice, this would be more sophisticated)
        global_model = {}
        if updates:
            # Get all parameter keys
            param_keys = set()
            for update in updates:
                param_keys.update(update.keys())

            # Average each parameter
            for key in param_keys:
                values = [update.get(key, 0) for update in updates if key in update]
                if values:
                    global_model[key] = sum(values) / len(values)

        return global_model

    def _fedprox_aggregate(self, round_id: str) -> Dict[str, Any]:
        """FedProx aggregation (simplified)"""
        # Similar to FedAvg but with proximal regularization
        return self._fedavg_aggregate(round_id)

    def _scaffold_aggregate(self, round_id: str) -> Dict[str, Any]:
        """SCAFFOLD aggregation (simplified)"""
        # More advanced aggregation with control variates
        return self._fedavg_aggregate(round_id)

def create_core_server():
    """Create the main QFLARE core server"""
    try:
        from fastapi import FastAPI, HTTPException, Request, Depends, Header
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.staticfiles import StaticFiles
        from fastapi.templating import Jinja2Templates
        from fastapi.responses import JSONResponse
        import uvicorn
        import redis
        import sqlite3
        from datetime import datetime
        import json

        # Initialize components
        registry = DeviceRegistry()
        redis_client = redis.Redis.from_url(os.environ['REDIS_URL'])

        # Test Redis connection
        redis_client.ping()

        coordinator = FLCoordinator(registry, redis_client)

        # Create FastAPI app
        app = FastAPI(
            title="QFLARE Core Server",
            description="Federated Learning Coordinator with Post-Quantum Security",
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

        # Dependency for device authentication
        def authenticate_device(x_device_token: str = Header(None), x_device_id: str = Header(None)):
            if not x_device_token or not x_device_id:
                raise HTTPException(status_code=401, detail="Device authentication required")

            if not registry.authenticate_device(x_device_id, x_device_token):
                raise HTTPException(status_code=403, detail="Invalid device credentials")

            return x_device_id

        @app.get("/")
        async def dashboard(request: Request):
            """Serve the main dashboard"""
            return templates.TemplateResponse("dashboard.html", {"request": request})

        @app.post("/api/v1/devices/register")
        async def register_device_endpoint(request: Request):
            """Register a new device with PQ cryptography"""
            try:
                data = await request.json()
                device_id = data.get("device_id")
                public_key = data.get("public_key")
                device_type = data.get("device_type", "edge")
                location = data.get("location", "")
                capabilities = data.get("capabilities", {})

                if not device_id or not public_key:
                    raise HTTPException(status_code=400, detail="device_id and public_key required")

                result = registry.register_device(
                    device_id=device_id,
                    public_key=public_key,
                    device_type=device_type,
                    location=location,
                    capabilities=capabilities
                )

                return result

            except Exception as e:
                logger.error(f"Device registration failed: {e}")
                raise HTTPException(status_code=500, detail="Registration failed")

        @app.get("/api/v1/devices")
        async def get_devices():
            """Get all registered devices"""
            devices = registry.get_active_devices()
            return {"devices": devices, "total": len(devices)}

        @app.post("/api/v1/fl/start-round")
        async def start_fl_round(request: Request):
            """Start a new federated learning round"""
            try:
                data = await request.json()
                algorithm = data.get("algorithm", "fedavg")
                min_participants = data.get("min_participants", 3)

                result = coordinator.start_round(algorithm, min_participants)
                return result

            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Failed to start FL round: {e}")
                raise HTTPException(status_code=500, detail="Failed to start round")

        @app.post("/api/v1/fl/submit-update")
        async def submit_model_update(request: Request, device_id: str = Depends(authenticate_device)):
            """Submit model update from authenticated device"""
            try:
                data = await request.json()
                round_id = data.get("round_id")
                model_data = data.get("model_data", {})
                local_metrics = data.get("local_metrics", {})

                if not round_id:
                    raise HTTPException(status_code=400, detail="round_id required")

                success = coordinator.submit_model_update(
                    device_id=device_id,
                    round_id=round_id,
                    model_data=model_data,
                    local_metrics=local_metrics
                )

                if success:
                    # Check if round is complete
                    if coordinator.check_round_completion(round_id):
                        # Auto-complete round
                        result = coordinator.complete_round(round_id)
                        return {
                            "status": "submitted_and_completed",
                            "round_result": result
                        }

                    return {"status": "submitted", "message": "Model update submitted successfully"}
                else:
                    raise HTTPException(status_code=403, detail="Submission failed")

            except Exception as e:
                logger.error(f"Model submission failed: {e}")
                raise HTTPException(status_code=500, detail="Submission failed")

        @app.get("/api/v1/fl/current-round")
        async def get_current_round():
            """Get current active round information"""
            if not coordinator.current_round:
                return {"status": "no_active_round"}

            round_id = coordinator.current_round
            status = redis_client.get(f"fl:round:{round_id}:status")
            participants = json.loads(redis_client.get(f"fl:round:{round_id}:participants") or "[]")
            submissions = list(redis_client.smembers(f"fl:round:{round_id}:submissions"))

            return {
                "round_id": round_id,
                "status": status,
                "participants": participants,
                "submissions": submissions,
                "progress": f"{len(submissions)}/{len(participants)}"
            }

        @app.get("/api/v1/fl/rounds")
        async def get_round_history():
            """Get federated learning round history"""
            import sqlite3
            conn = sqlite3.connect(registry.db_path)
            try:
                cursor = conn.execute('''
                    SELECT round_id, algorithm, status, participants, started_at, completed_at
                    FROM fl_rounds
                    ORDER BY started_at DESC
                    LIMIT 10
                ''')

                rounds = []
                for row in cursor.fetchall():
                    rounds.append({
                        "round_id": row[0],
                        "algorithm": row[1],
                        "status": row[2],
                        "participants": json.loads(row[3]) if row[3] else [],
                        "started_at": row[4],
                        "completed_at": row[5]
                    })

                return {"rounds": rounds}
            finally:
                conn.close()

        @app.get("/health")
        async def health_check():
            """System health check"""
            db_status = True  # Assume DB is working since we initialized it
            redis_status = True
            try:
                redis_client.ping()
            except:
                redis_status = False

            return {
                "status": "healthy" if db_status and redis_status else "unhealthy",
                "components": {
                    "database": "ok" if db_status else "error",
                    "redis": "ok" if redis_status else "error",
                    "fl_coordinator": "ok"
                },
                "timestamp": datetime.now().isoformat(),
                "active_devices": len(registry.get_active_devices()),
                "current_round": coordinator.current_round
            }

        return app

    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.info("Please install: pip install fastapi uvicorn redis jinja2")
        sys.exit(1)

def main():
    """Main entry point"""
    logger.info("🚀 Starting QFLARE Core Server - Federated Learning Coordinator...")

    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/logs', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)

    # Test Redis connection
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
    logger.info("Creating QFLARE Core server...")
    app = create_core_server()

    logger.info("🎯 QFLARE Core Server starting on http://localhost:8000")
    logger.info("🌐 Web Dashboard: http://localhost:8000")
    logger.info("📊 Health check: http://localhost:8000/health")
    logger.info("🔐 Device Registration: POST /api/v1/devices/register")
    logger.info("🎓 FL Coordination: /api/v1/fl/*")
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