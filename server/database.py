"""
Database models and operations for QFLARE server.
"""

from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from datetime import datetime
import logging
import os
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./qflare.db")

# Create database engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


class Device(Base):
    """Device model for storing device information."""
    __tablename__ = "devices"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, unique=True, index=True, nullable=False)
    status = Column(String, default="active")
    kem_public_key = Column(Text, nullable=True)
    signature_public_key = Column(Text, nullable=True)
    enrollment_token = Column(String, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_seen = Column(DateTime, default=func.now())
    device_metadata = Column(Text, nullable=True)  # JSON string for additional data


class KeyPair(Base):
    """Key pair model for storing generated keys."""
    __tablename__ = "key_pairs"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True, nullable=False)
    key_type = Column(String, nullable=False)  # "kem", "signature", "session"
    public_key = Column(Text, nullable=False)
    private_key = Column(Text, nullable=True)  # Only for server keys
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)


class Session(Base):
    """Session model for managing device sessions."""
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True, nullable=False)
    session_token = Column(String, unique=True, nullable=False)
    challenge = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True)


class ModelUpdate(Base):
    """Model update storage."""
    __tablename__ = "model_updates"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True, nullable=False)
    model_weights = Column(Text, nullable=False)  # Base64 encoded
    signature = Column(Text, nullable=False)
    model_metadata = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=func.now())
    aggregation_round = Column(Integer, default=1)
    is_aggregated = Column(Boolean, default=False)


# Create tables
Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DatabaseManager:
    """Database manager for QFLARE operations."""
    
    def __init__(self):
        self.db = SessionLocal()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()
    
    def register_device(self, device_id: str, kem_public_key: str = None, 
                       signature_public_key: str = None, metadata: Dict[str, Any] = None) -> bool:
        """Register a new device."""
        try:
            import json
            metadata_json = json.dumps(metadata) if metadata else None
            
            device = Device(
                device_id=device_id,
                kem_public_key=kem_public_key,
                signature_public_key=signature_public_key,
                device_metadata=metadata_json
            )
            
            self.db.add(device)
            self.db.commit()
            logger.info(f"Registered device {device_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error registering device {device_id}: {e}")
            return False
    
    def get_device(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get device information."""
        try:
            device = self.db.query(Device).filter(Device.device_id == device_id).first()
            if device:
                # Update last seen
                device.last_seen = datetime.now()
                self.db.commit()
                
                import json
                metadata = json.loads(device.device_metadata) if device.device_metadata else {}
                
                return {
                    "device_id": device.device_id,
                    "status": device.status,
                    "kem_public_key": device.kem_public_key,
                    "signature_public_key": device.signature_public_key,
                    "created_at": device.created_at,
                    "last_seen": device.last_seen,
                    "metadata": metadata
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting device {device_id}: {e}")
            return None
    
    def update_device_status(self, device_id: str, status: str) -> bool:
        """Update device status."""
        try:
            device = self.db.query(Device).filter(Device.device_id == device_id).first()
            if device:
                device.status = status
                device.last_seen = datetime.now()
                self.db.commit()
                logger.info(f"Updated status for device {device_id} to {status}")
                return True
            return False
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating status for device {device_id}: {e}")
            return False
    
    def get_all_devices(self) -> List[Dict[str, Any]]:
        """Get all registered devices."""
        try:
            devices = self.db.query(Device).all()
            result = []
            
            for device in devices:
                import json
                metadata = json.loads(device.device_metadata) if device.device_metadata else {}
                
                result.append({
                    "device_id": device.device_id,
                    "status": device.status,
                    "kem_public_key": device.kem_public_key,
                    "signature_public_key": device.signature_public_key,
                    "created_at": device.created_at,
                    "last_seen": device.last_seen,
                    "metadata": metadata
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting all devices: {e}")
            return []
    
    def store_key_pair(self, device_id: str, key_type: str, public_key: str, 
                      private_key: str = None, expires_at: datetime = None) -> bool:
        """Store a key pair."""
        try:
            key_pair = KeyPair(
                device_id=device_id,
                key_type=key_type,
                public_key=public_key,
                private_key=private_key,
                expires_at=expires_at
            )
            
            self.db.add(key_pair)
            self.db.commit()
            logger.info(f"Stored {key_type} key pair for device {device_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error storing key pair for device {device_id}: {e}")
            return False
    
    def get_device_keys(self, device_id: str, key_type: str = None) -> List[Dict[str, Any]]:
        """Get device keys."""
        try:
            query = self.db.query(KeyPair).filter(KeyPair.device_id == device_id)
            if key_type:
                query = query.filter(KeyPair.key_type == key_type)
            
            keys = query.filter(KeyPair.is_active == True).all()
            result = []
            
            for key in keys:
                result.append({
                    "key_type": key.key_type,
                    "public_key": key.public_key,
                    "private_key": key.private_key,
                    "created_at": key.created_at,
                    "expires_at": key.expires_at
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting keys for device {device_id}: {e}")
            return []
    
    def create_session(self, device_id: str, session_token: str, 
                      challenge: str = None, expires_at: datetime = None) -> bool:
        """Create a new session."""
        try:
            if not expires_at:
                from datetime import timedelta
                expires_at = datetime.now() + timedelta(hours=24)
            
            session = Session(
                device_id=device_id,
                session_token=session_token,
                challenge=challenge,
                expires_at=expires_at
            )
            
            self.db.add(session)
            self.db.commit()
            logger.info(f"Created session for device {device_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating session for device {device_id}: {e}")
            return False
    
    def get_active_session(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get active session for device."""
        try:
            session = self.db.query(Session).filter(
                Session.device_id == device_id,
                Session.is_active == True,
                Session.expires_at > datetime.now()
            ).first()
            
            if session:
                return {
                    "session_token": session.session_token,
                    "challenge": session.challenge,
                    "created_at": session.created_at,
                    "expires_at": session.expires_at
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting session for device {device_id}: {e}")
            return None
    
    def store_model_update(self, device_id: str, model_weights: str, 
                          signature: str, metadata: Dict[str, Any] = None) -> bool:
        """Store model update."""
        try:
            import json
            metadata_json = json.dumps(metadata) if metadata else None
            
            model_update = ModelUpdate(
                device_id=device_id,
                model_weights=model_weights,
                signature=signature,
                model_metadata=metadata_json
            )
            
            self.db.add(model_update)
            self.db.commit()
            logger.info(f"Stored model update for device {device_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error storing model update for device {device_id}: {e}")
            return False
    
    def get_device_statistics(self) -> Dict[str, Any]:
        """Get device statistics."""
        try:
            total_devices = self.db.query(Device).count()
            active_devices = self.db.query(Device).filter(Device.status == "active").count()
            recent_devices = self.db.query(Device).filter(
                Device.last_seen >= datetime.now() - timedelta(hours=24)
            ).count()
            
            return {
                "total_devices": total_devices,
                "active_devices": active_devices,
                "recent_devices": recent_devices,
                "inactive_devices": total_devices - active_devices
            }
            
        except Exception as e:
            logger.error(f"Error getting device statistics: {e}")
            return {
                "total_devices": 0,
                "active_devices": 0,
                "recent_devices": 0,
                "inactive_devices": 0
            }


# Global database manager instance
db_manager = DatabaseManager() 