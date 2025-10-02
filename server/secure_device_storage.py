#!/usr/bin/env python3
"""
QFLARE Secure Device Management - Production Implementation
Replaces in-memory storage with PostgreSQL + KMS encryption
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
import logging

# Database and encryption imports
try:
    from sqlalchemy import create_engine, Column, String, DateTime, Text, LargeBinary, Boolean, Integer, Float
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.dialects.postgresql import JSONB
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    import boto3
    HAS_SECURE_STORAGE = True
except ImportError:
    # Fallback to in-memory storage if dependencies not available
    HAS_SECURE_STORAGE = False

import base64
import uuid
import secrets

logger = logging.getLogger(__name__)

# Database Models
if HAS_SECURE_STORAGE:
    Base = declarative_base()

    class SecureDevice(Base):
        """Production device storage with encryption"""
        __tablename__ = 'secure_devices'
        
        # Primary identification
        device_id = Column(String(255), primary_key=True)
        device_name = Column(String(255), nullable=False, unique=True)
        device_type = Column(String(100), nullable=False)
        
        # Encrypted sensitive data
        encrypted_metadata = Column(LargeBinary)  # Capabilities, contact info, etc.
        encryption_context = Column(JSONB)       # KMS encryption context
        
        # Public searchable data
        status = Column(String(50), default='online')
        device_class = Column(String(50))        # mobile, desktop, server, iot
        location_region = Column(String(100))    # Geographic region only
        
        # Security and performance
        trust_score = Column(Float, default=0.0)
        security_level = Column(Integer, default=1)
        total_training_sessions = Column(Integer, default=0)
        total_training_time = Column(Float, default=0.0)
        success_rate = Column(Float, default=100.0)
        
        # Timestamps
        registered_at = Column(DateTime, default=datetime.utcnow)
        last_seen = Column(DateTime, default=datetime.utcnow)
        last_heartbeat = Column(DateTime)

    class SecureKeyMaterial(Base):
        """Encrypted cryptographic keys with envelope encryption"""
        __tablename__ = 'secure_keys'
        
        key_id = Column(String(255), primary_key=True)
        device_id = Column(String(255), nullable=False, index=True)
        key_type = Column(String(50), nullable=False)  # quantum_private, session, etc.
        
        # Envelope encryption
        encrypted_key_data = Column(LargeBinary, nullable=False)
        encryption_context = Column(JSONB)
        
        # Key metadata
        algorithm = Column(String(100))
        key_purpose = Column(String(100))
        created_at = Column(DateTime, default=datetime.utcnow)
        expires_at = Column(DateTime)
        status = Column(String(50), default='active')
        usage_count = Column(Integer, default=0)
        last_used = Column(DateTime)

class SecureDeviceStorage:
    """
    Production secure storage implementation
    Falls back to in-memory if secure dependencies unavailable
    """
    
    def __init__(self, database_url: Optional[str] = None, kms_key_id: Optional[str] = None):
        self.secure_mode = HAS_SECURE_STORAGE and database_url and kms_key_id
        
        if self.secure_mode:
            self._init_secure_storage(database_url, kms_key_id)
        else:
            self._init_fallback_storage()
            logger.warning("🔓 Using fallback in-memory storage - not secure for production!")
    
    def _init_secure_storage(self, database_url: str, kms_key_id: str):
        """Initialize PostgreSQL + KMS secure storage"""
        try:
            self.engine = create_engine(database_url, echo=False)
            Base.metadata.create_all(self.engine)
            self.SessionLocal = sessionmaker(bind=self.engine)
            
            # Initialize KMS (simplified - use AWS SDK configuration)
            self.kms_key_id = kms_key_id
            self.kms_client = boto3.client('kms')
            
            logger.info("🔐 Secure storage initialized with PostgreSQL + KMS")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize secure storage: {e}")
            logger.info("🔓 Falling back to in-memory storage")
            self.secure_mode = False
            self._init_fallback_storage()
    
    def _init_fallback_storage(self):
        """Initialize in-memory fallback storage"""
        self.devices: Dict[str, Dict] = {}
        self.keys: Dict[str, Dict] = {}
        logger.info("📝 In-memory storage initialized")
    
    def _generate_encryption_key(self) -> bytes:
        """Generate AES-256 key for data encryption"""
        return secrets.token_bytes(32)
    
    def _encrypt_data_secure(self, data: bytes, context: Dict[str, str]) -> Dict[str, Any]:
        """Encrypt data using KMS + AES-GCM envelope encryption"""
        if not self.secure_mode:
            # Fallback: base64 encode (not secure!)
            return {
                'data': base64.b64encode(data).decode(),
                'method': 'fallback'
            }
        
        try:
            # Generate data encryption key
            dek_response = self.kms_client.generate_data_key(
                KeyId=self.kms_key_id,
                KeySpec='AES_256',
                EncryptionContext=context
            )
            
            # Encrypt data with DEK
            aesgcm = AESGCM(dek_response['Plaintext'])
            nonce = os.urandom(12)
            ciphertext = aesgcm.encrypt(nonce, data, None)
            
            return {
                'encrypted_dek': base64.b64encode(dek_response['CiphertextBlob']).decode(),
                'nonce': base64.b64encode(nonce).decode(),
                'ciphertext': base64.b64encode(ciphertext).decode(),
                'method': 'kms_envelope'
            }
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def _decrypt_data_secure(self, encrypted_blob: Dict[str, Any], context: Dict[str, str]) -> bytes:
        """Decrypt data using KMS + AES-GCM"""
        if encrypted_blob.get('method') == 'fallback':
            return base64.b64decode(encrypted_blob['data'])
        
        try:
            # Decrypt DEK using KMS
            dek_response = self.kms_client.decrypt(
                CiphertextBlob=base64.b64decode(encrypted_blob['encrypted_dek']),
                EncryptionContext=context
            )
            
            # Decrypt data using DEK
            aesgcm = AESGCM(dek_response['Plaintext'])
            nonce = base64.b64decode(encrypted_blob['nonce'])
            ciphertext = base64.b64decode(encrypted_blob['ciphertext'])
            
            return aesgcm.decrypt(nonce, ciphertext, None)
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    async def store_device(self, device_data: Dict[str, Any]) -> str:
        """Store device with appropriate security level"""
        device_id = str(uuid.uuid4())
        
        if self.secure_mode:
            return await self._store_device_secure(device_id, device_data)
        else:
            return await self._store_device_fallback(device_id, device_data)
    
    async def _store_device_secure(self, device_id: str, device_data: Dict[str, Any]) -> str:
        """Store device using PostgreSQL + KMS encryption"""
        # Separate sensitive and public data
        sensitive_data = {
            'capabilities': device_data.get('capabilities', []),
            'contact_info': device_data.get('contact_info'),
            'location': device_data.get('location'),
            'hardware_specs': device_data.get('hardware_specs', {}),
            'max_concurrent_tasks': device_data.get('max_concurrent_tasks', 1),
            'preferred_schedule': device_data.get('preferred_schedule')
        }
        
        # Encrypt sensitive data
        encryption_context = {
            'device_id': device_id,
            'data_type': 'device_metadata'
        }
        
        sensitive_json = json.dumps(sensitive_data, default=str).encode('utf-8')
        encrypted_blob = self._encrypt_data_secure(sensitive_json, encryption_context)
        
        # Store in database
        with self.SessionLocal() as db:
            device = SecureDevice(
                device_id=device_id,
                device_name=device_data['device_name'],
                device_type=device_data['device_type'],
                encrypted_metadata=json.dumps(encrypted_blob).encode(),
                encryption_context=encryption_context,
                device_class=device_data.get('device_class', 'unknown'),
                location_region=device_data.get('location', {}).get('region', 'unknown'),
                security_level=device_data.get('security_level', 1)
            )
            
            db.add(device)
            db.commit()
            
        logger.info(f"🔐 Device stored securely: {device_id}")
        return device_id
    
    async def _store_device_fallback(self, device_id: str, device_data: Dict[str, Any]) -> str:
        """Store device using in-memory fallback"""
        self.devices[device_id] = {
            **device_data,
            'device_id': device_id,
            'registered_at': datetime.utcnow(),
            'last_seen': datetime.utcnow(),
            'status': 'online'
        }
        
        logger.info(f"📝 Device stored in memory: {device_id}")
        return device_id
    
    async def get_device(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve device with decryption if secure"""
        if self.secure_mode:
            return await self._get_device_secure(device_id)
        else:
            return self.devices.get(device_id)
    
    async def _get_device_secure(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve device from PostgreSQL with decryption"""
        with self.SessionLocal() as db:
            device = db.query(SecureDevice).filter(
                SecureDevice.device_id == device_id
            ).first()
            
            if not device:
                return None
            
            # Decrypt sensitive data
            try:
                encrypted_blob = json.loads(device.encrypted_metadata.decode())
                decrypted_data = self._decrypt_data_secure(
                    encrypted_blob, 
                    device.encryption_context
                )
                sensitive_data = json.loads(decrypted_data.decode())
                
            except Exception as e:
                logger.warning(f"Failed to decrypt device data: {e}")
                sensitive_data = {}
            
            return {
                'device_id': device.device_id,
                'device_name': device.device_name,
                'device_type': device.device_type,
                'status': device.status,
                'registered_at': device.registered_at,
                'last_seen': device.last_seen,
                'trust_score': device.trust_score,
                'total_training_sessions': device.total_training_sessions,
                'success_rate': device.success_rate,
                **sensitive_data
            }
    
    async def list_devices(self, limit: int = 50, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List devices with optional filtering"""
        if self.secure_mode:
            return await self._list_devices_secure(limit, status_filter)
        else:
            devices = list(self.devices.values())
            if status_filter:
                devices = [d for d in devices if d.get('status') == status_filter]
            return devices[:limit]
    
    async def _list_devices_secure(self, limit: int, status_filter: Optional[str]) -> List[Dict[str, Any]]:
        """List devices from PostgreSQL"""
        with self.SessionLocal() as db:
            query = db.query(SecureDevice)
            
            if status_filter:
                query = query.filter(SecureDevice.status == status_filter)
            
            devices = query.order_by(SecureDevice.last_seen.desc()).limit(limit).all()
            
            result = []
            for device in devices:
                # Return only public data for list view (performance)
                result.append({
                    'device_id': device.device_id,
                    'device_name': device.device_name,
                    'device_type': device.device_type,
                    'status': device.status,
                    'device_class': device.device_class,
                    'location_region': device.location_region,
                    'registered_at': device.registered_at,
                    'last_seen': device.last_seen,
                    'trust_score': device.trust_score,
                    'total_training_sessions': device.total_training_sessions,
                    'success_rate': device.success_rate
                })
            
            return result
    
    async def update_device_heartbeat(self, device_id: str) -> bool:
        """Update device last seen timestamp"""
        if self.secure_mode:
            with self.SessionLocal() as db:
                device = db.query(SecureDevice).filter(
                    SecureDevice.device_id == device_id
                ).first()
                
                if device:
                    device.last_seen = datetime.utcnow()
                    device.last_heartbeat = datetime.utcnow()
                    db.commit()
                    return True
                return False
        else:
            if device_id in self.devices:
                self.devices[device_id]['last_seen'] = datetime.utcnow()
                return True
            return False
    
    async def store_key_material(self, device_id: str, key_type: str, 
                               key_data: bytes, algorithm: str = 'kyber1024') -> str:
        """Store cryptographic key material securely"""
        key_id = str(uuid.uuid4())
        
        if self.secure_mode:
            return await self._store_key_secure(key_id, device_id, key_type, key_data, algorithm)
        else:
            return await self._store_key_fallback(key_id, device_id, key_type, key_data, algorithm)
    
    async def _store_key_secure(self, key_id: str, device_id: str, key_type: str, 
                              key_data: bytes, algorithm: str) -> str:
        """Store key using KMS encryption"""
        encryption_context = {
            'device_id': device_id,
            'key_id': key_id,
            'key_type': key_type,
            'algorithm': algorithm
        }
        
        encrypted_blob = self._encrypt_data_secure(key_data, encryption_context)
        
        with self.SessionLocal() as db:
            key_record = SecureKeyMaterial(
                key_id=key_id,
                device_id=device_id,
                key_type=key_type,
                encrypted_key_data=json.dumps(encrypted_blob).encode(),
                encryption_context=encryption_context,
                algorithm=algorithm,
                expires_at=datetime.utcnow() + timedelta(days=365)
            )
            
            db.add(key_record)
            db.commit()
        
        logger.info(f"🔐 Key stored securely: {key_id}")
        return key_id
    
    async def _store_key_fallback(self, key_id: str, device_id: str, key_type: str, 
                                key_data: bytes, algorithm: str) -> str:
        """Store key in memory (insecure fallback)"""
        self.keys[key_id] = {
            'key_id': key_id,
            'device_id': device_id,
            'key_type': key_type,
            'key_data': base64.b64encode(key_data).decode(),  # Not secure!
            'algorithm': algorithm,
            'created_at': datetime.utcnow(),
            'status': 'active'
        }
        
        logger.warning(f"🔓 Key stored in memory (insecure): {key_id}")
        return key_id
    
    async def get_storage_status(self) -> Dict[str, Any]:
        """Get storage system status and statistics"""
        if self.secure_mode:
            with self.SessionLocal() as db:
                device_count = db.query(SecureDevice).count()
                key_count = db.query(SecureKeyMaterial).count()
                
                return {
                    'storage_type': 'secure_postgresql_kms',
                    'secure_mode': True,
                    'device_count': device_count,
                    'key_count': key_count,
                    'encryption': 'KMS + AES-GCM envelope encryption',
                    'database': 'PostgreSQL with encryption at rest'
                }
        else:
            return {
                'storage_type': 'in_memory_fallback',
                'secure_mode': False,
                'device_count': len(self.devices),
                'key_count': len(self.keys),
                'encryption': 'None - development only',
                'warning': 'Not secure for production use'
            }

# Global storage instance
_storage_instance: Optional[SecureDeviceStorage] = None

def get_secure_storage() -> SecureDeviceStorage:
    """Get global secure storage instance"""
    global _storage_instance
    
    if _storage_instance is None:
        # Try to initialize with environment variables
        database_url = os.getenv('QFLARE_DATABASE_URL')
        kms_key_id = os.getenv('QFLARE_KMS_KEY_ID')
        
        _storage_instance = SecureDeviceStorage(database_url, kms_key_id)
    
    return _storage_instance

# Example usage and testing
async def test_secure_storage():
    """Test the secure storage implementation"""
    storage = get_secure_storage()
    
    print("🧪 Testing QFLARE Secure Storage")
    print("=" * 50)
    
    # Test device storage
    device_data = {
        'device_name': 'test_device_001',
        'device_type': 'smartphone',
        'capabilities': ['cpu', 'gpu'],
        'contact_info': 'test@example.com',
        'location': {'region': 'us-east-1', 'city': 'New York'},
        'hardware_specs': {'cpu_cores': 8, 'ram_gb': 6}
    }
    
    device_id = await storage.store_device(device_data)
    print(f"✅ Device stored: {device_id}")
    
    # Test device retrieval
    retrieved_device = await storage.get_device(device_id)
    print(f"✅ Device retrieved: {retrieved_device['device_name']}")
    
    # Test key storage
    key_data = os.urandom(64)  # 512-bit key
    key_id = await storage.store_key_material(device_id, 'quantum_private', key_data)
    print(f"🔐 Key stored: {key_id}")
    
    # Test storage status
    status = await storage.get_storage_status()
    print(f"📊 Storage status: {status['storage_type']}")
    print(f"🔒 Secure mode: {status['secure_mode']}")
    
    return storage

if __name__ == "__main__":
    asyncio.run(test_secure_storage())