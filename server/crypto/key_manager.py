#!/usr/bin/env python3
"""
Advanced Key Management System for QFLARE
Integrates with PostgreSQL database for comprehensive key lifecycle management
"""

import os
import asyncio
import asyncpg
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import uuid
import secrets
import hashlib
import hmac
import struct
from dataclasses import dataclass
from enum import Enum
import json
import base64

# Quantum-safe crypto imports
try:
    import oqs
    LIBOQS_AVAILABLE = True
except ImportError:
    LIBOQS_AVAILABLE = False

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.fernet import Fernet

from .quantum_key_exchange import LatticeKeyExchange, QuantumSafeEncryption

logger = logging.getLogger(__name__)

class KeyType(Enum):
    KYBER_PUBLIC = "KYBER_PUBLIC"
    KYBER_PRIVATE = "KYBER_PRIVATE"
    DILITHIUM_PUBLIC = "DILITHIUM_PUBLIC"
    DILITHIUM_PRIVATE = "DILITHIUM_PRIVATE"
    SESSION_KEY = "SESSION_KEY"
    DERIVED_KEY = "DERIVED_KEY"
    BACKUP_KEY = "BACKUP_KEY"

class KeyStatus(Enum):
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"
    COMPROMISED = "COMPROMISED"
    PENDING_ROTATION = "PENDING_ROTATION"

@dataclass
class KeyMetadata:
    key_id: str
    device_id: str
    key_type: KeyType
    algorithm: str
    status: KeyStatus
    created_at: datetime
    expires_at: Optional[datetime]
    usage_count: int = 0
    max_usage_count: Optional[int] = None

class QuantumKeyManager:
    """
    Advanced key management system with PostgreSQL integration.
    Handles quantum-safe key generation, rotation, and lifecycle management.
    """
    
    def __init__(self, db_connection_string: str, master_key: bytes = None):
        self.db_connection_string = db_connection_string
        self.master_key = master_key or self._derive_master_key()
        self.fernet = Fernet(base64.urlsafe_b64encode(self.master_key[:32]))
        self.key_exchange = LatticeKeyExchange()
        
    def _derive_master_key(self) -> bytes:
        """Derive master key from environment or generate new one"""
        master_key_env = os.getenv('QFLARE_MASTER_KEY')
        if master_key_env:
            return base64.b64decode(master_key_env)
        
        # Generate new master key (save this securely!)
        master_key = secrets.token_bytes(32)
        logger.warning("Generated new master key - ensure this is saved securely!")
        logger.info(f"Master key (base64): {base64.b64encode(master_key).decode()}")
        return master_key
    
    async def get_db_connection(self) -> asyncpg.Connection:
        """Get database connection"""
        return await asyncpg.connect(self.db_connection_string)
    
    async def generate_device_keypair(self, 
                                    device_id: str, 
                                    algorithm: str = "Kyber1024",
                                    expires_hours: int = 24) -> Dict[str, str]:
        """
        Generate a new quantum-safe keypair for a device.
        Returns key IDs for the public and private keys.
        """
        conn = await self.get_db_connection()
        try:
            # Generate keypair
            if LIBOQS_AVAILABLE and algorithm.startswith("Kyber"):
                kem = oqs.KeyEncapsulation(algorithm)
                public_key = kem.generate_keypair()
                private_key = kem.export_secret_key()
            else:
                # Fallback to RSA for development
                from cryptography.hazmat.primitives.asymmetric import rsa
                from cryptography.hazmat.primitives import serialization
                
                private_key_obj = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=4096
                )
                public_key = private_key_obj.public_key().public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
                private_key = private_key_obj.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
            
            # Encrypt private key
            encrypted_private_key = self.fernet.encrypt(private_key)
            
            # Generate key IDs
            public_key_id = str(uuid.uuid4())
            private_key_id = str(uuid.uuid4())
            
            # Calculate expiry
            expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
            
            # Store public key
            await conn.execute("""
                INSERT INTO cryptographic_keys 
                (key_id, device_id, key_type, algorithm, public_key, key_size, 
                 status, expires_at, generation_entropy)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, public_key_id, device_id, KeyType.KYBER_PUBLIC.value, algorithm,
                public_key, len(public_key), KeyStatus.ACTIVE.value, expires_at,
                secrets.token_bytes(32))
            
            # Store private key
            await conn.execute("""
                INSERT INTO cryptographic_keys 
                (key_id, device_id, key_type, algorithm, private_key_encrypted, key_size,
                 status, expires_at, generation_entropy)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, private_key_id, device_id, KeyType.KYBER_PRIVATE.value, algorithm,
                encrypted_private_key, len(private_key), KeyStatus.ACTIVE.value, expires_at,
                secrets.token_bytes(32))
            
            # Log key generation
            await self._log_audit_event(conn, device_id, "KEY_ROTATION", {
                "action": "keypair_generated",
                "algorithm": algorithm,
                "public_key_id": public_key_id,
                "private_key_id": private_key_id,
                "expires_at": expires_at.isoformat()
            })
            
            return {
                "public_key_id": public_key_id,
                "private_key_id": private_key_id,
                "public_key_b64": base64.b64encode(public_key).decode(),
                "expires_at": expires_at.isoformat()
            }
            
        finally:
            await conn.close()
    
    async def initiate_temporal_key_exchange(self, 
                                           device_id: str, 
                                           client_public_key: str) -> Dict:
        """
        Initiate quantum-safe key exchange with temporal mapping.
        Integrates with database for session tracking.
        """
        conn = await self.get_db_connection()
        try:
            # Decode client public key
            client_pub_key_bytes = base64.b64decode(client_public_key)
            
            # Perform key exchange using our quantum system
            exchange_data = self.key_exchange.initiate_key_exchange(
                device_id, client_pub_key_bytes
            )
            
            # Get server public key from database
            server_key_record = await conn.fetchrow("""
                SELECT id, public_key FROM cryptographic_keys 
                WHERE device_id IS NULL 
                AND key_type = 'KYBER_PUBLIC' 
                AND status = 'ACTIVE'
                ORDER BY created_at DESC LIMIT 1
            """)
            
            if not server_key_record:
                raise Exception("No server public key available")
            
            # Store session in database
            session_record = await conn.fetchrow("""
                INSERT INTO key_exchange_sessions 
                (session_id, device_id, status, algorithm, initiation_timestamp,
                 expiry_timestamp, server_public_key_id, client_public_key,
                 nonce, salt, time_window)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING id
            """, 
                exchange_data['session_id'],
                device_id,
                'PENDING',
                exchange_data['algorithm'],
                exchange_data['timestamp'],
                exchange_data['expiry_time'],
                server_key_record['id'],
                client_pub_key_bytes,
                base64.b64decode(exchange_data['nonce']),
                secrets.token_bytes(32),  # Additional salt
                300  # 5-minute time window
            )
            
            # Log key exchange initiation
            await self._log_audit_event(conn, device_id, "KEY_EXCHANGE", {
                "action": "initiated",
                "session_id": exchange_data['session_id'],
                "algorithm": exchange_data['algorithm'],
                "timestamp": exchange_data['timestamp']
            })
            
            return exchange_data
            
        finally:
            await conn.close()
    
    async def complete_key_exchange(self, 
                                  session_id: str, 
                                  client_response: str) -> bool:
        """
        Complete key exchange and derive session key.
        Updates database with completion status.
        """
        conn = await self.get_db_connection()
        try:
            # Get session from database
            session_record = await conn.fetchrow("""
                SELECT * FROM key_exchange_sessions 
                WHERE session_id = $1 AND status = 'PENDING'
            """, session_id)
            
            if not session_record:
                return False
            
            # Verify with key exchange system
            client_response_bytes = base64.b64decode(client_response)
            success = self.key_exchange.complete_key_exchange(
                session_id, client_response_bytes
            )
            
            if success:
                # Update session status
                await conn.execute("""
                    UPDATE key_exchange_sessions 
                    SET status = 'ACTIVE', completion_timestamp = $2
                    WHERE session_id = $1
                """, session_id, int(datetime.utcnow().timestamp()))
                
                # Generate derived session key
                session_key = self.key_exchange.get_session_key(session_id)
                if session_key:
                    # Store hashed version for verification
                    key_hash = hashlib.sha3_256(session_key).digest()
                    await conn.execute("""
                        UPDATE key_exchange_sessions 
                        SET derived_key_hash = $2
                        WHERE session_id = $1
                    """, session_id, key_hash)
                
                # Log successful completion
                await self._log_audit_event(conn, session_record['device_id'], 
                                          "KEY_EXCHANGE", {
                    "action": "completed",
                    "session_id": session_id,
                    "success": True
                })
                
                return True
            else:
                # Mark session as failed
                await conn.execute("""
                    UPDATE key_exchange_sessions 
                    SET status = 'TERMINATED'
                    WHERE session_id = $1
                """, session_id)
                
                # Log failure
                await self._log_audit_event(conn, session_record['device_id'], 
                                          "SECURITY_VIOLATION", {
                    "action": "key_exchange_failed",
                    "session_id": session_id,
                    "reason": "invalid_client_response"
                })
                
                return False
                
        finally:
            await conn.close()
    
    async def rotate_device_keys(self, device_id: str) -> Dict:
        """
        Rotate all keys for a device. Implements forward secrecy.
        """
        conn = await self.get_db_connection()
        try:
            # Mark existing keys as pending rotation
            await conn.execute("""
                UPDATE cryptographic_keys 
                SET status = 'PENDING_ROTATION', rotated_at = CURRENT_TIMESTAMP
                WHERE device_id = $1 AND status = 'ACTIVE'
            """, device_id)
            
            # Generate new keypair
            new_keys = await self.generate_device_keypair(device_id)
            
            # Revoke old keys after successful generation
            await conn.execute("""
                UPDATE cryptographic_keys 
                SET status = 'REVOKED'
                WHERE device_id = $1 AND status = 'PENDING_ROTATION'
            """, device_id)
            
            # Terminate active sessions (force re-authentication)
            await conn.execute("""
                UPDATE key_exchange_sessions 
                SET status = 'TERMINATED'
                WHERE device_id = $1 AND status = 'ACTIVE'
            """, device_id)
            
            # Log key rotation
            await self._log_audit_event(conn, device_id, "KEY_ROTATION", {
                "action": "keys_rotated",
                "new_public_key_id": new_keys["public_key_id"],
                "reason": "scheduled_rotation"
            })
            
            return new_keys
            
        finally:
            await conn.close()
    
    async def get_device_public_key(self, device_id: str) -> Optional[bytes]:
        """Get the active public key for a device"""
        conn = await self.get_db_connection()
        try:
            record = await conn.fetchrow("""
                SELECT public_key FROM cryptographic_keys 
                WHERE device_id = $1 
                AND key_type = 'KYBER_PUBLIC' 
                AND status = 'ACTIVE'
                ORDER BY created_at DESC LIMIT 1
            """, device_id)
            
            return record['public_key'] if record else None
            
        finally:
            await conn.close()
    
    async def verify_session_key(self, session_id: str, provided_key: bytes) -> bool:
        """Verify a session key against stored hash"""
        conn = await self.get_db_connection()
        try:
            record = await conn.fetchrow("""
                SELECT derived_key_hash FROM key_exchange_sessions 
                WHERE session_id = $1 AND status = 'ACTIVE'
                AND expiry_timestamp > $2
            """, session_id, int(datetime.utcnow().timestamp()))
            
            if not record or not record['derived_key_hash']:
                return False
            
            provided_hash = hashlib.sha3_256(provided_key).digest()
            return hmac.compare_digest(record['derived_key_hash'], provided_hash)
            
        finally:
            await conn.close()
    
    async def cleanup_expired_keys_and_sessions(self) -> Dict[str, int]:
        """Clean up expired keys and sessions"""
        conn = await self.get_db_connection()
        try:
            current_timestamp = int(datetime.utcnow().timestamp())
            
            # Expire old keys
            expired_keys = await conn.execute("""
                UPDATE cryptographic_keys 
                SET status = 'EXPIRED'
                WHERE expires_at < CURRENT_TIMESTAMP 
                AND status = 'ACTIVE'
            """)
            
            # Expire old sessions
            expired_sessions = await conn.execute("""
                UPDATE key_exchange_sessions 
                SET status = 'EXPIRED'
                WHERE expiry_timestamp < $1 
                AND status = 'ACTIVE'
            """, current_timestamp)
            
            # Log cleanup
            await self._log_audit_event(conn, None, "SYSTEM_ERROR", {
                "action": "cleanup_expired",
                "expired_keys": expired_keys,
                "expired_sessions": expired_sessions
            })
            
            return {
                "expired_keys": expired_keys,
                "expired_sessions": expired_sessions
            }
            
        finally:
            await conn.close()
    
    async def get_key_statistics(self, device_id: Optional[str] = None) -> Dict:
        """Get comprehensive key and session statistics"""
        conn = await self.get_db_connection()
        try:
            where_clause = "WHERE device_id = $1" if device_id else ""
            params = [device_id] if device_id else []
            
            # Key statistics
            key_stats = await conn.fetchrow(f"""
                SELECT 
                    COUNT(*) as total_keys,
                    COUNT(CASE WHEN status = 'ACTIVE' THEN 1 END) as active_keys,
                    COUNT(CASE WHEN status = 'EXPIRED' THEN 1 END) as expired_keys,
                    COUNT(CASE WHEN status = 'REVOKED' THEN 1 END) as revoked_keys
                FROM cryptographic_keys {where_clause}
            """, *params)
            
            # Session statistics  
            session_stats = await conn.fetchrow(f"""
                SELECT 
                    COUNT(*) as total_sessions,
                    COUNT(CASE WHEN status = 'ACTIVE' THEN 1 END) as active_sessions,
                    COUNT(CASE WHEN status = 'EXPIRED' THEN 1 END) as expired_sessions
                FROM key_exchange_sessions {where_clause}
            """, *params)
            
            return {
                "keys": dict(key_stats),
                "sessions": dict(session_stats),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            await conn.close()
    
    async def _log_audit_event(self, 
                             conn: asyncpg.Connection,
                             device_id: Optional[str],
                             event_type: str,
                             event_data: Dict) -> None:
        """Log an audit event"""
        await conn.execute("""
            INSERT INTO audit_logs (device_id, event_type, event_data, severity)
            VALUES ($1, $2, $3, $4)
        """, device_id, event_type, json.dumps(event_data), "INFO")

# Background task for key rotation and cleanup
class KeyMaintenanceService:
    """Background service for key maintenance tasks"""
    
    def __init__(self, key_manager: QuantumKeyManager):
        self.key_manager = key_manager
        self.running = False
    
    async def start(self):
        """Start the maintenance service"""
        self.running = True
        while self.running:
            try:
                # Clean up expired keys and sessions
                await self.key_manager.cleanup_expired_keys_and_sessions()
                
                # Check for keys that need rotation (based on usage or time)
                await self._check_key_rotation_needed()
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Key maintenance error: {e}")
                await asyncio.sleep(60)  # Shorter sleep on error
    
    def stop(self):
        """Stop the maintenance service"""
        self.running = False
    
    async def _check_key_rotation_needed(self):
        """Check if any keys need rotation"""
        conn = await self.key_manager.get_db_connection()
        try:
            # Find devices with keys expiring soon (24 hours)
            expiring_soon = await conn.fetch("""
                SELECT DISTINCT device_id 
                FROM cryptographic_keys 
                WHERE expires_at < CURRENT_TIMESTAMP + INTERVAL '24 hours'
                AND status = 'ACTIVE'
                AND device_id IS NOT NULL
            """)
            
            for record in expiring_soon:
                device_id = record['device_id']
                logger.info(f"Rotating keys for device {device_id} (expiring soon)")
                await self.key_manager.rotate_device_keys(device_id)
                
        finally:
            await conn.close()

# Example usage
async def main():
    """Example usage of the key management system"""
    
    # Database connection string
    db_url = "postgresql://qflare_user:secure_password@localhost:5432/qflare"
    
    # Initialize key manager
    key_manager = QuantumKeyManager(db_url)
    
    # Generate device keypair
    device_id = str(uuid.uuid4())
    keys = await key_manager.generate_device_keypair(device_id)
    print(f"Generated keys for device {device_id}")
    
    # Simulate key exchange
    client_public_key = keys["public_key_b64"]  # In real scenario, this comes from client
    exchange_data = await key_manager.initiate_temporal_key_exchange(
        device_id, client_public_key
    )
    print(f"Key exchange initiated: {exchange_data['session_id']}")
    
    # Get statistics
    stats = await key_manager.get_key_statistics()
    print(f"Key statistics: {stats}")

if __name__ == "__main__":
    asyncio.run(main())