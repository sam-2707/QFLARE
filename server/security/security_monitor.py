#!/usr/bin/env python3
"""
Comprehensive Security Monitoring and Logging System for QFLARE
Implements quantum-safe signatures and advanced threat detection
"""

import os
import time
import asyncio
import json
import hashlib
import hmac
from typing import Dict, List, Optional, Any, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncpg
import secrets
import base64

try:
    import oqs
    LIBOQS_AVAILABLE = True
except ImportError:
    LIBOQS_AVAILABLE = False

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    LOW = 1
    MEDIUM = 3
    HIGH = 7
    CRITICAL = 10

class EventSeverity(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class SecurityEventType(Enum):
    DEVICE_REGISTRATION = "DEVICE_REGISTRATION"
    DEVICE_AUTHENTICATION = "DEVICE_AUTHENTICATION"
    KEY_EXCHANGE = "KEY_EXCHANGE"
    KEY_ROTATION = "KEY_ROTATION"
    MODEL_SUBMISSION = "MODEL_SUBMISSION"
    MODEL_AGGREGATION = "MODEL_AGGREGATION"
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    QUANTUM_ATTACK_DETECTED = "QUANTUM_ATTACK_DETECTED"
    ANOMALOUS_BEHAVIOR = "ANOMALOUS_BEHAVIOR"

@dataclass
class SecurityEvent:
    """Represents a security event in the system"""
    event_id: str
    event_type: SecurityEventType
    severity: EventSeverity
    threat_level: ThreatLevel
    timestamp: datetime
    device_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    event_data: Dict[str, Any]
    signature: Optional[bytes] = None
    verified: bool = False

class QuantumSafeSignature:
    """
    Quantum-safe digital signature system using Dilithium or RSA fallback.
    Provides non-repudiation for security events and audit logs.
    """
    
    def __init__(self):
        self.algorithm = "Dilithium2"
        if LIBOQS_AVAILABLE:
            try:
                self.sig = oqs.Signature(self.algorithm)
                self.public_key = self.sig.generate_keypair()
                self.private_key = self.sig.export_secret_key()
                logger.info(f"Using quantum-safe signatures: {self.algorithm}")
            except Exception as e:
                logger.warning(f"Dilithium not available: {e}, using RSA fallback")
                self._init_rsa_fallback()
        else:
            self._init_rsa_fallback()
    
    def _init_rsa_fallback(self):
        """Initialize RSA fallback for signatures"""
        self.algorithm = "RSA-PSS-4096"
        self.private_key_obj = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        self.public_key_obj = self.private_key_obj.public_key()
        
        self.private_key = self.private_key_obj.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        self.public_key = self.public_key_obj.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    def sign_event(self, event: SecurityEvent) -> bytes:
        """Sign a security event for non-repudiation"""
        # Create canonical representation of event
        event_dict = asdict(event)
        event_dict.pop('signature', None)  # Remove signature field
        event_dict.pop('verified', None)   # Remove verified field
        
        # Convert to canonical JSON
        canonical_data = json.dumps(event_dict, sort_keys=True, default=str).encode()
        
        if LIBOQS_AVAILABLE and hasattr(self, 'sig'):
            # Use Dilithium for quantum-safe signatures
            signature = self.sig.sign(canonical_data)
        else:
            # Use RSA-PSS fallback
            signature = self.private_key_obj.sign(
                canonical_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA512()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA512()
            )
        
        return signature
    
    def verify_signature(self, event: SecurityEvent, signature: bytes) -> bool:
        """Verify the signature of a security event"""
        try:
            # Recreate canonical representation
            event_dict = asdict(event)
            event_dict.pop('signature', None)
            event_dict.pop('verified', None)
            
            canonical_data = json.dumps(event_dict, sort_keys=True, default=str).encode()
            
            if LIBOQS_AVAILABLE and hasattr(self, 'sig'):
                # Verify with Dilithium
                return self.sig.verify(canonical_data, signature, self.public_key)
            else:
                # Verify with RSA-PSS
                try:
                    self.public_key_obj.verify(
                        signature,
                        canonical_data,
                        padding.PSS(
                            mgf=padding.MGF1(hashes.SHA512()),
                            salt_length=padding.PSS.MAX_LENGTH
                        ),
                        hashes.SHA512()
                    )
                    return True
                except Exception:
                    return False
                    
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False

class ThreatDetector:
    """
    Advanced threat detection system for quantum and classical attacks.
    Monitors patterns and anomalies in cryptographic operations.
    """
    
    def __init__(self):
        self.suspicious_patterns = {}
        self.device_behavior_profiles = {}
        self.quantum_attack_indicators = []
        
    def analyze_key_exchange_pattern(self, 
                                   device_id: str, 
                                   exchange_data: Dict) -> ThreatLevel:
        """Analyze key exchange patterns for anomalies"""
        current_time = time.time()
        
        # Track device behavior
        if device_id not in self.device_behavior_profiles:
            self.device_behavior_profiles[device_id] = {
                'exchanges': [],
                'first_seen': current_time,
                'typical_interval': None
            }
        
        profile = self.device_behavior_profiles[device_id]
        profile['exchanges'].append({
            'timestamp': current_time,
            'algorithm': exchange_data.get('algorithm'),
            'session_duration': exchange_data.get('session_duration', 0)
        })
        
        # Keep only recent history (last 24 hours)
        profile['exchanges'] = [
            ex for ex in profile['exchanges'] 
            if current_time - ex['timestamp'] < 86400
        ]
        
        # Detect suspicious patterns
        threat_level = ThreatLevel.LOW
        
        # 1. Too many key exchanges in short time (potential DoS)
        recent_exchanges = [
            ex for ex in profile['exchanges']
            if current_time - ex['timestamp'] < 3600  # Last hour
        ]
        if len(recent_exchanges) > 50:
            threat_level = ThreatLevel.HIGH
            self._record_suspicious_pattern(device_id, "excessive_key_exchanges", len(recent_exchanges))
        
        # 2. Unusual timing patterns (potential timing attack)
        if len(profile['exchanges']) > 5:
            intervals = []
            for i in range(1, len(profile['exchanges'])):
                interval = profile['exchanges'][i]['timestamp'] - profile['exchanges'][i-1]['timestamp']
                intervals.append(interval)
            
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                recent_interval = intervals[-1]
                
                # Check for unusual timing
                if abs(recent_interval - avg_interval) > avg_interval * 2:
                    threat_level = max(threat_level, ThreatLevel.MEDIUM)
                    self._record_suspicious_pattern(device_id, "unusual_timing", recent_interval)
        
        # 3. Algorithm downgrade attempts
        recent_algorithms = [ex['algorithm'] for ex in recent_exchanges]
        if any(alg and ('rsa' in alg.lower() or 'classic' in alg.lower()) for alg in recent_algorithms):
            threat_level = max(threat_level, ThreatLevel.MEDIUM)
            self._record_suspicious_pattern(device_id, "algorithm_downgrade", recent_algorithms)
        
        return threat_level
    
    def detect_quantum_attack_indicators(self, 
                                       session_data: Dict) -> List[str]:
        """Detect potential quantum attack indicators"""
        indicators = []
        
        # 1. Unusual key sizes or parameters
        if 'key_size' in session_data:
            if session_data['key_size'] < 256:  # Too small for quantum resistance
                indicators.append("insufficient_key_size")
        
        # 2. Suspicious entropy patterns
        if 'entropy_data' in session_data:
            entropy = session_data['entropy_data']
            if self._analyze_entropy_quality(entropy) < 0.8:
                indicators.append("poor_entropy_quality")
        
        # 3. Repeated nonce or IV values (potential replay)
        if 'nonce' in session_data:
            nonce = session_data['nonce']
            if nonce in self.quantum_attack_indicators:
                indicators.append("nonce_reuse")
            else:
                self.quantum_attack_indicators.append(nonce)
                # Keep only recent nonces (last 1000)
                if len(self.quantum_attack_indicators) > 1000:
                    self.quantum_attack_indicators.pop(0)
        
        # 4. Suspicious timing correlations (potential side-channel attack)
        if 'processing_times' in session_data:
            times = session_data['processing_times']
            if self._detect_timing_correlation(times):
                indicators.append("timing_correlation")
        
        return indicators
    
    def _record_suspicious_pattern(self, device_id: str, pattern_type: str, data: Any):
        """Record a suspicious pattern for further analysis"""
        if device_id not in self.suspicious_patterns:
            self.suspicious_patterns[device_id] = []
        
        self.suspicious_patterns[device_id].append({
            'timestamp': time.time(),
            'pattern_type': pattern_type,
            'data': data
        })
    
    def _analyze_entropy_quality(self, entropy_data: bytes) -> float:
        """Analyze the quality of entropy data"""
        if len(entropy_data) == 0:
            return 0.0
        
        # Simple entropy calculation using Shannon entropy
        byte_counts = [0] * 256
        for byte in entropy_data:
            byte_counts[byte] += 1
        
        entropy = 0.0
        length = len(entropy_data)
        for count in byte_counts:
            if count > 0:
                p = count / length
                entropy -= p * (p.bit_length() - 1)
        
        # Normalize to 0-1 range (max entropy for bytes is 8 bits)
        return entropy / 8.0
    
    def _detect_timing_correlation(self, timing_data: List[float]) -> bool:
        """Detect timing correlations that might indicate side-channel attacks"""
        if len(timing_data) < 10:
            return False
        
        # Calculate variance in timing
        mean_time = sum(timing_data) / len(timing_data)
        variance = sum((t - mean_time) ** 2 for t in timing_data) / len(timing_data)
        std_dev = variance ** 0.5
        
        # If timing is too consistent, it might be suspicious
        coefficient_of_variation = std_dev / mean_time if mean_time > 0 else 0
        return coefficient_of_variation < 0.01  # Very low variation is suspicious

class SecurityMonitor:
    """
    Main security monitoring system that coordinates all security components.
    Handles event logging, threat detection, and incident response.
    """
    
    def __init__(self, db_connection_string: str):
        self.db_connection_string = db_connection_string
        self.signature_system = QuantumSafeSignature()
        self.threat_detector = ThreatDetector()
        self.incident_response_handlers = {}
        
    async def log_security_event(self, 
                                event_type: SecurityEventType,
                                severity: EventSeverity,
                                event_data: Dict,
                                device_id: Optional[str] = None,
                                session_id: Optional[str] = None,
                                ip_address: Optional[str] = None,
                                user_agent: Optional[str] = None) -> str:
        """Log a security event with quantum-safe signature"""
        
        # Create security event
        event = SecurityEvent(
            event_id=secrets.token_hex(16),
            event_type=event_type,
            severity=severity,
            threat_level=ThreatLevel.LOW,  # Will be updated by threat analysis
            timestamp=datetime.utcnow(),
            device_id=device_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            event_data=event_data
        )
        
        # Perform threat analysis
        if event_type == SecurityEventType.KEY_EXCHANGE and device_id:
            event.threat_level = self.threat_detector.analyze_key_exchange_pattern(
                device_id, event_data
            )
        
        # Check for quantum attack indicators
        quantum_indicators = self.threat_detector.detect_quantum_attack_indicators(event_data)
        if quantum_indicators:
            event.threat_level = max(event.threat_level, ThreatLevel.HIGH)
            event.event_data['quantum_indicators'] = quantum_indicators
        
        # Sign the event
        signature = self.signature_system.sign_event(event)
        event.signature = signature
        event.verified = True
        
        # Store in database
        await self._store_event_in_database(event)
        
        # Trigger incident response if needed
        if event.threat_level.value >= ThreatLevel.HIGH.value:
            await self._trigger_incident_response(event)
        
        logger.info(f"Logged security event {event.event_id} with threat level {event.threat_level.name}")
        return event.event_id
    
    async def verify_event_integrity(self, event_id: str) -> bool:
        """Verify the integrity of a stored security event"""
        conn = await asyncpg.connect(self.db_connection_string)
        try:
            # Retrieve event from database
            record = await conn.fetchrow("""
                SELECT * FROM audit_logs WHERE id = $1
            """, event_id)
            
            if not record:
                return False
            
            # Reconstruct event object
            event = SecurityEvent(
                event_id=str(record['id']),
                event_type=SecurityEventType(record['event_type']),
                severity=EventSeverity(record['severity']),
                threat_level=ThreatLevel(record['threat_level']),
                timestamp=record['event_timestamp'],
                device_id=str(record['device_id']) if record['device_id'] else None,
                session_id=str(record['session_id']) if record['session_id'] else None,
                ip_address=str(record['ip_address']) if record['ip_address'] else None,
                user_agent=record['user_agent'],
                event_data=record['event_data']
            )
            
            # Verify signature (stored in event_data for now)
            if 'signature' in record['event_data']:
                signature = base64.b64decode(record['event_data']['signature'])
                return self.signature_system.verify_signature(event, signature)
            
            return False
            
        finally:
            await conn.close()
    
    async def get_security_dashboard_data(self, 
                                        time_range_hours: int = 24) -> Dict:
        """Get comprehensive security dashboard data"""
        conn = await asyncpg.connect(self.db_connection_string)
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
            
            # Get event counts by type
            event_counts = await conn.fetch("""
                SELECT event_type, severity, COUNT(*) as count
                FROM audit_logs 
                WHERE event_timestamp >= $1
                GROUP BY event_type, severity
            """, cutoff_time)
            
            # Get threat level distribution
            threat_levels = await conn.fetch("""
                SELECT threat_level, COUNT(*) as count
                FROM audit_logs 
                WHERE event_timestamp >= $1
                GROUP BY threat_level
            """, cutoff_time)
            
            # Get active sessions
            active_sessions = await conn.fetchval("""
                SELECT COUNT(*) FROM key_exchange_sessions 
                WHERE status = 'ACTIVE'
                AND expiry_timestamp > $1
            """, int(time.time()))
            
            # Get device statistics
            device_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_devices,
                    COUNT(CASE WHEN status = 'ACTIVE' THEN 1 END) as active_devices,
                    COUNT(CASE WHEN status = 'SUSPENDED' THEN 1 END) as suspended_devices
                FROM devices
            """)
            
            return {
                "time_range_hours": time_range_hours,
                "event_counts": [dict(row) for row in event_counts],
                "threat_levels": [dict(row) for row in threat_levels],
                "active_sessions": active_sessions,
                "device_statistics": dict(device_stats),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            await conn.close()
    
    async def _store_event_in_database(self, event: SecurityEvent):
        """Store security event in database"""
        conn = await asyncpg.connect(self.db_connection_string)
        try:
            # Add signature to event data
            event_data_with_sig = event.event_data.copy()
            if event.signature:
                event_data_with_sig['signature'] = base64.b64encode(event.signature).decode()
            
            await conn.execute("""
                INSERT INTO audit_logs 
                (id, device_id, session_id, event_type, severity, event_data,
                 ip_address, user_agent, threat_level, event_timestamp)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, 
                event.event_id,
                event.device_id,
                event.session_id,
                event.event_type.value,
                event.severity.value,
                json.dumps(event_data_with_sig),
                event.ip_address,
                event.user_agent,
                event.threat_level.value,
                event.timestamp
            )
            
        finally:
            await conn.close()
    
    async def _trigger_incident_response(self, event: SecurityEvent):
        """Trigger incident response for high-threat events"""
        incident_type = f"{event.event_type.value}_{event.threat_level.name}"
        
        if incident_type in self.incident_response_handlers:
            handler = self.incident_response_handlers[incident_type]
            await handler(event)
        else:
            # Default incident response
            await self._default_incident_response(event)
    
    async def _default_incident_response(self, event: SecurityEvent):
        """Default incident response actions"""
        logger.critical(f"Security incident detected: {event.event_id}")
        
        # If it's a device-related incident, consider suspending the device
        if event.device_id and event.threat_level == ThreatLevel.CRITICAL:
            conn = await asyncpg.connect(self.db_connection_string)
            try:
                await conn.execute("""
                    UPDATE devices SET status = 'SUSPENDED'
                    WHERE device_id = $1
                """, event.device_id)
                
                logger.warning(f"Suspended device {event.device_id} due to critical security incident")
                
            finally:
                await conn.close()

# Background monitoring service
class SecurityMonitoringService:
    """Background service for continuous security monitoring"""
    
    def __init__(self, security_monitor: SecurityMonitor):
        self.security_monitor = security_monitor
        self.running = False
    
    async def start(self):
        """Start the monitoring service"""
        self.running = True
        
        while self.running:
            try:
                # Perform periodic security checks
                await self._check_system_health()
                await self._analyze_recent_events()
                await self._cleanup_old_data()
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Security monitoring error: {e}")
                await asyncio.sleep(60)
    
    def stop(self):
        """Stop the monitoring service"""
        self.running = False
    
    async def _check_system_health(self):
        """Check overall system health"""
        dashboard_data = await self.security_monitor.get_security_dashboard_data(1)
        
        # Check for anomalies
        critical_events = sum(
            row['count'] for row in dashboard_data['threat_levels'] 
            if row['threat_level'] >= ThreatLevel.CRITICAL.value
        )
        
        if critical_events > 10:  # More than 10 critical events in the last hour
            await self.security_monitor.log_security_event(
                SecurityEventType.SYSTEM_ERROR,
                EventSeverity.CRITICAL,
                {"anomaly": "high_critical_event_rate", "count": critical_events}
            )
    
    async def _analyze_recent_events(self):
        """Analyze recent events for patterns"""
        # This would contain more sophisticated analysis
        pass
    
    async def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        conn = await asyncpg.connect(self.security_monitor.db_connection_string)
        try:
            # Remove events older than 90 days
            cutoff_date = datetime.utcnow() - timedelta(days=90)
            
            deleted_count = await conn.execute("""
                DELETE FROM audit_logs 
                WHERE event_timestamp < $1 
                AND severity NOT IN ('ERROR', 'CRITICAL')
            """, cutoff_date)
            
            if deleted_count:
                logger.info(f"Cleaned up {deleted_count} old audit log entries")
                
        finally:
            await conn.close()

# Example usage
async def example_usage():
    """Example usage of the security monitoring system"""
    
    db_url = "postgresql://qflare_user:secure_password@localhost:5432/qflare"
    monitor = SecurityMonitor(db_url)
    
    # Log a security event
    event_id = await monitor.log_security_event(
        SecurityEventType.KEY_EXCHANGE,
        EventSeverity.INFO,
        {
            "algorithm": "Kyber1024",
            "session_duration": 3600,
            "key_size": 1568
        },
        device_id="device_001",
        ip_address="192.168.1.100"
    )
    
    print(f"Logged security event: {event_id}")
    
    # Verify event integrity
    is_valid = await monitor.verify_event_integrity(event_id)
    print(f"Event integrity verified: {is_valid}")
    
    # Get dashboard data
    dashboard = await monitor.get_security_dashboard_data()
    print(f"Dashboard data: {dashboard}")

if __name__ == "__main__":
    asyncio.run(example_usage())