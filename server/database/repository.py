#!/usr/bin/env python3
"""
QFLARE Database Repository Layer
High-level database operations for quantum key exchange system
"""

import asyncio
import json
import uuid
from datetime import datetime, UTC, timedelta
from typing import List, Optional, Dict, Any, Union
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import (
    Device, KeyExchangeSession, AuditLog, SecurityEvent,
    PerformanceMetric, EnrollmentToken, SystemConfiguration
)
from .connection import database_transaction

class DeviceRepository:
    """Device management operations"""
    
    @staticmethod
    async def create_device(
        device_id: str,
        device_type: str,
        organization: str,
        location: Optional[str] = None,
        capabilities: Optional[Dict] = None,
        public_key: Optional[bytes] = None,
        enrollment_token: Optional[str] = None
    ) -> Device:
        """Create a new device registration"""
        async with database_transaction() as session:
            device = Device(
                device_id=device_id,
                device_type=device_type,
                organization=organization,
                location=location,
                capabilities=json.dumps(capabilities) if capabilities else None,
                public_key=public_key,
                enrollment_token=enrollment_token,
                status='pending',
                trust_score=0.0,
                security_level=1
            )
            session.add(device)
            await session.flush()
            return device
    
    @staticmethod
    async def get_device(device_id: str) -> Optional[Device]:
        """Get device by ID"""
        async with database_transaction() as session:
            result = await session.execute(
                select(Device).where(Device.device_id == device_id)
            )
            return result.scalar_one_or_none()
    
    @staticmethod
    async def get_devices_by_organization(organization: str) -> List[Device]:
        """Get all devices for an organization"""
        async with database_transaction() as session:
            result = await session.execute(
                select(Device).where(Device.organization == organization)
                .order_by(Device.created_at.desc())
            )
            return list(result.scalars().all())
    
    @staticmethod
    async def get_all_devices() -> List[Device]:
        """Get all devices"""
        async with database_transaction() as session:
            result = await session.execute(
                select(Device).order_by(Device.created_at.desc())
            )
            return list(result.scalars().all())
    
    @staticmethod
    async def get_devices_by_status(status: str) -> List[Device]:
        """Get devices by status"""
        async with database_transaction() as session:
            result = await session.execute(
                select(Device).where(Device.status == status)
                .order_by(Device.created_at.desc())
            )
            return list(result.scalars().all())
    
    @staticmethod
    async def approve_device(device_id: str, approver: str = None) -> bool:
        """Approve a pending device"""
        async with database_transaction() as session:
            result = await session.execute(
                update(Device)
                .where(Device.device_id == device_id)
                .values(
                    status='approved',
                    approved_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC)
                )
            )
            return result.rowcount > 0
    
    @staticmethod
    async def update_device_trust_score(device_id: str, trust_score: float) -> bool:
        """Update device trust score"""
        async with database_transaction() as session:
            result = await session.execute(
                update(Device)
                .where(Device.device_id == device_id)
                .values(
                    trust_score=trust_score,
                    updated_at=datetime.now(UTC)
                )
            )
            return result.rowcount > 0
    
    @staticmethod
    async def update_last_seen(device_id: str) -> bool:
        """Update device last seen timestamp"""
        async with database_transaction() as session:
            result = await session.execute(
                update(Device)
                .where(Device.device_id == device_id)
                .values(
                    last_seen=datetime.now(UTC),
                    updated_at=datetime.now(UTC)
                )
            )
            return result.rowcount > 0

class KeyExchangeRepository:
    """Quantum key exchange session operations"""
    
    @staticmethod
    async def create_session(
        device_id: str,
        algorithm: str = 'Kyber1024',
        public_key: Optional[bytes] = None,
        encapsulated_secret: Optional[bytes] = None,
        shared_secret_hash: Optional[str] = None,
        expires_in_seconds: int = 3600,
        security_level: int = 5,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> KeyExchangeSession:
        """Create a new key exchange session"""
        session_id = str(uuid.uuid4())
        expires_at = datetime.now(UTC) + timedelta(seconds=expires_in_seconds)
        
        async with database_transaction() as session:
            key_session = KeyExchangeSession(
                session_id=session_id,
                device_id=device_id,
                algorithm=algorithm,
                public_key=public_key,
                encapsulated_secret=encapsulated_secret,
                shared_secret_hash=shared_secret_hash,
                expires_at=expires_at,
                security_level=security_level,
                client_ip=client_ip,
                user_agent=user_agent,
                status='active',
                timestamp_derived=True,
                nonce=str(uuid.uuid4()),
                time_window_seconds=300
            )
            session.add(key_session)
            await session.flush()
            return key_session
    
    @staticmethod
    async def get_session(session_id: str) -> Optional[KeyExchangeSession]:
        """Get key exchange session by ID"""
        async with database_transaction() as session:
            result = await session.execute(
                select(KeyExchangeSession)
                .options(selectinload(KeyExchangeSession.device))
                .where(KeyExchangeSession.session_id == session_id)
            )
            return result.scalar_one_or_none()
    
    @staticmethod
    async def get_active_sessions(device_id: Optional[str] = None) -> List[KeyExchangeSession]:
        """Get active key exchange sessions"""
        async with database_transaction() as session:
            query = select(KeyExchangeSession).where(
                and_(
                    KeyExchangeSession.status == 'active',
                    KeyExchangeSession.expires_at > datetime.now(UTC)
                )
            )
            
            if device_id:
                query = query.where(KeyExchangeSession.device_id == device_id)
            
            result = await session.execute(query.order_by(KeyExchangeSession.created_at.desc()))
            return list(result.scalars().all())
    
    @staticmethod
    async def expire_session(session_id: str, used_at: Optional[datetime] = None) -> bool:
        """Expire a key exchange session"""
        async with database_transaction() as session:
            values = {
                'status': 'expired',
                'used_at': used_at or datetime.now(UTC)
            }
            
            result = await session.execute(
                update(KeyExchangeSession)
                .where(KeyExchangeSession.session_id == session_id)
                .values(**values)
            )
            return result.rowcount > 0
    
    @staticmethod
    async def cleanup_expired_sessions() -> int:
        """Clean up expired sessions"""
        async with database_transaction() as session:
            result = await session.execute(
                update(KeyExchangeSession)
                .where(
                    and_(
                        KeyExchangeSession.status == 'active',
                        KeyExchangeSession.expires_at <= datetime.now(UTC)
                    )
                )
                .values(status='expired')
            )
            return result.rowcount
    
    @staticmethod
    async def get_session_statistics(
        device_id: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get key exchange session statistics"""
        async with database_transaction() as session:
            since = datetime.now(UTC) - timedelta(hours=hours)
            
            # Build base filter conditions
            conditions = [KeyExchangeSession.created_at >= since]
            if device_id:
                conditions.append(KeyExchangeSession.device_id == device_id)
            
            # Total sessions
            total_result = await session.execute(
                select(func.count(KeyExchangeSession.session_id))
                .where(and_(*conditions))
            )
            total_sessions = total_result.scalar()
            
            # Active sessions
            active_conditions = conditions + [KeyExchangeSession.status == 'active']
            active_result = await session.execute(
                select(func.count(KeyExchangeSession.session_id))
                .where(and_(*active_conditions))
            )
            active_sessions = active_result.scalar()
            
            # Average exchange duration
            duration_conditions = conditions + [KeyExchangeSession.exchange_duration_ms.isnot(None)]
            duration_result = await session.execute(
                select(func.avg(KeyExchangeSession.exchange_duration_ms))
                .where(and_(*duration_conditions))
            )
            avg_duration = duration_result.scalar() or 0
            
            return {
                'total_sessions': total_sessions,
                'active_sessions': active_sessions,
                'expired_sessions': total_sessions - active_sessions,
                'average_duration_ms': round(avg_duration, 2),
                'period_hours': hours
            }

class AuditRepository:
    """Audit logging operations"""
    
    @staticmethod
    async def log_event(
        event_type: str,
        event_category: str,
        severity: str,
        message: str,
        device_id: Optional[str] = None,
        session_id: Optional[str] = None,
        event_data: Optional[Dict] = None,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        endpoint: Optional[str] = None,
        threat_level: int = 1
    ) -> AuditLog:
        """Create audit log entry"""
        async with database_transaction() as session:
            audit_log = AuditLog(
                device_id=device_id,
                session_id=session_id,
                event_type=event_type,
                event_category=event_category,
                severity=severity,
                message=message,
                event_data=json.dumps(event_data) if event_data else None,
                client_ip=client_ip,
                user_agent=user_agent,
                endpoint=endpoint,
                threat_level=threat_level,
                event_timestamp=datetime.now(UTC)
            )
            session.add(audit_log)
            await session.flush()
            return audit_log
    
    @staticmethod
    async def get_recent_logs(
        limit: int = 100,
        device_id: Optional[str] = None,
        severity: Optional[str] = None,
        hours: int = 24
    ) -> List[AuditLog]:
        """Get recent audit logs"""
        async with database_transaction() as session:
            since = datetime.now(UTC) - timedelta(hours=hours)
            
            query = select(AuditLog).where(AuditLog.event_timestamp >= since)
            
            if device_id:
                query = query.where(AuditLog.device_id == device_id)
            
            if severity:
                query = query.where(AuditLog.severity == severity)
            
            query = query.order_by(AuditLog.event_timestamp.desc()).limit(limit)
            
            result = await session.execute(query)
            return list(result.scalars().all())
    
    @staticmethod
    async def get_security_summary(hours: int = 24) -> Dict[str, Any]:
        """Get security event summary"""
        async with database_transaction() as session:
            since = datetime.now(UTC) - timedelta(hours=hours)
            
            # Count by severity
            severity_result = await session.execute(
                select(AuditLog.severity, func.count())
                .where(AuditLog.event_timestamp >= since)
                .group_by(AuditLog.severity)
            )
            severity_counts = dict(severity_result.all())
            
            # Count by threat level
            threat_result = await session.execute(
                select(AuditLog.threat_level, func.count())
                .where(
                    and_(
                        AuditLog.event_timestamp >= since,
                        AuditLog.threat_level > 1
                    )
                )
                .group_by(AuditLog.threat_level)
            )
            threat_counts = dict(threat_result.all())
            
            return {
                'period_hours': hours,
                'severity_counts': severity_counts,
                'threat_level_counts': threat_counts,
                'total_events': sum(severity_counts.values()),
                'high_threat_events': sum(count for level, count in threat_counts.items() if level >= 7)
            }

class ConfigurationRepository:
    """System configuration operations"""
    
    @staticmethod
    async def get_config(key: str) -> Optional[SystemConfiguration]:
        """Get configuration value"""
        async with database_transaction() as session:
            result = await session.execute(
                select(SystemConfiguration).where(SystemConfiguration.key == key)
            )
            return result.scalar_one_or_none()
    
    @staticmethod
    async def set_config(
        key: str,
        value: str,
        category: str,
        data_type: str = 'string',
        description: Optional[str] = None,
        modified_by: Optional[str] = None
    ) -> SystemConfiguration:
        """Set configuration value"""
        async with database_transaction() as session:
            # Check if config exists
            result = await session.execute(
                select(SystemConfiguration).where(SystemConfiguration.key == key)
            )
            config = result.scalar_one_or_none()
            
            if config:
                # Update existing
                config.value = value
                config.data_type = data_type
                config.updated_at = datetime.now(UTC)
                config.last_modified_by = modified_by
            else:
                # Create new
                config = SystemConfiguration(
                    key=key,
                    value=value,
                    category=category,
                    data_type=data_type,
                    description=description,
                    last_modified_by=modified_by
                )
                session.add(config)
            
            await session.flush()
            return config
    
    @staticmethod
    async def get_configs_by_category(category: str) -> List[SystemConfiguration]:
        """Get all configurations in a category"""
        async with database_transaction() as session:
            result = await session.execute(
                select(SystemConfiguration)
                .where(SystemConfiguration.category == category)
                .order_by(SystemConfiguration.key)
            )
            return list(result.scalars().all())

# Convenience functions for common database operations
async def quick_device_lookup(device_id: str) -> Optional[Dict[str, Any]]:
    """Quick device information lookup"""
    device = await DeviceRepository.get_device(device_id)
    if not device:
        return None
    
    return {
        'device_id': device.device_id,
        'device_type': device.device_type,
        'organization': device.organization,
        'status': device.status,
        'trust_score': device.trust_score,
        'last_seen': device.last_seen.isoformat() if device.last_seen else None,
        'created_at': device.created_at.isoformat()
    }

async def quick_session_lookup(session_id: str) -> Optional[Dict[str, Any]]:
    """Quick session information lookup"""
    session = await KeyExchangeRepository.get_session(session_id)
    if not session:
        return None
    
    return {
        'session_id': session.session_id,
        'device_id': session.device_id,
        'algorithm': session.algorithm,
        'status': session.status,
        'security_level': session.security_level,
        'created_at': session.created_at.isoformat(),
        'expires_at': session.expires_at.isoformat(),
        'is_expired': session.expires_at.replace(tzinfo=UTC) <= datetime.now(UTC)
    }

async def security_audit_log(
    event_type: str,
    message: str,
    device_id: Optional[str] = None,
    session_id: Optional[str] = None,
    threat_level: int = 1,
    **kwargs
) -> None:
    """Quick security audit logging"""
    await AuditRepository.log_event(
        event_type=event_type,
        event_category='SECURITY',
        severity='INFO' if threat_level <= 3 else 'WARN' if threat_level <= 6 else 'ERROR',
        message=message,
        device_id=device_id,
        session_id=session_id,
        threat_level=threat_level,
        **kwargs
    )