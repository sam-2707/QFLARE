#!/usr/bin/env python3
"""
QFLARE Production Database Models
SQLAlchemy models for quantum key exchange system
"""

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, 
    Text, LargeBinary, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from datetime import datetime, UTC
import uuid

Base = declarative_base()

class Device(Base):
    """Device registration and management"""
    __tablename__ = 'devices'
    
    # Primary fields
    device_id = Column(String(255), primary_key=True)
    device_type = Column(String(50), nullable=False)  # EDGE_NODE, MOBILE_DEVICE, etc.
    organization = Column(String(255), nullable=False)
    location = Column(String(255))
    
    # Status and metadata
    status = Column(String(20), nullable=False, default='pending')  # pending, approved, suspended, revoked
    trust_score = Column(Float, default=0.0)
    public_key = Column(LargeBinary)
    capabilities = Column(Text)  # JSON string of device capabilities
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    last_seen = Column(DateTime(timezone=True))
    approved_at = Column(DateTime(timezone=True))
    
    # Security fields
    enrollment_token = Column(String(255))
    certificate_fingerprint = Column(String(255))
    security_level = Column(Integer, default=1)
    
    # Relationships
    key_exchanges = relationship("KeyExchangeSession", back_populates="device")
    audit_logs = relationship("AuditLog", back_populates="device")
    
    # Indexes
    __table_args__ = (
        Index('idx_device_status', 'status'),
        Index('idx_device_organization', 'organization'),
        Index('idx_device_created', 'created_at'),
        Index('idx_device_last_seen', 'last_seen'),
    )

class KeyExchangeSession(Base):
    """Quantum key exchange sessions"""
    __tablename__ = 'key_exchange_sessions'
    
    # Primary fields
    session_id = Column(String(255), primary_key=True)
    device_id = Column(String(255), ForeignKey('devices.device_id'), nullable=False)
    
    # Cryptographic details
    algorithm = Column(String(50), nullable=False, default='Kyber1024')
    public_key = Column(LargeBinary)
    encapsulated_secret = Column(LargeBinary)
    shared_secret_hash = Column(String(255))  # Hash of shared secret for verification
    
    # Session metadata
    status = Column(String(20), nullable=False, default='active')  # active, expired, revoked
    created_at = Column(DateTime(timezone=True), default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used_at = Column(DateTime(timezone=True))
    
    # Security and performance
    security_level = Column(Integer, default=5)  # NIST security level
    exchange_duration_ms = Column(Float)
    client_ip = Column(String(45))  # IPv4/IPv6 address
    user_agent = Column(String(500))
    
    # Temporal security
    timestamp_derived = Column(Boolean, default=True)
    nonce = Column(String(255))
    time_window_seconds = Column(Integer, default=300)
    
    # Relationships
    device = relationship("Device", back_populates="key_exchanges")
    
    # Indexes
    __table_args__ = (
        Index('idx_session_device', 'device_id'),
        Index('idx_session_status', 'status'),
        Index('idx_session_created', 'created_at'),
        Index('idx_session_expires', 'expires_at'),
        Index('idx_session_algorithm', 'algorithm'),
    )

class AuditLog(Base):
    """Comprehensive audit logging"""
    __tablename__ = 'audit_logs'
    
    # Primary fields
    id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    device_id = Column(String(255), ForeignKey('devices.device_id'))
    session_id = Column(String(255), ForeignKey('key_exchange_sessions.session_id'))
    
    # Event details
    event_type = Column(String(50), nullable=False)
    event_category = Column(String(50), nullable=False)  # SECURITY, PERFORMANCE, SYSTEM
    severity = Column(String(20), nullable=False)  # DEBUG, INFO, WARN, ERROR, CRITICAL
    message = Column(Text, nullable=False)
    
    # Context and metadata
    event_data = Column(Text)  # JSON string with additional data
    user_agent = Column(String(500))
    client_ip = Column(String(45))
    endpoint = Column(String(255))
    
    # Timestamps and identification
    event_timestamp = Column(DateTime(timezone=True), default=func.now())
    created_at = Column(DateTime(timezone=True), default=func.now())
    
    # Security fields
    threat_level = Column(Integer, default=1)  # 1-10 threat severity
    signature = Column(LargeBinary)  # Digital signature of log entry
    signature_verified = Column(Boolean, default=False)
    
    # Relationships
    device = relationship("Device", back_populates="audit_logs")
    
    # Indexes
    __table_args__ = (
        Index('idx_audit_device', 'device_id'),
        Index('idx_audit_event_type', 'event_type'),
        Index('idx_audit_severity', 'severity'),
        Index('idx_audit_timestamp', 'event_timestamp'),
        Index('idx_audit_threat_level', 'threat_level'),
        Index('idx_audit_category', 'event_category'),
    )

class SecurityEvent(Base):
    """Security incidents and threat detection"""
    __tablename__ = 'security_events'
    
    # Primary fields
    id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    device_id = Column(String(255), ForeignKey('devices.device_id'))
    session_id = Column(String(255), ForeignKey('key_exchange_sessions.session_id'))
    
    # Threat details
    threat_type = Column(String(50), nullable=False)  # QUANTUM_ATTACK, ANOMALOUS_BEHAVIOR, etc.
    threat_category = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    confidence = Column(Float, default=0.0)  # 0-1 confidence score
    
    # Detection details
    detection_method = Column(String(100))
    indicators = Column(Text)  # JSON string of threat indicators
    mitigation_applied = Column(Text)  # Actions taken
    
    # Status and resolution
    status = Column(String(20), default='detected')  # detected, investigating, resolved, false_positive
    resolved_at = Column(DateTime(timezone=True))
    resolution_notes = Column(Text)
    
    # Timestamps
    detected_at = Column(DateTime(timezone=True), default=func.now())
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    
    # Context
    context_data = Column(Text)  # JSON string with additional context
    
    # Indexes
    __table_args__ = (
        Index('idx_security_device', 'device_id'),
        Index('idx_security_threat_type', 'threat_type'),
        Index('idx_security_severity', 'severity'),
        Index('idx_security_detected', 'detected_at'),
        Index('idx_security_status', 'status'),
    )

class PerformanceMetric(Base):
    """System and cryptographic performance metrics"""
    __tablename__ = 'performance_metrics'
    
    # Primary fields
    id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    metric_type = Column(String(50), nullable=False)  # KEY_EXCHANGE, API_RESPONSE, SYSTEM_RESOURCE
    metric_name = Column(String(100), nullable=False)
    
    # Metric values
    value = Column(Float, nullable=False)
    unit = Column(String(20))  # ms, bytes, percent, count
    
    # Context
    device_id = Column(String(255), ForeignKey('devices.device_id'))
    session_id = Column(String(255), ForeignKey('key_exchange_sessions.session_id'))
    component = Column(String(100))  # Which system component
    
    # Additional data
    metric_metadata = Column(Text)  # JSON string with additional metric data
    
    # Timestamps
    measured_at = Column(DateTime(timezone=True), default=func.now())
    created_at = Column(DateTime(timezone=True), default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_metric_type', 'metric_type'),
        Index('idx_metric_name', 'metric_name'),
        Index('idx_metric_measured', 'measured_at'),
        Index('idx_metric_device', 'device_id'),
        Index('idx_metric_component', 'component'),
    )

class EnrollmentToken(Base):
    """Device enrollment tokens"""
    __tablename__ = 'enrollment_tokens'
    
    # Primary fields
    id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    token = Column(String(255), nullable=False, unique=True)
    
    # Token details
    organization = Column(String(255), nullable=False)
    device_type = Column(String(50), nullable=False)
    max_uses = Column(Integer, default=1)
    used_count = Column(Integer, default=0)
    
    # Status and expiry
    status = Column(String(20), default='active')  # active, expired, revoked, exhausted
    created_at = Column(DateTime(timezone=True), default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    
    # Usage tracking
    first_used_at = Column(DateTime(timezone=True))
    last_used_at = Column(DateTime(timezone=True))
    used_by_devices = Column(Text)  # JSON array of device IDs that used this token
    
    # Creator info
    created_by = Column(String(255))  # Admin user who created the token
    notes = Column(Text)
    
    # Indexes
    __table_args__ = (
        Index('idx_token_status', 'status'),
        Index('idx_token_organization', 'organization'),
        Index('idx_token_expires', 'expires_at'),
        Index('idx_token_created', 'created_at'),
        UniqueConstraint('token', name='uq_enrollment_token'),
    )

class SystemConfiguration(Base):
    """System-wide configuration and settings"""
    __tablename__ = 'system_configuration'
    
    # Primary fields
    key = Column(String(100), primary_key=True)
    value = Column(Text, nullable=False)
    data_type = Column(String(20), default='string')  # string, integer, float, boolean, json
    
    # Metadata
    category = Column(String(50), nullable=False)  # SECURITY, PERFORMANCE, QUANTUM, SYSTEM
    description = Column(Text)
    default_value = Column(Text)
    
    # Validation and constraints
    min_value = Column(Float)
    max_value = Column(Float)
    allowed_values = Column(Text)  # JSON array of allowed values
    validation_regex = Column(String(255))
    
    # Change tracking
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    last_modified_by = Column(String(255))
    
    # Security
    requires_restart = Column(Boolean, default=False)
    is_sensitive = Column(Boolean, default=False)  # Don't log changes to sensitive configs
    
    # Indexes
    __table_args__ = (
        Index('idx_config_category', 'category'),
        Index('idx_config_updated', 'updated_at'),
    )

# Database metadata for migrations
database_metadata = Base.metadata


class GlobalModel(Base):
    """Global federated learning model storage"""
    __tablename__ = 'global_models'
    
    # Primary fields
    id = Column(Integer, primary_key=True, autoincrement=True)
    round_number = Column(Integer, nullable=False)
    model_data = Column(LargeBinary, nullable=False)  # Serialized model weights
    model_hash = Column(String(64), nullable=False)  # SHA256 hash for integrity
    
    # Metadata
    algorithm = Column(String(50), default='FedAvg')
    num_participants = Column(Integer, nullable=False)
    accuracy = Column(Float)
    loss = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now())
    aggregated_at = Column(DateTime(timezone=True))
    
    # Performance metrics
    training_time = Column(Float)  # Training time in seconds
    model_size = Column(Integer)   # Model size in bytes
    
    # Indexes
    __table_args__ = (
        Index('idx_global_model_round', 'round_number'),
        Index('idx_global_model_created', 'created_at'),
        UniqueConstraint('round_number', name='uq_global_model_round'),
    )


class ModelUpdate(Base):
    """Individual device model updates for federated learning"""
    __tablename__ = 'model_updates'
    
    # Primary fields
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(255), ForeignKey('devices.device_id'), nullable=False)
    round_number = Column(Integer, nullable=False)
    
    # Model data
    model_delta = Column(LargeBinary, nullable=False)  # Model weight updates
    delta_hash = Column(String(64), nullable=False)   # SHA256 hash
    
    # Training metadata
    local_epochs = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    num_samples = Column(Integer, nullable=False)
    
    # Performance metrics
    local_accuracy = Column(Float)
    local_loss = Column(Float)
    training_time = Column(Float)
    
    # Status tracking
    status = Column(String(20), default='pending')  # pending, validated, rejected, aggregated
    validation_score = Column(Float)  # Cosine similarity or other validation metric
    
    # Timestamps
    submitted_at = Column(DateTime(timezone=True), default=func.now())
    processed_at = Column(DateTime(timezone=True))
    
    # Relationships
    device = relationship("Device", back_populates="model_updates")
    
    # Indexes
    __table_args__ = (
        Index('idx_model_update_device', 'device_id'),
        Index('idx_model_update_round', 'round_number'),
        Index('idx_model_update_status', 'status'),
        Index('idx_model_update_submitted', 'submitted_at'),
    )


class TrainingSession(Base):
    """Federated learning training session management"""
    __tablename__ = 'training_sessions'
    
    # Primary fields
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False)
    round_number = Column(Integer, nullable=False)
    
    # Session configuration
    target_participants = Column(Integer, nullable=False)
    min_participants = Column(Integer, nullable=False)
    max_participants = Column(Integer, nullable=False)
    
    # Training parameters
    global_epochs = Column(Integer, default=1)
    local_epochs = Column(Integer, default=1)
    learning_rate = Column(Float, default=0.01)
    batch_size = Column(Integer, default=32)
    
    # Status and progress
    status = Column(String(20), default='initializing')  # initializing, recruiting, training, aggregating, completed, failed
    enrolled_devices = Column(Integer, default=0)
    completed_updates = Column(Integer, default=0)
    
    # Timing
    started_at = Column(DateTime(timezone=True), default=func.now())
    deadline = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    # Results
    final_accuracy = Column(Float)
    final_loss = Column(Float)
    convergence_achieved = Column(Boolean, default=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_training_session_round', 'round_number'),
        Index('idx_training_session_status', 'status'),
        Index('idx_training_session_started', 'started_at'),
        UniqueConstraint('session_id', name='uq_training_session_id'),
    )


class UserToken(Base):
    """User authentication tokens"""
    __tablename__ = 'user_tokens'
    
    # Primary fields
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False)
    token_hash = Column(String(64), nullable=False)  # SHA256 hash of token
    
    # Token metadata
    token_type = Column(String(20), default='access')  # access, refresh, session
    scope = Column(String(255))  # Permissions scope
    
    # Expiration
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now())
    last_used = Column(DateTime(timezone=True))
    
    # Security
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    is_revoked = Column(Boolean, default=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_user_token_user', 'user_id'),
        Index('idx_user_token_expires', 'expires_at'),
        Index('idx_user_token_type', 'token_type'),
        UniqueConstraint('token_hash', name='uq_user_token_hash'),
    )


# Update the Device model to include the relationship
Device.model_updates = relationship("ModelUpdate", back_populates="device")