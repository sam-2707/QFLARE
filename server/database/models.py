"""
Database Models for QFLARE Federated Learning System

This module defines SQLAlchemy models for persistent storage of:
- Device registry and metadata
- Model updates and training history
- Aggregation rounds and global models
- User authentication and tokens
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, 
    LargeBinary, DateTime, JSON, ForeignKey, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Device(Base):
    """Device registry table"""
    __tablename__ = 'devices'
    
    device_id = Column(String(255), primary_key=True, index=True)
    status = Column(String(50), default='active', index=True)
    registered_at = Column(DateTime, default=func.now())
    last_seen = Column(DateTime, default=func.now())
    
    # Post-quantum cryptographic keys
    kem_public_key = Column(LargeBinary)
    sig_public_key = Column(LargeBinary)
    key_rotation_count = Column(Integer, default=0)
    
    # Device metadata
    device_type = Column(String(100))
    hardware_info = Column(JSON)
    network_info = Column(JSON)
    capabilities = Column(JSON)
    
    # Training configuration
    local_epochs = Column(Integer, default=1)
    batch_size = Column(Integer, default=32)
    learning_rate = Column(Float, default=0.01)
    
    # Relationships
    model_updates = relationship("ModelUpdate", back_populates="device")
    training_sessions = relationship("TrainingSession", back_populates="device")
    
    def __repr__(self):
        return f"<Device(device_id='{self.device_id}', status='{self.status}')>"


class GlobalModel(Base):
    """Global model versions and metadata"""
    __tablename__ = 'global_models'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    round_number = Column(Integer, index=True)
    model_weights = Column(LargeBinary)
    model_hash = Column(String(64), index=True)
    
    # Model metadata
    model_type = Column(String(100))
    model_architecture = Column(JSON)
    accuracy = Column(Float)
    loss = Column(Float)
    
    # Aggregation info
    num_participants = Column(Integer)
    aggregation_method = Column(String(50), default='fedavg')
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    model_updates = relationship("ModelUpdate", back_populates="global_model")
    
    def __repr__(self):
        return f"<GlobalModel(round={self.round_number}, participants={self.num_participants})>"


class ModelUpdate(Base):
    """Individual model updates from devices"""
    __tablename__ = 'model_updates'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(255), ForeignKey('devices.device_id'), index=True)
    global_model_id = Column(Integer, ForeignKey('global_models.id'), index=True)
    
    # Update data
    model_weights = Column(LargeBinary)
    model_hash = Column(String(64))
    signature = Column(LargeBinary)  # Post-quantum signature
    
    # Training metrics
    local_loss = Column(Float)
    local_accuracy = Column(Float)
    local_epochs = Column(Integer)
    samples_count = Column(Integer)
    training_time = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    aggregated_at = Column(DateTime)
    
    # Status tracking
    status = Column(String(50), default='pending')  # pending, validated, aggregated, rejected
    validation_score = Column(Float)
    
    # Relationships
    device = relationship("Device", back_populates="model_updates")
    global_model = relationship("GlobalModel", back_populates="model_updates")
    
    def __repr__(self):
        return f"<ModelUpdate(device='{self.device_id}', status='{self.status}')>"


class TrainingSession(Base):
    """Training sessions and rounds"""
    __tablename__ = 'training_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), unique=True, index=True)
    device_id = Column(String(255), ForeignKey('devices.device_id'), index=True)
    
    # Session configuration
    dataset_name = Column(String(100))
    model_type = Column(String(100))
    hyperparameters = Column(JSON)
    
    # Session status
    status = Column(String(50), default='active')  # active, completed, failed, suspended
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
    
    # Training progress
    current_round = Column(Integer, default=0)
    total_rounds = Column(Integer)
    last_update_at = Column(DateTime)
    
    # Performance metrics
    best_accuracy = Column(Float)
    current_loss = Column(Float)
    convergence_score = Column(Float)
    
    # Relationships
    device = relationship("Device", back_populates="training_sessions")
    
    def __repr__(self):
        return f"<TrainingSession(session='{self.session_id}', status='{self.status}')>"


class AuditLog(Base):
    """Security and operation audit logs"""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(255), index=True)
    
    # Event details
    event_type = Column(String(100), index=True)  # registration, update, aggregation, error
    event_description = Column(Text)
    event_data = Column(JSON)
    
    # Security context
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    auth_method = Column(String(50))
    
    # Timestamps
    timestamp = Column(DateTime, default=func.now(), index=True)
    
    # Risk assessment
    risk_level = Column(String(20), default='low')  # low, medium, high, critical
    
    def __repr__(self):
        return f"<AuditLog(event='{self.event_type}', risk='{self.risk_level}')>"


class UserToken(Base):
    """User authentication tokens and API keys"""
    __tablename__ = 'user_tokens'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    token_hash = Column(String(64), unique=True, index=True)
    device_id = Column(String(255), ForeignKey('devices.device_id'), index=True)
    
    # Token metadata
    token_type = Column(String(50))  # enrollment, api, refresh
    permissions = Column(JSON)
    
    # Lifecycle
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime)
    last_used_at = Column(DateTime)
    
    # Status
    is_active = Column(Boolean, default=True)
    revoked_at = Column(DateTime)
    revocation_reason = Column(String(255))
    
    def __repr__(self):
        return f"<UserToken(type='{self.token_type}', active={self.is_active})>"


# Indexes for performance optimization
Index('idx_device_status_lastseen', Device.status, Device.last_seen)
Index('idx_model_update_device_created', ModelUpdate.device_id, ModelUpdate.created_at)
Index('idx_global_model_round', GlobalModel.round_number)
Index('idx_audit_log_timestamp_type', AuditLog.timestamp, AuditLog.event_type)
Index('idx_training_session_device_status', TrainingSession.device_id, TrainingSession.status)