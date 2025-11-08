"""
SQLAlchemy database models for QFLARE system.

This module defines all database models including users, projects,
training sessions, metrics, and system configurations.
"""

import enum
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Column, Integer, String, DateTime, Float, Boolean, Text, JSON,
    ForeignKey, Enum as SQLEnum, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

Base = declarative_base()


class UserRole(enum.Enum):
    """User roles in the QFLARE system."""
    ADMIN = "admin"
    USER = "user"
    OPERATOR = "operator"
    OBSERVER = "observer"


class TrainingStatus(enum.Enum):
    """Training session status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class NodeStatus(enum.Enum):
    """Federated learning node status."""
    ONLINE = "online"
    OFFLINE = "offline"
    TRAINING = "training"
    IDLE = "idle"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class SecurityLevel(enum.Enum):
    """Security configuration levels."""
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"


class User(Base):
    """User accounts and authentication."""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    salt = Column(String(32), nullable=False)
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.USER)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Multi-factor authentication
    mfa_enabled = Column(Boolean, default=False, nullable=False)
    mfa_secret = Column(String(64), nullable=True)
    
    # Profile information
    first_name = Column(String(50), nullable=True)
    last_name = Column(String(50), nullable=True)
    organization = Column(String(100), nullable=True)
    
    # Security settings
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    password_changed_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    projects = relationship("Project", back_populates="owner")
    training_sessions = relationship("TrainingSession", back_populates="user")
    user_sessions = relationship("UserSession", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")
    
    __table_args__ = (
        Index('idx_users_email_active', 'email', 'is_active'),
        Index('idx_users_username_active', 'username', 'is_active'),
        CheckConstraint('failed_login_attempts >= 0', name='check_failed_login_attempts'),
    )


class Project(Base):
    """Federated learning projects."""
    __tablename__ = "projects"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Configuration
    model_config = Column(JSONB, nullable=False)
    privacy_config = Column(JSONB, nullable=False)
    security_level = Column(SQLEnum(SecurityLevel), nullable=False, default=SecurityLevel.STANDARD)
    
    # Settings
    max_clients = Column(Integer, nullable=False, default=100)
    min_clients = Column(Integer, nullable=False, default=2)
    rounds_per_session = Column(Integer, nullable=False, default=10)
    client_selection_strategy = Column(String(50), nullable=False, default="random")
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_public = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    owner = relationship("User", back_populates="projects")
    training_sessions = relationship("TrainingSession", back_populates="project")
    nodes = relationship("FederatedNode", back_populates="project")
    
    __table_args__ = (
        Index('idx_projects_owner_active', 'owner_id', 'is_active'),
        CheckConstraint('max_clients > min_clients', name='check_client_limits'),
        CheckConstraint('rounds_per_session > 0', name='check_positive_rounds'),
    )


class TrainingSession(Base):
    """Federated learning training sessions."""
    __tablename__ = "training_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Session details
    name = Column(String(100), nullable=False)
    status = Column(SQLEnum(TrainingStatus), nullable=False, default=TrainingStatus.PENDING)
    current_round = Column(Integer, default=0, nullable=False)
    total_rounds = Column(Integer, nullable=False)
    
    # Performance metrics
    initial_accuracy = Column(Float, nullable=True)
    current_accuracy = Column(Float, nullable=True)
    best_accuracy = Column(Float, nullable=True)
    current_loss = Column(Float, nullable=True)
    convergence_threshold = Column(Float, default=0.01, nullable=False)
    
    # Privacy metrics
    epsilon_spent = Column(Float, default=0.0, nullable=False)
    delta_spent = Column(Float, default=0.0, nullable=False)
    noise_scale = Column(Float, nullable=True)
    
    # Security metrics
    byzantine_clients_detected = Column(Integer, default=0, nullable=False)
    anomalous_updates_rejected = Column(Integer, default=0, nullable=False)
    
    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    estimated_completion = Column(DateTime(timezone=True), nullable=True)
    
    # Configuration snapshot
    config_snapshot = Column(JSONB, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="training_sessions")
    user = relationship("User", back_populates="training_sessions")
    metrics = relationship("TrainingMetric", back_populates="session")
    round_results = relationship("RoundResult", back_populates="session")
    
    __table_args__ = (
        Index('idx_training_sessions_project_status', 'project_id', 'status'),
        Index('idx_training_sessions_user_created', 'user_id', 'created_at'),
        CheckConstraint('total_rounds > 0', name='check_positive_total_rounds'),
        CheckConstraint('current_round >= 0', name='check_non_negative_current_round'),
        CheckConstraint('epsilon_spent >= 0', name='check_non_negative_epsilon'),
    )


class FederatedNode(Base):
    """Federated learning client nodes."""
    __tablename__ = "federated_nodes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    
    # Node identification
    node_id = Column(String(50), nullable=False, index=True)
    node_name = Column(String(100), nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 support
    port = Column(Integer, nullable=True)
    
    # Status and capabilities
    status = Column(SQLEnum(NodeStatus), nullable=False, default=NodeStatus.OFFLINE)
    last_seen = Column(DateTime(timezone=True), nullable=True)
    version = Column(String(20), nullable=True)
    capabilities = Column(JSONB, nullable=True)
    
    # Performance metrics
    compute_power = Column(Float, nullable=True)  # FLOPS or relative score
    bandwidth = Column(Float, nullable=True)  # Mbps
    reliability_score = Column(Float, default=1.0, nullable=False)
    
    # Dataset information
    dataset_size = Column(Integer, nullable=True)
    data_quality_score = Column(Float, nullable=True)
    data_distribution = Column(JSONB, nullable=True)
    
    # Security
    public_key = Column(Text, nullable=True)
    certificate = Column(Text, nullable=True)
    is_trusted = Column(Boolean, default=False, nullable=False)
    risk_score = Column(Float, default=0.0, nullable=False)
    
    # Timestamps
    registered_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="nodes")
    participations = relationship("ClientParticipation", back_populates="node")
    
    __table_args__ = (
        UniqueConstraint('project_id', 'node_id', name='unique_node_per_project'),
        Index('idx_nodes_project_status', 'project_id', 'status'),
        Index('idx_nodes_last_seen', 'last_seen'),
        CheckConstraint('reliability_score >= 0 AND reliability_score <= 1', name='check_reliability_score'),
        CheckConstraint('risk_score >= 0', name='check_non_negative_risk'),
    )


class TrainingMetric(Base):
    """Real-time training metrics and system performance."""
    __tablename__ = "training_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("training_sessions.id"), nullable=False)
    
    # Metric identification
    metric_type = Column(String(50), nullable=False)  # accuracy, loss, cpu, memory, etc.
    round_number = Column(Integer, nullable=False)
    client_id = Column(String(50), nullable=True)  # null for global metrics
    
    # Metric values
    value = Column(Float, nullable=False)
    metadata = Column(JSONB, nullable=True)  # Additional context
    
    # Timestamp
    recorded_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    
    # Relationships
    session = relationship("TrainingSession", back_populates="metrics")
    
    __table_args__ = (
        Index('idx_metrics_session_type_round', 'session_id', 'metric_type', 'round_number'),
        Index('idx_metrics_recorded_at', 'recorded_at'),
    )


class RoundResult(Base):
    """Results and aggregated data from training rounds."""
    __tablename__ = "round_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("training_sessions.id"), nullable=False)
    
    # Round information
    round_number = Column(Integer, nullable=False)
    participants_count = Column(Integer, nullable=False)
    selected_clients = Column(JSONB, nullable=False)  # List of client IDs
    
    # Performance metrics
    global_accuracy = Column(Float, nullable=True)
    global_loss = Column(Float, nullable=True)
    convergence_metric = Column(Float, nullable=True)
    
    # Aggregation details
    aggregation_method = Column(String(50), nullable=False)
    aggregation_time = Column(Float, nullable=False)  # seconds
    
    # Security results
    byzantine_detected = Column(Integer, default=0, nullable=False)
    outliers_rejected = Column(Integer, default=0, nullable=False)
    security_violations = Column(JSONB, nullable=True)
    
    # Privacy metrics for this round
    privacy_budget_spent = Column(Float, nullable=True)
    noise_added = Column(Float, nullable=True)
    
    # Model state (hash or version)
    model_hash = Column(String(64), nullable=True)
    model_size = Column(Integer, nullable=True)  # bytes
    
    # Timing
    round_start = Column(DateTime(timezone=True), nullable=False)
    round_end = Column(DateTime(timezone=True), nullable=False)
    
    # Additional data
    round_metadata = Column(JSONB, nullable=True)
    
    # Relationships
    session = relationship("TrainingSession", back_populates="round_results")
    client_participations = relationship("ClientParticipation", back_populates="round_result")
    
    __table_args__ = (
        UniqueConstraint('session_id', 'round_number', name='unique_round_per_session'),
        Index('idx_round_results_session_round', 'session_id', 'round_number'),
        CheckConstraint('participants_count > 0', name='check_positive_participants'),
        CheckConstraint('round_end >= round_start', name='check_round_timing'),
    )


class ClientParticipation(Base):
    """Individual client participation in training rounds."""
    __tablename__ = "client_participations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    round_result_id = Column(UUID(as_uuid=True), ForeignKey("round_results.id"), nullable=False)
    node_id = Column(UUID(as_uuid=True), ForeignKey("federated_nodes.id"), nullable=False)
    
    # Participation details
    selected = Column(Boolean, nullable=False)
    participated = Column(Boolean, nullable=False)
    update_quality = Column(Float, nullable=True)  # Quality score 0-1
    
    # Performance metrics
    local_accuracy = Column(Float, nullable=True)
    local_loss = Column(Float, nullable=True)
    training_time = Column(Float, nullable=True)  # seconds
    communication_time = Column(Float, nullable=True)  # seconds
    
    # Data contribution
    samples_used = Column(Integer, nullable=True)
    gradient_norm = Column(Float, nullable=True)
    
    # Security assessment
    is_byzantine = Column(Boolean, default=False, nullable=False)
    anomaly_score = Column(Float, nullable=True)
    rejection_reason = Column(String(100), nullable=True)
    
    # Timestamps
    selected_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    round_result = relationship("RoundResult", back_populates="client_participations")
    node = relationship("FederatedNode", back_populates="participations")
    
    __table_args__ = (
        Index('idx_participations_round_node', 'round_result_id', 'node_id'),
        Index('idx_participations_node_completed', 'node_id', 'completed_at'),
        CheckConstraint('update_quality IS NULL OR (update_quality >= 0 AND update_quality <= 1)', 
                       name='check_update_quality'),
    )


class UserSession(Base):
    """User login sessions for security tracking."""
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Session details
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    refresh_token = Column(String(255), unique=True, nullable=True, index=True)
    ip_address = Column(String(45), nullable=False)
    user_agent = Column(Text, nullable=True)
    
    # Security flags
    is_active = Column(Boolean, default=True, nullable=False)
    is_suspicious = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    last_activity = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="user_sessions")
    
    __table_args__ = (
        Index('idx_sessions_user_active', 'user_id', 'is_active'),
        Index('idx_sessions_expires_at', 'expires_at'),
    )


class SystemConfig(Base):
    """System-wide configuration settings."""
    __tablename__ = "system_configs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Configuration
    config_key = Column(String(100), unique=True, nullable=False, index=True)
    config_value = Column(JSONB, nullable=False)
    description = Column(Text, nullable=True)
    
    # Metadata
    is_sensitive = Column(Boolean, default=False, nullable=False)
    requires_restart = Column(Boolean, default=False, nullable=False)
    
    # Versioning
    version = Column(Integer, default=1, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class AuditLog(Base):
    """Security and audit logging."""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    
    # Event details
    event_type = Column(String(50), nullable=False, index=True)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(String(100), nullable=True)
    action = Column(String(50), nullable=False)
    
    # Context
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    correlation_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Results
    success = Column(Boolean, nullable=False)
    error_message = Column(Text, nullable=True)
    details = Column(JSONB, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    __table_args__ = (
        Index('idx_audit_logs_event_type_created', 'event_type', 'created_at'),
        Index('idx_audit_logs_user_created', 'user_id', 'created_at'),
        Index('idx_audit_logs_correlation_id', 'correlation_id'),
    )


# Additional utility models for caching and performance

class CacheEntry(Base):
    """Redis cache backup and metadata."""
    __tablename__ = "cache_entries"
    
    cache_key = Column(String(255), primary_key=True)
    cache_value = Column(JSONB, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index('idx_cache_expires_at', 'expires_at'),
    )


class PerformanceMetric(Base):
    """System performance monitoring."""
    __tablename__ = "performance_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Metric identification
    metric_name = Column(String(100), nullable=False, index=True)
    component = Column(String(50), nullable=False)  # frontend, backend, crypto, etc.
    
    # Values
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=False)  # ms, MB, %, ops/sec, etc.
    
    # Context
    labels = Column(JSONB, nullable=True)  # Additional labels for filtering
    
    # Timestamp
    recorded_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index('idx_perf_metrics_name_component_time', 'metric_name', 'component', 'recorded_at'),
    )