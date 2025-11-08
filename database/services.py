"""
Database service layer for QFLARE application.

This module provides high-level database operations and business logic
for user management, training sessions, and system monitoring.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy import func, and_, or_, desc
import hashlib
import secrets
import uuid

from .models import (
    User, UserRole, Project, TrainingSession, TrainingStatus,
    FederatedNode, NodeStatus, TrainingMetric, RoundResult,
    ClientParticipation, UserSession, AuditLog, SystemConfig,
    PerformanceMetric, SecurityLevel
)
from .connection import get_db_session, get_cache

logger = logging.getLogger(__name__)


class UserService:
    """User management service."""
    
    @staticmethod
    def create_user(
        session: Session,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.USER,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        organization: Optional[str] = None
    ) -> User:
        """Create a new user with hashed password."""
        try:
            # Generate salt and hash password
            salt = secrets.token_hex(16)
            password_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000  # iterations
            ).hex()
            
            user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                salt=salt,
                role=role,
                first_name=first_name,
                last_name=last_name,
                organization=organization
            )
            
            session.add(user)
            session.flush()  # Get user ID
            
            logger.info(f"User created: {username} ({user.id})")
            return user
            
        except IntegrityError as e:
            session.rollback()
            logger.error(f"Failed to create user {username}: {e}")
            raise ValueError("Username or email already exists")
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error creating user {username}: {e}")
            raise
    
    @staticmethod
    def authenticate_user(session: Session, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/email and password."""
        try:
            # Find user by username or email
            user = session.query(User).filter(
                and_(
                    or_(User.username == username, User.email == username),
                    User.is_active == True
                )
            ).first()
            
            if not user:
                logger.warning(f"Authentication failed: user not found: {username}")
                return None
            
            # Check if account is locked
            if user.locked_until and user.locked_until > datetime.utcnow():
                logger.warning(f"Authentication failed: account locked: {username}")
                return None
            
            # Verify password
            password_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                user.salt.encode('utf-8'),
                100000
            ).hex()
            
            if password_hash != user.password_hash:
                # Increment failed attempts
                user.failed_login_attempts += 1
                if user.failed_login_attempts >= 5:
                    user.locked_until = datetime.utcnow() + timedelta(minutes=30)
                    logger.warning(f"Account locked due to failed attempts: {username}")
                
                session.commit()
                logger.warning(f"Authentication failed: invalid password: {username}")
                return None
            
            # Reset failed attempts on successful login
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login_at = datetime.utcnow()
            session.commit()
            
            logger.info(f"User authenticated successfully: {username}")
            return user
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error during authentication: {e}")
            return None
    
    @staticmethod
    def get_user_by_id(session: Session, user_id: uuid.UUID) -> Optional[User]:
        """Get user by ID."""
        return session.query(User).filter(
            and_(User.id == user_id, User.is_active == True)
        ).first()
    
    @staticmethod
    def update_user_profile(
        session: Session,
        user_id: uuid.UUID,
        **updates
    ) -> Optional[User]:
        """Update user profile information."""
        try:
            user = UserService.get_user_by_id(session, user_id)
            if not user:
                return None
            
            # Update allowed fields
            allowed_fields = {
                'first_name', 'last_name', 'organization', 'email'
            }
            
            for field, value in updates.items():
                if field in allowed_fields and hasattr(user, field):
                    setattr(user, field, value)
            
            session.commit()
            logger.info(f"User profile updated: {user.username}")
            return user
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to update user profile: {e}")
            raise


class ProjectService:
    """Project management service."""
    
    @staticmethod
    def create_project(
        session: Session,
        owner_id: uuid.UUID,
        name: str,
        description: str,
        model_config: Dict[str, Any],
        privacy_config: Dict[str, Any],
        security_level: SecurityLevel = SecurityLevel.STANDARD,
        **kwargs
    ) -> Project:
        """Create a new federated learning project."""
        try:
            project = Project(
                name=name,
                description=description,
                owner_id=owner_id,
                model_config=model_config,
                privacy_config=privacy_config,
                security_level=security_level,
                **kwargs
            )
            
            session.add(project)
            session.flush()
            
            logger.info(f"Project created: {name} ({project.id}) by user {owner_id}")
            return project
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to create project {name}: {e}")
            raise
    
    @staticmethod
    def get_user_projects(session: Session, user_id: uuid.UUID) -> List[Project]:
        """Get all projects owned by a user."""
        return session.query(Project).filter(
            and_(
                Project.owner_id == user_id,
                Project.is_active == True
            )
        ).order_by(desc(Project.created_at)).all()
    
    @staticmethod
    def get_project_with_stats(session: Session, project_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get project with aggregated statistics."""
        try:
            project = session.query(Project).filter(Project.id == project_id).first()
            if not project:
                return None
            
            # Get statistics
            total_sessions = session.query(func.count(TrainingSession.id)).filter(
                TrainingSession.project_id == project_id
            ).scalar() or 0
            
            active_sessions = session.query(func.count(TrainingSession.id)).filter(
                and_(
                    TrainingSession.project_id == project_id,
                    TrainingSession.status == TrainingStatus.RUNNING
                )
            ).scalar() or 0
            
            total_nodes = session.query(func.count(FederatedNode.id)).filter(
                FederatedNode.project_id == project_id
            ).scalar() or 0
            
            online_nodes = session.query(func.count(FederatedNode.id)).filter(
                and_(
                    FederatedNode.project_id == project_id,
                    FederatedNode.status == NodeStatus.ONLINE
                )
            ).scalar() or 0
            
            return {
                'project': project,
                'stats': {
                    'total_sessions': total_sessions,
                    'active_sessions': active_sessions,
                    'total_nodes': total_nodes,
                    'online_nodes': online_nodes
                }
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to get project stats for {project_id}: {e}")
            return None


class TrainingService:
    """Training session management service."""
    
    @staticmethod
    def create_training_session(
        session: Session,
        project_id: uuid.UUID,
        user_id: uuid.UUID,
        name: str,
        total_rounds: int,
        config_snapshot: Dict[str, Any],
        **kwargs
    ) -> TrainingSession:
        """Create a new training session."""
        try:
            training_session = TrainingSession(
                project_id=project_id,
                user_id=user_id,
                name=name,
                total_rounds=total_rounds,
                config_snapshot=config_snapshot,
                **kwargs
            )
            
            session.add(training_session)
            session.flush()
            
            logger.info(f"Training session created: {name} ({training_session.id})")
            return training_session
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to create training session {name}: {e}")
            raise
    
    @staticmethod
    def update_training_progress(
        session: Session,
        session_id: uuid.UUID,
        round_number: int,
        accuracy: Optional[float] = None,
        loss: Optional[float] = None,
        **kwargs
    ) -> Optional[TrainingSession]:
        """Update training session progress."""
        try:
            training_session = session.query(TrainingSession).filter(
                TrainingSession.id == session_id
            ).first()
            
            if not training_session:
                return None
            
            training_session.current_round = round_number
            
            if accuracy is not None:
                training_session.current_accuracy = accuracy
                if not training_session.best_accuracy or accuracy > training_session.best_accuracy:
                    training_session.best_accuracy = accuracy
            
            if loss is not None:
                training_session.current_loss = loss
            
            # Update other fields
            for field, value in kwargs.items():
                if hasattr(training_session, field):
                    setattr(training_session, field, value)
            
            session.commit()
            return training_session
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to update training progress for session {session_id}: {e}")
            raise
    
    @staticmethod
    def get_active_sessions(session: Session, limit: int = 50) -> List[TrainingSession]:
        """Get all active training sessions."""
        return session.query(TrainingSession).filter(
            TrainingSession.status.in_([TrainingStatus.RUNNING, TrainingStatus.PENDING])
        ).options(
            joinedload(TrainingSession.project),
            joinedload(TrainingSession.user)
        ).order_by(desc(TrainingSession.created_at)).limit(limit).all()


class NodeService:
    """Federated node management service."""
    
    @staticmethod
    def register_node(
        session: Session,
        project_id: uuid.UUID,
        node_id: str,
        node_name: Optional[str] = None,
        ip_address: Optional[str] = None,
        capabilities: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> FederatedNode:
        """Register a new federated learning node."""
        try:
            # Check if node already exists
            existing_node = session.query(FederatedNode).filter(
                and_(
                    FederatedNode.project_id == project_id,
                    FederatedNode.node_id == node_id
                )
            ).first()
            
            if existing_node:
                # Update existing node
                existing_node.node_name = node_name or existing_node.node_name
                existing_node.ip_address = ip_address or existing_node.ip_address
                existing_node.capabilities = capabilities or existing_node.capabilities
                existing_node.status = NodeStatus.ONLINE
                existing_node.last_seen = datetime.utcnow()
                
                for field, value in kwargs.items():
                    if hasattr(existing_node, field):
                        setattr(existing_node, field, value)
                
                session.commit()
                logger.info(f"Node updated: {node_id} in project {project_id}")
                return existing_node
            
            # Create new node
            node = FederatedNode(
                project_id=project_id,
                node_id=node_id,
                node_name=node_name,
                ip_address=ip_address,
                capabilities=capabilities or {},
                status=NodeStatus.ONLINE,
                last_seen=datetime.utcnow(),
                **kwargs
            )
            
            session.add(node)
            session.flush()
            
            logger.info(f"Node registered: {node_id} ({node.id}) in project {project_id}")
            return node
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to register node {node_id}: {e}")
            raise
    
    @staticmethod
    def update_node_status(
        session: Session,
        node_id: uuid.UUID,
        status: NodeStatus,
        **kwargs
    ) -> Optional[FederatedNode]:
        """Update node status and metrics."""
        try:
            node = session.query(FederatedNode).filter(
                FederatedNode.id == node_id
            ).first()
            
            if not node:
                return None
            
            node.status = status
            node.last_seen = datetime.utcnow()
            
            # Update additional fields
            for field, value in kwargs.items():
                if hasattr(node, field):
                    setattr(node, field, value)
            
            session.commit()
            return node
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to update node status for {node_id}: {e}")
            raise
    
    @staticmethod
    def get_project_nodes(
        session: Session,
        project_id: uuid.UUID,
        status_filter: Optional[List[NodeStatus]] = None
    ) -> List[FederatedNode]:
        """Get all nodes for a project, optionally filtered by status."""
        query = session.query(FederatedNode).filter(
            FederatedNode.project_id == project_id
        )
        
        if status_filter:
            query = query.filter(FederatedNode.status.in_(status_filter))
        
        return query.order_by(desc(FederatedNode.last_seen)).all()


class MetricsService:
    """Training and system metrics service."""
    
    @staticmethod
    def record_training_metric(
        session: Session,
        session_id: uuid.UUID,
        metric_type: str,
        value: float,
        round_number: int,
        client_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a training metric."""
        try:
            metric = TrainingMetric(
                session_id=session_id,
                metric_type=metric_type,
                value=value,
                round_number=round_number,
                client_id=client_id,
                metadata=metadata or {}
            )
            
            session.add(metric)
            session.commit()
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to record training metric: {e}")
    
    @staticmethod
    def get_session_metrics(
        session: Session,
        session_id: uuid.UUID,
        metric_types: Optional[List[str]] = None,
        limit: int = 1000
    ) -> List[TrainingMetric]:
        """Get metrics for a training session."""
        query = session.query(TrainingMetric).filter(
            TrainingMetric.session_id == session_id
        )
        
        if metric_types:
            query = query.filter(TrainingMetric.metric_type.in_(metric_types))
        
        return query.order_by(
            TrainingMetric.round_number,
            TrainingMetric.recorded_at
        ).limit(limit).all()
    
    @staticmethod
    def record_performance_metric(
        session: Session,
        metric_name: str,
        component: str,
        value: float,
        unit: str,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a system performance metric."""
        try:
            metric = PerformanceMetric(
                metric_name=metric_name,
                component=component,
                value=value,
                unit=unit,
                labels=labels or {}
            )
            
            session.add(metric)
            session.commit()
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to record performance metric: {e}")


class AuditService:
    """Security and audit logging service."""
    
    @staticmethod
    def log_event(
        session: Session,
        event_type: str,
        action: str,
        success: bool,
        user_id: Optional[uuid.UUID] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        error_message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[uuid.UUID] = None
    ):
        """Log an audit event."""
        try:
            audit_log = AuditLog(
                event_type=event_type,
                action=action,
                success=success,
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                ip_address=ip_address,
                user_agent=user_agent,
                error_message=error_message,
                details=details or {},
                correlation_id=correlation_id or uuid.uuid4()
            )
            
            session.add(audit_log)
            session.commit()
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to log audit event: {e}")
            # Don't re-raise to avoid breaking main functionality
    
    @staticmethod
    def get_user_activity(
        session: Session,
        user_id: uuid.UUID,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get recent activity for a user."""
        return session.query(AuditLog).filter(
            AuditLog.user_id == user_id
        ).order_by(desc(AuditLog.created_at)).limit(limit).all()