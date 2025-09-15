"""
Database Service Layer for QFLARE

This module provides high-level database operations for:
- Device management
- Model updates and aggregation
- Training sessions
- Audit logging
"""

import time
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func

from .models import Device, GlobalModel, ModelUpdate, TrainingSession, AuditLog, UserToken
from .connection import get_database

logger = logging.getLogger(__name__)


class DeviceService:
    """Service for device management operations"""
    
    @staticmethod
    def register_device(device_id: str, device_info: Dict[str, Any]) -> bool:
        """
        Register a new device in the database.
        
        Args:
            device_id: Unique device identifier
            device_info: Device information and configuration
            
        Returns:
            True if device was registered successfully
        """
        try:
            db = get_database()
            with db.get_session() as session:
                # Check if device already exists
                existing_device = session.query(Device).filter(
                    Device.device_id == device_id
                ).first()
                
                if existing_device:
                    logger.warning(f"Device {device_id} already registered")
                    return False
                
                # Create new device record
                device = Device(
                    device_id=device_id,
                    device_type=device_info.get('device_type', 'unknown'),
                    hardware_info=device_info.get('hardware_info', {}),
                    network_info=device_info.get('network_info', {}),
                    capabilities=device_info.get('capabilities', {}),
                    kem_public_key=device_info.get('kem_public_key'),
                    sig_public_key=device_info.get('sig_public_key'),
                    local_epochs=device_info.get('local_epochs', 1),
                    batch_size=device_info.get('batch_size', 32),
                    learning_rate=device_info.get('learning_rate', 0.01)
                )
                
                session.add(device)
                
                # Log registration event
                AuditService.log_event(
                    session=session,
                    device_id=device_id,
                    event_type='device_registration',
                    event_description=f'Device {device_id} registered successfully',
                    event_data=device_info
                )
                
                logger.info(f"Device {device_id} registered successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error registering device {device_id}: {e}")
            return False
    
    @staticmethod
    def get_device(device_id: str) -> Optional[Dict[str, Any]]:
        """
        Get device information.
        
        Args:
            device_id: Device identifier
            
        Returns:
            Device information dictionary or None
        """
        try:
            db = get_database()
            with db.get_session() as session:
                device = session.query(Device).filter(
                    Device.device_id == device_id
                ).first()
                
                if not device:
                    return None
                
                # Update last seen timestamp
                device.last_seen = datetime.utcnow()
                
                return {
                    'device_id': device.device_id,
                    'status': device.status,
                    'registered_at': device.registered_at.isoformat(),
                    'last_seen': device.last_seen.isoformat(),
                    'device_type': device.device_type,
                    'hardware_info': device.hardware_info,
                    'network_info': device.network_info,
                    'capabilities': device.capabilities,
                    'training_config': {
                        'local_epochs': device.local_epochs,
                        'batch_size': device.batch_size,
                        'learning_rate': device.learning_rate
                    },
                    'key_rotation_count': device.key_rotation_count
                }
                
        except Exception as e:
            logger.error(f"Error getting device {device_id}: {e}")
            return None
    
    @staticmethod
    def update_device_status(device_id: str, status: str) -> bool:
        """
        Update device status.
        
        Args:
            device_id: Device identifier
            status: New status
            
        Returns:
            True if status was updated successfully
        """
        try:
            db = get_database()
            with db.get_session() as session:
                device = session.query(Device).filter(
                    Device.device_id == device_id
                ).first()
                
                if not device:
                    logger.warning(f"Device {device_id} not found for status update")
                    return False
                
                old_status = device.status
                device.status = status
                device.last_seen = datetime.utcnow()
                
                # Log status change
                AuditService.log_event(
                    session=session,
                    device_id=device_id,
                    event_type='status_change',
                    event_description=f'Device status changed from {old_status} to {status}',
                    event_data={'old_status': old_status, 'new_status': status}
                )
                
                logger.info(f"Device {device_id} status updated to {status}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating device status {device_id}: {e}")
            return False
    
    @staticmethod
    def list_active_devices() -> List[Dict[str, Any]]:
        """
        Get list of active devices.
        
        Returns:
            List of active device information
        """
        try:
            db = get_database()
            with db.get_session() as session:
                devices = session.query(Device).filter(
                    or_(Device.status == 'active', Device.status == 'training')
                ).all()
                
                return [
                    {
                        'device_id': device.device_id,
                        'last_seen': device.last_seen.isoformat(),
                        'device_type': device.device_type,
                        'capabilities': device.capabilities,
                        'status': device.status
                    }
                    for device in devices
                ]
                
        except Exception as e:
            logger.error(f"Error listing active devices: {e}")
            return []


class ModelService:
    """Service for model management operations"""
    
    @staticmethod
    def store_model_update(
        device_id: str,
        model_weights: bytes,
        signature: bytes,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Store a model update from a device.
        
        Args:
            device_id: Device identifier
            model_weights: Serialized model weights
            signature: Post-quantum signature
            metadata: Update metadata
            
        Returns:
            True if update was stored successfully
        """
        try:
            db = get_database()
            with db.get_session() as session:
                # Calculate model hash
                model_hash = hashlib.sha256(model_weights).hexdigest()
                
                # Get current global model
                current_global = session.query(GlobalModel).order_by(
                    desc(GlobalModel.round_number)
                ).first()
                
                global_model_id = current_global.id if current_global else None
                
                # Create model update record
                update = ModelUpdate(
                    device_id=device_id,
                    global_model_id=global_model_id,
                    model_weights=model_weights,
                    model_hash=model_hash,
                    signature=signature,
                    local_loss=metadata.get('local_loss'),
                    local_accuracy=metadata.get('local_accuracy'),
                    local_epochs=metadata.get('local_epochs'),
                    samples_count=metadata.get('samples_count'),
                    training_time=metadata.get('training_time'),
                    status='pending'
                )
                
                session.add(update)
                
                # Log model update event
                AuditService.log_event(
                    session=session,
                    device_id=device_id,
                    event_type='model_update',
                    event_description=f'Model update received from {device_id}',
                    event_data={
                        'model_hash': model_hash,
                        'samples_count': metadata.get('samples_count'),
                        'local_accuracy': metadata.get('local_accuracy')
                    }
                )
                
                logger.info(f"Model update stored for device {device_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing model update for {device_id}: {e}")
            return False
    
    @staticmethod
    def get_pending_updates() -> List[Dict[str, Any]]:
        """
        Get all pending model updates for aggregation.
        
        Returns:
            List of pending model updates
        """
        try:
            db = get_database()
            with db.get_session() as session:
                updates = session.query(ModelUpdate).filter(
                    ModelUpdate.status == 'pending'
                ).all()
                
                return [
                    {
                        'id': update.id,
                        'device_id': update.device_id,
                        'model_weights': update.model_weights,
                        'model_hash': update.model_hash,
                        'local_loss': update.local_loss,
                        'local_accuracy': update.local_accuracy,
                        'samples_count': update.samples_count,
                        'created_at': update.created_at.isoformat()
                    }
                    for update in updates
                ]
                
        except Exception as e:
            logger.error(f"Error getting pending updates: {e}")
            return []
    
    @staticmethod
    def store_global_model(
        round_number: int,
        model_weights: bytes,
        model_metadata: Dict[str, Any],
        participating_devices: List[str]
    ) -> bool:
        """
        Store a new global model after aggregation.
        
        Args:
            round_number: Aggregation round number
            model_weights: Aggregated model weights
            model_metadata: Model performance metrics
            participating_devices: List of devices that participated
            
        Returns:
            True if global model was stored successfully
        """
        try:
            db = get_database()
            with db.get_session() as session:
                # Calculate model hash
                model_hash = hashlib.sha256(model_weights).hexdigest()
                
                # Create global model record
                global_model = GlobalModel(
                    round_number=round_number,
                    model_weights=model_weights,
                    model_hash=model_hash,
                    model_type=model_metadata.get('model_type'),
                    model_architecture=model_metadata.get('architecture'),
                    accuracy=model_metadata.get('accuracy'),
                    loss=model_metadata.get('loss'),
                    num_participants=len(participating_devices),
                    aggregation_method=model_metadata.get('aggregation_method', 'fedavg')
                )
                
                session.add(global_model)
                session.flush()  # Get the ID
                
                # Mark corresponding updates as aggregated
                session.query(ModelUpdate).filter(
                    and_(
                        ModelUpdate.device_id.in_(participating_devices),
                        ModelUpdate.status == 'pending'
                    )
                ).update({
                    'status': 'aggregated',
                    'aggregated_at': datetime.utcnow(),
                    'global_model_id': global_model.id
                })
                
                # Log aggregation event
                AuditService.log_event(
                    session=session,
                    device_id=None,
                    event_type='model_aggregation',
                    event_description=f'Global model round {round_number} created',
                    event_data={
                        'round_number': round_number,
                        'participants': participating_devices,
                        'accuracy': model_metadata.get('accuracy'),
                        'model_hash': model_hash
                    }
                )
                
                logger.info(f"Global model round {round_number} stored successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error storing global model: {e}")
            return False
    
    @staticmethod
    def get_latest_global_model() -> Optional[Dict[str, Any]]:
        """
        Get the latest global model.
        
        Returns:
            Latest global model information or None
        """
        try:
            db = get_database()
            with db.get_session() as session:
                global_model = session.query(GlobalModel).order_by(
                    desc(GlobalModel.round_number)
                ).first()
                
                if not global_model:
                    return None
                
                return {
                    'id': global_model.id,
                    'round_number': global_model.round_number,
                    'model_weights': global_model.model_weights,
                    'model_hash': global_model.model_hash,
                    'accuracy': global_model.accuracy,
                    'loss': global_model.loss,
                    'num_participants': global_model.num_participants,
                    'created_at': global_model.created_at.isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting latest global model: {e}")
            return None


class AuditService:
    """Service for audit logging operations"""
    
    @staticmethod
    def log_event(
        session: Session,
        device_id: Optional[str],
        event_type: str,
        event_description: str,
        event_data: Optional[Dict[str, Any]] = None,
        risk_level: str = 'low',
        ip_address: Optional[str] = None
    ):
        """
        Log an audit event.
        
        Args:
            session: Database session
            device_id: Device identifier (optional)
            event_type: Type of event
            event_description: Human-readable description
            event_data: Additional event data
            risk_level: Risk level (low, medium, high, critical)
            ip_address: Client IP address
        """
        try:
            audit_log = AuditLog(
                device_id=device_id,
                event_type=event_type,
                event_description=event_description,
                event_data=event_data or {},
                risk_level=risk_level,
                ip_address=ip_address
            )
            
            session.add(audit_log)
            
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")
    
    @staticmethod
    def get_recent_events(
        device_id: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        hours: int = 24,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent audit events.
        
        Args:
            device_id: Filter by device ID
            event_types: Filter by event types
            hours: Number of hours to look back
            limit: Maximum number of events
            
        Returns:
            List of recent audit events
        """
        try:
            db = get_database()
            with db.get_session() as session:
                query = session.query(AuditLog)
                
                # Time filter
                since = datetime.utcnow() - timedelta(hours=hours)
                query = query.filter(AuditLog.timestamp >= since)
                
                # Device filter
                if device_id:
                    query = query.filter(AuditLog.device_id == device_id)
                
                # Event type filter
                if event_types:
                    query = query.filter(AuditLog.event_type.in_(event_types))
                
                events = query.order_by(desc(AuditLog.timestamp)).limit(limit).all()
                
                return [
                    {
                        'id': event.id,
                        'device_id': event.device_id,
                        'event_type': event.event_type,
                        'event_description': event.event_description,
                        'event_data': event.event_data,
                        'risk_level': event.risk_level,
                        'timestamp': event.timestamp.isoformat()
                    }
                    for event in events
                ]
                
        except Exception as e:
            logger.error(f"Error getting recent events: {e}")
            return []


class TrainingService:
    """Service for training session management"""
    
    @staticmethod
    def create_training_session(
        session_id: str,
        device_id: str,
        config: Dict[str, Any]
    ) -> bool:
        """
        Create a new training session.
        
        Args:
            session_id: Unique session identifier
            device_id: Device identifier
            config: Training configuration
            
        Returns:
            True if session was created successfully
        """
        try:
            db = get_database()
            with db.get_session() as session:
                training_session = TrainingSession(
                    session_id=session_id,
                    device_id=device_id,
                    dataset_name=config.get('dataset_name'),
                    model_type=config.get('model_type'),
                    hyperparameters=config.get('hyperparameters', {}),
                    total_rounds=config.get('total_rounds', 10)
                )
                
                session.add(training_session)
                
                logger.info(f"Training session {session_id} created for device {device_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating training session: {e}")
            return False
    
    @staticmethod
    def update_training_progress(
        session_id: str,
        current_round: int,
        metrics: Dict[str, Any]
    ) -> bool:
        """
        Update training session progress.
        
        Args:
            session_id: Session identifier
            current_round: Current training round
            metrics: Training metrics
            
        Returns:
            True if progress was updated successfully
        """
        try:
            db = get_database()
            with db.get_session() as session:
                training_session = session.query(TrainingSession).filter(
                    TrainingSession.session_id == session_id
                ).first()
                
                if not training_session:
                    logger.warning(f"Training session {session_id} not found")
                    return False
                
                training_session.current_round = current_round
                training_session.last_update_at = datetime.utcnow()
                training_session.current_loss = metrics.get('loss')
                
                if metrics.get('accuracy') and (
                    not training_session.best_accuracy or 
                    metrics['accuracy'] > training_session.best_accuracy
                ):
                    training_session.best_accuracy = metrics['accuracy']
                
                logger.info(f"Training session {session_id} updated to round {current_round}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating training progress: {e}")
            return False