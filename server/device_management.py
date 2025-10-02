"""
🔧 QFLARE Device Management System
Advanced device registration, monitoring, and control for federated learning
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import uuid
import asyncio
import logging

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/devices", tags=["Device Management"])

class DeviceStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    TRAINING = "training"
    IDLE = "idle"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class DeviceCapability(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    MOBILE = "mobile"
    EDGE = "edge"
    CLOUD = "cloud"

class DeviceRegistration(BaseModel):
    device_name: str = Field(..., min_length=3, max_length=50)
    device_type: str = Field(..., description="Type of device (mobile, desktop, server, etc.)")
    capabilities: List[DeviceCapability] = Field(default=[], description="Device computational capabilities")
    location: Optional[str] = Field(None, description="Geographic location or data center")
    contact_info: Optional[str] = Field(None, description="Contact information for device owner")
    max_concurrent_tasks: int = Field(default=1, ge=1, le=10)
    preferred_schedule: Optional[Dict[str, Any]] = Field(None, description="Preferred training schedule")
    
    @validator('device_name')
    def validate_device_name(cls, v):
        if not v.replace('_', '').replace('-', '').replace(' ', '').isalnum():
            raise ValueError('Device name must contain only alphanumeric characters, spaces, hyphens, and underscores')
        return v

class DeviceInfo(BaseModel):
    device_id: str
    device_name: str
    device_type: str
    status: DeviceStatus
    capabilities: List[DeviceCapability]
    location: Optional[str]
    contact_info: Optional[str]
    registered_at: datetime
    last_seen: datetime
    total_training_sessions: int = 0
    total_training_time: float = 0.0  # in hours
    success_rate: float = 100.0  # percentage
    current_task: Optional[str] = None
    performance_metrics: Dict[str, float] = Field(default_factory=dict)

class DeviceUpdate(BaseModel):
    device_name: Optional[str] = None
    device_type: Optional[str] = None
    capabilities: Optional[List[DeviceCapability]] = None
    location: Optional[str] = None
    contact_info: Optional[str] = None
    max_concurrent_tasks: Optional[int] = None
    preferred_schedule: Optional[Dict[str, Any]] = None

class TrainingTask(BaseModel):
    task_id: str
    device_id: str
    model_name: str
    dataset_name: str
    training_params: Dict[str, Any]
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    error_message: Optional[str] = None

# Import secure storage system
try:
    from secure_device_storage import get_secure_storage
    secure_storage = get_secure_storage()
    SECURE_STORAGE_AVAILABLE = True
    logger.info("🔐 Secure storage initialized for device management")
except ImportError as e:
    logger.warning(f"⚠️ Secure storage not available, using fallback: {e}")
    SECURE_STORAGE_AVAILABLE = False
    # Fallback to in-memory storage for development
    registered_devices: Dict[str, DeviceInfo] = {}
    secure_storage = None

device_tasks: Dict[str, List[TrainingTask]] = {}
active_connections: Dict[str, Any] = {}

@router.post("/register", response_model=Dict[str, str])
async def register_device(device: DeviceRegistration):
    """
    📱 Register a new device for federated learning
    
    This endpoint allows devices to register themselves in the QFLARE network.
    Each device receives a unique ID and can participate in training rounds.
    """
    
    if SECURE_STORAGE_AVAILABLE:
        # Use secure storage for production
        device_data = {
            'device_name': device.device_name,
            'device_type': device.device_type,
            'capabilities': [cap.value for cap in device.capabilities],
            'location': device.location,
            'contact_info': device.contact_info,
            'max_concurrent_tasks': device.max_concurrent_tasks,
            'preferred_schedule': device.preferred_schedule,
            'device_class': device.device_type.lower(),
            'security_level': 1
        }
        
        try:
            device_id = await secure_storage.store_device(device_data)
            device_tasks[device_id] = []
            
            logger.info(f"🔐 New device registered securely: {device.device_name} ({device_id})")
            
            return {
                "device_id": device_id,
                "message": f"Device '{device.device_name}' registered successfully!",
                "status": "registered",
                "storage": "secure"
            }
            
        except Exception as e:
            logger.error(f"❌ Secure registration failed: {e}")
            raise HTTPException(status_code=500, detail="Registration failed - please try again")
    
    else:
        # Fallback to in-memory storage
        device_id = str(uuid.uuid4())
        
        # Check for duplicate device names
        existing_names = [d.device_name for d in registered_devices.values()]
        if device.device_name in existing_names:
            raise HTTPException(
                status_code=400,
                detail=f"Device name '{device.device_name}' already exists. Please choose a different name."
            )
        
        # Create device info
        now = datetime.now()
        device_info = DeviceInfo(
            device_id=device_id,
            device_name=device.device_name,
            device_type=device.device_type,
            status=DeviceStatus.ONLINE,
            capabilities=device.capabilities,
            location=device.location,
            contact_info=device.contact_info,
            registered_at=now,
            last_seen=now
        )
        
        # Store device
        registered_devices[device_id] = device_info
        device_tasks[device_id] = []
        
        logger.info(f"📝 New device registered (in-memory): {device.device_name} ({device_id})")
        
        return {
            "device_id": device_id,
            "message": f"Device '{device.device_name}' registered successfully!",
            "status": "registered",
            "storage": "in_memory"
        }

@router.get("/", response_model=List[Dict[str, Any]])
async def list_devices(
    status: Optional[DeviceStatus] = None,
    capability: Optional[DeviceCapability] = None,
    limit: int = Query(default=50, le=100)
):
    """
    📋 List all registered devices with optional filtering
    """
    if SECURE_STORAGE_AVAILABLE:
        try:
            # Get devices from secure storage
            status_filter = status.value if status else None
            devices = await secure_storage.list_devices(limit, status_filter)
            
            # Apply capability filter if specified
            if capability:
                filtered_devices = []
                for device in devices:
                    # Get full device data to check capabilities
                    full_device = await secure_storage.get_device(device['device_id'])
                    if full_device and capability.value in full_device.get('capabilities', []):
                        filtered_devices.append(device)
                devices = filtered_devices
            
            return devices
            
        except Exception as e:
            logger.error(f"❌ Failed to list devices from secure storage: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve devices")
    
    else:
        # Fallback to in-memory storage
        devices = []
        for device_info in registered_devices.values():
            devices.append({
                'device_id': device_info.device_id,
                'device_name': device_info.device_name,
                'device_type': device_info.device_type,
                'status': device_info.status.value,
                'capabilities': [cap.value for cap in device_info.capabilities],
                'location': device_info.location,
                'registered_at': device_info.registered_at,
                'last_seen': device_info.last_seen,
                'total_training_sessions': device_info.total_training_sessions,
                'success_rate': device_info.success_rate
            })
        
        # Apply filters
        if status:
            devices = [d for d in devices if d['status'] == status.value]
        
        if capability:
            devices = [d for d in devices if capability.value in d['capabilities']]
        
        # Sort by last seen (most recent first)
        devices.sort(key=lambda x: x['last_seen'], reverse=True)
        
        return devices[:limit]

@router.get("/{device_id}", response_model=Dict[str, Any])
async def get_device(device_id: str):
    """
    🔍 Get detailed information about a specific device
    """
    if SECURE_STORAGE_AVAILABLE:
        try:
            device = await secure_storage.get_device(device_id)
            if not device:
                raise HTTPException(status_code=404, detail="Device not found")
            return device
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve device {device_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve device")
    
    else:
        # Fallback to in-memory storage
        if device_id not in registered_devices:
            raise HTTPException(status_code=404, detail="Device not found")
        
        device_info = registered_devices[device_id]
        return {
            'device_id': device_info.device_id,
            'device_name': device_info.device_name,
            'device_type': device_info.device_type,
            'status': device_info.status.value,
            'capabilities': [cap.value for cap in device_info.capabilities],
            'location': device_info.location,
            'contact_info': device_info.contact_info,
            'registered_at': device_info.registered_at,
            'last_seen': device_info.last_seen,
            'total_training_sessions': device_info.total_training_sessions,
            'total_training_time': device_info.total_training_time,
            'success_rate': device_info.success_rate,
            'current_task': device_info.current_task,
            'performance_metrics': device_info.performance_metrics
        }

@router.put("/{device_id}", response_model=DeviceInfo)
async def update_device(device_id: str, update: DeviceUpdate):
    """
    ✏️ Update device information
    """
    if device_id not in registered_devices:
        raise HTTPException(status_code=404, detail="Device not found")
    
    device = registered_devices[device_id]
    
    # Update only provided fields
    update_data = update.dict(exclude_unset=True)
    for field, value in update_data.items():
        if hasattr(device, field):
            setattr(device, field, value)
    
    device.last_seen = datetime.now()
    
    logger.info(f"📝 Device updated: {device.device_name} ({device_id})")
    
    return device

@router.delete("/{device_id}")
async def unregister_device(device_id: str):
    """
    🗑️ Unregister a device from the network
    """
    if device_id not in registered_devices:
        raise HTTPException(status_code=404, detail="Device not found")
    
    device_name = registered_devices[device_id].device_name
    
    # Clean up
    del registered_devices[device_id]
    if device_id in device_tasks:
        del device_tasks[device_id]
    if device_id in active_connections:
        del active_connections[device_id]
    
    logger.info(f"👋 Device unregistered: {device_name} ({device_id})")
    
    return {"message": f"Device '{device_name}' unregistered successfully"}

@router.post("/{device_id}/heartbeat")
async def device_heartbeat(device_id: str, status: Optional[DeviceStatus] = DeviceStatus.ONLINE):
    """
    💓 Device heartbeat to maintain connection status
    """
    if SECURE_STORAGE_AVAILABLE:
        try:
            # Update heartbeat in secure storage
            success = await secure_storage.update_device_heartbeat(device_id)
            if not success:
                raise HTTPException(status_code=404, detail="Device not found")
            
            logger.debug(f"💓 Heartbeat received from device {device_id}")
            
            return {
                "device_id": device_id,
                "status": "heartbeat_received",
                "storage": "secure",
                "timestamp": datetime.now()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Heartbeat update failed: {e}")
            raise HTTPException(status_code=500, detail="Heartbeat update failed")
    
    else:
        # Fallback to in-memory storage
        if device_id not in registered_devices:
            raise HTTPException(status_code=404, detail="Device not found")
        
        device = registered_devices[device_id]
        device.last_seen = datetime.now()
        device.status = status
        
        return {
            "device_id": device_id,
            "status": "heartbeat_received",
            "storage": "in_memory",
        "timestamp": device.last_seen
    }

@router.get("/{device_id}/tasks", response_model=List[TrainingTask])
async def get_device_tasks(device_id: str, limit: int = 20):
    """
    📋 Get training tasks for a specific device
    """
    if device_id not in registered_devices:
        raise HTTPException(status_code=404, detail="Device not found")
    
    tasks = device_tasks.get(device_id, [])
    return sorted(tasks, key=lambda x: x.created_at, reverse=True)[:limit]

@router.post("/{device_id}/tasks", response_model=TrainingTask)
async def assign_task(device_id: str, task_params: Dict[str, Any]):
    """
    🎯 Assign a training task to a device
    """
    if device_id not in registered_devices:
        raise HTTPException(status_code=404, detail="Device not found")
    
    device = registered_devices[device_id]
    
    # Check if device is available
    if device.status not in [DeviceStatus.ONLINE, DeviceStatus.IDLE]:
        raise HTTPException(
            status_code=400,
            detail=f"Device is not available for training. Current status: {device.status}"
        )
    
    # Create training task
    task = TrainingTask(
        task_id=str(uuid.uuid4()),
        device_id=device_id,
        model_name=task_params.get("model_name", "default_model"),
        dataset_name=task_params.get("dataset_name", "default_dataset"),
        training_params=task_params.get("training_params", {})
    )
    
    # Store task
    if device_id not in device_tasks:
        device_tasks[device_id] = []
    device_tasks[device_id].append(task)
    
    # Update device status
    device.status = DeviceStatus.TRAINING
    device.current_task = task.task_id
    
    logger.info(f"🎯 Task assigned to {device.device_name}: {task.task_id}")
    
    return task

@router.get("/stats/overview")
async def get_device_stats():
    """
    📊 Get overall device network statistics
    """
    total_devices = len(registered_devices)
    status_counts = {}
    capability_counts = {}
    
    for device in registered_devices.values():
        # Count by status
        status = device.status.value
        status_counts[status] = status_counts.get(status, 0) + 1
        
        # Count by capabilities
        for cap in device.capabilities:
            cap_name = cap.value
            capability_counts[cap_name] = capability_counts.get(cap_name, 0) + 1
    
    # Calculate uptime
    now = datetime.now()
    online_devices = [d for d in registered_devices.values() if d.status == DeviceStatus.ONLINE]
    avg_uptime = 0.0
    if online_devices:
        uptimes = [(now - d.last_seen).total_seconds() / 3600 for d in online_devices]  # in hours
        avg_uptime = sum(uptimes) / len(uptimes)
    
    return {
        "total_devices": total_devices,
        "status_distribution": status_counts,
        "capability_distribution": capability_counts,
        "average_uptime_hours": round(avg_uptime, 2),
        "active_training_sessions": len([d for d in registered_devices.values() if d.status == DeviceStatus.TRAINING]),
        "timestamp": now
    }

# Background task to check device health
async def check_device_health():
    """
    🏥 Background task to monitor device health and update status
    """
    while True:
        try:
            now = datetime.now()
            offline_threshold = timedelta(minutes=5)  # Consider offline after 5 minutes
            
            for device_id, device in registered_devices.items():
                if now - device.last_seen > offline_threshold and device.status != DeviceStatus.OFFLINE:
                    device.status = DeviceStatus.OFFLINE
                    device.current_task = None
                    logger.warning(f"📴 Device went offline: {device.device_name} ({device_id})")
            
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"❌ Error in device health check: {e}")
            await asyncio.sleep(60)

# Start background health check
@router.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(check_device_health())