"""
QFLARE Device Registry (Database-Integrated v2.0)
Unified device registration using production database backend
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, UTC

# Import the new unified database system
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database import (
    DeviceRepository, KeyExchangeRepository, AuditRepository,
    quick_device_lookup, security_audit_log
)

logger = logging.getLogger(__name__)

class DeviceRegistryDB:
    """Database-integrated device registry"""
    
    @staticmethod
    async def register_device(
        device_id: str, 
        device_info: Dict[str, Any] = None,
        organization: str = "UNKNOWN",
        location: str = "Unknown Location"
    ) -> bool:
        """
        Register a new device using unified database backend.
        
        Args:
            device_id: Unique device identifier
            device_info: Device information dictionary
            organization: Device organization
            location: Device location
            
        Returns:
            True if device was registered successfully, False otherwise
        """
        try:
            if device_info is None:
                device_info = {}
            
            # Extract device type from legacy format
            metadata = device_info.get("metadata", {})
            device_type = metadata.get("device_type", "EDGE_NODE")
            
            # Extract capabilities
            capabilities = {
                "hardware": metadata.get("hardware", {}),
                "network": metadata.get("network", {}),
                "capabilities": metadata.get("capabilities", {}),
                "training_config": device_info.get("training_config", {}),
                "public_keys": device_info.get("public_keys", {}),
                "quantum_ready": metadata.get("capabilities", {}).get("quantum_ready", True),
                "legacy_migration": True
            }
            
            # Extract public key (KEM key preferred)
            public_keys = device_info.get("public_keys", {})
            public_key = None
            if "kem_public_key" in public_keys:
                # Convert base64 string to bytes if needed
                kem_key = public_keys["kem_public_key"]
                if isinstance(kem_key, str):
                    import base64
                    try:
                        public_key = base64.b64decode(kem_key)
                    except:
                        public_key = kem_key.encode()
                else:
                    public_key = kem_key
            
            # Create device in database
            device = await DeviceRepository.create_device(
                device_id=device_id,
                device_type=device_type,
                organization=organization,
                location=location,
                capabilities=capabilities,
                public_key=public_key
            )
            
            # Auto-approve device (for now)
            await DeviceRepository.approve_device(device_id)
            await DeviceRepository.update_device_trust_score(device_id, 7.5)
            
            # Log registration event
            await security_audit_log(
                event_type="DEVICE_REGISTRATION",
                message=f"Device {device_id} registered successfully",
                device_id=device_id,
                threat_level=1
            )
            
            logger.info(f"Successfully registered device {device_id} in unified database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register device {device_id}: {e}")
            return False
    
    @staticmethod
    async def get_device(device_id: str) -> Optional[Dict[str, Any]]:
        """
        Get device information from database.
        
        Args:
            device_id: Device identifier
            
        Returns:
            Device information dictionary or None if not found
        """
        try:
            device_info = await quick_device_lookup(device_id)
            if device_info:
                # Update last seen
                await DeviceRepository.update_last_seen(device_id)
                
            return device_info
            
        except Exception as e:
            logger.error(f"Failed to get device {device_id}: {e}")
            return None
    
    @staticmethod
    async def get_all_devices() -> List[Dict[str, Any]]:
        """
        Get all registered devices.
        
        Returns:
            List of device information dictionaries
        """
        try:
            # Get approved devices
            approved_devices = await DeviceRepository.get_devices_by_status('approved')
            
            devices = []
            for device in approved_devices:
                device_data = {
                    "device_id": device.device_id,
                    "device_type": device.device_type,
                    "organization": device.organization,
                    "location": device.location,
                    "status": device.status,
                    "trust_score": device.trust_score,
                    "security_level": device.security_level,
                    "created_at": device.created_at.isoformat(),
                    "last_seen": device.last_seen.isoformat() if device.last_seen else None,
                    "capabilities": device.capabilities
                }
                devices.append(device_data)
            
            return devices
            
        except Exception as e:
            logger.error(f"Failed to get all devices: {e}")
            return []
    
    @staticmethod
    async def remove_device(device_id: str) -> bool:
        """
        Remove device from registry.
        
        Args:
            device_id: Device identifier
            
        Returns:
            True if device was removed successfully, False otherwise
        """
        try:
            # For now, we'll suspend the device instead of deleting
            device = await DeviceRepository.get_device(device_id)
            if not device:
                return False
            
            # Update device status to suspended
            from database.repository import database_transaction
            from database.models import Device
            from sqlalchemy import update
            
            async with database_transaction() as session:
                result = await session.execute(
                    update(Device)
                    .where(Device.device_id == device_id)
                    .values(status='suspended', updated_at=datetime.now(UTC))
                )
                success = result.rowcount > 0
            
            if success:
                # Log removal event
                await security_audit_log(
                    event_type="DEVICE_REMOVED",
                    message=f"Device {device_id} suspended from registry",
                    device_id=device_id,
                    threat_level=3
                )
                
                logger.info(f"Device {device_id} suspended from registry")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to remove device {device_id}: {e}")
            return False
    
    @staticmethod
    async def update_device_status(device_id: str, status: str) -> bool:
        """
        Update device status.
        
        Args:
            device_id: Device identifier
            status: New status
            
        Returns:
            True if status was updated successfully, False otherwise
        """
        try:
            from database.repository import database_transaction
            from database.models import Device
            from sqlalchemy import update
            
            async with database_transaction() as session:
                result = await session.execute(
                    update(Device)
                    .where(Device.device_id == device_id)
                    .values(status=status, updated_at=datetime.now(UTC))
                )
                success = result.rowcount > 0
            
            if success:
                # Log status update
                await security_audit_log(
                    event_type="DEVICE_STATUS_UPDATED",
                    message=f"Device {device_id} status updated to {status}",
                    device_id=device_id,
                    threat_level=2
                )
                
                logger.info(f"Device {device_id} status updated to {status}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update device {device_id} status: {e}")
            return False

# Legacy sync wrappers for backward compatibility
def register_device(device_id: str, device_info: Dict[str, Any] = None) -> bool:
    """Legacy sync wrapper for device registration"""
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            DeviceRegistryDB.register_device(device_id, device_info)
        )
    except RuntimeError:
        # No event loop running, create new one
        return asyncio.run(
            DeviceRegistryDB.register_device(device_id, device_info)
        )

def get_device(device_id: str) -> Optional[Dict[str, Any]]:
    """Legacy sync wrapper for getting device"""
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            DeviceRegistryDB.get_device(device_id)
        )
    except RuntimeError:
        return asyncio.run(
            DeviceRegistryDB.get_device(device_id)
        )

def get_all_devices() -> List[Dict[str, Any]]:
    """Legacy sync wrapper for getting all devices"""
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            DeviceRegistryDB.get_all_devices()
        )
    except RuntimeError:
        return asyncio.run(
            DeviceRegistryDB.get_all_devices()
        )

def remove_device(device_id: str) -> bool:
    """Legacy sync wrapper for removing device"""
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            DeviceRegistryDB.remove_device(device_id)
        )
    except RuntimeError:
        return asyncio.run(
            DeviceRegistryDB.remove_device(device_id)
        )

def update_device_status(device_id: str, status: str) -> bool:
    """Legacy sync wrapper for updating device status"""
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            DeviceRegistryDB.update_device_status(device_id, status)
        )
    except RuntimeError:
        return asyncio.run(
            DeviceRegistryDB.update_device_status(device_id, status)
        )

# Additional utility functions
def get_device_count() -> int:
    """Get total number of registered devices"""
    devices = get_all_devices()
    return len(devices)

def get_active_devices() -> List[Dict[str, Any]]:
    """Get only active/approved devices"""
    devices = get_all_devices()
    return [d for d in devices if d.get('status') == 'approved']

def device_exists(device_id: str) -> bool:
    """Check if device exists in registry"""
    device = get_device(device_id)
    return device is not None

def get_device_by_type(device_type: str) -> List[Dict[str, Any]]:
    """Get devices by type"""
    devices = get_all_devices()
    return [d for d in devices if d.get('device_type') == device_type]

# Database initialization for server startup
async def initialize_registry():
    """Initialize the device registry database"""
    try:
        from database import init_database
        await init_database()
        logger.info("Device registry database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize device registry database: {e}")
        return False

def init_registry_sync():
    """Sync wrapper for registry initialization"""
    return asyncio.run(initialize_registry())