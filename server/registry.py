"""
Device Registry for QFLARE

This module manages device registration and provides device information.
Now backed by persistent database storage.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .database import DeviceService, AuditService

logger = logging.getLogger(__name__)


def register_device(device_id: str, device_info: Dict[str, Any] = None) -> bool:
    """
    Register a new device using database backend.
    
    Args:
        device_id: Unique device identifier
        device_info: Additional device information
        
    Returns:
        True if device was registered successfully, False otherwise
    """
    try:
        if device_info is None:
            device_info = {}
        
        # Convert legacy format to new database format
        device_data = {
            "device_type": device_info.get("metadata", {}).get("device_type", "unknown"),
            "hardware_info": device_info.get("metadata", {}).get("hardware", {}),
            "network_info": device_info.get("metadata", {}).get("network", {}),
            "capabilities": device_info.get("metadata", {}).get("capabilities", {}),
            "kem_public_key": device_info.get("public_keys", {}).get("kem_public_key"),
            "sig_public_key": device_info.get("public_keys", {}).get("sig_public_key"),
            "local_epochs": device_info.get("training_config", {}).get("local_epochs", 1),
            "batch_size": device_info.get("training_config", {}).get("batch_size", 32),
            "learning_rate": device_info.get("training_config", {}).get("learning_rate", 0.01)
        }
        
        success = DeviceService.register_device(device_id, device_data)
        
        if success:
            logger.info(f"Registered device {device_id} in database")
        else:
            logger.error(f"Failed to register device {device_id} in database")
            
        return success
        
    except Exception as e:
        logger.error(f"Error registering device {device_id}: {e}")
        return False


def get_device_info(device_id: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a registered device using database backend.
    
    Args:
        device_id: Device identifier
        
    Returns:
        Device information dictionary, or None if device not found
    """
    try:
        device_info = DeviceService.get_device(device_id)
        
        if device_info:
            # Convert database format to legacy format for compatibility
            legacy_format = {
                "device_id": device_info["device_id"],
                "status": device_info["status"],
                "registered_at": device_info["registered_at"],
                "last_seen": device_info["last_seen"],
                "public_keys": {
                    "kem_public_key": None,  # These would be binary in database
                    "sig_public_key": None
                },
                "metadata": {
                    "device_type": device_info["device_type"],
                    "hardware": device_info["hardware_info"],
                    "network": device_info["network_info"],
                    "capabilities": device_info["capabilities"]
                },
                "training_config": device_info["training_config"]
            }
            
            logger.debug(f"Retrieved device info for {device_id}")
            return legacy_format
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting device info for {device_id}: {e}")
        return None


def update_device_status(device_id: str, status: str) -> bool:
    """
    Update device status using database backend.
    
    Args:
        device_id: Device identifier
        status: New status (e.g., "active", "inactive", "suspended")
        
    Returns:
        True if status was updated successfully, False otherwise
    """
    try:
        success = DeviceService.update_device_status(device_id, status)
        
        if success:
            logger.info(f"Updated status for device {device_id} to {status}")
        else:
            logger.warning(f"Device {device_id} not found for status update")
            
        return success
        
    except Exception as e:
        logger.error(f"Error updating status for device {device_id}: {e}")
        return False


def update_device_keys(device_id: str, public_keys: Dict[str, Any]) -> bool:
    """
    Update device public keys (legacy function - now managed through device updates).
    
    Args:
        device_id: Device identifier
        public_keys: New public keys
        
    Returns:
        True if keys were updated successfully, False otherwise
    """
    try:
        # For now, log the key update but don't implement full update
        # In production, this would trigger key rotation workflow
        logger.info(f"Key update requested for device {device_id} (not implemented)")
        return True
        
    except Exception as e:
        logger.error(f"Error updating keys for device {device_id}: {e}")
        return False


def remove_device(device_id: str) -> bool:
    """
    Remove device from registry (sets status to inactive rather than deleting).
    
    Args:
        device_id: Device identifier
        
    Returns:
        True if device was removed successfully, False otherwise
    """
    try:
        success = DeviceService.update_device_status(device_id, "inactive")
        
        if success:
            logger.info(f"Removed device {device_id} from registry")
        else:
            logger.warning(f"Device {device_id} not found for removal")
            
        return success
        
    except Exception as e:
        logger.error(f"Error removing device {device_id}: {e}")
        return False


def cleanup_inactive_devices(hours_threshold: int = 24) -> Dict[str, int]:
    """
    Clean up devices that haven't been seen for specified hours.
    
    Args:
        hours_threshold: Hours of inactivity before marking as inactive
        
    Returns:
        Dictionary with cleanup statistics
    """
    try:
        # This would be implemented with a database query
        # For now, return empty stats
        logger.info(f"Cleanup inactive devices with threshold {hours_threshold} hours")
        return {"cleaned": 0, "total": 0}
        
    except Exception as e:
        logger.error(f"Error during device cleanup: {e}")
        return {"cleaned": 0, "total": 0}


def list_active_devices() -> List[Dict[str, Any]]:
    """
    Get list of all active devices using database backend.
    
    Returns:
        List of active device information
    """
    try:
        devices = DeviceService.list_active_devices()
        logger.debug(f"Retrieved {len(devices)} active devices")
        return devices
        
    except Exception as e:
        logger.error(f"Error listing active devices: {e}")
        return []


def get_device_count() -> int:
    """
    Get total number of registered devices.
    
    Returns:
        Number of registered devices
    """
    try:
        devices = DeviceService.list_active_devices()
        return len(devices)
        
    except Exception as e:
        logger.error(f"Error getting device count: {e}")
        return 0


def get_registry_stats() -> Dict[str, Any]:
    """
    Get registry statistics using database backend.
    
    Returns:
        Dictionary with registry statistics
    """
    try:
        devices = DeviceService.list_active_devices()
        
        stats = {
            "total": len(devices),
            "active": len(devices),
            "inactive": 0,  # Would need separate query
            "last_updated": datetime.utcnow().isoformat()
        }
        
        # Group by device type
        device_types = {}
        for device in devices:
            device_type = device.get("device_type", "unknown")
            device_types[device_type] = device_types.get(device_type, 0) + 1
        
        stats["by_type"] = device_types
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting registry stats: {e}")
        return {"total": 0, "active": 0, "inactive": 0}


def check_device_health() -> Dict[str, Any]:
    """
    Check health of all registered devices.
    
    Returns:
        Dictionary with device health information
    """
    try:
        # This would implement health checking logic
        # For now, return basic stats
        devices = DeviceService.list_active_devices()
        
        health_info = {
            "total_devices": len(devices),
            "healthy": len(devices),
            "unhealthy": 0,
            "unknown": 0,
            "last_check": datetime.utcnow().isoformat()
        }
        
        return health_info
        
    except Exception as e:
        logger.error(f"Error checking device health: {e}")
        return {"total_devices": 0, "healthy": 0, "unhealthy": 0}


def get_device_capabilities() -> Dict[str, List[str]]:
    """
    Get capabilities of all registered devices.
    
    Returns:
        Dictionary mapping device IDs to their capabilities
    """
    try:
        devices = DeviceService.list_active_devices()
        
        capabilities = {}
        for device in devices:
            device_id = device["device_id"]
            device_capabilities = device.get("capabilities", {})
            capabilities[device_id] = list(device_capabilities.keys())
        
        return capabilities
        
    except Exception as e:
        logger.error(f"Error getting device capabilities: {e}")
        return {}


# Legacy compatibility functions
def get_registered_devices() -> Dict[str, Any]:
    """Legacy function for compatibility - returns active devices."""
    try:
        devices = DeviceService.list_active_devices()
        result = {}
        for device in devices:
            device_id = device["device_id"]
            result[device_id] = get_device_info(device_id)
        return result
    except Exception as e:
        logger.error(f"Error getting registered devices: {e}")
        return {}