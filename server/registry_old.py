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
    Get information about a registered device.
    
    Args:
        device_id: Device identifier
        
    Returns:
        Device information dictionary, or None if device not found
    """
    try:
        device_info = registered_devices.get(device_id)
        if device_info:
            # Update last seen timestamp
            device_info["last_seen"] = time.time()
            registered_devices[device_id] = device_info
            
        return device_info
        
    except Exception as e:
        logger.error(f"Error getting device info for {device_id}: {e}")
        return None


def update_device_status(device_id: str, status: str) -> bool:
    """
    Update device status.
    
    Args:
        device_id: Device identifier
        status: New status (e.g., "active", "inactive", "suspended")
        
    Returns:
        True if status was updated successfully, False otherwise
    """
    try:
        if device_id not in registered_devices:
            logger.warning(f"Device {device_id} not found for status update")
            return False
        
        registered_devices[device_id]["status"] = status
        registered_devices[device_id]["last_seen"] = time.time()
        
        logger.info(f"Updated status for device {device_id} to {status}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating status for device {device_id}: {e}")
        return False


def update_device_keys(device_id: str, public_keys: Dict[str, str]) -> bool:
    """
    Update device public keys.
    
    Args:
        device_id: Device identifier
        public_keys: Dictionary of public keys
        
    Returns:
        True if keys were updated successfully, False otherwise
    """
    try:
        if device_id not in registered_devices:
            logger.warning(f"Device {device_id} not found for key update")
            return False
        
        registered_devices[device_id]["public_keys"] = public_keys
        registered_devices[device_id]["last_seen"] = time.time()
        
        logger.info(f"Updated keys for device {device_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating keys for device {device_id}: {e}")
        return False


def remove_device(device_id: str) -> bool:
    """
    Remove a device from the registry.
    
    Args:
        device_id: Device identifier
        
    Returns:
        True if device was removed successfully, False otherwise
    """
    try:
        if device_id in registered_devices:
            del registered_devices[device_id]
            logger.info(f"Removed device {device_id} from registry")
            return True
        else:
            logger.warning(f"Device {device_id} not found for removal")
            return False
            
    except Exception as e:
        logger.error(f"Error removing device {device_id}: {e}")
        return False


def get_registered_devices() -> Dict[str, Dict[str, Any]]:
    """
    Get all registered devices.
    
    Returns:
        Dictionary of all registered devices
    """
    try:
        # Update last seen timestamps for active devices
        current_time = time.time()
        for device_id, device_info in registered_devices.items():
            if device_info.get("status") == "active":
                device_info["last_seen"] = current_time
                registered_devices[device_id] = device_info
        
        return registered_devices.copy()
        
    except Exception as e:
        logger.error(f"Error getting registered devices: {e}")
        return {}


def get_active_devices() -> List[Dict[str, Any]]:
    """
    Get list of active devices.
    
    Returns:
        List of active device information
    """
    try:
        active_devices = []
        current_time = time.time()
        
        for device_id, device_info in registered_devices.items():
            if device_info.get("status") == "active":
                device_info["last_seen"] = current_time
                registered_devices[device_id] = device_info
                active_devices.append(device_info)
        
        return active_devices
        
    except Exception as e:
        logger.error(f"Error getting active devices: {e}")
        return []


def get_device_count() -> Dict[str, int]:
    """
    Get device count statistics.
    
    Returns:
        Dictionary with device counts by status
    """
    try:
        counts = {
            "total": len(registered_devices),
            "active": 0,
            "inactive": 0,
            "suspended": 0
        }
        
        for device_info in registered_devices.values():
            status = device_info.get("status", "unknown")
            if status in counts:
                counts[status] += 1
        
        return counts
        
    except Exception as e:
        logger.error(f"Error getting device count: {e}")
        return {"total": 0, "active": 0, "inactive": 0, "suspended": 0}


def cleanup_inactive_devices(max_inactive_days: int = 30) -> int:
    """
    Remove devices that have been inactive for too long.
    
    Args:
        max_inactive_days: Maximum number of days of inactivity
        
    Returns:
        Number of devices removed
    """
    try:
        current_time = time.time()
        max_inactive_seconds = max_inactive_days * 24 * 3600
        removed_count = 0
        
        devices_to_remove = []
        
        for device_id, device_info in registered_devices.items():
            last_seen = device_info.get("last_seen", 0)
            if current_time - last_seen > max_inactive_seconds:
                devices_to_remove.append(device_id)
        
        for device_id in devices_to_remove:
            if remove_device(device_id):
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} inactive devices")
        
        return removed_count
        
    except Exception as e:
        logger.error(f"Error cleaning up inactive devices: {e}")
        return 0


def get_device_statistics() -> Dict[str, Any]:
    """
    Get comprehensive device statistics.
    
    Returns:
        Dictionary with device statistics
    """
    try:
        current_time = time.time()
        stats = {
            "total_devices": len(registered_devices),
            "active_devices": 0,
            "inactive_devices": 0,
            "suspended_devices": 0,
            "recent_activity": 0,  # Devices active in last 24 hours
            "avg_last_seen": 0,
            "oldest_device": None,
            "newest_device": None
        }
        
        last_seen_times = []
        oldest_time = current_time
        newest_time = 0
        
        for device_info in registered_devices.values():
            status = device_info.get("status", "unknown")
            if status in stats:
                stats[f"{status}_devices"] += 1
            
            last_seen = device_info.get("last_seen", 0)
            last_seen_times.append(last_seen)
            
            # Check recent activity (last 24 hours)
            if current_time - last_seen < 86400:  # 24 hours
                stats["recent_activity"] += 1
            
            # Track oldest and newest devices
            if last_seen < oldest_time:
                oldest_time = last_seen
                stats["oldest_device"] = device_info.get("device_id")
            
            if last_seen > newest_time:
                newest_time = last_seen
                stats["newest_device"] = device_info.get("device_id")
        
        if last_seen_times:
            stats["avg_last_seen"] = sum(last_seen_times) / len(last_seen_times)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting device statistics: {e}")
        return {
            "total_devices": 0,
            "active_devices": 0,
            "inactive_devices": 0,
            "suspended_devices": 0,
            "recent_activity": 0,
            "avg_last_seen": 0,
            "oldest_device": None,
            "newest_device": None
        }


# Legacy function for backward compatibility
def store_key(device_id: str, qkey: str) -> bool:
    """
    Legacy key storage function (deprecated).
    
    This function is kept for backward compatibility but should not be used
    in new code. Use the secure enrollment mechanism instead.
    """
    logger.warning("Legacy key storage called - use secure enrollment instead")
    return register_device(device_id, {"legacy_qkey": qkey})


def get_key(device_id: str) -> Optional[str]:
    """
    Legacy key retrieval function (deprecated).
    
    This function is kept for backward compatibility but should not be used
    in new code. Use the secure key management instead.
    """
    logger.warning("Legacy key retrieval called - use secure key management instead")
    device_info = get_device_info(device_id)
    if device_info:
        return device_info.get("legacy_qkey")
    return None


def revoke_key(device_id: str) -> bool:
    """
    Legacy key revocation function (deprecated).
    
    This function is kept for backward compatibility but should not be used
    in new code. Use the secure key management instead.
    """
    logger.warning("Legacy key revocation called - use secure key management instead")
    return update_device_status(device_id, "suspended") 