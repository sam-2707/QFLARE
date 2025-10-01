"""
QFLARE Security Module for Federated Learning

This module provides security validation and monitoring for FL operations.
"""

import logging
import hashlib
import hmac
import time
from typing import Dict, Any, Optional, List, Tuple
import json

logger = logging.getLogger(__name__)

class ModelValidator:
    """
    Validates model updates for security and integrity.
    """
    
    def __init__(self, 
                 max_model_size: int = 50 * 1024 * 1024,  # 50MB
                 allowed_formats: List[str] = None):
        """
        Initialize model validator.
        
        Args:
            max_model_size: Maximum allowed model size in bytes
            allowed_formats: List of allowed model formats
        """
        self.max_model_size = max_model_size
        self.allowed_formats = allowed_formats or ['.pt', '.pth', '.pkl']
        
        logger.info(f"ModelValidator initialized (max_size={max_model_size//1024//1024}MB)")
    
    def validate_model_update(self, 
                            model_data: bytes, 
                            device_id: str,
                            expected_checksum: Optional[str] = None) -> bool:
        """
        Validate a model update for security and integrity.
        
        Args:
            model_data: Serialized model data
            device_id: ID of submitting device
            expected_checksum: Expected checksum (optional)
            
        Returns:
            True if model is valid, False otherwise
        """
        try:
            # Size validation
            if len(model_data) > self.max_model_size:
                logger.warning(f"Model from {device_id} rejected: size {len(model_data)} > {self.max_model_size}")
                return False
            
            if len(model_data) == 0:
                logger.warning(f"Model from {device_id} rejected: empty data")
                return False
            
            # Checksum validation
            if expected_checksum:
                actual_checksum = hashlib.sha256(model_data).hexdigest()
                if actual_checksum != expected_checksum:
                    logger.warning(f"Model from {device_id} rejected: checksum mismatch")
                    return False
            
            # Basic format validation
            if not self._validate_model_format(model_data):
                logger.warning(f"Model from {device_id} rejected: invalid format")
                return False
            
            # Malicious content detection
            if not self._scan_for_malicious_content(model_data):
                logger.warning(f"Model from {device_id} rejected: potential malicious content")
                return False
            
            logger.debug(f"Model from {device_id} passed validation")
            return True
            
        except Exception as e:
            logger.error(f"Error validating model from {device_id}: {e}")
            return False
    
    def _validate_model_format(self, model_data: bytes) -> bool:
        """Basic format validation for model data."""
        try:
            # Check for common serialization headers
            if model_data.startswith(b'PK'):  # ZIP-based formats (PyTorch)
                return True
            elif model_data.startswith(b'\x80\x03'):  # Pickle protocol 3
                return True
            elif model_data.startswith(b'\x80\x04'):  # Pickle protocol 4
                return True
            elif model_data.startswith(b'\x80\x05'):  # Pickle protocol 5
                return True
            else:
                # Try to deserialize as a test
                import pickle
                import io
                try:
                    buffer = io.BytesIO(model_data)
                    pickle.load(buffer)
                    return True
                except:
                    pass
                
                # Try PyTorch if available
                try:
                    import torch
                    buffer = io.BytesIO(model_data)
                    torch.load(buffer, map_location='cpu')
                    return True
                except:
                    pass
            
            return False
            
        except Exception as e:
            logger.warning(f"Format validation error: {e}")
            return False
    
    def _scan_for_malicious_content(self, model_data: bytes) -> bool:
        """Scan for potentially malicious content in model data."""
        try:
            # Convert to string for text-based scanning (limited effectiveness)
            data_str = str(model_data)
            
            # List of suspicious patterns
            suspicious_patterns = [
                'exec(',
                'eval(',
                'import os',
                'import subprocess',
                'import sys',
                '__import__',
                'open(',
                'file(',
                'input(',
                'raw_input(',
                'execfile(',
                'compile(',
            ]
            
            # Check for suspicious patterns
            for pattern in suspicious_patterns:
                if pattern in data_str:
                    logger.warning(f"Suspicious pattern found: {pattern}")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Malicious content scan error: {e}")
            return True  # Default to safe if scan fails


class SecurityMonitor:
    """
    Monitors FL system for security threats and anomalies.
    """
    
    def __init__(self, 
                 alert_threshold: int = 5,
                 monitoring_window: int = 3600):  # 1 hour
        """
        Initialize security monitor.
        
        Args:
            alert_threshold: Number of suspicious events to trigger alert
            monitoring_window: Time window for monitoring in seconds
        """
        self.alert_threshold = alert_threshold
        self.monitoring_window = monitoring_window
        
        # Event tracking
        self.security_events = []
        self.device_reputation = {}
        
        logger.info(f"SecurityMonitor initialized (threshold={alert_threshold})")
    
    def log_security_event(self, 
                          event_type: str,
                          device_id: str,
                          details: Dict[str, Any]):
        """
        Log a security event.
        
        Args:
            event_type: Type of security event
            device_id: ID of device involved
            details: Additional event details
        """
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'device_id': device_id,
            'details': details
        }
        
        self.security_events.append(event)
        
        # Update device reputation
        if device_id not in self.device_reputation:
            self.device_reputation[device_id] = {
                'suspicious_events': 0,
                'total_events': 0,
                'reputation_score': 1.0
            }
        
        self.device_reputation[device_id]['total_events'] += 1
        
        if event_type in ['model_validation_failed', 'suspicious_activity', 'malicious_content']:
            self.device_reputation[device_id]['suspicious_events'] += 1
        
        # Update reputation score
        device_rep = self.device_reputation[device_id]
        if device_rep['total_events'] > 0:
            device_rep['reputation_score'] = 1.0 - (device_rep['suspicious_events'] / device_rep['total_events'])
        
        # Clean old events
        self._cleanup_old_events()
        
        # Check for alerts
        self._check_security_alerts()
        
        logger.info(f"Security event logged: {event_type} from {device_id}")
    
    def get_device_reputation(self, device_id: str) -> float:
        """Get reputation score for a device (0.0 = bad, 1.0 = good)."""
        if device_id not in self.device_reputation:
            return 1.0
        return self.device_reputation[device_id]['reputation_score']
    
    def is_device_trusted(self, device_id: str, trust_threshold: float = 0.7) -> bool:
        """Check if a device is trusted based on reputation."""
        reputation = self.get_device_reputation(device_id)
        return reputation >= trust_threshold
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security monitoring summary."""
        current_time = time.time()
        recent_events = [
            event for event in self.security_events
            if current_time - event['timestamp'] <= self.monitoring_window
        ]
        
        event_types = {}
        for event in recent_events:
            event_type = event['event_type']
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return {
            'total_events': len(self.security_events),
            'recent_events': len(recent_events),
            'event_types': event_types,
            'monitored_devices': len(self.device_reputation),
            'low_reputation_devices': len([
                device_id for device_id, rep in self.device_reputation.items()
                if rep['reputation_score'] < 0.5
            ]),
            'monitoring_window_hours': self.monitoring_window / 3600
        }
    
    def _cleanup_old_events(self):
        """Remove events older than monitoring window."""
        current_time = time.time()
        cutoff_time = current_time - self.monitoring_window
        
        self.security_events = [
            event for event in self.security_events
            if event['timestamp'] > cutoff_time
        ]
    
    def _check_security_alerts(self):
        """Check if security alerts should be triggered."""
        current_time = time.time()
        recent_events = [
            event for event in self.security_events
            if current_time - event['timestamp'] <= self.monitoring_window
        ]
        
        # Alert if too many suspicious events
        suspicious_events = [
            event for event in recent_events
            if event['event_type'] in ['model_validation_failed', 'suspicious_activity', 'malicious_content']
        ]
        
        if len(suspicious_events) >= self.alert_threshold:
            logger.warning(f"SECURITY ALERT: {len(suspicious_events)} suspicious events in last {self.monitoring_window/3600:.1f} hours")


class DeviceAuthenticator:
    """
    Handles device authentication for FL participation.
    """
    
    def __init__(self, secret_key: str = None):
        """
        Initialize device authenticator.
        
        Args:
            secret_key: Secret key for HMAC authentication
        """
        self.secret_key = secret_key or "qflare_default_secret_key_2023"
        logger.info("DeviceAuthenticator initialized")
    
    def generate_session_token(self, device_id: str, validity_period: int = 3600) -> str:
        """
        Generate a session token for a device.
        
        Args:
            device_id: Device identifier
            validity_period: Token validity in seconds
            
        Returns:
            Session token string
        """
        expiry_time = int(time.time()) + validity_period
        message = f"{device_id}:{expiry_time}"
        
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        token = f"{message}:{signature}"
        return token
    
    def validate_session_token(self, token: str, device_id: str) -> bool:
        """
        Validate a session token.
        
        Args:
            token: Session token to validate
            device_id: Expected device ID
            
        Returns:
            True if token is valid, False otherwise
        """
        try:
            parts = token.split(':')
            if len(parts) != 3:
                return False
            
            token_device_id, expiry_time_str, signature = parts
            
            # Check device ID match
            if token_device_id != device_id:
                return False
            
            # Check expiry
            expiry_time = int(expiry_time_str)
            if time.time() > expiry_time:
                return False
            
            # Verify signature
            message = f"{token_device_id}:{expiry_time_str}"
            expected_signature = hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.warning(f"Token validation error: {e}")
            return False
    
    def get_device_challenge(self, device_id: str) -> str:
        """
        Generate a challenge for device authentication.
        
        Args:
            device_id: Device identifier
            
        Returns:
            Challenge string
        """
        timestamp = int(time.time())
        nonce = hashlib.sha256(f"{device_id}:{timestamp}:{self.secret_key}".encode()).hexdigest()[:16]
        
        challenge = f"{timestamp}:{nonce}"
        return challenge
    
    def validate_challenge_response(self, 
                                  device_id: str,
                                  challenge: str,
                                  response: str) -> bool:
        """
        Validate a challenge response.
        
        Args:
            device_id: Device identifier
            challenge: Original challenge
            response: Device response
            
        Returns:
            True if response is valid, False otherwise
        """
        try:
            # Extract challenge components
            timestamp_str, nonce = challenge.split(':')
            timestamp = int(timestamp_str)
            
            # Check challenge age (valid for 5 minutes)
            if time.time() - timestamp > 300:
                return False
            
            # Generate expected response
            expected_response = hmac.new(
                self.secret_key.encode(),
                f"{device_id}:{challenge}".encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(response, expected_response)
            
        except Exception as e:
            logger.warning(f"Challenge response validation error: {e}")
            return False