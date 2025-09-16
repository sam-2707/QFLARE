"""
Production Intel SGX Integration

This module provides real Intel SGX hardware integration for QFLARE federated learning,
replacing simulation mode with actual hardware-based Trusted Execution Environment.

Features:
- Real SGX enclave loading and management
- Hardware-based secure aggregation
- Remote attestation with Intel Attestation Service (IAS)
- Data sealing and unsealing
- Byzantine fault tolerance within TEE
"""

import os
import sys
import json
import logging
import hashlib
import time
import threading
import ctypes
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess
import struct

logger = logging.getLogger(__name__)

# Try to import Intel SGX SDK Python bindings
try:
    # Real SGX SDK imports
    import sgx_python as sgx
    from sgx_python import (
        SGXEnclave, SGXError, SGXErrorCode,
        create_enclave, destroy_enclave,
        SGX_SUCCESS, SGX_ERROR_INVALID_PARAMETER,
        SGX_ERROR_OUT_OF_MEMORY, SGX_ERROR_ENCLAVE_LOST
    )
    from sgx_python.attestation import (
        generate_quote, verify_quote, 
        AttestationService, IASClient
    )
    from sgx_python.sealing import seal_data, unseal_data
    SGX_HARDWARE_AVAILABLE = True
    logger.info("Intel SGX SDK Python bindings loaded successfully")
except ImportError as e:
    logger.warning(f"SGX SDK not available: {e}. Using simulation mode.")
    SGX_HARDWARE_AVAILABLE = False
    
    # Mock implementations for development
    class SGXEnclave:
        def __init__(self, *args, **kwargs):
            self.enclave_id = 12345
    
    class SGXError(Exception):
        pass
    
    SGX_SUCCESS = 0


@dataclass
class SGXHardwareConfig:
    """Enhanced SGX hardware configuration."""
    enclave_file: str = "enclaves/qflare_enclave.signed.so"
    token_file: str = "enclaves/enclave.token"
    debug_mode: bool = False
    
    # Attestation configuration
    attestation_enabled: bool = True
    ias_spid: Optional[str] = None
    ias_primary_key: Optional[str] = None
    ias_secondary_key: Optional[str] = None
    linkable_attestation: bool = True
    
    # Performance tuning
    enclave_heap_size: int = 64 * 1024 * 1024  # 64MB
    enclave_stack_size: int = 1024 * 1024      # 1MB
    max_threads: int = 16
    
    # Security parameters
    poison_threshold: float = 0.8
    byzantine_tolerance: float = 0.3
    max_model_size: int = 100 * 1024 * 1024
    
    # Sealing configuration
    enable_sealing: bool = True
    sealed_data_dir: str = "enclaves/sealed"


class SGXHardwareEnclave:
    """Production SGX enclave with real hardware integration."""
    
    def __init__(self, config: SGXHardwareConfig):
        self.config = config
        self.enclave = None
        self.enclave_id = None
        self.attestation_service = None
        self.sealed_keys = {}
        self.aggregation_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'rejected_updates': 0,
            'attestation_verifications': 0
        }
        self._lock = threading.RLock()
        self._initialize_hardware_enclave()
    
    def _initialize_hardware_enclave(self) -> None:
        """Initialize real SGX hardware enclave."""
        try:
            if not SGX_HARDWARE_AVAILABLE:
                raise SGXError("SGX hardware not available")
            
            # Check if enclave file exists
            enclave_path = Path(self.config.enclave_file)
            if not enclave_path.exists():
                raise SGXError(f"Enclave file not found: {enclave_path}")
            
            # Load launch token if available
            token_path = Path(self.config.token_file)
            launch_token = None
            if token_path.exists():
                with open(token_path, 'rb') as f:
                    launch_token = f.read()
            
            # Create the enclave
            logger.info(f"Creating SGX enclave from {enclave_path}")
            self.enclave, self.enclave_id = create_enclave(
                str(enclave_path),
                debug=self.config.debug_mode,
                launch_token=launch_token
            )
            
            # Save updated launch token
            if launch_token and token_path.parent.exists():
                with open(token_path, 'wb') as f:
                    f.write(launch_token)
            
            # Initialize enclave state
            self._ecall_initialize_enclave()
            
            # Initialize attestation service
            if self.config.attestation_enabled:
                self._initialize_attestation_service()
            
            # Create sealed data directory
            if self.config.enable_sealing:
                Path(self.config.sealed_data_dir).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"SGX enclave initialized successfully with ID: {self.enclave_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SGX hardware enclave: {e}")
            raise SGXError(f"Hardware enclave initialization failed: {e}")
    
    def _initialize_attestation_service(self) -> None:
        """Initialize Intel Attestation Service client."""
        try:
            if not self.config.ias_spid or not self.config.ias_primary_key:
                logger.warning("IAS credentials not configured, attestation disabled")
                return
            
            self.attestation_service = IASClient(
                spid=self.config.ias_spid,
                primary_key=self.config.ias_primary_key,
                secondary_key=self.config.ias_secondary_key,
                linkable=self.config.linkable_attestation
            )
            
            logger.info("Intel Attestation Service client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize attestation service: {e}")
            self.attestation_service = None
    
    def _ecall_initialize_enclave(self) -> None:
        """Call enclave initialization function."""
        if not SGX_HARDWARE_AVAILABLE:
            logger.info("Mock ECALL: initialize_enclave")
            return
        
        try:
            # Convert float to ctypes
            poison_threshold = ctypes.c_float(self.config.poison_threshold)
            byzantine_tolerance = ctypes.c_float(self.config.byzantine_tolerance)
            max_model_size = ctypes.c_uint32(self.config.max_model_size)
            
            # Call enclave function
            ret = sgx.ecall_initialize_enclave(
                self.enclave_id,
                poison_threshold,
                byzantine_tolerance,
                max_model_size
            )
            
            if ret != SGX_SUCCESS:
                raise SGXError(f"Enclave initialization failed with code: {ret}")
            
            logger.info("Enclave initialization ECALL completed successfully")
            
        except Exception as e:
            logger.error(f"ECALL initialize_enclave failed: {e}")
            raise SGXError(f"Enclave initialization ECALL failed: {e}")
    
    def generate_attestation_quote(self, report_data: bytes = None) -> Tuple[bytes, bytes]:
        """Generate SGX quote for remote attestation."""
        if not SGX_HARDWARE_AVAILABLE:
            # Mock quote for simulation
            mock_quote = b"MOCK_SGX_QUOTE_" + str(self.enclave_id).encode()
            mock_report = b"MOCK_SGX_REPORT_" + str(int(time.time())).encode()
            return mock_report, mock_quote
        
        try:
            with self._lock:
                # Prepare report data (64 bytes)
                if report_data is None:
                    report_data = hashlib.sha256(f"qflare_attestation_{time.time()}".encode()).digest()[:32]
                    report_data = report_data.ljust(64, b'\x00')
                
                # Generate quote
                report, quote = generate_quote(self.enclave_id, report_data)
                
                self.aggregation_metrics['attestation_verifications'] += 1
                logger.info("SGX attestation quote generated successfully")
                
                return report, quote
                
        except Exception as e:
            logger.error(f"Failed to generate attestation quote: {e}")
            raise SGXError(f"Attestation quote generation failed: {e}")
    
    def verify_remote_attestation(self, quote: bytes, expected_mr_enclave: bytes = None) -> Dict:
        """Verify remote attestation quote using IAS."""
        if not self.attestation_service:
            logger.warning("Attestation service not available, skipping verification")
            return {"status": "skipped", "reason": "no_attestation_service"}
        
        try:
            # Verify quote with Intel Attestation Service
            verification_result = self.attestation_service.verify_quote(quote)
            
            # Additional checks
            if expected_mr_enclave:
                enclave_hash = verification_result.get('mr_enclave')
                if enclave_hash != expected_mr_enclave.hex():
                    logger.error("Enclave measurement mismatch")
                    return {"status": "failed", "reason": "measurement_mismatch"}
            
            logger.info("Remote attestation verification completed successfully")
            return {"status": "verified", "details": verification_result}
            
        except Exception as e:
            logger.error(f"Remote attestation verification failed: {e}")
            return {"status": "failed", "reason": str(e)}
    
    def secure_aggregate(self, model_updates: List[Dict], global_model: bytes = None) -> Dict:
        """Perform secure aggregation within SGX enclave."""
        if not SGX_HARDWARE_AVAILABLE:
            return self._simulate_hardware_aggregation(model_updates, global_model)
        
        try:
            with self._lock:
                start_time = time.time()
                
                # Prepare update data for enclave
                serialized_updates = self._serialize_updates(model_updates)
                
                # Prepare buffers for results
                max_result_size = len(serialized_updates) + 1024
                result_buffer = (ctypes.c_ubyte * max_result_size)()
                result_size = ctypes.c_size_t(max_result_size)
                
                metadata_buffer = (ctypes.c_ubyte * 4096)()
                metadata_size = ctypes.c_size_t(4096)
                
                # Call secure aggregation ECALL
                ret = sgx.ecall_secure_aggregate(
                    self.enclave_id,
                    serialized_updates,
                    len(serialized_updates),
                    len(model_updates),
                    global_model or b'',
                    len(global_model) if global_model else 0,
                    result_buffer,
                    ctypes.byref(result_size),
                    metadata_buffer,
                    ctypes.byref(metadata_size)
                )
                
                if ret != SGX_SUCCESS:
                    raise SGXError(f"Secure aggregation ECALL failed with code: {ret}")
                
                # Extract results
                aggregated_weights = bytes(result_buffer[:result_size.value])
                metadata = json.loads(bytes(metadata_buffer[:metadata_size.value]).decode())
                
                # Update metrics
                operation_time = time.time() - start_time
                self.aggregation_metrics['total_operations'] += 1
                self.aggregation_metrics['successful_operations'] += 1
                self.aggregation_metrics['rejected_updates'] += metadata.get('num_rejected', 0)
                
                logger.info(f"Hardware secure aggregation completed in {operation_time:.2f}s")
                
                return {
                    'aggregated_weights': aggregated_weights,
                    'num_processed': metadata.get('num_processed', 0),
                    'num_rejected': metadata.get('num_rejected', 0),
                    'rejected_devices': metadata.get('rejected_devices', []),
                    'aggregation_hash': hashlib.sha256(aggregated_weights).hexdigest(),
                    'timestamp': time.time(),
                    'byzantine_detected': metadata.get('byzantine_detected', []),
                    'operation_time': operation_time
                }
                
        except Exception as e:
            logger.error(f"Hardware secure aggregation failed: {e}")
            self.aggregation_metrics['total_operations'] += 1
            raise SGXError(f"Secure aggregation failed: {e}")
    
    def _serialize_updates(self, model_updates: List[Dict]) -> bytes:
        """Serialize model updates for enclave processing."""
        try:
            # Create a compact binary format for enclave
            serialized_data = b''
            
            for update in model_updates:
                device_id = update['device_id'].encode()[:63].ljust(64, b'\x00')
                weights = update['encrypted_weights']
                timestamp = struct.pack('<Q', int(update.get('timestamp', time.time())))
                weights_size = struct.pack('<I', len(weights))
                
                # Pack: device_id(64) + timestamp(8) + weights_size(4) + weights
                update_data = device_id + timestamp + weights_size + weights
                serialized_data += update_data
            
            return serialized_data
            
        except Exception as e:
            logger.error(f"Failed to serialize updates: {e}")
            raise SGXError(f"Update serialization failed: {e}")
    
    def _simulate_hardware_aggregation(self, model_updates: List[Dict], global_model: bytes) -> Dict:
        """Simulate hardware aggregation for development."""
        logger.info("Performing simulated hardware aggregation")
        
        # Enhanced simulation with more realistic processing
        valid_updates = []
        rejected_updates = []
        
        for update in model_updates:
            # Simulate Byzantine detection with more sophisticated criteria
            device_id = update['device_id']
            weights = update['encrypted_weights']
            
            # Simulate various rejection criteria
            if len(weights) == 0:
                rejected_updates.append(device_id)
                continue
            
            # Simulate poison detection
            weight_hash = hashlib.sha256(weights).hexdigest()
            if int(weight_hash[:8], 16) % 100 < 10:  # 10% rejection rate
                rejected_updates.append(device_id)
                continue
            
            valid_updates.append(update)
        
        # Simulate aggregation
        if valid_updates:
            # Create mock aggregated weights
            combined_weights = b''
            for update in valid_updates:
                combined_weights += update['encrypted_weights']
            
            aggregated_weights = hashlib.sha256(combined_weights).digest() * 32  # 1KB result
        else:
            aggregated_weights = b''
        
        return {
            'aggregated_weights': aggregated_weights,
            'num_processed': len(valid_updates),
            'num_rejected': len(rejected_updates),
            'rejected_devices': rejected_updates,
            'aggregation_hash': hashlib.sha256(aggregated_weights).hexdigest(),
            'timestamp': time.time(),
            'byzantine_detected': rejected_updates,
            'operation_time': 0.1  # Simulate processing time
        }
    
    def seal_sensitive_data(self, data: bytes, key_id: str) -> bytes:
        """Seal data using SGX sealing capabilities."""
        if not SGX_HARDWARE_AVAILABLE:
            # Mock sealing - just base64 encode for simulation
            import base64
            return base64.b64encode(data)
        
        try:
            # Use SGX sealing to encrypt data with enclave-specific key
            sealed_data = seal_data(self.enclave_id, data)
            
            # Store sealed data
            if self.config.enable_sealing:
                sealed_path = Path(self.config.sealed_data_dir) / f"{key_id}.sealed"
                with open(sealed_path, 'wb') as f:
                    f.write(sealed_data)
            
            self.sealed_keys[key_id] = sealed_data
            logger.info(f"Data sealed successfully with key ID: {key_id}")
            
            return sealed_data
            
        except Exception as e:
            logger.error(f"Data sealing failed: {e}")
            raise SGXError(f"Sealing failed: {e}")
    
    def unseal_sensitive_data(self, sealed_data: bytes, key_id: str = None) -> bytes:
        """Unseal data using SGX unsealing capabilities."""
        if not SGX_HARDWARE_AVAILABLE:
            # Mock unsealing
            import base64
            return base64.b64decode(sealed_data)
        
        try:
            # Use SGX unsealing to decrypt data
            unsealed_data = unseal_data(self.enclave_id, sealed_data)
            
            logger.info(f"Data unsealed successfully for key ID: {key_id}")
            return unsealed_data
            
        except Exception as e:
            logger.error(f"Data unsealing failed: {e}")
            raise SGXError(f"Unsealing failed: {e}")
    
    def get_enclave_status(self) -> Dict:
        """Get comprehensive enclave status and metrics."""
        return {
            'enclave_type': 'sgx_hardware_enclave',
            'hardware_available': SGX_HARDWARE_AVAILABLE,
            'enclave_id': self.enclave_id,
            'attestation_enabled': self.config.attestation_enabled,
            'sealing_enabled': self.config.enable_sealing,
            'metrics': self.aggregation_metrics.copy(),
            'config': asdict(self.config)
        }
    
    def destroy_enclave(self) -> None:
        """Safely destroy the enclave and cleanup resources."""
        try:
            if self.enclave and SGX_HARDWARE_AVAILABLE:
                destroy_enclave(self.enclave_id)
                logger.info("SGX enclave destroyed successfully")
            
            self.enclave = None
            self.enclave_id = None
            
        except Exception as e:
            logger.error(f"Failed to destroy enclave: {e}")


def create_hardware_sgx_enclave(config: SGXHardwareConfig = None) -> SGXHardwareEnclave:
    """Factory function to create hardware SGX enclave."""
    if config is None:
        config = SGXHardwareConfig()
    
    return SGXHardwareEnclave(config)


def is_sgx_hardware_available() -> bool:
    """Check if SGX hardware is available on this system."""
    return SGX_HARDWARE_AVAILABLE


def get_sgx_capabilities() -> Dict:
    """Get SGX hardware capabilities and status."""
    capabilities = {
        'hardware_available': SGX_HARDWARE_AVAILABLE,
        'sgx_enabled': False,
        'memory_encryption': False,
        'attestation_support': False
    }
    
    if SGX_HARDWARE_AVAILABLE:
        try:
            # Query SGX capabilities
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            
            capabilities.update({
                'sgx_enabled': 'sgx' in cpu_info.get('flags', []),
                'memory_encryption': True,
                'attestation_support': True,
                'cpu_model': cpu_info.get('brand_raw', 'Unknown')
            })
            
        except Exception as e:
            logger.warning(f"Failed to query SGX capabilities: {e}")
    
    return capabilities