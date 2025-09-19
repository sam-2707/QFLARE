"""
Unified Trusted Execution Environment (TEE) Manager

This module provides a unified interface for managing different TEE implementations
including Intel SGX, AMD SEV, and mock enclaves. It automatically detects available
hardware and provides a consistent API for secure federated learning operations.

Features:
- Automatic hardware detection and TEE selection
- Unified interface for SGX, SEV, and mock enclaves
- Load balancing across multiple TEE instances
- Fallback mechanisms and error handling
- Performance monitoring and optimization
"""

import os
import sys
import json
import logging
import platform
import subprocess
import threading
import time
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import cpuinfo
from .mock_enclave import MockSecureEnclave, ModelUpdate
from .sgx_enclave import (
    SGXSecureEnclave, SGXConfig, SecureModelUpdate, 
    AggregationResult, EnclaveError, EnclaveStatus,
    get_sgx_enclave, destroy_global_enclave
)
from .sev_enclave import (
    SEVSecureEnclave, SEVConfig,
    get_sev_enclave, destroy_global_sev_enclave
)

logger = logging.getLogger(__name__)


class TEEType(Enum):
    """Supported TEE types."""
    MOCK = "mock"
    INTEL_SGX = "intel_sgx"
    AMD_SEV = "amd_sev"
    ARM_TRUSTZONE = "arm_trustzone"  # Future support
    RISC_V_KEYSTONE = "riscv_keystone"  # Future support


class TEECapability(Enum):
    """TEE capability flags."""
    MEMORY_ENCRYPTION = "memory_encryption"
    ATTESTATION = "attestation"
    SEALING = "sealing"
    SIDE_CHANNEL_RESISTANCE = "side_channel_resistance"
    BYZANTINE_TOLERANCE = "byzantine_tolerance"
    HIGH_PERFORMANCE = "high_performance"


@dataclass
class TEEInfo:
    """Information about a TEE implementation."""
    tee_type: TEEType
    available: bool
    capabilities: List[TEECapability]
    performance_score: float
    hardware_version: Optional[str] = None
    software_version: Optional[str] = None
    max_memory: Optional[int] = None
    error_message: Optional[str] = None


@dataclass
class UnifiedTEEConfig:
    """Unified configuration for TEE management."""
    preferred_tee: Optional[TEEType] = None
    fallback_enabled: bool = True
    load_balancing: bool = True
    max_concurrent_operations: int = 10
    performance_monitoring: bool = True
    auto_failover: bool = True
    
    # TEE-specific configurations
    sgx_config: Optional[SGXConfig] = None
    sev_config: Optional[SEVConfig] = None
    
    # Performance thresholds
    max_aggregation_time: float = 30.0
    min_success_rate: float = 0.95
    
    def __post_init__(self):
        if self.sgx_config is None:
            self.sgx_config = SGXConfig()
        if self.sev_config is None:
            self.sev_config = SEVConfig()


class TEEManager:
    """
    Unified Trusted Execution Environment Manager.
    
    Manages multiple TEE implementations and provides a unified interface
    for secure federated learning operations.
    """
    
    def __init__(self, config: UnifiedTEEConfig):
        self.config = config
        self.available_tees: Dict[TEEType, TEEInfo] = {}
        self.active_enclaves: Dict[TEEType, Any] = {}
        self.performance_metrics: Dict[TEEType, Dict] = {}
        self.operation_queue: List = []
        self._lock = threading.RLock()
        
        # Global metrics
        self.global_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'fallback_activations': 0,
            'average_operation_time': 0.0
        }
        
        logger.info("Initializing Unified TEE Manager")
        self._detect_available_tees()
        self._initialize_preferred_tees()
    
    def _detect_available_tees(self) -> None:
        """Detect all available TEE implementations on the system."""
        logger.info("Detecting available TEE implementations...")
        
        # Always available: Mock TEE
        self.available_tees[TEEType.MOCK] = TEEInfo(
            tee_type=TEEType.MOCK,
            available=True,
            capabilities=[
                TEECapability.BYZANTINE_TOLERANCE,
                TEECapability.HIGH_PERFORMANCE
            ],
            performance_score=0.5,
            software_version="1.0.0"
        )
        
        # Detect Intel SGX
        sgx_info = self._detect_intel_sgx()
        if sgx_info:
            self.available_tees[TEEType.INTEL_SGX] = sgx_info
        
        # Detect AMD SEV
        sev_info = self._detect_amd_sev()
        if sev_info:
            self.available_tees[TEEType.AMD_SEV] = sev_info
        
        # Log available TEEs
        available_count = sum(1 for tee in self.available_tees.values() if tee.available)
        logger.info(f"Detected {available_count} available TEE implementations:")
        
        for tee_type, tee_info in self.available_tees.items():
            if tee_info.available:
                logger.info(f"  {tee_type.value}: {tee_info.capabilities}")
            else:
                logger.warning(f"  {tee_type.value}: Not available - {tee_info.error_message}")
    
    def _detect_intel_sgx(self) -> Optional[TEEInfo]:
        """Detect Intel SGX support."""
        try:
            # Check CPU support
            cpu_info = cpuinfo.get_cpu_info()
            cpu_vendor = cpu_info.get('vendor_id', '').lower()
            
            if 'intel' not in cpu_vendor:
                return TEEInfo(
                    tee_type=TEEType.INTEL_SGX,
                    available=False,
                    capabilities=[],
                    performance_score=0.0,
                    error_message="Intel CPU required for SGX"
                )
            
            # Check for SGX feature flags
            sgx_supported = False
            if 'flags' in cpu_info:
                flags = cpu_info['flags']
                sgx_supported = 'sgx' in flags or 'sgx1' in flags or 'sgx2' in flags
            
            if not sgx_supported:
                return TEEInfo(
                    tee_type=TEEType.INTEL_SGX,
                    available=False,
                    capabilities=[],
                    performance_score=0.0,
                    error_message="SGX not supported by CPU"
                )
            
            # Check SGX driver and SDK
            sgx_driver_available = self._check_sgx_driver()
            
            capabilities = [
                TEECapability.MEMORY_ENCRYPTION,
                TEECapability.ATTESTATION,
                TEECapability.SEALING,
                TEECapability.SIDE_CHANNEL_RESISTANCE,
                TEECapability.BYZANTINE_TOLERANCE
            ]
            
            performance_score = 0.9 if sgx_driver_available else 0.7
            
            return TEEInfo(
                tee_type=TEEType.INTEL_SGX,
                available=sgx_driver_available,
                capabilities=capabilities,
                performance_score=performance_score,
                hardware_version=cpu_info.get('brand', 'Unknown'),
                max_memory=128 * 1024 * 1024,  # Typical SGX EPC size
                error_message=None if sgx_driver_available else "SGX driver not available"
            )
            
        except Exception as e:
            logger.error(f"Error detecting Intel SGX: {e}")
            return TEEInfo(
                tee_type=TEEType.INTEL_SGX,
                available=False,
                capabilities=[],
                performance_score=0.0,
                error_message=f"Detection error: {e}"
            )
    
    def _detect_amd_sev(self) -> Optional[TEEInfo]:
        """Detect AMD SEV support."""
        try:
            # Check CPU support
            cpu_info = cpuinfo.get_cpu_info()
            cpu_vendor = cpu_info.get('vendor_id', '').lower()
            
            if 'amd' not in cpu_vendor:
                return TEEInfo(
                    tee_type=TEEType.AMD_SEV,
                    available=False,
                    capabilities=[],
                    performance_score=0.0,
                    error_message="AMD CPU required for SEV"
                )
            
            # Check for SEV support
            sev_supported = self._check_sev_support()
            
            if not sev_supported:
                return TEEInfo(
                    tee_type=TEEType.AMD_SEV,
                    available=False,
                    capabilities=[],
                    performance_score=0.0,
                    error_message="SEV not supported by CPU/kernel"
                )
            
            capabilities = [
                TEECapability.MEMORY_ENCRYPTION,
                TEECapability.ATTESTATION,
                TEECapability.BYZANTINE_TOLERANCE,
                TEECapability.HIGH_PERFORMANCE
            ]
            
            return TEEInfo(
                tee_type=TEEType.AMD_SEV,
                available=True,
                capabilities=capabilities,
                performance_score=0.85,
                hardware_version=cpu_info.get('brand', 'Unknown'),
                max_memory=512 * 1024 * 1024,  # Typical SEV VM memory
            )
            
        except Exception as e:
            logger.error(f"Error detecting AMD SEV: {e}")
            return TEEInfo(
                tee_type=TEEType.AMD_SEV,
                available=False,
                capabilities=[],
                performance_score=0.0,
                error_message=f"Detection error: {e}"
            )
    
    def _check_sgx_driver(self) -> bool:
        """Check if SGX driver is available."""
        try:
            # Check for SGX device files
            sgx_devices = ['/dev/sgx_enclave', '/dev/sgx/enclave', '/dev/isgx']
            
            for device in sgx_devices:
                if Path(device).exists():
                    return True
            
            # Check for SGX modules
            try:
                result = subprocess.run(['lsmod'], capture_output=True, text=True, timeout=5)
                if 'intel_sgx' in result.stdout:
                    return True
            except:
                pass
            
            return False
            
        except Exception as e:
            logger.debug(f"SGX driver check failed: {e}")
            return False
    
    def _check_sev_support(self) -> bool:
        """Check if SEV is supported."""
        try:
            # Check for SEV in /proc/cpuinfo
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo_content = f.read()
                    if 'sev' in cpuinfo_content.lower():
                        return True
            except:
                pass
            
            # Check for KVM SEV support
            sev_files = ['/sys/module/kvm_amd/parameters/sev',
                        '/dev/sev']
            
            for sev_file in sev_files:
                if Path(sev_file).exists():
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"SEV support check failed: {e}")
            return False
    
    def _initialize_preferred_tees(self) -> None:
        """Initialize preferred TEE implementations."""
        try:
            # Determine the best available TEE
            if self.config.preferred_tee:
                preferred = self.config.preferred_tee
                if preferred in self.available_tees and self.available_tees[preferred].available:
                    logger.info(f"Using preferred TEE: {preferred.value}")
                    self._initialize_tee(preferred)
                else:
                    logger.warning(f"Preferred TEE {preferred.value} not available, selecting best available")
                    self._select_best_tee()
            else:
                self._select_best_tee()
            
        except Exception as e:
            logger.error(f"Error initializing TEEs: {e}")
            # Fallback to mock if everything fails
            if TEEType.MOCK not in self.active_enclaves:
                self._initialize_tee(TEEType.MOCK)
    
    def _select_best_tee(self) -> None:
        """Select the best available TEE based on performance score."""
        available_tees = [(tee_type, tee_info) for tee_type, tee_info in self.available_tees.items() 
                         if tee_info.available]
        
        if not available_tees:
            raise EnclaveError("No TEE implementations available")
        
        # Sort by performance score
        available_tees.sort(key=lambda x: x[1].performance_score, reverse=True)
        
        # Initialize the best TEE
        best_tee = available_tees[0][0]
        logger.info(f"Selected best available TEE: {best_tee.value}")
        self._initialize_tee(best_tee)
        
        # Initialize additional TEEs if load balancing is enabled
        if self.config.load_balancing and len(available_tees) > 1:
            for tee_type, _ in available_tees[1:]:
                try:
                    self._initialize_tee(tee_type)
                    logger.info(f"Initialized additional TEE for load balancing: {tee_type.value}")
                except Exception as e:
                    logger.warning(f"Failed to initialize additional TEE {tee_type.value}: {e}")
    
    def _initialize_tee(self, tee_type: TEEType) -> None:
        """Initialize a specific TEE implementation."""
        try:
            if tee_type == TEEType.MOCK:
                enclave = MockSecureEnclave()
            elif tee_type == TEEType.INTEL_SGX:
                enclave = get_sgx_enclave(self.config.sgx_config)
            elif tee_type == TEEType.AMD_SEV:
                enclave = get_sev_enclave(self.config.sev_config)
            else:
                raise EnclaveError(f"Unsupported TEE type: {tee_type}")
            
            self.active_enclaves[tee_type] = enclave
            self.performance_metrics[tee_type] = {
                'operations_count': 0,
                'success_count': 0,
                'failure_count': 0,
                'average_time': 0.0,
                'last_operation_time': 0.0
            }
            
            logger.info(f"Successfully initialized TEE: {tee_type.value}")
            
        except Exception as e:
            logger.error(f"Failed to initialize TEE {tee_type.value}: {e}")
            raise
    
    def get_available_tees(self) -> Dict[TEEType, TEEInfo]:
        """Get information about all available TEEs."""
        return self.available_tees.copy()
    
    def get_active_tees(self) -> List[TEEType]:
        """Get list of currently active TEEs."""
        return list(self.active_enclaves.keys())
    
    def secure_aggregate(self, model_updates: List[Union[ModelUpdate, SecureModelUpdate]],
                        global_model_weights: Optional[bytes] = None,
                        preferred_tee: Optional[TEEType] = None) -> AggregationResult:
        """
        Perform secure aggregation using the best available TEE.
        
        Args:
            model_updates: List of model updates from clients
            global_model_weights: Current global model for comparison
            preferred_tee: Preferred TEE for this operation
            
        Returns:
            AggregationResult with aggregated model and metadata
        """
        with self._lock:
            start_time = time.time()
            
            try:
                # Select TEE for this operation
                selected_tee = self._select_tee_for_operation(preferred_tee)
                enclave = self.active_enclaves[selected_tee]
                
                # Convert model updates to appropriate format
                converted_updates = self._convert_model_updates(model_updates, selected_tee)
                
                logger.info(f"Performing secure aggregation using {selected_tee.value}")
                
                # Perform aggregation
                if selected_tee == TEEType.MOCK:
                    result = self._mock_aggregate(enclave, converted_updates, global_model_weights)
                else:
                    result = enclave.secure_aggregate(converted_updates, global_model_weights)
                
                # Update metrics
                operation_time = time.time() - start_time
                self._update_performance_metrics(selected_tee, True, operation_time)
                self._update_global_metrics(True, operation_time)
                
                logger.info(f"Secure aggregation completed in {operation_time:.2f}s using {selected_tee.value}")
                return result
                
            except Exception as e:
                operation_time = time.time() - start_time
                logger.error(f"Secure aggregation failed: {e}")
                
                # Try fallback if enabled
                if self.config.fallback_enabled and len(self.active_enclaves) > 1:
                    try:
                        logger.info("Attempting fallback aggregation...")
                        fallback_result = self._fallback_aggregate(model_updates, global_model_weights, preferred_tee)
                        self.global_metrics['fallback_activations'] += 1
                        self._update_global_metrics(True, time.time() - start_time)
                        return fallback_result
                    except Exception as fallback_error:
                        logger.error(f"Fallback aggregation also failed: {fallback_error}")
                
                # Update failure metrics
                if 'selected_tee' in locals():
                    self._update_performance_metrics(selected_tee, False, operation_time)
                self._update_global_metrics(False, operation_time)
                
                raise EnclaveError(f"Secure aggregation failed: {e}")
    
    def _select_tee_for_operation(self, preferred_tee: Optional[TEEType] = None) -> TEEType:
        """Select the best TEE for a specific operation."""
        if not self.active_enclaves:
            raise EnclaveError("No active TEE implementations available")
        
        # Use preferred TEE if specified and available
        if preferred_tee and preferred_tee in self.active_enclaves:
            return preferred_tee
        
        # If only one TEE is active, use it
        if len(self.active_enclaves) == 1:
            return list(self.active_enclaves.keys())[0]
        
        # Select based on load balancing and performance
        if self.config.load_balancing:
            return self._load_balance_tee_selection()
        else:
            # Select the best performing TEE
            best_tee = None
            best_score = -1
            
            for tee_type in self.active_enclaves.keys():
                tee_info = self.available_tees[tee_type]
                metrics = self.performance_metrics[tee_type]
                
                # Calculate dynamic score based on performance and availability
                success_rate = 1.0
                if metrics['operations_count'] > 0:
                    success_rate = metrics['success_count'] / metrics['operations_count']
                
                score = tee_info.performance_score * success_rate
                
                if score > best_score:
                    best_score = score
                    best_tee = tee_type
            
            return best_tee
    
    def _load_balance_tee_selection(self) -> TEEType:
        """Select TEE based on load balancing."""
        # Simple round-robin load balancing
        tee_loads = {}
        
        for tee_type in self.active_enclaves.keys():
            metrics = self.performance_metrics[tee_type]
            tee_loads[tee_type] = metrics['operations_count']
        
        # Select TEE with lowest load
        return min(tee_loads.keys(), key=lambda x: tee_loads[x])
    
    def _convert_model_updates(self, model_updates: List[Union[ModelUpdate, SecureModelUpdate]],
                             target_tee: TEEType) -> List:
        """Convert model updates to the appropriate format for the target TEE."""
        converted_updates = []
        
        for update in model_updates:
            if target_tee == TEEType.MOCK:
                # Convert to MockEnclave format
                if isinstance(update, ModelUpdate):
                    converted_updates.append(update)
                else:
                    # Convert SecureModelUpdate to ModelUpdate
                    mock_update = ModelUpdate(
                        device_id=update.device_id,
                        model_weights=update.encrypted_weights,
                        signature=update.signature,
                        timestamp=update.timestamp,
                        metadata=update.metadata or {}
                    )
                    converted_updates.append(mock_update)
            else:
                # Convert to SecureModelUpdate format
                if isinstance(update, SecureModelUpdate):
                    converted_updates.append(update)
                else:
                    # Convert ModelUpdate to SecureModelUpdate
                    secure_update = SecureModelUpdate(
                        device_id=update.device_id,
                        encrypted_weights=update.model_weights,
                        signature=update.signature,
                        timestamp=update.timestamp,
                        metadata=update.metadata
                    )
                    converted_updates.append(secure_update)
        
        return converted_updates
    
    def _mock_aggregate(self, enclave: MockSecureEnclave, updates: List[ModelUpdate],
                       global_weights: Optional[bytes]) -> AggregationResult:
        """Perform aggregation using mock enclave and convert result."""
        aggregated_weights, metadata = enclave.aggregate_models(updates, global_weights)
        
        return AggregationResult(
            aggregated_weights=aggregated_weights,
            num_updates_processed=metadata['num_updates'],
            num_updates_rejected=metadata['num_rejected'],
            rejected_device_ids=metadata['rejected_devices'],
            aggregation_hash=metadata['global_model_hash'],
            timestamp=time.time(),
            byzantine_detected=metadata['rejected_devices']
        )
    
    def _fallback_aggregate(self, model_updates: List, global_model_weights: Optional[bytes],
                          failed_tee: Optional[TEEType]) -> AggregationResult:
        """Perform fallback aggregation using alternative TEE."""
        available_fallbacks = [tee for tee in self.active_enclaves.keys() if tee != failed_tee]
        
        if not available_fallbacks:
            raise EnclaveError("No fallback TEE available")
        
        # Try each fallback TEE
        for fallback_tee in available_fallbacks:
            try:
                logger.info(f"Attempting fallback aggregation with {fallback_tee.value}")
                return self.secure_aggregate(model_updates, global_model_weights, fallback_tee)
            except Exception as e:
                logger.warning(f"Fallback TEE {fallback_tee.value} also failed: {e}")
                continue
        
        raise EnclaveError("All fallback TEEs failed")
    
    def _update_performance_metrics(self, tee_type: TEEType, success: bool, operation_time: float) -> None:
        """Update performance metrics for a specific TEE."""
        metrics = self.performance_metrics[tee_type]
        
        metrics['operations_count'] += 1
        metrics['last_operation_time'] = operation_time
        
        if success:
            metrics['success_count'] += 1
        else:
            metrics['failure_count'] += 1
        
        # Update rolling average
        prev_avg = metrics['average_time']
        total_ops = metrics['operations_count']
        metrics['average_time'] = ((prev_avg * (total_ops - 1)) + operation_time) / total_ops
    
    def _update_global_metrics(self, success: bool, operation_time: float) -> None:
        """Update global performance metrics."""
        self.global_metrics['total_operations'] += 1
        
        if success:
            self.global_metrics['successful_operations'] += 1
        else:
            self.global_metrics['failed_operations'] += 1
        
        # Update rolling average
        prev_avg = self.global_metrics['average_operation_time']
        total_ops = self.global_metrics['total_operations']
        self.global_metrics['average_operation_time'] = ((prev_avg * (total_ops - 1)) + operation_time) / total_ops
    
    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics."""
        return {
            'global_metrics': self.global_metrics.copy(),
            'tee_metrics': {tee_type.value: metrics.copy() 
                           for tee_type, metrics in self.performance_metrics.items()},
            'available_tees': {tee_type.value: asdict(tee_info) 
                              for tee_type, tee_info in self.available_tees.items()},
            'active_tees': [tee.value for tee in self.active_enclaves.keys()]
        }
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        return {
            'tee_manager_status': 'operational',
            'available_tees': len([t for t in self.available_tees.values() if t.available]),
            'active_tees': len(self.active_enclaves),
            'performance_metrics': self.get_performance_metrics(),
            'configuration': asdict(self.config)
        }
    
    def shutdown(self) -> None:
        """Safely shutdown all TEE implementations."""
        logger.info("Shutting down TEE Manager...")
        
        for tee_type, enclave in self.active_enclaves.items():
            try:
                if hasattr(enclave, 'destroy_enclave'):
                    enclave.destroy_enclave()
                logger.info(f"Shutdown TEE: {tee_type.value}")
            except Exception as e:
                logger.error(f"Error shutting down TEE {tee_type.value}: {e}")
        
        # Destroy global instances
        try:
            destroy_global_enclave()
            destroy_global_sev_enclave()
        except Exception as e:
            logger.error(f"Error destroying global enclaves: {e}")
        
        self.active_enclaves.clear()
        logger.info("TEE Manager shutdown completed")


# Global TEE manager instance
_tee_manager: Optional[TEEManager] = None
_tee_manager_lock = threading.Lock()


def get_tee_manager(config: Optional[UnifiedTEEConfig] = None) -> TEEManager:
    """Get or create the global TEE manager instance."""
    global _tee_manager
    
    with _tee_manager_lock:
        if _tee_manager is None:
            if config is None:
                config = UnifiedTEEConfig()
            _tee_manager = TEEManager(config)
        
        return _tee_manager


def shutdown_tee_manager() -> None:
    """Shutdown the global TEE manager instance."""
    global _tee_manager
    
    with _tee_manager_lock:
        if _tee_manager:
            _tee_manager.shutdown()
            _tee_manager = None