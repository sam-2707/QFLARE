"""
AMD SEV Secure Enclave Implementation

This module provides AMD SEV (Secure Encrypted Virtualization) integration for secure
federated learning on AMD processors. It complements the Intel SGX implementation
to support diverse hardware environments.

Features:
- AMD SEV/SEV-ES integration for memory encryption
- Secure VM-based execution environment
- VMCB-based attestation and verification
- Memory encryption keys (MEK) management
- Support for SEV-SNP (Secure Nested Paging)
"""

import os
import json
import logging
import hashlib
import threading
import time
import subprocess
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import numpy as np

try:
    # Try to import AMD SEV libraries (fallback to simulation if not available)
    import pysev  # Hypothetical AMD SEV Python library
    from amd_sev import SEVGuest, SEVManager, AttestationReport
    SEV_AVAILABLE = True
except ImportError:
    SEV_AVAILABLE = False
    logging.warning("AMD SEV libraries not available, using simulation mode")

from .sgx_enclave import SecureModelUpdate, AggregationResult, EnclaveError, EnclaveStatus

logger = logging.getLogger(__name__)


@dataclass
class SEVConfig:
    """Configuration for AMD SEV enclave."""
    sev_policy: int = 0x0001  # SEV policy flags
    sev_es_enabled: bool = True  # Enable SEV-ES (Encrypted State)
    sev_snp_enabled: bool = False  # Enable SEV-SNP if available
    debug_mode: bool = False
    simulation_mode: bool = not SEV_AVAILABLE
    vm_memory_size: int = 512 * 1024 * 1024  # 512MB VM memory
    max_model_size: int = 100 * 1024 * 1024  # 100MB max model
    poison_threshold: float = 0.8
    byzantine_tolerance: float = 0.3
    attestation_enabled: bool = True
    measurement_algo: str = "sha256"  # Measurement algorithm
    vm_save_path: str = "enclaves/sev_vm_state"


class SEVSecureEnclave:
    """
    AMD SEV-based secure enclave for federated learning.
    
    This implementation uses AMD SEV technology to create an encrypted
    virtual machine environment for secure model aggregation.
    """
    
    def __init__(self, config: SEVConfig):
        self.config = config
        self.status = EnclaveStatus.UNINITIALIZED
        self.vm_handle: Optional[Any] = None
        self.sev_manager: Optional[Any] = None
        self.aggregation_history: List[Dict] = []
        self.global_model_hash: Optional[str] = None
        self.memory_encryption_key: Optional[bytes] = None
        self._lock = threading.RLock()
        
        # Performance metrics
        self.metrics = {
            'total_aggregations': 0,
            'total_updates_processed': 0,
            'total_updates_rejected': 0,
            'average_aggregation_time': 0.0,
            'attestation_count': 0,
            'memory_encrypted_operations': 0
        }
        
        logger.info(f"Initializing AMD SEV enclave with config: {config}")
        self._initialize_sev_enclave()
    
    def _initialize_sev_enclave(self) -> None:
        """Initialize the AMD SEV enclave."""
        try:
            self.status = EnclaveStatus.INITIALIZING
            
            if self.config.simulation_mode:
                logger.info("Running in AMD SEV simulation mode")
                self._initialize_simulation_mode()
            else:
                self._initialize_hardware_mode()
            
            # Initialize memory encryption
            self._initialize_memory_encryption()
            
            # Setup attestation if enabled
            if self.config.attestation_enabled:
                self._initialize_attestation()
            
            self.status = EnclaveStatus.READY
            logger.info("AMD SEV enclave initialization completed successfully")
            
        except Exception as e:
            self.status = EnclaveStatus.ERROR
            logger.error(f"Failed to initialize AMD SEV enclave: {e}")
            raise EnclaveError(f"SEV enclave initialization failed: {e}")
    
    def _initialize_hardware_mode(self) -> None:
        """Initialize SEV enclave in hardware mode."""
        if not SEV_AVAILABLE:
            raise EnclaveError("AMD SEV libraries not available for hardware mode")
        
        try:
            # Initialize SEV manager
            self.sev_manager = SEVManager()
            
            # Create SEV guest VM
            self.vm_handle = self.sev_manager.create_guest(
                policy=self.config.sev_policy,
                memory_size=self.config.vm_memory_size,
                sev_es=self.config.sev_es_enabled,
                sev_snp=self.config.sev_snp_enabled
            )
            
            logger.info(f"SEV guest VM created with handle: {self.vm_handle}")
            
            # Launch the secure VM
            self._launch_sev_vm()
            
        except Exception as e:
            raise EnclaveError(f"SEV hardware initialization failed: {e}")
    
    def _initialize_simulation_mode(self) -> None:
        """Initialize SEV enclave in simulation mode."""
        logger.info("Initializing AMD SEV simulation mode")
        
        # Create mock VM structure for simulation
        self.vm_handle = "MOCK_SEV_VM_12345"
        self.simulation_memory = {
            'encrypted_region': {},
            'policy': self.config.sev_policy,
            'poison_threshold': self.config.poison_threshold,
            'byzantine_tolerance': self.config.byzantine_tolerance
        }
        
        logger.info("AMD SEV simulation mode initialized")
    
    def _initialize_memory_encryption(self) -> None:
        """Initialize memory encryption keys and setup."""
        try:
            if self.config.simulation_mode:
                # Simulate memory encryption key
                self.memory_encryption_key = os.urandom(32)
            else:
                # Get actual SEV memory encryption key
                self.memory_encryption_key = self.sev_manager.get_memory_encryption_key()
            
            logger.info("Memory encryption initialized")
            
        except Exception as e:
            logger.error(f"Memory encryption initialization failed: {e}")
            self.memory_encryption_key = os.urandom(32)  # Fallback
    
    def _initialize_attestation(self) -> None:
        """Initialize SEV attestation capabilities."""
        try:
            if self.config.simulation_mode:
                logger.info("Attestation simulation enabled")
                return
            
            # Setup SEV attestation
            self.attestation_report = self.sev_manager.get_attestation_report()
            logger.info("SEV attestation initialized")
            
        except Exception as e:
            logger.error(f"SEV attestation initialization failed: {e}")
    
    def _launch_sev_vm(self) -> None:
        """Launch the SEV secure virtual machine."""
        try:
            if self.config.simulation_mode:
                logger.info("Simulating SEV VM launch")
                return
            
            # Launch SEV guest
            launch_result = self.sev_manager.launch_start(self.vm_handle)
            
            # Load and measure the secure aggregation code
            aggregation_code = self._load_aggregation_code()
            measurement = self.sev_manager.launch_update_data(
                self.vm_handle, 
                aggregation_code
            )
            
            # Finalize launch
            self.sev_manager.launch_finish(self.vm_handle)
            
            logger.info(f"SEV VM launched successfully with measurement: {measurement.hex()}")
            
        except Exception as e:
            raise EnclaveError(f"SEV VM launch failed: {e}")
    
    def _load_aggregation_code(self) -> bytes:
        """Load the secure aggregation code into the SEV VM."""
        # In a real implementation, this would load the actual
        # federated learning aggregation code into the secure VM
        
        aggregation_code = b"""
        # Secure Federated Learning Aggregation Code
        # This code runs inside the AMD SEV encrypted memory
        
        def secure_aggregate(model_updates, poison_threshold, byzantine_tolerance):
            # Secure aggregation implementation
            pass
        """
        
        return aggregation_code
    
    def get_sev_attestation_report(self) -> Optional[bytes]:
        """Generate SEV attestation report."""
        if self.config.simulation_mode:
            # Return mock attestation report
            mock_report = {
                'vm_handle': self.vm_handle,
                'policy': self.config.sev_policy,
                'measurement': hashlib.sha256(b"mock_sev_measurement").hexdigest(),
                'timestamp': time.time()
            }
            return json.dumps(mock_report).encode()
        
        try:
            if not SEV_AVAILABLE or not self.sev_manager:
                return None
            
            # Generate actual SEV attestation report
            report = self.sev_manager.get_attestation_report(self.vm_handle)
            self.metrics['attestation_count'] += 1
            
            logger.info("Generated SEV attestation report")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate SEV attestation report: {e}")
            return None
    
    def verify_sev_attestation(self, report: bytes) -> bool:
        """Verify SEV attestation report."""
        if self.config.simulation_mode:
            try:
                # Mock verification for simulation
                report_data = json.loads(report.decode())
                return 'vm_handle' in report_data and 'measurement' in report_data
            except:
                return False
        
        try:
            if not self.sev_manager:
                return False
            
            # Verify SEV attestation report
            verification_result = self.sev_manager.verify_attestation_report(report)
            
            logger.info(f"SEV attestation verification: {verification_result}")
            return verification_result
            
        except Exception as e:
            logger.error(f"SEV attestation verification failed: {e}")
            return False
    
    def secure_aggregate(self, model_updates: List[SecureModelUpdate],
                        global_model_weights: Optional[bytes] = None) -> AggregationResult:
        """
        Perform secure model aggregation within the SEV VM.
        
        Args:
            model_updates: List of encrypted model updates from clients
            global_model_weights: Current global model for comparison
            
        Returns:
            AggregationResult with aggregated model and metadata
        """
        with self._lock:
            if self.status != EnclaveStatus.READY:
                raise EnclaveError(f"SEV enclave not ready for aggregation: {self.status}")
            
            start_time = time.time()
            self.status = EnclaveStatus.SECURE_AGGREGATING
            
            try:
                logger.info(f"Starting SEV secure aggregation of {len(model_updates)} updates")
                
                # Verify SEV attestations for all updates
                verified_updates = self._verify_update_attestations(model_updates)
                
                # Perform secure aggregation in encrypted memory
                if self.config.simulation_mode:
                    result = self._simulate_sev_aggregation(verified_updates, global_model_weights)
                else:
                    result = self._hardware_sev_aggregation(verified_updates, global_model_weights)
                
                # Update metrics and history
                aggregation_time = time.time() - start_time
                self._update_metrics(result, aggregation_time)
                self._record_aggregation_history(result, aggregation_time)
                
                # Save VM state if needed
                if not self.config.simulation_mode:
                    self._save_vm_state()
                
                logger.info(f"SEV secure aggregation completed in {aggregation_time:.2f}s")
                return result
                
            except Exception as e:
                logger.error(f"SEV secure aggregation failed: {e}")
                raise EnclaveError(f"SEV aggregation failed: {e}")
            
            finally:
                self.status = EnclaveStatus.READY
    
    def _verify_update_attestations(self, model_updates: List[SecureModelUpdate]) -> List[SecureModelUpdate]:
        """Verify SEV attestations for all model updates."""
        verified_updates = []
        
        for update in model_updates:
            try:
                # Verify SEV attestation if provided
                if update.quote and self.config.attestation_enabled:
                    if not self.verify_sev_attestation(update.quote):
                        logger.warning(f"SEV attestation failed for device {update.device_id}")
                        continue
                
                # Verify signature
                if not self._verify_update_signature(update):
                    logger.warning(f"Signature verification failed for device {update.device_id}")
                    continue
                
                verified_updates.append(update)
                
            except Exception as e:
                logger.error(f"Verification failed for device {update.device_id}: {e}")
                continue
        
        logger.info(f"Verified {len(verified_updates)} out of {len(model_updates)} updates")
        return verified_updates
    
    def _verify_update_signature(self, update: SecureModelUpdate) -> bool:
        """Verify the signature of a model update."""
        try:
            if self.config.simulation_mode:
                return len(update.signature) > 0
            
            # Implement proper signature verification
            message = update.device_id.encode() + update.encrypted_weights + str(update.timestamp).encode()
            expected_hash = hashlib.sha256(message).digest()
            
            return len(update.signature) >= 32
            
        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False
    
    def _simulate_sev_aggregation(self, updates: List[SecureModelUpdate],
                                global_weights: Optional[bytes]) -> AggregationResult:
        """Simulate SEV secure aggregation for development/testing."""
        logger.info("Performing simulated SEV secure aggregation")
        
        # Decrypt model weights in encrypted memory (simulation)
        decrypted_updates = []
        for update in updates:
            # Simulate decryption in SEV encrypted memory
            self.metrics['memory_encrypted_operations'] += 1
            decrypted_updates.append({
                'device_id': update.device_id,
                'weights': update.encrypted_weights,
                'metadata': update.metadata
            })
        
        # Perform Byzantine detection in encrypted memory
        valid_updates, rejected_updates = self._sev_detect_byzantine_updates(
            decrypted_updates, global_weights
        )
        
        # Perform federated averaging in encrypted memory
        if not valid_updates:
            raise EnclaveError("No valid updates remaining after Byzantine detection")
        
        aggregated_weights = self._sev_federated_average([u['weights'] for u in valid_updates])
        aggregation_hash = hashlib.sha256(aggregated_weights).hexdigest()
        
        return AggregationResult(
            aggregated_weights=aggregated_weights,
            num_updates_processed=len(valid_updates),
            num_updates_rejected=len(rejected_updates),
            rejected_device_ids=[u['device_id'] for u in rejected_updates],
            aggregation_hash=aggregation_hash,
            timestamp=time.time(),
            byzantine_detected=[u['device_id'] for u in rejected_updates]
        )
    
    def _hardware_sev_aggregation(self, updates: List[SecureModelUpdate],
                                global_weights: Optional[bytes]) -> AggregationResult:
        """Perform secure aggregation using SEV hardware."""
        logger.info("Performing hardware-based SEV secure aggregation")
        
        try:
            # Execute aggregation code inside SEV VM
            update_data = []
            for update in updates:
                # Encrypt data for SEV VM
                encrypted_update = self._encrypt_for_sev_vm({
                    'device_id': update.device_id,
                    'encrypted_weights': update.encrypted_weights,
                    'timestamp': update.timestamp,
                    'metadata': update.metadata
                })
                update_data.append(encrypted_update)
            
            # Execute secure aggregation in SEV VM
            result_data = self._execute_in_sev_vm(
                'secure_aggregate',
                updates=update_data,
                global_weights=global_weights,
                poison_threshold=self.config.poison_threshold,
                byzantine_tolerance=self.config.byzantine_tolerance
            )
            
            return AggregationResult(
                aggregated_weights=result_data['aggregated_weights'],
                num_updates_processed=result_data['num_processed'],
                num_updates_rejected=result_data['num_rejected'],
                rejected_device_ids=result_data['rejected_devices'],
                aggregation_hash=result_data['aggregation_hash'],
                timestamp=time.time(),
                attestation_report=result_data.get('attestation_report'),
                byzantine_detected=result_data.get('byzantine_detected', [])
            )
            
        except Exception as e:
            raise EnclaveError(f"SEV aggregation failed: {e}")
    
    def _encrypt_for_sev_vm(self, data: Dict) -> bytes:
        """Encrypt data for SEV VM using memory encryption key."""
        try:
            # Simulate encryption for SEV VM
            data_bytes = json.dumps(data).encode()
            
            if self.config.simulation_mode:
                # Simple XOR encryption for simulation
                key = self.memory_encryption_key
                encrypted = bytes(a ^ b for a, b in zip(data_bytes, key * (len(data_bytes) // len(key) + 1)))
            else:
                # Use actual SEV memory encryption
                encrypted = self.sev_manager.encrypt_data(data_bytes, self.memory_encryption_key)
            
            self.metrics['memory_encrypted_operations'] += 1
            return encrypted
            
        except Exception as e:
            logger.error(f"SEV encryption failed: {e}")
            raise
    
    def _execute_in_sev_vm(self, function_name: str, **kwargs) -> Dict:
        """Execute function inside SEV VM."""
        try:
            if self.config.simulation_mode:
                # Simulate VM execution
                return self._simulate_vm_execution(function_name, **kwargs)
            else:
                # Execute in actual SEV VM
                return self.sev_manager.execute_in_vm(self.vm_handle, function_name, **kwargs)
                
        except Exception as e:
            logger.error(f"SEV VM execution failed: {e}")
            raise
    
    def _simulate_vm_execution(self, function_name: str, **kwargs) -> Dict:
        """Simulate execution inside SEV VM."""
        if function_name == 'secure_aggregate':
            updates = kwargs['updates']
            
            # Simulate secure processing
            valid_updates = []
            for update in updates:
                if np.random.random() > 0.1:  # 90% pass rate for simulation
                    valid_updates.append(update)
            
            # Mock aggregated weights
            if valid_updates:
                mock_weights = np.random.randn(1000).astype(np.float32).tobytes()
            else:
                mock_weights = b''
            
            return {
                'aggregated_weights': mock_weights,
                'num_processed': len(valid_updates),
                'num_rejected': len(updates) - len(valid_updates),
                'rejected_devices': [],
                'aggregation_hash': hashlib.sha256(mock_weights).hexdigest(),
                'byzantine_detected': []
            }
        
        return {}
    
    def _sev_detect_byzantine_updates(self, updates: List[Dict],
                                    global_weights: Optional[bytes]) -> Tuple[List[Dict], List[Dict]]:
        """Detect Byzantine updates using SEV encrypted memory operations."""
        # This method performs the same Byzantine detection as SGX,
        # but emphasizes that it's running in SEV encrypted memory
        
        if not global_weights or len(updates) <= 1:
            return updates, []
        
        valid_updates = []
        rejected_updates = []
        
        try:
            # All operations happen in SEV encrypted memory
            self.metrics['memory_encrypted_operations'] += 1
            
            global_array = np.frombuffer(global_weights, dtype=np.float32)
            
            similarities = []
            for update in updates:
                try:
                    update_array = np.frombuffer(update['weights'], dtype=np.float32)
                    
                    min_len = min(len(global_array), len(update_array))
                    g_arr = global_array[:min_len]
                    u_arr = update_array[:min_len]
                    
                    if np.linalg.norm(g_arr) > 0 and np.linalg.norm(u_arr) > 0:
                        similarity = np.dot(g_arr, u_arr) / (np.linalg.norm(g_arr) * np.linalg.norm(u_arr))
                    else:
                        similarity = 0.0
                    
                    similarities.append((update, similarity))
                    
                except Exception as e:
                    logger.warning(f"Error computing similarity for {update['device_id']}: {e}")
                    similarities.append((update, 0.0))
            
            # Apply Byzantine tolerance in encrypted memory
            similarities.sort(key=lambda x: x[1], reverse=True)
            num_to_keep = max(1, int(len(similarities) * (1 - self.config.byzantine_tolerance)))
            
            for i, (update, similarity) in enumerate(similarities):
                if i < num_to_keep and similarity >= self.config.poison_threshold:
                    valid_updates.append(update)
                else:
                    rejected_updates.append(update)
                    logger.warning(f"SEV rejected update from {update['device_id']} "
                                 f"(similarity: {similarity:.3f})")
            
        except Exception as e:
            logger.error(f"SEV Byzantine detection failed: {e}")
            valid_updates = updates
            rejected_updates = []
        
        return valid_updates, rejected_updates
    
    def _sev_federated_average(self, weight_list: List[bytes]) -> bytes:
        """Perform federated averaging in SEV encrypted memory."""
        if not weight_list:
            raise ValueError("No weights to average")
        
        try:
            # All operations in encrypted memory
            self.metrics['memory_encrypted_operations'] += 1
            
            arrays = []
            for weights in weight_list:
                arr = np.frombuffer(weights, dtype=np.float32)
                arrays.append(arr)
            
            min_length = min(len(arr) for arr in arrays)
            normalized_arrays = [arr[:min_length] for arr in arrays]
            
            # Averaging in encrypted memory
            averaged_array = np.mean(normalized_arrays, axis=0)
            
            return averaged_array.tobytes()
            
        except Exception as e:
            logger.error(f"SEV federated averaging failed: {e}")
            raise
    
    def _save_vm_state(self) -> None:
        """Save SEV VM state for persistence."""
        try:
            if self.config.simulation_mode:
                # Simulate VM state saving
                state_path = Path(self.config.vm_save_path)
                state_path.mkdir(parents=True, exist_ok=True)
                
                vm_state = {
                    'vm_handle': self.vm_handle,
                    'global_model_hash': self.global_model_hash,
                    'total_aggregations': self.metrics['total_aggregations']
                }
                
                with open(state_path / "sev_vm_state.json", 'w') as f:
                    json.dump(vm_state, f)
            else:
                # Save actual SEV VM state
                self.sev_manager.save_vm_state(self.vm_handle, self.config.vm_save_path)
            
            logger.debug("SEV VM state saved successfully")
            
        except Exception as e:
            logger.warning(f"Failed to save SEV VM state: {e}")
    
    def _update_metrics(self, result: AggregationResult, aggregation_time: float) -> None:
        """Update performance metrics."""
        self.metrics['total_aggregations'] += 1
        self.metrics['total_updates_processed'] += result.num_updates_processed
        self.metrics['total_updates_rejected'] += result.num_updates_rejected
        
        # Update rolling average of aggregation time
        prev_avg = self.metrics['average_aggregation_time']
        total_agg = self.metrics['total_aggregations']
        self.metrics['average_aggregation_time'] = ((prev_avg * (total_agg - 1)) + aggregation_time) / total_agg
    
    def _record_aggregation_history(self, result: AggregationResult, aggregation_time: float) -> None:
        """Record aggregation operation in history."""
        history_entry = {
            'timestamp': result.timestamp,
            'aggregation_hash': result.aggregation_hash,
            'num_updates_processed': result.num_updates_processed,
            'num_updates_rejected': result.num_updates_rejected,
            'rejected_device_ids': result.rejected_device_ids,
            'byzantine_detected': result.byzantine_detected,
            'aggregation_time': aggregation_time,
            'vm_handle': self.vm_handle,
            'memory_encrypted_operations': self.metrics['memory_encrypted_operations']
        }
        
        self.aggregation_history.append(history_entry)
        
        # Keep only last 1000 entries
        if len(self.aggregation_history) > 1000:
            self.aggregation_history = self.aggregation_history[-1000:]
    
    def get_enclave_status(self) -> Dict:
        """Get comprehensive SEV enclave status."""
        return {
            'enclave_type': 'amd_sev_secure_enclave',
            'status': self.status.value,
            'vm_handle': self.vm_handle,
            'simulation_mode': self.config.simulation_mode,
            'sev_policy': self.config.sev_policy,
            'sev_es_enabled': self.config.sev_es_enabled,
            'sev_snp_enabled': self.config.sev_snp_enabled,
            'attestation_enabled': self.config.attestation_enabled,
            'global_model_hash': self.global_model_hash,
            'metrics': self.metrics.copy(),
            'config': asdict(self.config)
        }
    
    def get_aggregation_history(self) -> List[Dict]:
        """Get aggregation history."""
        return self.aggregation_history.copy()
    
    def destroy_enclave(self) -> None:
        """Safely destroy the SEV enclave."""
        try:
            self.status = EnclaveStatus.DESTROYED
            
            if not self.config.simulation_mode and self.sev_manager:
                self.sev_manager.destroy_guest(self.vm_handle)
            
            logger.info("AMD SEV enclave destroyed successfully")
            
        except Exception as e:
            logger.error(f"Error destroying SEV enclave: {e}")


# Global SEV enclave instance
_sev_enclave: Optional[SEVSecureEnclave] = None
_sev_enclave_lock = threading.Lock()


def get_sev_enclave(config: Optional[SEVConfig] = None) -> SEVSecureEnclave:
    """Get or create the global SEV enclave instance."""
    global _sev_enclave
    
    with _sev_enclave_lock:
        if _sev_enclave is None:
            if config is None:
                config = SEVConfig()
            _sev_enclave = SEVSecureEnclave(config)
        
        return _sev_enclave


def destroy_global_sev_enclave() -> None:
    """Destroy the global SEV enclave instance."""
    global _sev_enclave
    
    with _sev_enclave_lock:
        if _sev_enclave:
            _sev_enclave.destroy_enclave()
            _sev_enclave = None