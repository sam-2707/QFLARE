"""
QFlare Secure Enclave Module

This module provides access to trusted execution environments (TEEs) for secure
federated learning aggregation. It supports multiple TEE implementations including
Intel SGX, AMD SEV, and mock enclaves for development.

The module automatically detects available hardware and provides a unified interface
for secure operations regardless of the underlying TEE technology.
"""

from .tee_manager import (
    get_tee_manager, 
    shutdown_tee_manager,
    TEEManager,
    UnifiedTEEConfig,
    TEEType,
    TEECapability,
    TEEInfo
)

from .sgx_enclave import (
    SGXSecureEnclave,
    SGXConfig,
    SecureModelUpdate,
    AggregationResult,
    EnclaveError,
    EnclaveStatus
)

from .sev_enclave import (
    SEVSecureEnclave,
    SEVConfig
)

from .mock_enclave import (
    MockSecureEnclave,
    ModelUpdate
)

# Default TEE manager instance
_default_tee_config = UnifiedTEEConfig(
    preferred_tee=None,  # Auto-detect best TEE
    fallback_enabled=True,
    load_balancing=True,
    performance_monitoring=True,
    auto_failover=True
)

def get_default_tee_manager():
    """Get the default TEE manager with automatic configuration."""
    return get_tee_manager(_default_tee_config)

def secure_aggregate(model_updates, global_model_weights=None, preferred_tee=None):
    """
    Convenient function for secure aggregation using the default TEE manager.
    
    Args:
        model_updates: List of model updates from clients
        global_model_weights: Current global model weights (optional)
        preferred_tee: Preferred TEE type (optional)
        
    Returns:
        AggregationResult with aggregated model and metadata
    """
    tee_manager = get_default_tee_manager()
    return tee_manager.secure_aggregate(model_updates, global_model_weights, preferred_tee)

def get_enclave_status():
    """Get status of all available enclaves."""
    tee_manager = get_default_tee_manager()
    return tee_manager.get_system_status()

__all__ = [
    'get_tee_manager',
    'shutdown_tee_manager',
    'get_default_tee_manager',
    'secure_aggregate',
    'get_enclave_status',
    'TEEManager',
    'UnifiedTEEConfig',
    'TEEType',
    'TEECapability',
    'TEEInfo',
    'SGXSecureEnclave',
    'SEVSecureEnclave',
    'MockSecureEnclave',
    'SGXConfig',
    'SEVConfig',
    'SecureModelUpdate',
    'ModelUpdate',
    'AggregationResult',
    'EnclaveError',
    'EnclaveStatus'
]