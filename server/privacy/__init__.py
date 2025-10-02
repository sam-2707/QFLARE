"""
Privacy Package Initialization

Initializes the privacy module for QFLARE's differential privacy implementation.
"""

from .differential_privacy import (
    DifferentialPrivacyConfig,
    GaussianMechanism,
    PrivacyEngine,
    create_strong_privacy_config,
    create_moderate_privacy_config,
    create_weak_privacy_config,
    create_privacy_engine
)

from .private_trainer import PrivateFederatedTrainer
from .private_fl_controller import PrivateFLController, create_private_fl_controller

__all__ = [
    # Core privacy classes
    "DifferentialPrivacyConfig",
    "GaussianMechanism", 
    "PrivacyEngine",
    
    # Privacy configuration factories
    "create_strong_privacy_config",
    "create_moderate_privacy_config", 
    "create_weak_privacy_config",
    "create_privacy_engine",
    
    # Privacy-aware FL components
    "PrivateFederatedTrainer",
    "PrivateFLController",
    "create_private_fl_controller"
]

# Version info
__version__ = "1.0.0"