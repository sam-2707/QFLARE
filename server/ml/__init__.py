"""
QFLARE Machine Learning Package

This package contains real ML implementations for federated learning.
Includes PyTorch models, training engines, and federated algorithms.
"""

from .models import (
    MNISTNet,
    CIFAR10Net,
    SimpleMLPNet,
    create_model,
    get_model_info,
    serialize_model_weights,
    deserialize_model_weights,
    calculate_model_similarity
)

from .training import (
    FederatedDataset,
    FederatedTrainer,
    create_real_fl_trainer
)

__all__ = [
    "MNISTNet",
    "CIFAR10Net", 
    "SimpleMLPNet",
    "create_model",
    "get_model_info",
    "serialize_model_weights",
    "deserialize_model_weights",
    "calculate_model_similarity",
    "FederatedDataset",
    "FederatedTrainer",
    "create_real_fl_trainer"
]