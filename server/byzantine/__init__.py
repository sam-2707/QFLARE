"""
Byzantine Fault Tolerance Package Initialization

Initializes the Byzantine fault tolerance module for QFLARE's robust federated learning.
"""

from .detection import (
    ByzantineDetectionConfig,
    ByzantineDetector,
    create_krum_detector,
    create_multi_krum_detector,
    create_trimmed_mean_detector,
    create_clustering_detector
)

from .robust_aggregator import (
    ByzantineRobustAggregator,
    create_krum_aggregator,
    create_multi_krum_aggregator,
    create_trimmed_mean_aggregator,
    create_clustering_aggregator
)

from .byzantine_fl_controller import (
    ByzantineFLController,
    create_byzantine_fl_controller
)

__all__ = [
    # Byzantine detection classes
    "ByzantineDetectionConfig",
    "ByzantineDetector",
    
    # Detection factory functions
    "create_krum_detector",
    "create_multi_krum_detector",
    "create_trimmed_mean_detector", 
    "create_clustering_detector",
    
    # Robust aggregation classes
    "ByzantineRobustAggregator",
    
    # Aggregation factory functions
    "create_krum_aggregator",
    "create_multi_krum_aggregator",
    "create_trimmed_mean_aggregator",
    "create_clustering_aggregator",
    
    # Byzantine-aware FL controller
    "ByzantineFLController",
    "create_byzantine_fl_controller"
]

# Version info
__version__ = "1.0.0"