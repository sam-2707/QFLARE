"""
Byzantine Fault Tolerance for QFLARE Federated Learning

This module implements robust aggregation algorithms that can handle malicious clients
attempting to poison the global model. Supports up to 33% malicious participants.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
import math
from datetime import datetime
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)


class ByzantineDetectionConfig:
    """Configuration for Byzantine fault tolerance mechanisms."""
    
    def __init__(self,
                 max_malicious_ratio: float = 0.33,
                 detection_method: str = "krum",
                 outlier_threshold: float = 2.0,
                 clustering_eps: float = 0.5,
                 clustering_min_samples: int = 2,
                 statistical_threshold: float = 3.0):
        """
        Initialize Byzantine detection configuration.
        
        Args:
            max_malicious_ratio: Maximum ratio of malicious clients (default: 33%)
            detection_method: Detection method ("krum", "trimmed_mean", "median", "clustering")
            outlier_threshold: Threshold for outlier detection (standard deviations)
            clustering_eps: DBSCAN epsilon parameter for clustering
            clustering_min_samples: DBSCAN minimum samples per cluster
            statistical_threshold: Statistical outlier threshold
        """
        self.max_malicious_ratio = max_malicious_ratio
        self.detection_method = detection_method
        self.outlier_threshold = outlier_threshold
        self.clustering_eps = clustering_eps
        self.clustering_min_samples = clustering_min_samples
        self.statistical_threshold = statistical_threshold
        
        # Validation
        if not 0 < max_malicious_ratio < 0.5:
            raise ValueError("max_malicious_ratio must be between 0 and 0.5")
        
        if detection_method not in ["krum", "trimmed_mean", "median", "clustering", "multi_krum"]:
            raise ValueError(f"Unsupported detection method: {detection_method}")
        
        logger.info(f"Byzantine detection configured: {detection_method}, max malicious: {max_malicious_ratio:.1%}")


class ByzantineDetector:
    """
    Detects and filters Byzantine (malicious) model updates.
    Implements multiple robust aggregation algorithms.
    """
    
    def __init__(self, config: ByzantineDetectionConfig):
        self.config = config
        self.detection_history = []
        self.malicious_clients = set()
        self.client_reputation = {}
        
    def detect_byzantine_updates(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect Byzantine clients from model updates.
        
        Args:
            client_updates: List of client update dictionaries
            
        Returns:
            Detection result with identified malicious clients
        """
        if len(client_updates) < 3:
            logger.warning("Too few clients for Byzantine detection")
            return {
                "method": "insufficient_clients",
                "malicious_clients": [],
                "honest_clients": [update["client_id"] for update in client_updates],
                "detection_confidence": 0.0
            }
        
        # Extract model updates and client IDs
        model_updates = []
        client_ids = []
        
        for update in client_updates:
            if "model_update" in update and update.get("success", False):
                model_updates.append(update["model_update"])
                client_ids.append(update["client_id"])
        
        if len(model_updates) < 3:
            logger.warning("Too few successful updates for Byzantine detection")
            return {
                "method": "insufficient_updates",
                "malicious_clients": [],
                "honest_clients": client_ids,
                "detection_confidence": 0.0
            }
        
        # Apply detection method
        if self.config.detection_method == "krum":
            return self._krum_detection(model_updates, client_ids)
        elif self.config.detection_method == "multi_krum":
            return self._multi_krum_detection(model_updates, client_ids)
        elif self.config.detection_method == "trimmed_mean":
            return self._trimmed_mean_detection(model_updates, client_ids)
        elif self.config.detection_method == "median":
            return self._median_detection(model_updates, client_ids)
        elif self.config.detection_method == "clustering":
            return self._clustering_detection(model_updates, client_ids)
        else:
            raise ValueError(f"Unknown detection method: {self.config.detection_method}")
    
    def _flatten_model_update(self, model_update: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten model update to 1D tensor for analysis."""
        flattened_params = []
        for param_name, param_tensor in model_update.items():
            if param_tensor is not None:
                flattened_params.append(param_tensor.flatten())
        
        if not flattened_params:
            return torch.tensor([])
        
        return torch.cat(flattened_params)
    
    def _compute_pairwise_distances(self, flattened_updates: List[torch.Tensor]) -> torch.Tensor:
        """Compute pairwise L2 distances between flattened updates."""
        n = len(flattened_updates)
        distances = torch.zeros(n, n)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(flattened_updates[i] - flattened_updates[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _krum_detection(self, model_updates: List[Dict[str, torch.Tensor]], 
                       client_ids: List[int]) -> Dict[str, Any]:
        """
        Krum algorithm for Byzantine detection.
        Selects the update closest to its k-nearest neighbors.
        """
        n = len(model_updates)
        f = int(self.config.max_malicious_ratio * n)  # Max number of malicious clients
        k = n - f - 2  # Number of closest clients to consider
        
        if k <= 0:
            logger.warning("Krum: Too many potential malicious clients")
            return {
                "method": "krum",
                "malicious_clients": [],
                "honest_clients": client_ids,
                "detection_confidence": 0.0,
                "selected_client": client_ids[0] if client_ids else None
            }
        
        # Flatten all updates
        flattened_updates = [self._flatten_model_update(update) for update in model_updates]
        
        # Compute pairwise distances
        distances = self._compute_pairwise_distances(flattened_updates)
        
        # For each client, compute sum of distances to k closest clients
        krum_scores = []
        for i in range(n):
            # Get distances to all other clients
            client_distances = distances[i].clone()
            client_distances[i] = float('inf')  # Exclude self
            
            # Get k smallest distances
            k_distances, _ = torch.topk(client_distances, min(k, n-1), largest=False)
            krum_score = torch.sum(k_distances)
            krum_scores.append(krum_score)
        
        # Select client with minimum Krum score (most similar to neighbors)
        krum_scores_tensor = torch.tensor(krum_scores)
        selected_idx = torch.argmin(krum_scores_tensor).item()
        selected_client = client_ids[selected_idx]
        
        # Detect outliers based on Krum scores
        mean_score = torch.mean(krum_scores_tensor)
        std_score = torch.std(krum_scores_tensor)
        threshold = mean_score + self.config.statistical_threshold * std_score
        
        malicious_clients = []
        honest_clients = []
        
        for i, (client_id, score) in enumerate(zip(client_ids, krum_scores)):
            if score > threshold:
                malicious_clients.append(client_id)
                self.malicious_clients.add(client_id)
            else:
                honest_clients.append(client_id)
        
        detection_confidence = min(1.0, std_score / max(mean_score, 1e-6))
        
        logger.info(f"Krum detection: {len(malicious_clients)} malicious, {len(honest_clients)} honest")
        
        return {
            "method": "krum",
            "malicious_clients": malicious_clients,
            "honest_clients": honest_clients,
            "detection_confidence": detection_confidence,
            "selected_client": selected_client,
            "krum_scores": krum_scores
        }
    
    def _multi_krum_detection(self, model_updates: List[Dict[str, torch.Tensor]], 
                             client_ids: List[int]) -> Dict[str, Any]:
        """
        Multi-Krum algorithm: selects multiple honest clients and averages them.
        """
        n = len(model_updates)
        f = int(self.config.max_malicious_ratio * n)
        m = n - f  # Number of clients to select
        
        if m <= 0:
            return self._krum_detection(model_updates, client_ids)
        
        # Use Krum to get scores
        krum_result = self._krum_detection(model_updates, client_ids)
        krum_scores = krum_result["krum_scores"]
        
        # Select m clients with lowest Krum scores
        krum_scores_tensor = torch.tensor(krum_scores)
        _, selected_indices = torch.topk(krum_scores_tensor, min(m, n), largest=False)
        
        honest_clients = [client_ids[i] for i in selected_indices.tolist()]
        malicious_clients = [client_id for client_id in client_ids if client_id not in honest_clients]
        
        for client_id in malicious_clients:
            self.malicious_clients.add(client_id)
        
        logger.info(f"Multi-Krum: selected {len(honest_clients)} honest clients")
        
        return {
            "method": "multi_krum",
            "malicious_clients": malicious_clients,
            "honest_clients": honest_clients,
            "detection_confidence": krum_result["detection_confidence"],
            "selected_clients": honest_clients
        }
    
    def _trimmed_mean_detection(self, model_updates: List[Dict[str, torch.Tensor]], 
                               client_ids: List[int]) -> Dict[str, Any]:
        """
        Trimmed mean: removes extreme values and computes mean of remaining updates.
        """
        n = len(model_updates)
        f = int(self.config.max_malicious_ratio * n)
        
        # Flatten updates for analysis
        flattened_updates = [self._flatten_model_update(update) for update in model_updates]
        
        if not flattened_updates or len(flattened_updates[0]) == 0:
            return {
                "method": "trimmed_mean",
                "malicious_clients": [],
                "honest_clients": client_ids,
                "detection_confidence": 0.0
            }
        
        # Stack all updates
        stacked_updates = torch.stack(flattened_updates)
        
        # Compute distances from median
        median_update = torch.median(stacked_updates, dim=0)[0]
        distances_from_median = []
        
        for i, update in enumerate(flattened_updates):
            dist = torch.norm(update - median_update)
            distances_from_median.append(dist.item())
        
        # Sort clients by distance from median
        client_distances = list(zip(client_ids, distances_from_median))
        client_distances.sort(key=lambda x: x[1])
        
        # Select clients with smallest distances (trim outliers)
        num_to_keep = n - 2 * f  # Remove f from each end
        num_to_keep = max(1, num_to_keep)  # Keep at least one
        
        honest_clients = [client_id for client_id, _ in client_distances[:num_to_keep]]
        malicious_clients = [client_id for client_id, _ in client_distances[num_to_keep:]]
        
        for client_id in malicious_clients:
            self.malicious_clients.add(client_id)
        
        # Compute detection confidence based on distance variance
        distances = [dist for _, dist in client_distances]
        detection_confidence = np.std(distances) / max(np.mean(distances), 1e-6)
        detection_confidence = min(1.0, detection_confidence)
        
        logger.info(f"Trimmed mean: {len(malicious_clients)} outliers removed")
        
        return {
            "method": "trimmed_mean",
            "malicious_clients": malicious_clients,
            "honest_clients": honest_clients,
            "detection_confidence": detection_confidence,
            "distances_from_median": distances_from_median
        }
    
    def _median_detection(self, model_updates: List[Dict[str, torch.Tensor]], 
                         client_ids: List[int]) -> Dict[str, Any]:
        """
        Coordinate-wise median: detects outliers in each parameter dimension.
        """
        # For simplicity, use median-based outlier detection
        flattened_updates = [self._flatten_model_update(update) for update in model_updates]
        
        if not flattened_updates or len(flattened_updates[0]) == 0:
            return {
                "method": "median",
                "malicious_clients": [],
                "honest_clients": client_ids,
                "detection_confidence": 0.0
            }
        
        # Stack updates and compute coordinate-wise statistics
        stacked_updates = torch.stack(flattened_updates)
        median_values = torch.median(stacked_updates, dim=0)[0]
        mad_values = torch.median(torch.abs(stacked_updates - median_values), dim=0)[0]
        
        # Detect outliers using median absolute deviation
        outlier_scores = []
        for update in flattened_updates:
            # Modified Z-score using MAD
            mad_score = torch.median(torch.abs(update - median_values) / (mad_values + 1e-6))
            outlier_scores.append(mad_score.item())
        
        # Classify clients based on outlier scores
        threshold = self.config.statistical_threshold
        malicious_clients = []
        honest_clients = []
        
        for client_id, score in zip(client_ids, outlier_scores):
            if score > threshold:
                malicious_clients.append(client_id)
                self.malicious_clients.add(client_id)
            else:
                honest_clients.append(client_id)
        
        detection_confidence = np.std(outlier_scores) / max(np.mean(outlier_scores), 1e-6)
        detection_confidence = min(1.0, detection_confidence)
        
        logger.info(f"Median detection: {len(malicious_clients)} outliers detected")
        
        return {
            "method": "median",
            "malicious_clients": malicious_clients,
            "honest_clients": honest_clients,
            "detection_confidence": detection_confidence,
            "outlier_scores": outlier_scores
        }
    
    def _clustering_detection(self, model_updates: List[Dict[str, torch.Tensor]], 
                             client_ids: List[int]) -> Dict[str, Any]:
        """
        Clustering-based detection: groups similar updates and identifies outliers.
        """
        # Flatten updates for clustering
        flattened_updates = [self._flatten_model_update(update) for update in model_updates]
        
        if not flattened_updates or len(flattened_updates[0]) == 0:
            return {
                "method": "clustering",
                "malicious_clients": [],
                "honest_clients": client_ids,
                "detection_confidence": 0.0
            }
        
        # Convert to numpy for sklearn
        updates_np = torch.stack(flattened_updates).numpy()
        
        # Standardize features for clustering
        scaler = StandardScaler()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress sklearn warnings
            updates_scaled = scaler.fit_transform(updates_np)
        
        # Apply DBSCAN clustering
        dbscan = DBSCAN(
            eps=self.config.clustering_eps,
            min_samples=self.config.clustering_min_samples
        )
        
        cluster_labels = dbscan.fit_predict(updates_scaled)
        
        # Analyze clusters
        unique_labels = set(cluster_labels)
        cluster_sizes = {label: np.sum(cluster_labels == label) for label in unique_labels}
        
        # Largest cluster is assumed to be honest clients
        if -1 in cluster_sizes:  # -1 indicates outliers in DBSCAN
            honest_cluster = max([label for label in unique_labels if label != -1], 
                                key=lambda x: cluster_sizes[x], default=0)
        else:
            honest_cluster = max(unique_labels, key=lambda x: cluster_sizes[x])
        
        # Classify clients
        honest_clients = []
        malicious_clients = []
        
        for i, (client_id, label) in enumerate(zip(client_ids, cluster_labels)):
            if label == honest_cluster:
                honest_clients.append(client_id)
            else:
                malicious_clients.append(client_id)
                self.malicious_clients.add(client_id)
        
        # Detection confidence based on cluster separation
        if len(unique_labels) > 1:
            detection_confidence = 1.0 - (cluster_sizes[honest_cluster] / len(client_ids))
        else:
            detection_confidence = 0.0
        
        logger.info(f"Clustering detection: {len(unique_labels)} clusters, "
                   f"{len(malicious_clients)} outliers")
        
        return {
            "method": "clustering",
            "malicious_clients": malicious_clients,
            "honest_clients": honest_clients,
            "detection_confidence": detection_confidence,
            "cluster_labels": cluster_labels.tolist(),
            "cluster_sizes": cluster_sizes
        }
    
    def update_client_reputation(self, client_id: int, is_malicious: bool):
        """Update client reputation based on detection results."""
        if client_id not in self.client_reputation:
            self.client_reputation[client_id] = {"honest_count": 0, "malicious_count": 0}
        
        if is_malicious:
            self.client_reputation[client_id]["malicious_count"] += 1
        else:
            self.client_reputation[client_id]["honest_count"] += 1
    
    def get_client_trust_score(self, client_id: int) -> float:
        """Get trust score for a client (0.0 = malicious, 1.0 = trusted)."""
        if client_id not in self.client_reputation:
            return 0.5  # Neutral for new clients
        
        rep = self.client_reputation[client_id]
        total = rep["honest_count"] + rep["malicious_count"]
        
        if total == 0:
            return 0.5
        
        return rep["honest_count"] / total
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of Byzantine detection history."""
        return {
            "total_detections": len(self.detection_history),
            "unique_malicious_clients": len(self.malicious_clients),
            "malicious_clients": list(self.malicious_clients),
            "client_reputation": self.client_reputation.copy(),
            "detection_methods_used": list(set(
                event.get("method", "unknown") for event in self.detection_history
            ))
        }


# Factory functions for common configurations
def create_krum_detector(max_malicious_ratio: float = 0.33) -> ByzantineDetector:
    """Create Krum-based Byzantine detector."""
    config = ByzantineDetectionConfig(
        max_malicious_ratio=max_malicious_ratio,
        detection_method="krum"
    )
    return ByzantineDetector(config)


def create_multi_krum_detector(max_malicious_ratio: float = 0.33) -> ByzantineDetector:
    """Create Multi-Krum Byzantine detector."""
    config = ByzantineDetectionConfig(
        max_malicious_ratio=max_malicious_ratio,
        detection_method="multi_krum"
    )
    return ByzantineDetector(config)


def create_trimmed_mean_detector(max_malicious_ratio: float = 0.33) -> ByzantineDetector:
    """Create trimmed mean Byzantine detector."""
    config = ByzantineDetectionConfig(
        max_malicious_ratio=max_malicious_ratio,
        detection_method="trimmed_mean"
    )
    return ByzantineDetector(config)


def create_clustering_detector(max_malicious_ratio: float = 0.33) -> ByzantineDetector:
    """Create clustering-based Byzantine detector."""
    config = ByzantineDetectionConfig(
        max_malicious_ratio=max_malicious_ratio,
        detection_method="clustering",
        clustering_eps=0.3,
        clustering_min_samples=2
    )
    return ByzantineDetector(config)


if __name__ == "__main__":
    # Test Byzantine detection
    print("Testing QFLARE Byzantine Fault Tolerance...")
    
    # Create detector
    detector = create_krum_detector(max_malicious_ratio=0.3)
    print(f"✓ Byzantine detector created: {detector.config.detection_method}")
    
    # Create mock client updates
    mock_updates = []
    for i in range(6):
        # Create normal update
        if i < 4:  # 4 honest clients
            update = {
                "client_id": i,
                "success": True,
                "model_update": {
                    "layer1": torch.randn(5, 3) * 0.1,  # Small normal updates
                    "layer2": torch.randn(2, 5) * 0.1
                }
            }
        else:  # 2 malicious clients
            update = {
                "client_id": i,
                "success": True,
                "model_update": {
                    "layer1": torch.randn(5, 3) * 2.0,  # Large malicious updates
                    "layer2": torch.randn(2, 5) * 2.0
                }
            }
        mock_updates.append(update)
    
    # Test detection
    result = detector.detect_byzantine_updates(mock_updates)
    print(f"✓ Detection completed: {len(result['malicious_clients'])} malicious detected")
    print(f"  Method: {result['method']}")
    print(f"  Malicious clients: {result['malicious_clients']}")
    print(f"  Honest clients: {result['honest_clients']}")
    print(f"  Confidence: {result['detection_confidence']:.3f}")
    
    print("Byzantine fault tolerance test completed!")