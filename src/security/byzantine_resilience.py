"""
QFLARE Byzantine Resilience System
Implements robust aggregation methods to handle malicious participants
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import json
import math
from scipy import stats
from collections import defaultdict
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ByzantineAttack(Enum):
    """Types of Byzantine attacks"""
    HONEST = "honest"
    RANDOM = "random"
    SIGN_FLIP = "sign_flip"
    GAUSSIAN = "gaussian"
    LABEL_FLIP = "label_flip"
    BACKDOOR = "backdoor"
    MODEL_POISONING = "model_poisoning"
    SYBIL = "sybil"

class AggregationMethod(Enum):
    """Robust aggregation methods"""
    FEDAVG = "fedavg"  # Standard FedAvg (not Byzantine-robust)
    KRUM = "krum"
    MULTI_KRUM = "multi_krum"
    TRIMMED_MEAN = "trimmed_mean"
    MEDIAN = "coordinate_median"
    PHOCAS = "phocas"
    FLAME = "flame"
    FOOLSGOLD = "foolsgold"
    SPECTRAL = "spectral"

@dataclass
class ClientReputation:
    """Tracks client reputation and behavior"""
    client_id: str
    reputation_score: float = 1.0
    total_submissions: int = 0
    accepted_submissions: int = 0
    rejected_submissions: int = 0
    anomaly_score: float = 0.0
    last_update_time: float = field(default_factory=time.time)
    historical_gradients: List[torch.Tensor] = field(default_factory=list)
    cosine_similarities: List[float] = field(default_factory=list)
    
    def update_reputation(self, accepted: bool, anomaly_score: float = 0.0):
        """Update client reputation based on submission acceptance"""
        self.total_submissions += 1
        self.anomaly_score = anomaly_score
        
        if accepted:
            self.accepted_submissions += 1
            # Increase reputation for accepted submissions
            self.reputation_score = min(1.0, self.reputation_score + 0.1)
        else:
            self.rejected_submissions += 1
            # Decrease reputation for rejected submissions
            self.reputation_score = max(0.0, self.reputation_score - 0.2)
        
        self.last_update_time = time.time()
        
        logger.debug(f"Client {self.client_id} reputation updated: {self.reputation_score:.3f}")
    
    def acceptance_rate(self) -> float:
        """Calculate acceptance rate of submissions"""
        if self.total_submissions == 0:
            return 1.0
        return self.accepted_submissions / self.total_submissions
    
    def to_dict(self) -> Dict:
        """Export reputation to dictionary"""
        return {
            'client_id': self.client_id,
            'reputation_score': self.reputation_score,
            'total_submissions': self.total_submissions,
            'accepted_submissions': self.accepted_submissions,
            'rejected_submissions': self.rejected_submissions,
            'acceptance_rate': self.acceptance_rate(),
            'anomaly_score': self.anomaly_score,
            'last_update_time': self.last_update_time
        }

class ByzantineDetector:
    """Detects Byzantine clients using statistical methods"""
    
    def __init__(self, detection_threshold: float = 2.0):
        """
        Initialize Byzantine detector
        
        Args:
            detection_threshold: Z-score threshold for anomaly detection
        """
        self.detection_threshold = detection_threshold
        self.gradient_history = defaultdict(list)
    
    def detect_statistical_outliers(self, gradients: List[torch.Tensor], 
                                  client_ids: List[str]) -> List[bool]:
        """
        Detect statistical outliers in gradient updates
        
        Args:
            gradients: List of gradient tensors
            client_ids: Corresponding client IDs
            
        Returns:
            List of boolean flags indicating outliers
        """
        if len(gradients) < 3:
            return [False] * len(gradients)
        
        # Flatten gradients for analysis
        flattened_grads = [grad.flatten() for grad in gradients]
        
        # Calculate pairwise distances
        distances = []
        for i, grad_i in enumerate(flattened_grads):
            client_distances = []
            for j, grad_j in enumerate(flattened_grads):
                if i != j:
                    # Cosine distance
                    cos_sim = torch.nn.functional.cosine_similarity(
                        grad_i.unsqueeze(0), grad_j.unsqueeze(0)
                    ).item()
                    distance = 1 - cos_sim
                    client_distances.append(distance)
            distances.append(np.mean(client_distances))
        
        # Detect outliers using z-score
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        outliers = []
        for i, distance in enumerate(distances):
            z_score = abs(distance - mean_distance) / (std_distance + 1e-8)
            is_outlier = z_score > self.detection_threshold
            outliers.append(is_outlier)
            
            if is_outlier:
                logger.warning(f"Client {client_ids[i]} detected as statistical outlier (z-score: {z_score:.3f})")
        
        return outliers
    
    def detect_sign_flip_attack(self, gradients: List[torch.Tensor], 
                               client_ids: List[str]) -> List[bool]:
        """
        Detect sign-flipping attacks
        
        Args:
            gradients: List of gradient tensors
            client_ids: Corresponding client IDs
            
        Returns:
            List of boolean flags indicating sign-flip attackers
        """
        if len(gradients) < 3:
            return [False] * len(gradients)
        
        # Calculate mean gradient
        mean_grad = torch.stack(gradients).mean(dim=0)
        
        attackers = []
        for i, grad in enumerate(gradients):
            # Check if gradient is approximately negative of mean
            dot_product = torch.dot(grad.flatten(), mean_grad.flatten()).item()
            grad_norm = torch.norm(grad).item()
            mean_norm = torch.norm(mean_grad).item()
            
            # Normalize dot product
            normalized_dot = dot_product / (grad_norm * mean_norm + 1e-8)
            
            # Sign flip detection: strong negative correlation
            is_sign_flip = normalized_dot < -0.5
            attackers.append(is_sign_flip)
            
            if is_sign_flip:
                logger.warning(f"Client {client_ids[i]} detected as sign-flip attacker (correlation: {normalized_dot:.3f})")
        
        return attackers
    
    def detect_backdoor_attack(self, gradients: List[torch.Tensor], 
                              client_ids: List[str],
                              model: nn.Module = None) -> List[bool]:
        """
        Detect potential backdoor attacks using gradient analysis
        
        Args:
            gradients: List of gradient tensors
            client_ids: Corresponding client IDs
            model: Current model (optional, for more sophisticated detection)
            
        Returns:
            List of boolean flags indicating backdoor attackers
        """
        # Simplified backdoor detection based on gradient magnitude anomalies
        grad_norms = [torch.norm(grad).item() for grad in gradients]
        
        # Detect unusually large gradients (potential backdoor injection)
        mean_norm = np.mean(grad_norms)
        std_norm = np.std(grad_norms)
        
        attackers = []
        for i, norm in enumerate(grad_norms):
            z_score = abs(norm - mean_norm) / (std_norm + 1e-8)
            is_backdoor = norm > mean_norm + 3 * std_norm
            attackers.append(is_backdoor)
            
            if is_backdoor:
                logger.warning(f"Client {client_ids[i]} detected as potential backdoor attacker (norm: {norm:.3f}, z-score: {z_score:.3f})")
        
        return attackers

class RobustAggregator:
    """Implements robust aggregation methods for Byzantine fault tolerance"""
    
    def __init__(self, method: AggregationMethod = AggregationMethod.KRUM, 
                 byzantine_ratio: float = 0.33):
        """
        Initialize robust aggregator
        
        Args:
            method: Aggregation method to use
            byzantine_ratio: Expected ratio of Byzantine clients
        """
        self.method = method
        self.byzantine_ratio = byzantine_ratio
        self.detector = ByzantineDetector()
        
        logger.info(f"Robust aggregator initialized: {method.value}, Byzantine ratio: {byzantine_ratio}")
    
    def aggregate(self, gradients: List[torch.Tensor], 
                 client_ids: List[str],
                 reputations: Optional[Dict[str, ClientReputation]] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Perform robust aggregation
        
        Args:
            gradients: List of gradient tensors
            client_ids: Corresponding client IDs
            reputations: Client reputation scores
            
        Returns:
            Tuple of (aggregated_gradient, aggregation_info)
        """
        start_time = time.time()
        
        if len(gradients) == 0:
            raise ValueError("No gradients provided for aggregation")
        
        # Initialize aggregation info
        agg_info = {
            'method': self.method.value,
            'total_clients': len(gradients),
            'selected_clients': [],
            'rejected_clients': [],
            'byzantine_detected': [],
            'aggregation_time': 0.0
        }
        
        # Apply selected aggregation method
        if self.method == AggregationMethod.FEDAVG:
            aggregated_grad, info = self._fedavg(gradients, client_ids, reputations)
        elif self.method == AggregationMethod.KRUM:
            aggregated_grad, info = self._krum(gradients, client_ids)
        elif self.method == AggregationMethod.MULTI_KRUM:
            aggregated_grad, info = self._multi_krum(gradients, client_ids)
        elif self.method == AggregationMethod.TRIMMED_MEAN:
            aggregated_grad, info = self._trimmed_mean(gradients, client_ids)
        elif self.method == AggregationMethod.MEDIAN:
            aggregated_grad, info = self._coordinate_median(gradients, client_ids)
        elif self.method == AggregationMethod.PHOCAS:
            aggregated_grad, info = self._phocas(gradients, client_ids, reputations)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.method}")
        
        # Update aggregation info
        agg_info.update(info)
        agg_info['aggregation_time'] = time.time() - start_time
        
        logger.info(f"Robust aggregation completed: {self.method.value}, "
                   f"selected {len(agg_info['selected_clients'])}/{len(gradients)} clients, "
                   f"time: {agg_info['aggregation_time']:.3f}s")
        
        return aggregated_grad, agg_info
    
    def _fedavg(self, gradients: List[torch.Tensor], client_ids: List[str],
                reputations: Optional[Dict[str, ClientReputation]]) -> Tuple[torch.Tensor, Dict]:
        """Standard FedAvg aggregation (not Byzantine-robust)"""
        
        if reputations:
            # Weighted by reputation
            weights = [reputations.get(cid, ClientReputation(cid)).reputation_score for cid in client_ids]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        else:
            # Uniform weights
            weights = [1.0 / len(gradients)] * len(gradients)
        
        # Weighted average
        aggregated = torch.zeros_like(gradients[0])
        for grad, weight in zip(gradients, weights):
            aggregated += weight * grad
        
        info = {
            'selected_clients': client_ids,
            'rejected_clients': [],
            'byzantine_detected': []
        }
        
        return aggregated, info
    
    def _krum(self, gradients: List[torch.Tensor], client_ids: List[str]) -> Tuple[torch.Tensor, Dict]:
        """Krum aggregation - selects most representative gradient"""
        
        n = len(gradients)
        f = int(n * self.byzantine_ratio)  # Expected number of Byzantine clients
        
        if n < 2 * f + 1:
            logger.warning("Insufficient honest clients for Krum guarantee")
        
        # Calculate pairwise distances
        distances = torch.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(gradients[i] - gradients[j]).item()
                distances[i, j] = dist
                distances[j, i] = dist
        
        # For each client, sum distances to k closest neighbors
        k = n - f - 1
        scores = []
        
        for i in range(n):
            client_distances = distances[i].clone()
            client_distances[i] = float('inf')  # Exclude self
            
            # Sum k smallest distances
            k_distances, _ = torch.topk(client_distances, k, largest=False)
            score = k_distances.sum().item()
            scores.append(score)
        
        # Select client with smallest score
        selected_idx = np.argmin(scores)
        selected_client = client_ids[selected_idx]
        
        rejected_clients = [cid for i, cid in enumerate(client_ids) if i != selected_idx]
        
        info = {
            'selected_clients': [selected_client],
            'rejected_clients': rejected_clients,
            'byzantine_detected': [],
            'krum_scores': scores
        }
        
        logger.debug(f"Krum selected client {selected_client} with score {scores[selected_idx]:.3f}")
        
        return gradients[selected_idx], info
    
    def _multi_krum(self, gradients: List[torch.Tensor], client_ids: List[str]) -> Tuple[torch.Tensor, Dict]:
        """Multi-Krum aggregation - averages multiple selected gradients"""
        
        n = len(gradients)
        f = int(n * self.byzantine_ratio)
        m = n - f  # Number of gradients to select
        
        # Calculate pairwise distances (same as Krum)
        distances = torch.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(gradients[i] - gradients[j]).item()
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Calculate scores for each client
        k = n - f - 1
        scores = []
        
        for i in range(n):
            client_distances = distances[i].clone()
            client_distances[i] = float('inf')
            
            k_distances, _ = torch.topk(client_distances, k, largest=False)
            score = k_distances.sum().item()
            scores.append(score)
        
        # Select m clients with lowest scores
        selected_indices = np.argpartition(scores, m)[:m]
        selected_clients = [client_ids[i] for i in selected_indices]
        rejected_clients = [client_ids[i] for i in range(n) if i not in selected_indices]
        
        # Average selected gradients
        selected_gradients = [gradients[i] for i in selected_indices]
        aggregated = torch.stack(selected_gradients).mean(dim=0)
        
        info = {
            'selected_clients': selected_clients,
            'rejected_clients': rejected_clients,
            'byzantine_detected': [],
            'multikrum_scores': scores
        }
        
        logger.debug(f"Multi-Krum selected {len(selected_clients)} clients")
        
        return aggregated, info
    
    def _trimmed_mean(self, gradients: List[torch.Tensor], client_ids: List[str]) -> Tuple[torch.Tensor, Dict]:
        """Trimmed mean aggregation - removes extreme values coordinate-wise"""
        
        f = int(len(gradients) * self.byzantine_ratio)
        
        # Stack gradients
        stacked_grads = torch.stack(gradients)
        
        # Apply trimmed mean coordinate-wise
        sorted_grads, _ = torch.sort(stacked_grads, dim=0)
        
        # Remove f largest and f smallest values for each coordinate
        if f > 0 and len(gradients) > 2 * f:
            trimmed_grads = sorted_grads[f:-f]
        else:
            trimmed_grads = sorted_grads
        
        # Calculate mean of remaining gradients
        aggregated = trimmed_grads.mean(dim=0)
        
        # All clients contribute, but extreme values are trimmed
        info = {
            'selected_clients': client_ids,
            'rejected_clients': [],
            'byzantine_detected': [],
            'trimmed_count': f
        }
        
        logger.debug(f"Trimmed mean: removed {f} extreme values per coordinate")
        
        return aggregated, info
    
    def _coordinate_median(self, gradients: List[torch.Tensor], client_ids: List[str]) -> Tuple[torch.Tensor, Dict]:
        """Coordinate-wise median aggregation"""
        
        # Stack gradients
        stacked_grads = torch.stack(gradients)
        
        # Calculate median for each coordinate
        aggregated = torch.median(stacked_grads, dim=0)[0]
        
        info = {
            'selected_clients': client_ids,
            'rejected_clients': [],
            'byzantine_detected': []
        }
        
        logger.debug("Coordinate-wise median aggregation completed")
        
        return aggregated, info
    
    def _phocas(self, gradients: List[torch.Tensor], client_ids: List[str],
               reputations: Optional[Dict[str, ClientReputation]]) -> Tuple[torch.Tensor, Dict]:
        """PHOCAS aggregation with reputation-based weighting"""
        
        # Detect Byzantine clients
        outliers = self.detector.detect_statistical_outliers(gradients, client_ids)
        sign_flippers = self.detector.detect_sign_flip_attack(gradients, client_ids)
        
        byzantine_detected = []
        honest_gradients = []
        honest_clients = []
        
        for i, (grad, client_id) in enumerate(zip(gradients, client_ids)):
            is_byzantine = outliers[i] or sign_flippers[i]
            
            if is_byzantine:
                byzantine_detected.append(client_id)
                logger.warning(f"PHOCAS detected Byzantine client: {client_id}")
            else:
                honest_gradients.append(grad)
                honest_clients.append(client_id)
        
        if not honest_gradients:
            # Fallback to trimmed mean if all clients are flagged
            logger.warning("PHOCAS flagged all clients as Byzantine, falling back to trimmed mean")
            return self._trimmed_mean(gradients, client_ids)
        
        # Reputation-weighted aggregation of honest clients
        if reputations:
            weights = [reputations.get(cid, ClientReputation(cid)).reputation_score for cid in honest_clients]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights] if total_weight > 0 else [1.0 / len(weights)] * len(weights)
        else:
            weights = [1.0 / len(honest_gradients)] * len(honest_gradients)
        
        # Weighted aggregation
        aggregated = torch.zeros_like(honest_gradients[0])
        for grad, weight in zip(honest_gradients, weights):
            aggregated += weight * grad
        
        rejected_clients = [cid for cid in client_ids if cid not in honest_clients]
        
        info = {
            'selected_clients': honest_clients,
            'rejected_clients': rejected_clients,
            'byzantine_detected': byzantine_detected
        }
        
        logger.info(f"PHOCAS aggregation: {len(honest_clients)} honest, {len(byzantine_detected)} Byzantine")
        
        return aggregated, info

class ByzantineResilienceSystem:
    """
    Complete Byzantine resilience system for QFLARE
    """
    
    def __init__(self, aggregation_method: AggregationMethod = AggregationMethod.KRUM,
                 byzantine_ratio: float = 0.33,
                 reputation_decay: float = 0.95):
        """
        Initialize Byzantine resilience system
        
        Args:
            aggregation_method: Robust aggregation method
            byzantine_ratio: Expected ratio of Byzantine clients  
            reputation_decay: Decay factor for reputation scores
        """
        self.aggregator = RobustAggregator(aggregation_method, byzantine_ratio)
        self.detector = ByzantineDetector()
        self.client_reputations: Dict[str, ClientReputation] = {}
        self.reputation_decay = reputation_decay
        self.round_counter = 0
        
        logger.info(f"Byzantine resilience system initialized: {aggregation_method.value}")
    
    def process_round(self, gradients: List[torch.Tensor], client_ids: List[str]) -> Dict:
        """
        Process a complete federated learning round with Byzantine resilience
        
        Args:
            gradients: Client gradient updates
            client_ids: Corresponding client identifiers
            
        Returns:
            Round processing results including aggregated gradient and detection info
        """
        self.round_counter += 1
        start_time = time.time()
        
        # Initialize reputations for new clients
        for client_id in client_ids:
            if client_id not in self.client_reputations:
                self.client_reputations[client_id] = ClientReputation(client_id)
        
        # Detect Byzantine behavior
        outliers = self.detector.detect_statistical_outliers(gradients, client_ids)
        sign_flippers = self.detector.detect_sign_flip_attack(gradients, client_ids)
        backdoor_attackers = self.detector.detect_backdoor_attack(gradients, client_ids)
        
        # Update reputations based on detections
        for i, client_id in enumerate(client_ids):
            is_malicious = outliers[i] or sign_flippers[i] or backdoor_attackers[i]
            
            anomaly_score = 0.0
            if outliers[i]:
                anomaly_score += 0.3
            if sign_flippers[i]:
                anomaly_score += 0.4
            if backdoor_attackers[i]:
                anomaly_score += 0.5
            
            self.client_reputations[client_id].update_reputation(
                accepted=not is_malicious,
                anomaly_score=anomaly_score
            )
        
        # Perform robust aggregation
        aggregated_gradient, agg_info = self.aggregator.aggregate(
            gradients, client_ids, self.client_reputations
        )
        
        # Apply reputation decay
        self._decay_reputations()
        
        # Compile results
        results = {
            'round': self.round_counter,
            'aggregated_gradient': aggregated_gradient,
            'aggregation_info': agg_info,
            'detection_results': {
                'statistical_outliers': dict(zip(client_ids, outliers)),
                'sign_flippers': dict(zip(client_ids, sign_flippers)),
                'backdoor_attackers': dict(zip(client_ids, backdoor_attackers))
            },
            'client_reputations': {cid: rep.to_dict() for cid, rep in self.client_reputations.items()},
            'processing_time': time.time() - start_time,
            'byzantine_ratio_detected': len(agg_info['byzantine_detected']) / len(client_ids) if client_ids else 0.0
        }
        
        logger.info(f"Round {self.round_counter} processed: "
                   f"{len(agg_info['byzantine_detected'])}/{len(client_ids)} Byzantine clients detected")
        
        return results
    
    def _decay_reputations(self):
        """Apply exponential decay to all reputation scores"""
        for reputation in self.client_reputations.values():
            reputation.reputation_score *= self.reputation_decay
    
    def get_defense_effectiveness(self, attack_success_rate: float) -> Dict:
        """
        Calculate defense effectiveness metrics
        
        Args:
            attack_success_rate: Rate of successful attacks without defense
            
        Returns:
            Defense effectiveness statistics
        """
        total_rounds = self.round_counter
        if total_rounds == 0:
            return {'error': 'No rounds processed yet'}
        
        # Calculate detection accuracy
        total_detections = 0
        correct_detections = 0
        
        for reputation in self.client_reputations.values():
            if reputation.total_submissions > 0:
                # Assume clients with low acceptance rate are attackers
                if reputation.acceptance_rate() < 0.5:
                    total_detections += 1
                    if reputation.anomaly_score > 0:
                        correct_detections += 1
        
        detection_accuracy = correct_detections / total_detections if total_detections > 0 else 0.0
        
        # Calculate defense success rate (1 - attack success rate with defense)
        defense_success_rate = 1.0 - (attack_success_rate * (1.0 - detection_accuracy))
        
        return {
            'total_rounds': total_rounds,
            'total_clients': len(self.client_reputations),
            'detection_accuracy': detection_accuracy,
            'defense_success_rate': defense_success_rate,
            'attack_mitigation': (defense_success_rate - (1.0 - attack_success_rate)) / attack_success_rate if attack_success_rate > 0 else 0.0,
            'aggregation_method': self.aggregator.method.value,
            'reputation_system_active': True
        }

def demo_byzantine_resilience():
    """Demonstration of QFLARE Byzantine resilience"""
    print("🛡️ QFLARE Byzantine Resilience System Demo")
    print("=" * 50)
    
    # Initialize system
    system = ByzantineResilienceSystem(
        aggregation_method=AggregationMethod.KRUM,
        byzantine_ratio=0.33
    )
    
    # Simulate federated learning scenario
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_clients = 10
    grad_size = 100
    num_rounds = 5
    
    print(f"Simulating {num_rounds} rounds with {num_clients} clients")
    print(f"Expected Byzantine ratio: {system.aggregator.byzantine_ratio:.0%}")
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num} ---")
        
        # Generate client gradients
        gradients = []
        client_ids = [f"client_{i:02d}" for i in range(num_clients)]
        
        for i in range(num_clients):
            if i < 7:  # Honest clients
                # Normal gradient with some noise
                grad = torch.randn(grad_size) * 0.1
            elif i == 7:  # Sign-flip attacker
                grad = -torch.randn(grad_size) * 0.5  # Opposite direction
            elif i == 8:  # Gaussian attacker  
                grad = torch.randn(grad_size) * 2.0   # High variance noise
            else:  # Random attacker
                grad = torch.randn(grad_size) * 5.0   # Very high variance
            
            gradients.append(grad)
        
        # Process round with Byzantine resilience
        results = system.process_round(gradients, client_ids)
        
        # Display results
        agg_info = results['aggregation_info']
        detection = results['detection_results']
        
        print(f"Selected clients: {len(agg_info['selected_clients'])}")
        print(f"Rejected clients: {agg_info['rejected_clients']}")
        print(f"Byzantine detected: {agg_info['byzantine_detected']}")
        
        # Show reputation scores
        print("Client reputations:")
        for client_id, rep_dict in results['client_reputations'].items():
            print(f"  {client_id}: {rep_dict['reputation_score']:.3f} "
                  f"(acceptance: {rep_dict['acceptance_rate']:.2f})")
        
        print(f"Processing time: {results['processing_time']:.3f}s")
    
    # Demonstrate different aggregation methods
    print(f"\n--- Aggregation Method Comparison ---")
    
    # Generate test gradients with known Byzantine clients
    test_gradients = []
    test_client_ids = [f"test_client_{i}" for i in range(8)]
    
    # 5 honest clients
    for i in range(5):
        grad = torch.randn(50) * 0.1 + torch.ones(50) * 0.05  # Consistent direction
        test_gradients.append(grad)
    
    # 3 Byzantine clients
    test_gradients.append(-torch.randn(50) * 0.3)  # Sign-flip
    test_gradients.append(torch.randn(50) * 3.0)   # High variance
    test_gradients.append(torch.randn(50) * 3.0)   # High variance
    
    methods = [AggregationMethod.FEDAVG, AggregationMethod.KRUM, 
               AggregationMethod.TRIMMED_MEAN, AggregationMethod.MEDIAN]
    
    for method in methods:
        aggregator = RobustAggregator(method, byzantine_ratio=0.33)
        agg_grad, info = aggregator.aggregate(test_gradients, test_client_ids)
        
        print(f"{method.value:15s}: selected {len(info['selected_clients'])}/8 clients")
    
    # Calculate defense effectiveness
    print(f"\n--- Defense Effectiveness Analysis ---")
    
    # Simulate attack success rate without defense
    baseline_attack_success = 0.8  # 80% attack success without defense
    
    effectiveness = system.get_defense_effectiveness(baseline_attack_success)
    
    print(f"Total rounds processed: {effectiveness['total_rounds']}")
    print(f"Detection accuracy: {effectiveness['detection_accuracy']:.1%}")
    print(f"Defense success rate: {effectiveness['defense_success_rate']:.1%}")
    print(f"Attack mitigation: {effectiveness['attack_mitigation']:.1%}")
    
    print(f"\n🛡️ Byzantine resilience provides {effectiveness['defense_success_rate']:.1%} defense success rate!")
    print(f"   Aggregation method: {effectiveness['aggregation_method']}")

if __name__ == "__main__":
    demo_byzantine_resilience()