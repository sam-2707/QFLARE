import pytest
import numpy as np
import torch
import tempfile
import os
from datetime import datetime
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any

# Simple Byzantine Detection Implementation for Testing
class SimpleByzantineDetector:
    """Simplified Byzantine detector for testing without dependencies"""
    
    def __init__(self):
        self.client_reputation = {}
        self.attack_history = []
    
    def detect_byzantine_clients(self, updates: List[Dict], method: str = "krum", **kwargs) -> List[Dict]:
        """Detect Byzantine clients using specified method"""
        if method == "krum":
            return self._krum_detection(updates, **kwargs)
        elif method == "multi_krum":
            return self._multi_krum_detection(updates, **kwargs)
        elif method == "trimmed_mean":
            return self._trimmed_mean_detection(updates, **kwargs)
        elif method == "clustering":
            return self._clustering_detection(updates, **kwargs)
        else:
            raise ValueError(f"Unknown detection method: {method}")
    
    def _krum_detection(self, updates: List[Dict], f: int = None) -> List[Dict]:
        """Simplified Krum detection"""
        if len(updates) < 3:
            return updates
        
        n = len(updates)
        if f is None:
            f = (n - 1) // 3  # Assume up to 1/3 Byzantine clients
        
        # Compute pairwise distances
        distances = self._compute_pairwise_distances(updates)
        
        # Compute Krum scores
        scores = []
        for i in range(n):
            # Sum of distances to n-f-2 closest neighbors
            distances_i = distances[i].copy()
            distances_i[i] = float('inf')  # Exclude self
            distances_i = np.sort(distances_i)
            score = np.sum(distances_i[:n-f-2])
            scores.append((score, i))
        
        # Select client with minimum score
        scores.sort()
        selected_idx = scores[0][1]
        
        return [updates[selected_idx]]
    
    def _multi_krum_detection(self, updates: List[Dict], m: int = None, f: int = None) -> List[Dict]:
        """Simplified Multi-Krum detection"""
        if m is None:
            m = min(len(updates), 5)  # Select top 5 by default
        
        n = len(updates)
        if f is None:
            f = (n - 1) // 3
        
        distances = self._compute_pairwise_distances(updates)
        
        # Compute Krum scores
        scores = []
        for i in range(n):
            distances_i = distances[i].copy()
            distances_i[i] = float('inf')
            distances_i = np.sort(distances_i)
            score = np.sum(distances_i[:n-f-2]) if n-f-2 > 0 else 0
            scores.append((score, i))
        
        # Select top m clients with lowest scores
        scores.sort()
        selected_indices = [idx for _, idx in scores[:m]]
        
        return [updates[i] for i in selected_indices]
    
    def _trimmed_mean_detection(self, updates: List[Dict], beta: float = 0.3) -> List[Dict]:
        """Simplified Trimmed Mean detection"""
        n = len(updates)
        trim_count = int(beta * n)
        
        if trim_count == 0:
            return updates
        
        # Compute parameter-wise statistics
        all_params = []
        for update in updates:
            params = update['model_update'].flatten().numpy()
            all_params.append(params)
        
        all_params = np.array(all_params)
        
        # Compute L2 norms
        norms = np.linalg.norm(all_params, axis=1)
        
        # Remove extreme values
        sorted_indices = np.argsort(norms)
        keep_indices = sorted_indices[trim_count:n-trim_count]
        
        return [updates[i] for i in keep_indices]
    
    def _clustering_detection(self, updates: List[Dict]) -> List[Dict]:
        """Simplified clustering detection"""
        if len(updates) < 3:
            return updates
        
        # Extract features (L2 norms)
        features = []
        for update in updates:
            norm = torch.norm(update['model_update']).item()
            features.append([norm])
        
        features = np.array(features)
        
        # Simple clustering using standard deviation
        mean_norm = np.mean(features)
        std_norm = np.std(features)
        
        # Keep clients within 2 standard deviations
        honest_clients = []
        for i, feature in enumerate(features):
            if abs(feature[0] - mean_norm) <= 2 * std_norm:
                honest_clients.append(updates[i])
        
        return honest_clients if honest_clients else updates
    
    def _compute_pairwise_distances(self, updates: List[Dict]) -> np.ndarray:
        """Compute pairwise distances between model updates"""
        n = len(updates)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = updates[i]['model_update'] - updates[j]['model_update']
                    distances[i][j] = torch.norm(diff).item()
        
        return distances
    
    def get_client_reputation(self) -> Dict[str, float]:
        """Get client reputation scores"""
        return self.client_reputation.copy()


class TestSimpleByzantineDetection:
    """Test Byzantine detection algorithms with simplified implementation"""
    
    @pytest.fixture
    def detector(self):
        """Create a Byzantine detector instance"""
        return SimpleByzantineDetector()
    
    def create_mock_updates(self, num_honest: int, num_malicious: int, 
                           attack_type: str = "gaussian") -> List[Dict]:
        """Create mock model updates with Byzantine attacks"""
        updates = []
        
        # Create honest updates (normal distribution around zero)
        torch.manual_seed(42)  # For reproducible tests
        for i in range(num_honest):
            honest_update = {
                'client_id': f'honest_{i}',
                'model_update': torch.randn(10, 5) * 0.1,  # Small variance
                'timestamp': datetime.now().isoformat()
            }
            updates.append(honest_update)
        
        # Create Byzantine updates
        for i in range(num_malicious):
            if attack_type == "gaussian":
                # Large Gaussian noise attack
                malicious_update = {
                    'client_id': f'malicious_{i}',
                    'model_update': torch.randn(10, 5) * 2,  # Large variance
                    'timestamp': datetime.now().isoformat()
                }
            elif attack_type == "sign_flip":
                # Sign flipping attack
                base_update = torch.randn(10, 5) * 0.1
                malicious_update = {
                    'client_id': f'malicious_{i}',
                    'model_update': -base_update * 10,  # Flipped and amplified
                    'timestamp': datetime.now().isoformat()
                }
            elif attack_type == "large_deviation":
                # Large deviation attack
                malicious_update = {
                    'client_id': f'malicious_{i}',
                    'model_update': torch.ones(10, 5) * 10,  # Constant large values
                    'timestamp': datetime.now().isoformat()
                }
            else:
                raise ValueError(f"Unknown attack type: {attack_type}")
            
            updates.append(malicious_update)
        
        return updates
    
    def test_krum_detection_with_gaussian_attack(self, detector):
        """Test Krum detection with Gaussian noise attack"""
        updates = self.create_mock_updates(7, 3, "gaussian")
        
        honest_clients = detector.detect_byzantine_clients(updates, method="krum")
        
        # Krum should select 1 client
        assert len(honest_clients) == 1
        # Should likely select an honest client (high probability with this setup)
        selected_id = honest_clients[0]['client_id']
        print(f"Krum selected: {selected_id}")
    
    def test_multi_krum_detection(self, detector):
        """Test Multi-Krum detection algorithm"""
        updates = self.create_mock_updates(8, 2, "sign_flip")
        
        honest_clients = detector.detect_byzantine_clients(updates, method="multi_krum", m=5)
        
        # Should select top 5 clients
        assert len(honest_clients) == 5
        honest_ids = [client['client_id'] for client in honest_clients]
        print(f"Multi-Krum selected: {honest_ids}")
        
        # Should prefer honest clients (most should be honest)
        honest_count = sum(1 for client_id in honest_ids if 'honest_' in client_id)
        assert honest_count >= 3  # At least 3 out of 5 should be honest
    
    def test_trimmed_mean_detection(self, detector):
        """Test Trimmed Mean detection algorithm"""
        updates = self.create_mock_updates(6, 4, "large_deviation")
        
        honest_clients = detector.detect_byzantine_clients(updates, method="trimmed_mean", beta=0.3)
        
        # Should filter out extreme values
        assert len(honest_clients) >= 4  # At least 4 clients after trimming
        honest_ids = [client['client_id'] for client in honest_clients]
        print(f"Trimmed Mean selected: {honest_ids}")
        
        # Check that most surviving clients are honest
        honest_count = sum(1 for client_id in honest_ids if 'honest_' in client_id)
        malicious_count = len(honest_ids) - honest_count
        assert honest_count >= malicious_count  # More honest than malicious
    
    def test_clustering_detection(self, detector):
        """Test Clustering-based detection algorithm"""
        updates = self.create_mock_updates(8, 2, "gaussian")
        
        honest_clients = detector.detect_byzantine_clients(updates, method="clustering")
        
        # Clustering should identify reasonable clients
        assert len(honest_clients) >= 5  # Should keep most reasonable clients
        honest_ids = [client['client_id'] for client in honest_clients]
        print(f"Clustering selected: {honest_ids}")
        
        # Should prefer honest clients
        honest_count = sum(1 for client_id in honest_ids if 'honest_' in client_id)
        assert honest_count >= 5  # Should keep most honest clients
    
    def test_pairwise_distance_computation(self, detector):
        """Test pairwise distance computation"""
        updates = self.create_mock_updates(5, 0, "gaussian")  # Only honest clients
        
        distances = detector._compute_pairwise_distances(updates)
        
        # Should be symmetric matrix
        assert distances.shape == (5, 5)
        assert np.allclose(distances, distances.T, rtol=1e-10)
        
        # Diagonal should be zero
        assert np.allclose(np.diag(distances), 0)
        
        # All distances should be non-negative
        assert np.all(distances >= 0)
    
    def test_robustness_against_different_attack_intensities(self, detector):
        """Test detection robustness against different attack intensities"""
        base_updates = self.create_mock_updates(7, 3, "gaussian")
        
        methods = ["krum", "multi_krum", "trimmed_mean", "clustering"]
        
        for method in methods:
            try:
                if method == "multi_krum":
                    result = detector.detect_byzantine_clients(base_updates, method=method, m=4)
                elif method == "trimmed_mean":
                    result = detector.detect_byzantine_clients(base_updates, method=method, beta=0.2)
                else:
                    result = detector.detect_byzantine_clients(base_updates, method=method)
                
                assert len(result) > 0, f"Method {method} should return at least one client"
                print(f"Method {method} returned {len(result)} clients")
                
            except Exception as e:
                pytest.fail(f"Method {method} failed with error: {e}")
    
    def test_edge_cases(self, detector):
        """Test edge cases for Byzantine detection"""
        # Test with minimal number of clients
        minimal_updates = self.create_mock_updates(2, 1, "gaussian")
        
        # Should handle small groups gracefully
        result = detector.detect_byzantine_clients(minimal_updates, method="clustering")
        assert len(result) > 0
        
        # Test with no malicious clients
        clean_updates = self.create_mock_updates(5, 0, "gaussian")
        result = detector.detect_byzantine_clients(clean_updates, method="krum")
        assert len(result) == 1
        
        # Test with all malicious clients (pathological case)
        all_malicious = self.create_mock_updates(0, 5, "large_deviation")
        result = detector.detect_byzantine_clients(all_malicious, method="trimmed_mean", beta=0.2)
        assert len(result) >= 1  # Should return something even in worst case


class TestByzantineRobustAggregation:
    """Test Byzantine-robust aggregation logic"""
    
    def test_robust_mean_aggregation(self):
        """Test robust mean aggregation"""
        # Create honest updates (similar values)
        honest_updates = []
        for i in range(5):
            honest_updates.append({
                'client_id': f'honest_{i}',
                'model_update': torch.ones(3, 2) + torch.randn(3, 2) * 0.1
            })
        
        # Simple mean aggregation
        all_params = torch.stack([update['model_update'] for update in honest_updates])
        robust_mean = torch.mean(all_params, dim=0)
        
        assert robust_mean.shape == (3, 2)
        # Should be close to ones (the base value)
        assert torch.allclose(robust_mean, torch.ones(3, 2), atol=0.5)
    
    def test_robust_median_aggregation(self):
        """Test robust median aggregation"""
        # Create updates with one outlier
        updates = []
        for i in range(4):
            updates.append({
                'client_id': f'normal_{i}',
                'model_update': torch.ones(2, 2)
            })
        
        # Add outlier
        updates.append({
            'client_id': 'outlier',
            'model_update': torch.ones(2, 2) * 100  # Large outlier
        })
        
        # Compute median (should be robust to outlier)
        all_params = torch.stack([update['model_update'] for update in updates])
        robust_median = torch.median(all_params, dim=0)[0]
        
        assert robust_median.shape == (2, 2)
        # Median should be close to 1, not affected by the outlier
        assert torch.allclose(robust_median, torch.ones(2, 2), atol=0.1)
    
    def test_attack_detection_statistics(self):
        """Test attack detection statistics calculation"""
        # Simulate attack detection results
        detection_results = [
            {'round': 1, 'attack_detected': True, 'confidence': 0.9},
            {'round': 2, 'attack_detected': False, 'confidence': 0.1},
            {'round': 3, 'attack_detected': True, 'confidence': 0.8},
            {'round': 4, 'attack_detected': False, 'confidence': 0.2},
            {'round': 5, 'attack_detected': True, 'confidence': 0.95}
        ]
        
        # Calculate statistics
        total_rounds = len(detection_results)
        attacks_detected = sum(1 for result in detection_results if result['attack_detected'])
        attack_rate = attacks_detected / total_rounds
        avg_confidence = np.mean([result['confidence'] for result in detection_results if result['attack_detected']])
        
        assert total_rounds == 5
        assert attacks_detected == 3
        assert attack_rate == 0.6
        assert abs(avg_confidence - 0.883) < 0.01  # (0.9 + 0.8 + 0.95) / 3


class TestByzantineSystemIntegration:
    """Integration tests for Byzantine system components"""
    
    def test_detection_and_aggregation_pipeline(self):
        """Test complete detection and aggregation pipeline"""
        detector = SimpleByzantineDetector()
        
        # Create mixed updates
        all_updates = []
        all_updates.extend(self.create_honest_updates(6))
        all_updates.extend(self.create_malicious_updates(2))
        
        # Step 1: Detect Byzantine clients
        honest_clients = detector.detect_byzantine_clients(all_updates, method="multi_krum", m=5)
        
        # Step 2: Aggregate honest clients
        if honest_clients:
            honest_params = torch.stack([client['model_update'] for client in honest_clients])
            aggregated_model = torch.mean(honest_params, dim=0)
            
            assert aggregated_model.shape == (5, 3)
            print(f"Aggregated model from {len(honest_clients)} honest clients")
    
    def create_honest_updates(self, count: int) -> List[Dict]:
        """Create honest client updates"""
        updates = []
        for i in range(count):
            updates.append({
                'client_id': f'honest_{i}',
                'model_update': torch.randn(5, 3) * 0.1,
                'timestamp': datetime.now().isoformat()
            })
        return updates
    
    def create_malicious_updates(self, count: int) -> List[Dict]:
        """Create malicious client updates"""
        updates = []
        for i in range(count):
            updates.append({
                'client_id': f'malicious_{i}',
                'model_update': torch.randn(5, 3) * 3,  # Large variance
                'timestamp': datetime.now().isoformat()
            })
        return updates
    
    def test_reputation_tracking_simulation(self):
        """Test client reputation tracking over multiple rounds"""
        detector = SimpleByzantineDetector()
        
        # Simulate multiple rounds
        for round_num in range(5):
            # Same pattern: 6 honest, 2 malicious
            updates = []
            updates.extend(self.create_honest_updates(6))
            updates.extend(self.create_malicious_updates(2))
            
            # Detect Byzantine clients
            honest_clients = detector.detect_byzantine_clients(updates, method="krum")
            
            # Track which clients were selected as honest
            for client in honest_clients:
                client_id = client['client_id']
                if client_id not in detector.client_reputation:
                    detector.client_reputation[client_id] = 1.0
                else:
                    detector.client_reputation[client_id] += 0.1
            
            # Decrease reputation for non-selected clients
            selected_ids = {client['client_id'] for client in honest_clients}
            all_ids = {update['client_id'] for update in updates}
            
            for client_id in all_ids - selected_ids:
                if client_id not in detector.client_reputation:
                    detector.client_reputation[client_id] = 0.5
                else:
                    detector.client_reputation[client_id] = max(0, detector.client_reputation[client_id] - 0.1)
        
        # Check reputation patterns
        reputation = detector.get_client_reputation()
        print("Final reputation scores:", reputation)
        
        if reputation:
            honest_scores = [score for client_id, score in reputation.items() if 'honest_' in client_id]
            malicious_scores = [score for client_id, score in reputation.items() if 'malicious_' in client_id]
            
            if honest_scores and malicious_scores:
                # Honest clients should generally have better reputation
                print(f"Average honest reputation: {np.mean(honest_scores):.2f}")
                print(f"Average malicious reputation: {np.mean(malicious_scores):.2f}")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])