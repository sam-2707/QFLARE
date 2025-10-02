import pytest
import asyncio
import numpy as np
import torch
import tempfile
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from server.byzantine.detection import ByzantineDetector
from server.byzantine.robust_aggregator import ByzantineRobustAggregator
from server.byzantine.byzantine_fl_controller import ByzantineFLController
from server.api.byzantine_endpoints import router as byzantine_router
from server.aggregator_real import FederatedAggregator
from fastapi.testclient import TestClient
from fastapi import FastAPI

class TestByzantineDetection:
    """Test Byzantine detection algorithms"""
    
    @pytest.fixture
    def detector(self):
        """Create a Byzantine detector instance"""
        return ByzantineDetector()
    
    def create_mock_updates(self, num_honest: int, num_malicious: int, 
                           attack_type: str = "gaussian") -> list:
        """Create mock model updates with Byzantine attacks"""
        updates = []
        
        # Create honest updates (normal distribution around zero)
        for i in range(num_honest):
            honest_update = {
                'client_id': f'honest_{i}',
                'model_update': torch.randn(100, 10),  # 100x10 model
                'timestamp': datetime.now().isoformat()
            }
            updates.append(honest_update)
        
        # Create Byzantine updates
        for i in range(num_malicious):
            if attack_type == "gaussian":
                # Large Gaussian noise attack
                malicious_update = {
                    'client_id': f'malicious_{i}',
                    'model_update': torch.randn(100, 10) * 10,  # 10x larger noise
                    'timestamp': datetime.now().isoformat()
                }
            elif attack_type == "sign_flip":
                # Sign flipping attack
                base_update = torch.randn(100, 10)
                malicious_update = {
                    'client_id': f'malicious_{i}',
                    'model_update': -base_update,
                    'timestamp': datetime.now().isoformat()
                }
            elif attack_type == "large_deviation":
                # Large deviation attack
                malicious_update = {
                    'client_id': f'malicious_{i}',
                    'model_update': torch.ones(100, 10) * 100,  # Constant large values
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
        
        # Should detect most malicious clients
        honest_ids = [client['client_id'] for client in honest_clients]
        assert len(honest_clients) >= 5  # At least 5 honest clients detected
        assert all('honest_' in client_id for client_id in honest_ids[:5])
    
    def test_multi_krum_detection(self, detector):
        """Test Multi-Krum detection algorithm"""
        updates = self.create_mock_updates(8, 2, "sign_flip")
        
        honest_clients = detector.detect_byzantine_clients(updates, method="multi_krum", m=5)
        
        # Should select top 5 honest clients
        assert len(honest_clients) == 5
        honest_ids = [client['client_id'] for client in honest_clients]
        assert all('honest_' in client_id for client_id in honest_ids)
    
    def test_trimmed_mean_detection(self, detector):
        """Test Trimmed Mean detection algorithm"""
        updates = self.create_mock_updates(6, 4, "large_deviation")
        
        honest_clients = detector.detect_byzantine_clients(updates, method="trimmed_mean", beta=0.3)
        
        # Should filter out extreme values
        assert len(honest_clients) >= 4  # At least 4 honest clients
        honest_ids = [client['client_id'] for client in honest_clients]
        # Check that most honest clients are preserved
        honest_count = sum(1 for client_id in honest_ids if 'honest_' in client_id)
        assert honest_count >= 4
    
    def test_clustering_detection(self, detector):
        """Test Clustering-based detection algorithm"""
        updates = self.create_mock_updates(8, 2, "gaussian")
        
        honest_clients = detector.detect_byzantine_clients(updates, method="clustering")
        
        # Clustering should identify the main cluster of honest clients
        assert len(honest_clients) >= 6  # Most honest clients should be in main cluster
        honest_ids = [client['client_id'] for client in honest_clients]
        honest_count = sum(1 for client_id in honest_ids if 'honest_' in client_id)
        assert honest_count >= 6
    
    def test_client_reputation_tracking(self, detector):
        """Test client reputation tracking over multiple rounds"""
        # Round 1: Some clients are malicious
        updates_r1 = self.create_mock_updates(5, 2, "gaussian")
        detector.detect_byzantine_clients(updates_r1, method="krum")
        
        # Round 2: Same malicious clients
        updates_r2 = self.create_mock_updates(5, 2, "sign_flip")
        detector.detect_byzantine_clients(updates_r2, method="krum")
        
        # Check reputation scores
        reputation = detector.get_client_reputation()
        
        # Honest clients should have better reputation than malicious ones
        honest_scores = [score for client_id, score in reputation.items() if 'honest_' in client_id]
        malicious_scores = [score for client_id, score in reputation.items() if 'malicious_' in client_id]
        
        if honest_scores and malicious_scores:
            assert np.mean(honest_scores) > np.mean(malicious_scores)
    
    def test_pairwise_distance_computation(self, detector):
        """Test pairwise distance computation"""
        updates = self.create_mock_updates(5, 0, "gaussian")  # Only honest clients
        
        distances = detector._compute_pairwise_distances(updates)
        
        # Should be symmetric matrix
        assert distances.shape == (5, 5)
        assert np.allclose(distances, distances.T)
        
        # Diagonal should be zero
        assert np.allclose(np.diag(distances), 0)


class TestByzantineRobustAggregator:
    """Test Byzantine-robust aggregation"""
    
    @pytest.fixture
    def base_aggregator_mock(self):
        """Create mock base aggregator"""
        mock = MagicMock()
        mock.get_database_connection.return_value = MagicMock()
        mock.get_database_connection.return_value.cursor.return_value = MagicMock()
        return mock
    
    @pytest.fixture
    def robust_aggregator(self, base_aggregator_mock):
        """Create robust aggregator instance"""
        return ByzantineRobustAggregator(base_aggregator_mock)
    
    def create_mock_updates(self, num_honest: int, num_malicious: int) -> list:
        """Create mock updates for testing"""
        updates = []
        
        # Honest updates (similar parameters)
        for i in range(num_honest):
            updates.append({
                'client_id': f'honest_{i}',
                'model_update': torch.randn(50, 10) * 0.1,  # Small variance
                'timestamp': datetime.now().isoformat()
            })
        
        # Malicious updates (large deviation)
        for i in range(num_malicious):
            updates.append({
                'client_id': f'malicious_{i}',
                'model_update': torch.randn(50, 10) * 5,  # Large variance
                'timestamp': datetime.now().isoformat()
            })
        
        return updates
    
    @pytest.mark.asyncio
    async def test_robust_aggregate_with_attack_detection(self, robust_aggregator):
        """Test robust aggregation with attack detection"""
        mock_updates = self.create_mock_updates(7, 3)
        
        # Mock database operations
        with patch.object(robust_aggregator.base_aggregator, 'get_database_connection') as mock_db:
            mock_conn = MagicMock()
            mock_db.return_value = mock_conn
            
            result = await robust_aggregator.robust_aggregate(
                mock_updates, 
                round_number=1,
                detection_method="krum",
                aggregation_method="mean"
            )
        
        # Check result structure
        assert 'aggregated_model' in result
        assert 'robust_stats' in result
        assert 'honest_clients' in result
        
        # Check robust stats
        stats = result['robust_stats']
        assert stats['total_clients'] == 10
        assert stats['attack_detected'] in [True, False]
        assert 0 <= stats['detection_confidence'] <= 1
    
    @pytest.mark.asyncio
    async def test_different_aggregation_methods(self, robust_aggregator):
        """Test different aggregation methods"""
        mock_updates = self.create_mock_updates(5, 2)
        
        methods = ["mean", "median", "trimmed_mean"]
        
        for method in methods:
            with patch.object(robust_aggregator.base_aggregator, 'get_database_connection') as mock_db:
                mock_conn = MagicMock()
                mock_db.return_value = mock_conn
                
                result = await robust_aggregator.robust_aggregate(
                    mock_updates,
                    round_number=1,
                    aggregation_method=method
                )
                
                assert 'aggregated_model' in result
                assert result['robust_stats']['aggregation_method'] == method
    
    @pytest.mark.asyncio
    async def test_attack_statistics_tracking(self, robust_aggregator):
        """Test attack statistics tracking"""
        # Run multiple rounds with attacks
        for round_num in range(3):
            mock_updates = self.create_mock_updates(6, 4)  # High attack ratio
            
            with patch.object(robust_aggregator.base_aggregator, 'get_database_connection') as mock_db:
                mock_conn = MagicMock()
                mock_db.return_value = mock_conn
                
                await robust_aggregator.robust_aggregate(
                    mock_updates,
                    round_number=round_num + 1
                )
        
        # Get attack statistics
        stats = robust_aggregator.get_attack_statistics()
        
        assert 'total_rounds' in stats
        assert 'attacks_detected' in stats
        assert 'attack_rate' in stats
        assert stats['total_rounds'] == 3


class TestByzantineFLController:
    """Test Byzantine FL Controller"""
    
    @pytest.fixture
    def mock_websocket_manager(self):
        """Create mock WebSocket manager"""
        mock = AsyncMock()
        mock.broadcast_message = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_robust_aggregator(self):
        """Create mock robust aggregator"""
        mock = AsyncMock()
        mock.robust_aggregate = AsyncMock(return_value={
            'aggregated_model': torch.randn(10, 5),
            'robust_stats': {
                'total_clients': 10,
                'honest_clients': 7,
                'malicious_clients': 3,
                'attack_detected': True,
                'detection_confidence': 0.85,
                'detection_method': 'krum',
                'aggregation_method': 'mean'
            },
            'honest_clients': [{'client_id': f'client_{i}'} for i in range(7)]
        })
        return mock
    
    @pytest.fixture
    def byzantine_controller(self, mock_websocket_manager, mock_robust_aggregator):
        """Create Byzantine FL controller"""
        controller = ByzantineFLController(mock_websocket_manager)
        controller.robust_aggregator = mock_robust_aggregator
        return controller
    
    @pytest.mark.asyncio
    async def test_byzantine_robust_training_round(self, byzantine_controller):
        """Test Byzantine-robust training round"""
        # Mock client updates
        mock_updates = [
            {'client_id': f'client_{i}', 'model_update': torch.randn(10, 5)}
            for i in range(10)
        ]
        
        result = await byzantine_controller.run_byzantine_robust_training_round(
            mock_updates,
            round_number=1
        )
        
        # Check result structure
        assert 'global_model' in result
        assert 'byzantine_stats' in result
        assert 'round_info' in result
        
        # Verify WebSocket broadcast was called
        byzantine_controller.websocket_manager.broadcast_message.assert_called()
    
    @pytest.mark.asyncio
    async def test_attack_simulation(self, byzantine_controller):
        """Test attack simulation functionality"""
        # Test different attack types
        attack_types = ["gaussian", "sign_flip", "large_deviation"]
        
        for attack_type in attack_types:
            byzantine_updates = byzantine_controller.simulate_byzantine_attack(
                num_clients=5,
                attack_type=attack_type,
                attack_intensity=0.5
            )
            
            assert len(byzantine_updates) == 5
            assert all('client_id' in update for update in byzantine_updates)
            assert all('model_update' in update for update in byzantine_updates)
    
    def test_dashboard_data_generation(self, byzantine_controller):
        """Test dashboard data generation"""
        # Mock some attack history
        byzantine_controller.attack_history = [
            {
                'round': 1,
                'attack_detected': True,
                'confidence': 0.9,
                'honest_clients': 7,
                'malicious_clients': 3
            },
            {
                'round': 2,
                'attack_detected': False,
                'confidence': 0.1,
                'honest_clients': 10,
                'malicious_clients': 0
            }
        ]
        
        dashboard_data = byzantine_controller.get_dashboard_data()
        
        assert 'attack_detection_history' in dashboard_data
        assert 'client_trust_scores' in dashboard_data
        assert 'robustness_metrics' in dashboard_data
        
        # Check attack detection history
        history = dashboard_data['attack_detection_history']
        assert len(history) == 2
        assert history[0]['attack_detected'] == True
        assert history[1]['attack_detected'] == False


class TestByzantineAPI:
    """Test Byzantine API endpoints"""
    
    @pytest.fixture
    def test_app(self):
        """Create test FastAPI app"""
        app = FastAPI()
        app.include_router(byzantine_router, prefix="/api/byzantine")
        return app
    
    @pytest.fixture
    def client(self, test_app):
        """Create test client"""
        return TestClient(test_app)
    
    @pytest.fixture
    def mock_controller(self):
        """Create mock Byzantine controller"""
        mock = AsyncMock()
        mock.get_status.return_value = {
            'byzantine_protection_enabled': True,
            'detection_method': 'krum',
            'total_rounds': 5,
            'attacks_detected': 2
        }
        mock.get_configuration.return_value = {
            'detection_method': 'krum',
            'aggregation_method': 'mean',
            'detection_threshold': 0.5
        }
        return mock
    
    def test_get_byzantine_status(self, client):
        """Test getting Byzantine protection status"""
        with patch('server.api.byzantine_endpoints.get_byzantine_controller') as mock_get:
            mock_controller = AsyncMock()
            mock_controller.get_status.return_value = {
                'byzantine_protection_enabled': True,
                'detection_method': 'krum'
            }
            mock_get.return_value = mock_controller
            
            response = client.get("/api/byzantine/status")
            assert response.status_code == 200
            data = response.json()
            assert data['byzantine_protection_enabled'] == True
    
    def test_update_byzantine_configuration(self, client):
        """Test updating Byzantine configuration"""
        config_data = {
            'detection_method': 'multi_krum',
            'aggregation_method': 'trimmed_mean',
            'detection_threshold': 0.7
        }
        
        with patch('server.api.byzantine_endpoints.get_byzantine_controller') as mock_get:
            mock_controller = AsyncMock()
            mock_controller.update_configuration = AsyncMock()
            mock_get.return_value = mock_controller
            
            response = client.post("/api/byzantine/configuration", json=config_data)
            assert response.status_code == 200
            
            # Verify controller method was called
            mock_controller.update_configuration.assert_called_once_with(config_data)
    
    def test_get_attack_history(self, client):
        """Test getting attack history"""
        with patch('server.api.byzantine_endpoints.get_byzantine_controller') as mock_get:
            mock_controller = AsyncMock()
            mock_controller.get_attack_history.return_value = [
                {
                    'round': 1,
                    'timestamp': '2024-01-01T10:00:00',
                    'attack_detected': True,
                    'confidence': 0.9
                }
            ]
            mock_get.return_value = mock_controller
            
            response = client.get("/api/byzantine/attack-history")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]['attack_detected'] == True
    
    def test_simulate_attack_endpoint(self, client):
        """Test attack simulation endpoint"""
        attack_config = {
            'attack_type': 'gaussian',
            'num_malicious_clients': 3,
            'attack_intensity': 0.5
        }
        
        with patch('server.api.byzantine_endpoints.get_byzantine_controller') as mock_get:
            mock_controller = AsyncMock()
            mock_controller.simulate_attack = AsyncMock(return_value={
                'attack_id': 'attack_123',
                'status': 'simulated'
            })
            mock_get.return_value = mock_controller
            
            response = client.post("/api/byzantine/simulate-attack", json=attack_config)
            assert response.status_code == 200
            data = response.json()
            assert 'attack_id' in data


@pytest.mark.integration
class TestByzantineIntegration:
    """Integration tests for Byzantine fault tolerance"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_byzantine_protection(self):
        """Test complete Byzantine protection workflow"""
        # Create components
        detector = ByzantineDetector()
        
        # Mock base aggregator
        base_aggregator = MagicMock()
        base_aggregator.get_database_connection.return_value = MagicMock()
        base_aggregator.get_database_connection.return_value.cursor.return_value = MagicMock()
        
        robust_aggregator = ByzantineRobustAggregator(base_aggregator)
        websocket_manager = AsyncMock()
        fl_controller = ByzantineFLController(websocket_manager)
        fl_controller.robust_aggregator = robust_aggregator
        
        # Create mixed updates (honest + malicious)
        honest_updates = [
            {
                'client_id': f'honest_{i}',
                'model_update': torch.randn(20, 5) * 0.1,
                'timestamp': datetime.now().isoformat()
            }
            for i in range(7)
        ]
        
        malicious_updates = [
            {
                'client_id': f'malicious_{i}',
                'model_update': torch.randn(20, 5) * 2,  # Large noise
                'timestamp': datetime.now().isoformat()
            }
            for i in range(3)
        ]
        
        all_updates = honest_updates + malicious_updates
        
        # Run Byzantine-robust training round
        result = await fl_controller.run_byzantine_robust_training_round(
            all_updates,
            round_number=1
        )
        
        # Verify results
        assert 'global_model' in result
        assert 'byzantine_stats' in result
        
        byzantine_stats = result['byzantine_stats']
        assert byzantine_stats['total_clients'] == 10
        assert byzantine_stats['honest_clients'] <= 10
        assert byzantine_stats['malicious_clients'] >= 0
        
        # Verify WebSocket notification was sent
        websocket_manager.broadcast_message.assert_called()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])