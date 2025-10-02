"""
Byzantine-Robust Aggregation for QFLARE Federated Learning

This module implements robust aggregation algorithms that can handle malicious clients.
Integrates Byzantine detection with secure model aggregation.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import copy

from .detection import ByzantineDetector, ByzantineDetectionConfig
from ..fl_core.aggregator_real import RealModelAggregator
from ..database import get_database

logger = logging.getLogger(__name__)


class ByzantineRobustAggregator:
    """
    Robust aggregation that handles Byzantine (malicious) clients.
    Combines Byzantine detection with secure aggregation algorithms.
    """
    
    def __init__(self, 
                 detection_method: str = "krum",
                 max_malicious_ratio: float = 0.33,
                 aggregation_method: str = "trimmed_mean",
                 model_type: str = "mnist"):
        """
        Initialize Byzantine-robust aggregator.
        
        Args:
            detection_method: Byzantine detection method
            max_malicious_ratio: Maximum ratio of malicious clients
            aggregation_method: Robust aggregation method
            model_type: Type of model being aggregated
        """
        # Initialize Byzantine detector
        detection_config = ByzantineDetectionConfig(
            max_malicious_ratio=max_malicious_ratio,
            detection_method=detection_method
        )
        self.byzantine_detector = ByzantineDetector(detection_config)
        
        # Initialize base aggregator for comparison
        self.base_aggregator = RealModelAggregator(model_type=model_type)
        
        self.detection_method = detection_method
        self.aggregation_method = aggregation_method
        self.model_type = model_type
        self.max_malicious_ratio = max_malicious_ratio
        
        # Robust aggregation history
        self.robust_aggregation_history = []
        self.detected_attacks = []
        
        logger.info(f"Byzantine-robust aggregator initialized: "
                   f"detection={detection_method}, aggregation={aggregation_method}")
    
    async def robust_aggregate(self, client_updates: List[Dict[str, Any]], 
                              round_number: int) -> Dict[str, Any]:
        """
        Perform Byzantine-robust aggregation of client updates.
        
        Args:
            client_updates: List of client update dictionaries
            round_number: Current training round number
            
        Returns:
            Robust aggregation result with attack detection info
        """
        if not client_updates:
            return {
                "success": False,
                "error": "No client updates provided",
                "robust_aggregation": False
            }
        
        # Filter successful updates
        successful_updates = [update for update in client_updates if update.get("success", False)]
        
        if len(successful_updates) < 3:
            logger.warning("Too few successful updates for robust aggregation")
            # Fall back to standard aggregation
            return await self._fallback_aggregation(successful_updates, round_number)
        
        logger.info(f"Starting robust aggregation for {len(successful_updates)} clients")
        
        try:
            # Step 1: Detect Byzantine clients
            detection_start = datetime.now()
            detection_result = self.byzantine_detector.detect_byzantine_updates(successful_updates)
            detection_time = (datetime.now() - detection_start).total_seconds()
            
            malicious_clients = detection_result["malicious_clients"]
            honest_clients = detection_result["honest_clients"]
            
            # Log attack detection
            if malicious_clients:
                attack_info = {
                    "round_number": round_number,
                    "timestamp": datetime.now().isoformat(),
                    "malicious_clients": malicious_clients,
                    "honest_clients": honest_clients,
                    "detection_method": detection_result["method"],
                    "detection_confidence": detection_result["detection_confidence"],
                    "attack_severity": len(malicious_clients) / len(successful_updates)
                }
                self.detected_attacks.append(attack_info)
                
                logger.warning(f"Byzantine attack detected! "
                             f"Malicious clients: {malicious_clients}, "
                             f"Attack severity: {attack_info['attack_severity']:.1%}")
            
            # Step 2: Filter honest updates
            honest_updates = [
                update for update in successful_updates 
                if update["client_id"] in honest_clients
            ]
            
            if not honest_updates:
                logger.error("No honest clients found - this should not happen")
                return await self._fallback_aggregation(successful_updates, round_number)
            
            # Step 3: Perform robust aggregation on honest updates
            aggregation_start = datetime.now()
            
            if self.aggregation_method == "trimmed_mean":
                aggregated_model = await self._trimmed_mean_aggregation(honest_updates)
            elif self.aggregation_method == "median":
                aggregated_model = await self._median_aggregation(honest_updates)
            elif self.aggregation_method == "krum":
                aggregated_model = await self._krum_aggregation(honest_updates, detection_result)
            elif self.aggregation_method == "multi_krum":
                aggregated_model = await self._multi_krum_aggregation(honest_updates, detection_result)
            else:
                # Default to federated averaging on honest clients
                aggregated_model = await self._federated_averaging(honest_updates)
            
            aggregation_time = (datetime.now() - aggregation_start).total_seconds()
            
            # Step 4: Update client reputations
            for client_id in malicious_clients:
                self.byzantine_detector.update_client_reputation(client_id, is_malicious=True)
            for client_id in honest_clients:
                self.byzantine_detector.update_client_reputation(client_id, is_malicious=False)
            
            # Step 5: Store robust aggregation record
            robust_record = {
                "round_number": round_number,
                "timestamp": datetime.now().isoformat(),
                "total_clients": len(successful_updates),
                "honest_clients": len(honest_clients),
                "malicious_clients": len(malicious_clients),
                "detection_method": detection_result["method"],
                "aggregation_method": self.aggregation_method,
                "detection_time": detection_time,
                "aggregation_time": aggregation_time,
                "attack_detected": len(malicious_clients) > 0,
                "detection_confidence": detection_result["detection_confidence"]
            }
            self.robust_aggregation_history.append(robust_record)
            
            # Step 6: Store in database
            try:
                # Create robust_aggregations table if it doesn't exist
                conn = self.base_aggregator.get_database_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS robust_aggregations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        round_number INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        total_clients INTEGER NOT NULL,
                        honest_clients INTEGER NOT NULL,
                        malicious_clients INTEGER NOT NULL,
                        detection_method TEXT NOT NULL,
                        aggregation_method TEXT NOT NULL,
                        attack_detected BOOLEAN NOT NULL,
                        detection_confidence REAL NOT NULL
                    )
                """)
                
                cursor.execute("""
                    INSERT INTO robust_aggregations 
                    (round_number, timestamp, total_clients, honest_clients, malicious_clients,
                     detection_method, aggregation_method, attack_detected, detection_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    round_number,
                    robust_record["timestamp"],
                    robust_record["total_clients"],
                    robust_record["honest_clients"],
                    robust_record["malicious_clients"],
                    robust_record["detection_method"],
                    robust_record["aggregation_method"],
                    robust_record["attack_detected"],
                    robust_record["detection_confidence"]
                ))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.warning(f"Failed to store robust aggregation record: {e}")
            
            logger.info(f"Robust aggregation completed: "
                       f"{len(honest_clients)}/{len(successful_updates)} honest clients used")
            
            return {
                "success": True,
                "aggregated_model": aggregated_model,
                "round_number": round_number,
                "robust_aggregation": True,
                "byzantine_detection": detection_result,
                "aggregation_stats": robust_record,
                "honest_clients": honest_clients,
                "malicious_clients": malicious_clients,
                "attack_detected": len(malicious_clients) > 0,
                "total_clients": len(successful_updates),
                "aggregation_method": self.aggregation_method
            }
            
        except Exception as e:
            logger.error(f"Robust aggregation failed: {str(e)}")
            # Fall back to standard aggregation
            return await self._fallback_aggregation(successful_updates, round_number)
    
    async def _fallback_aggregation(self, client_updates: List[Dict[str, Any]], 
                                   round_number: int) -> Dict[str, Any]:
        """Fallback to standard aggregation when robust aggregation fails."""
        logger.info("Falling back to standard aggregation")
        
        # Use base aggregator
        result = await self.base_aggregator.aggregate_models(client_updates, round_number)
        
        # Add robust aggregation metadata
        result.update({
            "robust_aggregation": False,
            "fallback_used": True,
            "byzantine_detection": {
                "method": "none",
                "malicious_clients": [],
                "honest_clients": [update["client_id"] for update in client_updates],
                "detection_confidence": 0.0
            }
        })
        
        return result
    
    async def _federated_averaging(self, honest_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Standard federated averaging on honest clients only."""
        if not honest_updates:
            raise ValueError("No honest updates for aggregation")
        
        # Extract model updates
        model_updates = [update["model_update"] for update in honest_updates]
        
        # Perform federated averaging
        aggregated_model = {}
        param_names = model_updates[0].keys()
        
        for param_name in param_names:
            param_updates = []
            for model_update in model_updates:
                if param_name in model_update and model_update[param_name] is not None:
                    param_updates.append(model_update[param_name])
            
            if param_updates:
                # Simple average
                stacked_params = torch.stack(param_updates)
                aggregated_model[param_name] = torch.mean(stacked_params, dim=0)
        
        return aggregated_model
    
    async def _trimmed_mean_aggregation(self, honest_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Trimmed mean aggregation - removes extreme values before averaging."""
        if not honest_updates:
            raise ValueError("No honest updates for trimmed mean aggregation")
        
        model_updates = [update["model_update"] for update in honest_updates]
        
        if len(model_updates) < 3:
            # Not enough for trimming, use regular average
            return await self._federated_averaging(honest_updates)
        
        aggregated_model = {}
        param_names = model_updates[0].keys()
        
        # Trim ratio (remove 20% from each end)
        trim_ratio = 0.2
        n = len(model_updates)
        trim_count = int(trim_ratio * n)
        
        for param_name in param_names:
            param_updates = []
            for model_update in model_updates:
                if param_name in model_update and model_update[param_name] is not None:
                    param_updates.append(model_update[param_name])
            
            if param_updates and len(param_updates) > 2 * trim_count:
                # Stack parameters
                stacked_params = torch.stack(param_updates)
                
                # Sort along client dimension and trim
                sorted_params, _ = torch.sort(stacked_params, dim=0)
                
                if trim_count > 0:
                    trimmed_params = sorted_params[trim_count:-trim_count]
                else:
                    trimmed_params = sorted_params
                
                # Compute mean of trimmed values
                aggregated_model[param_name] = torch.mean(trimmed_params, dim=0)
            elif param_updates:
                # Fall back to regular mean if not enough samples
                stacked_params = torch.stack(param_updates)
                aggregated_model[param_name] = torch.mean(stacked_params, dim=0)
        
        return aggregated_model
    
    async def _median_aggregation(self, honest_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Coordinate-wise median aggregation."""
        if not honest_updates:
            raise ValueError("No honest updates for median aggregation")
        
        model_updates = [update["model_update"] for update in honest_updates]
        
        aggregated_model = {}
        param_names = model_updates[0].keys()
        
        for param_name in param_names:
            param_updates = []
            for model_update in model_updates:
                if param_name in model_update and model_update[param_name] is not None:
                    param_updates.append(model_update[param_name])
            
            if param_updates:
                # Stack parameters and compute coordinate-wise median
                stacked_params = torch.stack(param_updates)
                aggregated_model[param_name] = torch.median(stacked_params, dim=0)[0]
        
        return aggregated_model
    
    async def _krum_aggregation(self, honest_updates: List[Dict[str, Any]], 
                               detection_result: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Use Krum-selected client's model for aggregation."""
        if "selected_client" not in detection_result:
            # Fall back to federated averaging
            return await self._federated_averaging(honest_updates)
        
        selected_client = detection_result["selected_client"]
        
        # Find the selected client's update
        for update in honest_updates:
            if update["client_id"] == selected_client:
                return update["model_update"]
        
        # If selected client not found, use first honest client
        return honest_updates[0]["model_update"]
    
    async def _multi_krum_aggregation(self, honest_updates: List[Dict[str, Any]], 
                                     detection_result: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Use Multi-Krum selected clients for aggregation."""
        if "selected_clients" not in detection_result:
            return await self._federated_averaging(honest_updates)
        
        selected_clients = detection_result["selected_clients"]
        
        # Filter updates from selected clients
        selected_updates = [
            update for update in honest_updates 
            if update["client_id"] in selected_clients
        ]
        
        if not selected_updates:
            return await self._federated_averaging(honest_updates)
        
        # Average the selected clients' updates
        return await self._federated_averaging(selected_updates)
    
    def get_attack_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected attacks."""
        if not self.detected_attacks:
            return {
                "total_attacks": 0,
                "attack_rate": 0.0,
                "average_attack_severity": 0.0,
                "most_malicious_clients": [],
                "detection_methods_used": []
            }
        
        total_rounds = len(self.robust_aggregation_history)
        total_attacks = len(self.detected_attacks)
        
        # Calculate attack severity statistics
        attack_severities = [attack["attack_severity"] for attack in self.detected_attacks]
        avg_severity = np.mean(attack_severities) if attack_severities else 0.0
        
        # Find most frequently malicious clients
        malicious_client_counts = {}
        for attack in self.detected_attacks:
            for client_id in attack["malicious_clients"]:
                malicious_client_counts[client_id] = malicious_client_counts.get(client_id, 0) + 1
        
        most_malicious = sorted(malicious_client_counts.items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        
        # Collection detection methods used
        detection_methods = list(set(attack["detection_method"] for attack in self.detected_attacks))
        
        return {
            "total_attacks": total_attacks,
            "total_rounds": total_rounds,
            "attack_rate": total_attacks / max(total_rounds, 1),
            "average_attack_severity": avg_severity,
            "max_attack_severity": max(attack_severities) if attack_severities else 0.0,
            "most_malicious_clients": most_malicious,
            "detection_methods_used": detection_methods,
            "recent_attacks": self.detected_attacks[-5:]  # Last 5 attacks
        }
    
    def get_robustness_report(self) -> Dict[str, Any]:
        """Generate comprehensive robustness report."""
        attack_stats = self.get_attack_statistics()
        detection_summary = self.byzantine_detector.get_detection_summary()
        
        # Calculate aggregation statistics
        total_aggregations = len(self.robust_aggregation_history)
        successful_aggregations = sum(
            1 for record in self.robust_aggregation_history 
            if not record.get("attack_detected", False)
        )
        
        return {
            "robustness_overview": {
                "total_aggregations": total_aggregations,
                "successful_aggregations": successful_aggregations,
                "robustness_rate": successful_aggregations / max(total_aggregations, 1),
                "detection_method": self.detection_method,
                "aggregation_method": self.aggregation_method,
                "max_malicious_ratio": self.max_malicious_ratio
            },
            "attack_statistics": attack_stats,
            "detection_summary": detection_summary,
            "aggregation_history": self.robust_aggregation_history[-10:],  # Last 10 rounds
            "client_trust_scores": {
                client_id: self.byzantine_detector.get_client_trust_score(client_id)
                for client_id in detection_summary.get("malicious_clients", [])
            }
        }


# Factory functions for different robust aggregators
def create_krum_aggregator(max_malicious_ratio: float = 0.33, model_type: str = "mnist") -> ByzantineRobustAggregator:
    """Create Krum-based robust aggregator."""
    return ByzantineRobustAggregator(
        detection_method="krum",
        max_malicious_ratio=max_malicious_ratio,
        aggregation_method="krum",
        model_type=model_type
    )


def create_multi_krum_aggregator(max_malicious_ratio: float = 0.33, model_type: str = "mnist") -> ByzantineRobustAggregator:
    """Create Multi-Krum robust aggregator."""
    return ByzantineRobustAggregator(
        detection_method="multi_krum",
        max_malicious_ratio=max_malicious_ratio,
        aggregation_method="multi_krum",
        model_type=model_type
    )


def create_trimmed_mean_aggregator(max_malicious_ratio: float = 0.33, model_type: str = "mnist") -> ByzantineRobustAggregator:
    """Create trimmed mean robust aggregator."""
    return ByzantineRobustAggregator(
        detection_method="trimmed_mean",
        max_malicious_ratio=max_malicious_ratio,
        aggregation_method="trimmed_mean",
        model_type=model_type
    )


def create_clustering_aggregator(max_malicious_ratio: float = 0.33, model_type: str = "mnist") -> ByzantineRobustAggregator:
    """Create clustering-based robust aggregator."""
    return ByzantineRobustAggregator(
        detection_method="clustering",
        max_malicious_ratio=max_malicious_ratio,
        aggregation_method="trimmed_mean",
        model_type=model_type
    )


if __name__ == "__main__":
    # Test Byzantine-robust aggregation
    import asyncio
    
    async def test_robust_aggregation():
        print("Testing Byzantine-Robust Aggregation...")
        
        # Create robust aggregator
        aggregator = create_krum_aggregator(max_malicious_ratio=0.3)
        print("✓ Robust aggregator created")
        
        # Create mock client updates (4 honest, 2 malicious)
        client_updates = []
        for i in range(6):
            if i < 4:  # Honest clients
                update = {
                    "client_id": i,
                    "success": True,
                    "model_update": {
                        "layer1": torch.randn(3, 2) * 0.1,
                        "layer2": torch.randn(1, 3) * 0.1
                    },
                    "training_loss": 0.5 + np.random.normal(0, 0.1)
                }
            else:  # Malicious clients
                update = {
                    "client_id": i,
                    "success": True,
                    "model_update": {
                        "layer1": torch.randn(3, 2) * 5.0,  # Large malicious update
                        "layer2": torch.randn(1, 3) * 5.0
                    },
                    "training_loss": 10.0  # Suspicious high loss
                }
            client_updates.append(update)
        
        # Test robust aggregation
        result = await aggregator.robust_aggregate(client_updates, round_number=1)
        
        print(f"✓ Robust aggregation completed: {result['success']}")
        print(f"  Attack detected: {result.get('attack_detected', False)}")
        print(f"  Malicious clients: {result.get('malicious_clients', [])}")
        print(f"  Honest clients used: {len(result.get('honest_clients', []))}")
        
        # Test robustness report
        report = aggregator.get_robustness_report()
        print(f"✓ Robustness report generated")
        print(f"  Total aggregations: {report['robustness_overview']['total_aggregations']}")
        print(f"  Attack rate: {report['attack_statistics']['attack_rate']:.1%}")
        
        print("Byzantine-robust aggregation test completed!")
    
    asyncio.run(test_robust_aggregation())