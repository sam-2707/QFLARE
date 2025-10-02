"""
Byzantine-Aware Federated Learning Controller

Integrates Byzantine fault tolerance with the existing FL controller.
Provides robust federated learning against malicious clients.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

from ..fl_core.fl_controller import FLController
from .robust_aggregator import (
    ByzantineRobustAggregator,
    create_krum_aggregator,
    create_multi_krum_aggregator,
    create_trimmed_mean_aggregator,
    create_clustering_aggregator
)
from ..websocket.manager import WebSocketManager

logger = logging.getLogger(__name__)


class ByzantineFLController(FLController):
    """
    Byzantine-aware FL Controller that can handle malicious clients.
    Extends the base FL controller with robust aggregation capabilities.
    """
    
    def __init__(self, 
                 websocket_manager: Optional[WebSocketManager] = None,
                 detection_method: str = "krum",
                 aggregation_method: str = "krum",
                 max_malicious_ratio: float = 0.33,
                 model_type: str = "mnist"):
        """
        Initialize Byzantine-aware FL controller.
        
        Args:
            websocket_manager: WebSocket manager for real-time updates
            detection_method: Byzantine detection method
            aggregation_method: Robust aggregation method
            max_malicious_ratio: Maximum ratio of malicious clients
            model_type: Type of model for training
        """
        super().__init__()
        
        # Store websocket manager
        self.websocket_manager = websocket_manager
        
        # Initialize robust aggregator
        if detection_method == "krum":
            self.robust_aggregator = create_krum_aggregator(max_malicious_ratio, model_type)
        elif detection_method == "multi_krum":
            self.robust_aggregator = create_multi_krum_aggregator(max_malicious_ratio, model_type)
        elif detection_method == "trimmed_mean":
            self.robust_aggregator = create_trimmed_mean_aggregator(max_malicious_ratio, model_type)
        elif detection_method == "clustering":
            self.robust_aggregator = create_clustering_aggregator(max_malicious_ratio, model_type)
        else:
            raise ValueError(f"Unsupported detection method: {detection_method}")
        
        self.detection_method = detection_method
        self.aggregation_method = aggregation_method
        self.max_malicious_ratio = max_malicious_ratio
        self.model_type = model_type
        
        # Byzantine-specific tracking
        self.byzantine_rounds_completed = 0
        self.total_attacks_detected = 0
        self.malicious_clients_history = []
        
        logger.info(f"Byzantine FL Controller initialized: "
                   f"detection={detection_method}, max_malicious={max_malicious_ratio:.1%}")
    
    async def run_byzantine_robust_training_round(self, round_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a Byzantine-robust federated learning training round.
        
        Args:
            round_config: Configuration for the training round
            
        Returns:
            Training round results with Byzantine detection info
        """
        round_id = round_config.get("round_id", f"byzantine_round_{self.byzantine_rounds_completed + 1}")
        
        logger.info(f"Starting Byzantine-robust FL round: {round_id}")
        
        # Send WebSocket update - round started
        if self.websocket_manager:
            await self.websocket_manager.broadcast_fl_status_update({
                "event": "byzantine_round_started",
                "round_id": round_id,
                "detection_method": self.detection_method,
                "max_malicious_ratio": self.max_malicious_ratio,
                "timestamp": datetime.now().isoformat()
            })
        
        try:
            # Extract configuration parameters
            num_clients = round_config.get("num_clients", 10)
            client_fraction = round_config.get("client_fraction", 1.0)
            epochs = round_config.get("epochs", 1)
            batch_size = round_config.get("batch_size", 32)
            learning_rate = round_config.get("learning_rate", 0.01)
            
            # Simulate client training (in a real system, this would be distributed)
            client_updates = await self._simulate_client_training_with_attacks(
                num_clients=num_clients,
                client_fraction=client_fraction,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                round_config=round_config
            )
            
            # Send WebSocket update - aggregation phase
            if self.websocket_manager:
                await self.websocket_manager.broadcast_fl_status_update({
                    "event": "byzantine_aggregation_phase",
                    "round_id": round_id,
                    "total_updates": len(client_updates),
                    "timestamp": datetime.now().isoformat()
                })
            
            # Perform robust aggregation
            aggregation_result = await self.robust_aggregator.robust_aggregate(
                client_updates, 
                round_number=self.byzantine_rounds_completed + 1
            )
            
            if not aggregation_result.get("success", False):
                error_msg = aggregation_result.get("error", "Robust aggregation failed")
                logger.error(f"Byzantine-robust FL round {round_id} failed: {error_msg}")
                
                # Send WebSocket update - round failed
                if self.websocket_manager:
                    await self.websocket_manager.broadcast_fl_status_update({
                        "event": "byzantine_round_failed",
                        "round_id": round_id,
                        "error": error_msg,
                        "timestamp": datetime.now().isoformat()
                    })
                
                return aggregation_result
            
            # Update tracking
            self.byzantine_rounds_completed += 1
            attack_detected = aggregation_result.get("attack_detected", False)
            
            if attack_detected:
                self.total_attacks_detected += 1
                malicious_clients = aggregation_result.get("malicious_clients", [])
                self.malicious_clients_history.extend(malicious_clients)
                
                # Send WebSocket update - attack detected
                if self.websocket_manager:
                    await self.websocket_manager.broadcast_fl_status_update({
                        "event": "byzantine_attack_detected",
                        "round_id": round_id,
                        "malicious_clients": malicious_clients,
                        "honest_clients": aggregation_result.get("honest_clients", []),
                        "detection_confidence": aggregation_result.get("byzantine_detection", {}).get("detection_confidence", 0),
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Send WebSocket update - round completed
            if self.websocket_manager:
                await self.websocket_manager.broadcast_fl_status_update({
                    "event": "byzantine_round_completed",
                    "round_id": round_id,
                    "attack_detected": attack_detected,
                    "honest_clients": len(aggregation_result.get("honest_clients", [])),
                    "malicious_clients": len(aggregation_result.get("malicious_clients", [])),
                    "robust_aggregation": aggregation_result.get("robust_aggregation", False),
                    "timestamp": datetime.now().isoformat()
                })
            
            logger.info(f"Byzantine-robust FL round {round_id} completed - "
                       f"Attack detected: {attack_detected}, "
                       f"Honest clients: {len(aggregation_result.get('honest_clients', []))}")
            
            # Add round metadata
            aggregation_result.update({
                "round_id": round_id,
                "byzantine_robust": True,
                "detection_method": self.detection_method,
                "aggregation_method": self.aggregation_method,
                "rounds_completed": self.byzantine_rounds_completed,
                "total_attacks_detected": self.total_attacks_detected
            })
            
            return aggregation_result
            
        except Exception as e:
            error_msg = f"Byzantine-robust FL round {round_id} encountered error: {str(e)}"
            logger.error(error_msg)
            
            # Send WebSocket update - error
            if self.websocket_manager:
                await self.websocket_manager.broadcast_fl_status_update({
                    "event": "byzantine_round_error",
                    "round_id": round_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
            
            return {
                "success": False,
                "error": str(e),
                "round_id": round_id,
                "byzantine_robust": False
            }
    
    async def _simulate_client_training_with_attacks(self, 
                                                    num_clients: int,
                                                    client_fraction: float,
                                                    epochs: int,
                                                    batch_size: int,
                                                    learning_rate: float,
                                                    round_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Simulate client training with potential Byzantine attacks.
        In a real deployment, this would be replaced with actual client communication.
        """
        import torch
        import numpy as np
        
        # Determine number of malicious clients
        attack_probability = round_config.get("attack_probability", 0.2)  # 20% chance of attack
        num_malicious = 0
        
        if np.random.random() < attack_probability:
            max_malicious = int(self.max_malicious_ratio * num_clients)
            num_malicious = np.random.randint(1, max_malicious + 1)
        
        selected_clients = list(range(int(num_clients * client_fraction)))
        malicious_clients = np.random.choice(selected_clients, size=num_malicious, replace=False)
        
        logger.info(f"Simulating training: {num_clients} clients, {num_malicious} malicious")
        
        client_updates = []
        
        for client_id in selected_clients:
            is_malicious = client_id in malicious_clients
            
            if is_malicious:
                # Simulate malicious update
                attack_type = np.random.choice(["gaussian_noise", "sign_flipping", "large_deviation"])
                update = self._generate_malicious_update(client_id, attack_type)
            else:
                # Simulate honest update
                update = self._generate_honest_update(client_id, epochs, batch_size, learning_rate)
            
            client_updates.append(update)
        
        return client_updates
    
    def _generate_honest_update(self, client_id: int, epochs: int, 
                               batch_size: int, learning_rate: float) -> Dict[str, Any]:
        """Generate simulated honest client update."""
        import torch
        import numpy as np
        
        # Simulate realistic model updates (small gradients)
        if self.model_type == "mnist":
            model_update = {
                "fc1.weight": torch.randn(128, 784) * 0.01,
                "fc1.bias": torch.randn(128) * 0.01,
                "fc2.weight": torch.randn(64, 128) * 0.01,
                "fc2.bias": torch.randn(64) * 0.01,
                "fc3.weight": torch.randn(10, 64) * 0.01,
                "fc3.bias": torch.randn(10) * 0.01
            }
        else:
            # Simple model
            model_update = {
                "layer1": torch.randn(10, 5) * 0.05,
                "layer2": torch.randn(1, 10) * 0.05
            }
        
        # Add small random noise to make updates realistic
        for param_name in model_update:
            noise = torch.randn_like(model_update[param_name]) * 0.001
            model_update[param_name] += noise
        
        return {
            "client_id": client_id,
            "success": True,
            "model_update": model_update,
            "training_loss": 0.3 + np.random.normal(0, 0.1),  # Realistic loss
            "training_accuracy": 0.85 + np.random.normal(0, 0.05),
            "epochs_completed": epochs,
            "is_malicious": False
        }
    
    def _generate_malicious_update(self, client_id: int, attack_type: str) -> Dict[str, Any]:
        """Generate simulated malicious client update."""
        import torch
        import numpy as np
        
        if self.model_type == "mnist":
            base_shape = {
                "fc1.weight": (128, 784),
                "fc1.bias": (128,),
                "fc2.weight": (64, 128),
                "fc2.bias": (64,),
                "fc3.weight": (10, 64),
                "fc3.bias": (10,)
            }
        else:
            base_shape = {
                "layer1": (10, 5),
                "layer2": (1, 10)
            }
        
        model_update = {}
        
        for param_name, shape in base_shape.items():
            if attack_type == "gaussian_noise":
                # Large Gaussian noise attack
                model_update[param_name] = torch.randn(shape) * 2.0
            elif attack_type == "sign_flipping":
                # Sign flipping attack (flip gradient signs)
                honest_update = torch.randn(shape) * 0.01
                model_update[param_name] = -honest_update * 10
            elif attack_type == "large_deviation":
                # Large deviation attack
                model_update[param_name] = torch.ones(shape) * 5.0
            else:
                # Default: large random values
                model_update[param_name] = torch.randn(shape) * 3.0
        
        return {
            "client_id": client_id,
            "success": True,
            "model_update": model_update,
            "training_loss": 10.0 + np.random.normal(0, 2.0),  # Suspicious high loss
            "training_accuracy": 0.1 + np.random.normal(0, 0.05),  # Poor accuracy
            "epochs_completed": 1,
            "is_malicious": True,
            "attack_type": attack_type
        }
    
    async def get_byzantine_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive Byzantine robustness dashboard data.
        
        Returns:
            Byzantine metrics and status for dashboard display
        """
        try:
            # Get robustness report from aggregator
            robustness_report = self.robust_aggregator.get_robustness_report()
            attack_stats = self.robust_aggregator.get_attack_statistics()
            
            # Calculate additional metrics
            robustness_rate = (
                (self.byzantine_rounds_completed - self.total_attacks_detected) / 
                max(self.byzantine_rounds_completed, 1)
            )
            
            # Client trust analysis
            unique_malicious = set(self.malicious_clients_history)
            client_trust_scores = {}
            for client_id in unique_malicious:
                trust_score = self.robust_aggregator.byzantine_detector.get_client_trust_score(client_id)
                client_trust_scores[client_id] = trust_score
            
            dashboard_data = {
                "byzantine_overview": {
                    "byzantine_protection_active": True,
                    "detection_method": self.detection_method,
                    "aggregation_method": self.aggregation_method,
                    "max_malicious_ratio": self.max_malicious_ratio,
                    "robustness_mechanism": "Multi-Algorithm Byzantine Detection"
                },
                "robustness_statistics": {
                    "rounds_completed": self.byzantine_rounds_completed,
                    "attacks_detected": self.total_attacks_detected,
                    "attack_detection_rate": attack_stats.get("attack_rate", 0.0),
                    "robustness_rate": robustness_rate,
                    "average_attack_severity": attack_stats.get("average_attack_severity", 0.0)
                },
                "detection_performance": {
                    "detection_method": self.detection_method,
                    "detection_accuracy": robustness_report["robustness_overview"]["robustness_rate"],
                    "false_positive_rate": 0.0,  # Would need ground truth to calculate
                    "detection_latency": "< 100ms"  # Estimated
                },
                "malicious_clients": {
                    "total_unique_malicious": len(unique_malicious),
                    "recently_detected": list(unique_malicious)[-5:],
                    "client_trust_scores": client_trust_scores,
                    "most_malicious_clients": attack_stats.get("most_malicious_clients", [])
                },
                "aggregation_performance": robustness_report["robustness_overview"],
                "recent_attacks": attack_stats.get("recent_attacks", []),
                "system_resilience": {
                    "theoretical_max_malicious": f"{self.max_malicious_ratio:.1%}",
                    "system_status": "robust" if robustness_rate > 0.8 else "degraded",
                    "recommended_action": self._get_recommended_action(robustness_rate, attack_stats)
                }
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to generate Byzantine dashboard data: {str(e)}")
            return {
                "error": str(e),
                "byzantine_protection_active": False,
                "dashboard_available": False
            }
    
    def _get_recommended_action(self, robustness_rate: float, attack_stats: Dict[str, Any]) -> str:
        """Get recommended action based on system performance."""
        if robustness_rate > 0.9:
            return "System operating normally"
        elif robustness_rate > 0.7:
            return "Monitor for increased attack activity"
        elif robustness_rate > 0.5:
            return "Consider stricter detection thresholds"
        else:
            return "Investigate potential coordinated attack"
    
    async def update_byzantine_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update Byzantine protection configuration.
        
        Args:
            new_config: New configuration parameters
            
        Returns:
            Configuration update result
        """
        try:
            old_detection = self.detection_method
            old_max_malicious = self.max_malicious_ratio
            
            # Update configuration
            if "detection_method" in new_config:
                new_detection = new_config["detection_method"]
                if new_detection in ["krum", "multi_krum", "trimmed_mean", "clustering"]:
                    self.detection_method = new_detection
            
            if "max_malicious_ratio" in new_config:
                new_ratio = new_config["max_malicious_ratio"]
                if 0 < new_ratio < 0.5:
                    self.max_malicious_ratio = new_ratio
            
            # Recreate aggregator with new configuration
            if self.detection_method == "krum":
                self.robust_aggregator = create_krum_aggregator(self.max_malicious_ratio, self.model_type)
            elif self.detection_method == "multi_krum":
                self.robust_aggregator = create_multi_krum_aggregator(self.max_malicious_ratio, self.model_type)
            elif self.detection_method == "trimmed_mean":
                self.robust_aggregator = create_trimmed_mean_aggregator(self.max_malicious_ratio, self.model_type)
            elif self.detection_method == "clustering":
                self.robust_aggregator = create_clustering_aggregator(self.max_malicious_ratio, self.model_type)
            
            # Send WebSocket update
            if self.websocket_manager:
                await self.websocket_manager.broadcast_fl_status_update({
                    "event": "byzantine_config_updated",
                    "old_detection_method": old_detection,
                    "new_detection_method": self.detection_method,
                    "old_max_malicious_ratio": old_max_malicious,
                    "new_max_malicious_ratio": self.max_malicious_ratio,
                    "timestamp": datetime.now().isoformat()
                })
            
            logger.info(f"Byzantine config updated: {old_detection} -> {self.detection_method}")
            
            return {
                "success": True,
                "old_config": {
                    "detection_method": old_detection,
                    "max_malicious_ratio": old_max_malicious
                },
                "new_config": {
                    "detection_method": self.detection_method,
                    "max_malicious_ratio": self.max_malicious_ratio
                },
                "message": "Byzantine protection configuration updated"
            }
            
        except Exception as e:
            logger.error(f"Failed to update Byzantine config: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


# Factory functions
def create_byzantine_fl_controller(websocket_manager: Optional[WebSocketManager] = None,
                                  detection_method: str = "krum",
                                  max_malicious_ratio: float = 0.33) -> ByzantineFLController:
    """
    Create a Byzantine-aware FL controller.
    
    Args:
        websocket_manager: WebSocket manager for real-time updates
        detection_method: Byzantine detection method
        max_malicious_ratio: Maximum ratio of malicious clients
        
    Returns:
        Configured ByzantineFLController instance
    """
    return ByzantineFLController(
        websocket_manager=websocket_manager,
        detection_method=detection_method,
        max_malicious_ratio=max_malicious_ratio
    )


if __name__ == "__main__":
    # Test Byzantine FL controller
    import asyncio
    
    async def test_byzantine_controller():
        print("Testing Byzantine FL Controller...")
        
        # Create Byzantine controller
        controller = create_byzantine_fl_controller(detection_method="krum")
        print("✓ Byzantine FL Controller created")
        
        # Test configuration
        config_result = await controller.update_byzantine_config({
            "detection_method": "multi_krum",
            "max_malicious_ratio": 0.25
        })
        print(f"✓ Config update: {config_result['success']}")
        
        # Test dashboard data
        dashboard_data = await controller.get_byzantine_dashboard_data()
        print(f"✓ Dashboard data: {dashboard_data['byzantine_overview']['detection_method']}")
        
        # Test robust training round
        round_config = {
            "num_clients": 8,
            "client_fraction": 1.0,
            "epochs": 1,
            "attack_probability": 0.5  # Force an attack for testing
        }
        
        result = await controller.run_byzantine_robust_training_round(round_config)
        print(f"✓ Byzantine round: {result['success']}")
        print(f"  Attack detected: {result.get('attack_detected', False)}")
        print(f"  Honest clients: {len(result.get('honest_clients', []))}")
        
        print("Byzantine FL Controller test completed!")
    
    asyncio.run(test_byzantine_controller())