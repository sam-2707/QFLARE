"""
Privacy-aware FL Controller

Integrates differential privacy into the main federated learning controller.
Manages privacy-preserving training rounds and privacy budget allocation.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

from ..fl_core.fl_controller import FLController
from .private_trainer import PrivateFederatedTrainer
from ..websocket.manager import WebSocketManager

logger = logging.getLogger(__name__)


class PrivateFLController(FLController):
    """
    Enhanced FL Controller with differential privacy support.
    Manages privacy-preserving federated learning rounds.
    """
    
    def __init__(self, websocket_manager: Optional[WebSocketManager] = None,
                 privacy_level: str = "strong"):
        """
        Initialize private FL controller.
        
        Args:
            websocket_manager: WebSocket manager for real-time updates
            privacy_level: Privacy level ("strong", "moderate", "weak")
        """
        super().__init__()
        
        # Store websocket manager
        self.websocket_manager = websocket_manager
        
        # Initialize private trainer
        self.private_trainer = PrivateFederatedTrainer(
            model_type="mnist",  # Default to MNIST
            privacy_level=privacy_level
        )
        
        self.privacy_level = privacy_level
        self.privacy_rounds_completed = 0
        self.privacy_history = []
        
        logger.info(f"Private FL Controller initialized with {privacy_level} privacy")
    
    async def run_private_training_round(self, round_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a privacy-preserving federated learning round.
        
        Args:
            round_config: Configuration for the training round
            
        Returns:
            Training round results with privacy metrics
        """
        round_id = round_config.get("round_id", f"private_round_{self.privacy_rounds_completed + 1}")
        
        logger.info(f"Starting private FL round: {round_id}")
        
        # Send WebSocket update - round started
        if self.websocket_manager:
            await self.websocket_manager.broadcast_fl_status_update({
                "event": "private_round_started",
                "round_id": round_id,
                "privacy_level": self.privacy_level,
                "timestamp": datetime.now().isoformat()
            })
        
        try:
            # Extract configuration parameters
            num_clients = round_config.get("num_clients", 5)
            client_fraction = round_config.get("client_fraction", 1.0)
            epochs = round_config.get("epochs", 1)
            batch_size = round_config.get("batch_size", 32)
            learning_rate = round_config.get("learning_rate", 0.01)
            
            # Check privacy budget before starting
            privacy_status = self.private_trainer.get_privacy_status()
            if not privacy_status["budget_valid"]:
                error_msg = "Privacy budget exhausted - cannot start new round"
                logger.warning(error_msg)
                
                if self.websocket_manager:
                    await self.websocket_manager.broadcast_fl_status_update({
                        "event": "privacy_budget_exhausted",
                        "round_id": round_id,
                        "error": error_msg,
                        "timestamp": datetime.now().isoformat()
                    })
                
                return {
                    "success": False,
                    "error": error_msg,
                    "privacy_budget_exhausted": True,
                    "privacy_status": privacy_status
                }
            
            # Send WebSocket update - training phase
            if self.websocket_manager:
                await self.websocket_manager.broadcast_fl_status_update({
                    "event": "private_training_phase",
                    "round_id": round_id,
                    "num_clients": num_clients,
                    "privacy_budget_remaining": privacy_status["remaining_budget"],
                    "timestamp": datetime.now().isoformat()
                })
            
            # Run private training round
            round_result = self.private_trainer.run_private_training_round(
                num_clients=num_clients,
                client_fraction=client_fraction,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            if round_result.get("success", False):
                # Update round counter
                self.privacy_rounds_completed += 1
                
                # Store privacy metrics
                privacy_event = {
                    "round_id": round_id,
                    "timestamp": datetime.now().isoformat(),
                    "privacy_level": self.privacy_level,
                    "epsilon_spent": round_result.get("total_epsilon_spent", 0),
                    "clients_participated": round_result.get("successful_clients", 0),
                    "average_loss": round_result.get("average_loss", 0)
                }
                self.privacy_history.append(privacy_event)
                
                # Send WebSocket update - round completed
                if self.websocket_manager:
                    await self.websocket_manager.broadcast_fl_status_update({
                        "event": "private_round_completed",
                        "round_id": round_id,
                        "successful_clients": round_result.get("successful_clients", 0),
                        "average_loss": round_result.get("average_loss", 0),
                        "epsilon_spent": round_result.get("total_epsilon_spent", 0),
                        "privacy_guaranteed": True,
                        "timestamp": datetime.now().isoformat()
                    })
                
                logger.info(f"Private FL round {round_id} completed successfully - "
                           f"ε spent: {round_result.get('total_epsilon_spent', 0):.4f}")
                
                # Add privacy metadata to result
                round_result.update({
                    "round_id": round_id,
                    "privacy_level": self.privacy_level,
                    "privacy_guaranteed": True,
                    "privacy_rounds_completed": self.privacy_rounds_completed
                })
                
                return round_result
                
            else:
                error_msg = round_result.get("error", "Private training round failed")
                logger.error(f"Private FL round {round_id} failed: {error_msg}")
                
                # Send WebSocket update - round failed
                if self.websocket_manager:
                    await self.websocket_manager.broadcast_fl_status_update({
                        "event": "private_round_failed",
                        "round_id": round_id,
                        "error": error_msg,
                        "timestamp": datetime.now().isoformat()
                    })
                
                return round_result
                
        except Exception as e:
            error_msg = f"Private FL round {round_id} encountered error: {str(e)}"
            logger.error(error_msg)
            
            # Send WebSocket update - error
            if self.websocket_manager:
                await self.websocket_manager.broadcast_fl_status_update({
                    "event": "private_round_error",
                    "round_id": round_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
            
            return {
                "success": False,
                "error": str(e),
                "round_id": round_id,
                "privacy_guaranteed": False
            }
    
    async def get_privacy_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive privacy dashboard data.
        
        Returns:
            Privacy metrics and status for dashboard display
        """
        try:
            # Get current privacy status
            privacy_status = self.private_trainer.get_privacy_status()
            
            # Calculate privacy statistics
            total_epsilon_spent = sum(
                event.get("epsilon_spent", 0) for event in self.privacy_history
            )
            
            average_loss = sum(
                event.get("average_loss", 0) for event in self.privacy_history
            ) / max(len(self.privacy_history), 1)
            
            total_clients_participated = sum(
                event.get("clients_participated", 0) for event in self.privacy_history
            )
            
            # Privacy budget utilization
            current_budget = privacy_status["remaining_budget"]
            initial_epsilon = privacy_status["privacy_config"]["target_epsilon"] * 10  # Assume 10x budget
            budget_utilization = (total_epsilon_spent / initial_epsilon) * 100 if initial_epsilon > 0 else 0
            
            dashboard_data = {
                "privacy_overview": {
                    "privacy_level": self.privacy_level,
                    "privacy_enabled": True,
                    "differential_privacy_active": True,
                    "privacy_mechanism": "Gaussian Mechanism"
                },
                "privacy_parameters": privacy_status["privacy_config"],
                "privacy_budget": {
                    "epsilon_remaining": current_budget["epsilon_remaining"],
                    "delta_remaining": current_budget["delta_remaining"],
                    "epsilon_spent": total_epsilon_spent,
                    "budget_utilization_percent": min(budget_utilization, 100),
                    "budget_valid": privacy_status["budget_valid"]
                },
                "training_statistics": {
                    "privacy_rounds_completed": self.privacy_rounds_completed,
                    "total_clients_participated": total_clients_participated,
                    "average_loss": round(average_loss, 4),
                    "privacy_guaranteed_rounds": self.privacy_rounds_completed
                },
                "privacy_history": self.privacy_history[-10:],  # Last 10 rounds
                "privacy_metrics": privacy_status["privacy_metrics"],
                "system_status": {
                    "privacy_engine_status": "active",
                    "last_private_round": self.privacy_history[-1] if self.privacy_history else None,
                    "next_round_possible": privacy_status["budget_valid"]
                }
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to generate privacy dashboard data: {str(e)}")
            return {
                "error": str(e),
                "privacy_enabled": False,
                "privacy_dashboard_available": False
            }
    
    async def set_privacy_level(self, new_privacy_level: str) -> Dict[str, Any]:
        """
        Change the privacy level for future training rounds.
        
        Args:
            new_privacy_level: New privacy level ("strong", "moderate", "weak")
            
        Returns:
            Result of privacy level change
        """
        try:
            if new_privacy_level not in ["strong", "moderate", "weak"]:
                return {
                    "success": False,
                    "error": f"Invalid privacy level: {new_privacy_level}"
                }
            
            # Create new private trainer with new privacy level
            old_privacy_level = self.privacy_level
            self.private_trainer = PrivateFederatedTrainer(
                model_type="mnist",
                privacy_level=new_privacy_level
            )
            self.privacy_level = new_privacy_level
            
            # Send WebSocket update
            if self.websocket_manager:
                await self.websocket_manager.broadcast_fl_status_update({
                    "event": "privacy_level_changed",
                    "old_privacy_level": old_privacy_level,
                    "new_privacy_level": new_privacy_level,
                    "timestamp": datetime.now().isoformat()
                })
            
            logger.info(f"Privacy level changed from {old_privacy_level} to {new_privacy_level}")
            
            return {
                "success": True,
                "old_privacy_level": old_privacy_level,
                "new_privacy_level": new_privacy_level,
                "message": f"Privacy level updated to {new_privacy_level}"
            }
            
        except Exception as e:
            logger.error(f"Failed to change privacy level: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def validate_privacy_parameters(self, epsilon: float, delta: float) -> Dict[str, Any]:
        """
        Validate proposed differential privacy parameters.
        
        Args:
            epsilon: Proposed epsilon value
            delta: Proposed delta value
            
        Returns:
            Validation result with recommendations
        """
        try:
            validation_result = {
                "valid": True,
                "warnings": [],
                "recommendations": [],
                "privacy_strength": "unknown"
            }
            
            # Validate epsilon
            if epsilon <= 0:
                validation_result["valid"] = False
                validation_result["warnings"].append("Epsilon must be positive")
            elif epsilon < 0.1:
                validation_result["privacy_strength"] = "very_strong"
                validation_result["recommendations"].append("Very strong privacy - may impact model utility")
            elif epsilon < 1.0:
                validation_result["privacy_strength"] = "strong"
                validation_result["recommendations"].append("Strong privacy protection")
            elif epsilon < 5.0:
                validation_result["privacy_strength"] = "moderate"
                validation_result["recommendations"].append("Moderate privacy protection")
            else:
                validation_result["privacy_strength"] = "weak"
                validation_result["warnings"].append("Weak privacy protection - consider lower epsilon")
            
            # Validate delta
            if delta <= 0:
                validation_result["valid"] = False
                validation_result["warnings"].append("Delta must be positive")
            elif delta >= 1:
                validation_result["valid"] = False
                validation_result["warnings"].append("Delta must be less than 1")
            elif delta > 1e-3:
                validation_result["warnings"].append("Delta is quite large - consider smaller value")
            
            # Additional recommendations
            if validation_result["valid"]:
                if epsilon > 1.0 and delta > 1e-4:
                    validation_result["recommendations"].append(
                        "Both epsilon and delta are large - privacy protection may be limited"
                    )
                
                # Noise multiplier estimation
                import math
                estimated_noise = (2 * math.log(1.25 / delta) / epsilon)
                validation_result["estimated_noise_multiplier"] = round(estimated_noise, 3)
                
                if estimated_noise > 10:
                    validation_result["warnings"].append(
                        "High noise multiplier may significantly impact training"
                    )
            
            return validation_result
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }


# Factory function for easy initialization
def create_private_fl_controller(websocket_manager: Optional[WebSocketManager] = None,
                                privacy_level: str = "strong") -> PrivateFLController:
    """
    Create a privacy-aware FL controller.
    
    Args:
        websocket_manager: WebSocket manager for real-time updates
        privacy_level: Privacy level ("strong", "moderate", "weak")
        
    Returns:
        Configured PrivateFLController instance
    """
    return PrivateFLController(websocket_manager, privacy_level)


if __name__ == "__main__":
    # Test the private FL controller
    import asyncio
    
    async def test_private_controller():
        print("Testing Private FL Controller...")
        
        # Create private controller
        controller = create_private_fl_controller(privacy_level="strong")
        print("✓ Private FL Controller created")
        
        # Test privacy dashboard data
        dashboard_data = await controller.get_privacy_dashboard_data()
        print(f"✓ Privacy dashboard data: {dashboard_data['privacy_overview']['privacy_level']}")
        
        # Test privacy parameter validation
        validation = await controller.validate_privacy_parameters(epsilon=0.1, delta=1e-6)
        print(f"✓ Privacy validation: {validation['privacy_strength']}")
        
        # Test privacy level change
        change_result = await controller.set_privacy_level("moderate")
        print(f"✓ Privacy level change: {change_result['success']}")
        
        print("Private FL Controller test completed!")
    
    asyncio.run(test_private_controller())