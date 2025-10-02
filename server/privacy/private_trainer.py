"""
Privacy-Preserving Federated Learning Trainer

Integrates differential privacy with the existing federated learning training pipeline.
Provides privacy-preserving model updates while maintaining training effectiveness.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from typing import Dict, Any, List, Tuple, Optional
import time
import numpy as np

from ..ml.training import FederatedTrainer
from ..ml.models import create_model
from .differential_privacy import PrivacyEngine, DifferentialPrivacyConfig, create_privacy_engine

logger = logging.getLogger(__name__)


class PrivateFederatedTrainer(FederatedTrainer):
    """
    Enhanced federated trainer with differential privacy guarantees.
    Extends the base FederatedTrainer with privacy-preserving mechanisms.
    """
    
    def __init__(self, model_type: str = "mnist", privacy_level: str = "strong"):
        """
        Initialize private federated trainer.
        
        Args:
            model_type: Type of model to train ("mnist", "cifar10", "simple")
            privacy_level: Privacy level ("strong", "moderate", "weak")
        """
        super().__init__(model_type)
        
        # Initialize privacy engine
        self.privacy_engine = create_privacy_engine(privacy_level)
        self.privacy_level = privacy_level
        
        # Privacy metrics
        self.privacy_metrics = {
            "total_privatization_events": 0,
            "gradient_clipping_events": 0,
            "noise_addition_events": 0,
            "privacy_budget_warnings": 0
        }
        
        logger.info(f"Private FL Trainer initialized with {privacy_level} privacy")
    
    def train_client_private(self, client_id: int, epochs: int = 1, 
                           batch_size: int = 32, learning_rate: float = 0.01) -> Dict[str, Any]:
        """
        Train a client model with differential privacy guarantees.
        
        Args:
            client_id: ID of the client
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            
        Returns:
            Training results with privacy metrics
        """
        logger.info(f"Starting private training for client {client_id}")
        
        # Check privacy budget before training
        if not self.privacy_engine.validate_privacy_budget():
            logger.warning(f"Privacy budget exhausted for client {client_id}")
            self.privacy_metrics["privacy_budget_warnings"] += 1
            return {
                "success": False,
                "error": "Privacy budget exhausted",
                "privacy_budget_exhausted": True
            }
        
        try:
            # Get client data
            train_loader = self.get_client_data(client_id, batch_size, training=True)
            
            # Create fresh model for client
            model = create_model(self.model_type)
            
            # Apply global model weights if available
            if hasattr(self, 'global_model_state'):
                model.load_state_dict(self.global_model_state)
            
            # Setup optimizer
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # Store initial model state
            initial_state = {name: param.clone() for name, param in model.named_parameters()}
            
            # Training loop
            model.train()
            epoch_losses = []
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                batch_count = 0
                
                for batch_data, batch_labels in train_loader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Get gradients before clipping
                    gradients = {name: param.grad.clone() if param.grad is not None else None 
                               for name, param in model.named_parameters()}
                    
                    # Apply differential privacy to gradients
                    private_gradients = self.privacy_engine.privatize_model_update(gradients)
                    
                    # Apply privatized gradients
                    for name, param in model.named_parameters():
                        if param.grad is not None and name in private_gradients:
                            param.grad = private_gradients[name]
                    
                    # Update model with private gradients
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                    # Update privacy metrics
                    self.privacy_metrics["gradient_clipping_events"] += 1
                    self.privacy_metrics["noise_addition_events"] += 1
                
                avg_epoch_loss = epoch_loss / max(batch_count, 1)
                epoch_losses.append(avg_epoch_loss)
                
                logger.debug(f"Client {client_id}, Epoch {epoch+1}: Loss = {avg_epoch_loss:.4f}")
            
            # Calculate model update (difference from initial)
            model_update = {}
            for name, param in model.named_parameters():
                if name in initial_state:
                    model_update[name] = param.data - initial_state[name]
                else:
                    model_update[name] = param.data.clone()
            
            # Apply differential privacy to final model update
            private_model_update = self.privacy_engine.privatize_model_update(model_update)
            self.privacy_metrics["total_privatization_events"] += 1
            
            # Get privacy report
            privacy_report = self.privacy_engine.get_privacy_report()
            
            logger.info(f"Private training completed for client {client_id} - "
                       f"ε spent: {privacy_report['privacy_spent']['epsilon_spent']:.4f}")
            
            return {
                "success": True,
                "client_id": client_id,
                "model_update": private_model_update,
                "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
                "epoch_losses": epoch_losses,
                "training_time": time.time(),
                "privacy_report": privacy_report,
                "privacy_guaranteed": True,
                "epsilon_spent": privacy_report['privacy_spent']['epsilon_spent'],
                "delta_spent": privacy_report['privacy_spent']['delta_spent']
            }
            
        except Exception as e:
            logger.error(f"Private training failed for client {client_id}: {str(e)}")
            return {
                "success": False,
                "client_id": client_id,
                "error": str(e),
                "privacy_guaranteed": False
            }
    
    def aggregate_private_updates(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate model updates from multiple clients with privacy preservation.
        
        Args:
            client_updates: List of client update dictionaries
            
        Returns:
            Aggregated model update with privacy guarantees
        """
        if not client_updates:
            return {"success": False, "error": "No client updates to aggregate"}
        
        # Filter successful updates
        successful_updates = [update for update in client_updates if update.get("success", False)]
        
        if not successful_updates:
            return {"success": False, "error": "No successful client updates"}
        
        logger.info(f"Aggregating {len(successful_updates)} private client updates")
        
        try:
            # Extract model updates
            model_updates = [update["model_update"] for update in successful_updates]
            
            # Perform federated averaging
            aggregated_update = {}
            
            # Get parameter names from first update
            param_names = model_updates[0].keys()
            
            for param_name in param_names:
                # Stack all client updates for this parameter
                param_updates = []
                for client_update in model_updates:
                    if param_name in client_update and client_update[param_name] is not None:
                        param_updates.append(client_update[param_name])
                
                if param_updates:
                    # Average the parameter updates
                    stacked_updates = torch.stack(param_updates)
                    aggregated_param = torch.mean(stacked_updates, dim=0)
                    aggregated_update[param_name] = aggregated_param
            
            # Apply additional privacy protection to aggregated result
            private_aggregated = self.privacy_engine.privatize_aggregated_model(
                aggregated_update, len(successful_updates)
            )
            
            # Calculate aggregation metrics
            total_epsilon_spent = sum(
                update.get("epsilon_spent", 0) for update in successful_updates
            )
            total_delta_spent = sum(
                update.get("delta_spent", 0) for update in successful_updates
            )
            
            # Generate comprehensive privacy report
            privacy_report = self.privacy_engine.get_privacy_report()
            
            logger.info(f"Private aggregation completed - "
                       f"Total ε spent: {total_epsilon_spent:.4f}")
            
            return {
                "success": True,
                "aggregated_update": private_aggregated,
                "num_clients": len(successful_updates),
                "privacy_report": privacy_report,
                "total_epsilon_spent": total_epsilon_spent,
                "total_delta_spent": total_delta_spent,
                "privacy_guaranteed": True,
                "aggregation_time": time.time()
            }
            
        except Exception as e:
            logger.error(f"Private aggregation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "privacy_guaranteed": False
            }
    
    def run_private_training_round(self, num_clients: int = 5, 
                                  client_fraction: float = 1.0,
                                  epochs: int = 1, 
                                  batch_size: int = 32,
                                  learning_rate: float = 0.01) -> Dict[str, Any]:
        """
        Run a complete federated learning round with differential privacy.
        
        Args:
            num_clients: Total number of clients
            client_fraction: Fraction of clients to select
            epochs: Number of local training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            
        Returns:
            Complete training round results with privacy metrics
        """
        logger.info(f"Starting private FL round with {num_clients} clients")
        
        # Check overall privacy budget
        if not self.privacy_engine.validate_privacy_budget():
            return {
                "success": False,
                "error": "Global privacy budget exhausted",
                "privacy_budget_exhausted": True
            }
        
        round_start_time = time.time()
        
        # Select clients for this round
        selected_clients = self.select_clients(num_clients, client_fraction)
        logger.info(f"Selected {len(selected_clients)} clients for private training")
        
        # Train clients with privacy
        client_results = []
        for client_id in selected_clients:
            result = self.train_client_private(
                client_id=client_id,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            client_results.append(result)
        
        # Aggregate private updates
        aggregation_result = self.aggregate_private_updates(client_results)
        
        if not aggregation_result.get("success", False):
            return {
                "success": False,
                "error": "Aggregation failed",
                "client_results": client_results,
                "aggregation_result": aggregation_result
            }
        
        # Update global model if available
        if hasattr(self, 'global_model_state') and "aggregated_update" in aggregation_result:
            for param_name, update in aggregation_result["aggregated_update"].items():
                if param_name in self.global_model_state:
                    self.global_model_state[param_name] += update
        
        round_time = time.time() - round_start_time
        
        # Compile comprehensive results
        successful_clients = sum(1 for r in client_results if r.get("success", False))
        average_loss = np.mean([
            r.get("final_loss", 0) for r in client_results if r.get("success", False)
        ]) if successful_clients > 0 else 0.0
        
        total_privacy_spent = aggregation_result.get("total_epsilon_spent", 0)
        
        logger.info(f"Private FL round completed - "
                   f"{successful_clients}/{len(selected_clients)} clients succeeded, "
                   f"avg loss: {average_loss:.4f}, "
                   f"ε spent: {total_privacy_spent:.4f}")
        
        return {
            "success": True,
            "round_time": round_time,
            "selected_clients": selected_clients,
            "successful_clients": successful_clients,
            "average_loss": average_loss,
            "client_results": client_results,
            "aggregation_result": aggregation_result,
            "privacy_metrics": self.privacy_metrics.copy(),
            "privacy_guaranteed": True,
            "total_epsilon_spent": total_privacy_spent,
            "privacy_level": self.privacy_level
        }
    
    def get_privacy_status(self) -> Dict[str, Any]:
        """Get current privacy status and metrics."""
        privacy_report = self.privacy_engine.get_privacy_report()
        
        return {
            "privacy_engine_active": True,
            "privacy_level": self.privacy_level,
            "privacy_config": privacy_report["privacy_parameters"],
            "privacy_spent": privacy_report["privacy_spent"],
            "remaining_budget": privacy_report["remaining_budget"],
            "privacy_metrics": self.privacy_metrics,
            "budget_valid": self.privacy_engine.validate_privacy_budget()
        }


if __name__ == "__main__":
    # Test the private federated trainer
    print("Testing Private Federated Trainer...")
    
    # Create private trainer
    trainer = PrivateFederatedTrainer(model_type="mnist", privacy_level="strong")
    print("✓ Private trainer created")
    
    # Test privacy status
    status = trainer.get_privacy_status()
    print(f"✓ Privacy status: ε={status['privacy_config']['target_epsilon']}")
    
    # Test single client training (with mock data)
    print("Testing private client training...")
    try:
        result = trainer.train_client_private(client_id=0, epochs=1)
        if result.get("success", False):
            print("✓ Private client training works")
        else:
            print(f"⚠ Private client training failed: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"⚠ Private client training test failed: {str(e)}")
    
    print("Private federated trainer test completed!")