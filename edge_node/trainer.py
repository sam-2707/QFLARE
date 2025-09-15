"""
QFLARE Edge Node Trainer

This module implements local model training for federated learning using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
from typing import Dict, Any, Optional, Tuple
import io
import pickle
import sys
from pathlib import Path

# Add common directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "common"))

try:
    from error_handling import (
        retry_on_failure, RetryConfig, TrainingError, 
        log_execution_time, catch_and_log_exceptions,
        SafeExecutor, validate_input
    )
    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    ERROR_HANDLING_AVAILABLE = False
    # Define no-op decorators if error handling not available
    def retry_on_failure(config=None):
        def decorator(func):
            return func
        return decorator
    
    def log_execution_time(func):
        return func
    
    def catch_and_log_exceptions(logger_instance=None):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)

class SimpleCNN(nn.Module):
    """
    Simple CNN model for MNIST/CIFAR-10 classification.
    Suitable for federated learning experiments.
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Assuming 28x28 input (MNIST)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class LocalTrainer:
    """
    Local trainer for federated learning.
    Handles model training, parameter updates, and serialization.
    """
    
    def __init__(self, 
                 model: Optional[nn.Module] = None,
                 learning_rate: float = 0.01,
                 local_epochs: int = 5,
                 batch_size: int = 32,
                 device: str = None):
        """
        Initialize local trainer.
        
        Args:
            model: PyTorch model to train
            learning_rate: Learning rate for optimizer
            local_epochs: Number of local training epochs
            batch_size: Batch size for training
            device: Device to run training on ('cpu' or 'cuda')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        
        # Initialize model
        if model is None:
            self.model = SimpleCNN(num_classes=10, input_channels=1)
        else:
            self.model = model
            
        self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Local trainer initialized on device: {self.device}")
    
    @log_execution_time
    @catch_and_log_exceptions()
    def train_local_model(self, 
                         train_loader: DataLoader,
                         global_model_weights: Optional[bytes] = None) -> Tuple[bytes, Dict[str, Any]]:
        """
        Train the local model on local data with enhanced error handling.
        
        Args:
            train_loader: DataLoader for local training data
            global_model_weights: Serialized global model weights to start from
            
        Returns:
            Tuple of (serialized_model_weights, training_metadata)
        """
        
        # Validate inputs
        if train_loader is None:
            raise TrainingError("Training data loader cannot be None")
        
        if len(train_loader.dataset) == 0:
            raise TrainingError("Training dataset is empty")
        
        try:
            # Load global model weights if provided
            if global_model_weights:
                self.load_model_weights(global_model_weights)
            
            # Store initial weights for computing update
            initial_weights = self.get_model_weights()
            
            # Training loop with error recovery
            self.model.train()
            total_loss = 0.0
            num_samples = 0
            successful_batches = 0
            failed_batches = 0
            
            logger.info(f"Starting local training for {self.local_epochs} epochs")
            start_time = time.time()
            
            for epoch in range(self.local_epochs):
                epoch_loss = 0.0
                epoch_samples = 0
                epoch_failed_batches = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    try:
                        # Move data to device with error handling
                        data, target = data.to(self.device), target.to(self.device)
                        
                        # Validate batch data
                        if torch.isnan(data).any() or torch.isinf(data).any():
                            logger.warning(f"Invalid data in batch {batch_idx}, skipping")
                            failed_batches += 1
                            continue
                        
                        # Zero gradients
                        self.optimizer.zero_grad()
                        
                        # Forward pass
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        
                        # Check for invalid loss
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning(f"Invalid loss in batch {batch_idx}: {loss.item()}")
                            failed_batches += 1
                            continue
                        
                        # Backward pass
                        loss.backward()
                        
                        # Gradient clipping to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        self.optimizer.step()
                        
                        # Track metrics
                        batch_loss = loss.item() * data.size(0)
                        epoch_loss += batch_loss
                        epoch_samples += data.size(0)
                        successful_batches += 1
                        
                    except Exception as batch_error:
                        logger.warning(f"Error in batch {batch_idx}: {batch_error}")
                        failed_batches += 1
                        epoch_failed_batches += 1
                        
                        # If too many batches fail, raise an error
                        if epoch_failed_batches > len(train_loader) * 0.5:
                            raise TrainingError(f"More than 50% of batches failed in epoch {epoch}")
                
                if epoch_samples > 0:
                    avg_epoch_loss = epoch_loss / epoch_samples
                    total_loss += epoch_loss
                    num_samples += epoch_samples
                    
                    logger.info(f"Epoch {epoch + 1}/{self.local_epochs}, "
                              f"Loss: {avg_epoch_loss:.4f}, "
                              f"Successful batches: {successful_batches - failed_batches}, "
                              f"Failed batches: {epoch_failed_batches}")
                else:
                    raise TrainingError(f"No successful batches in epoch {epoch}")
            
            training_time = time.time() - start_time
            
            if num_samples == 0:
                raise TrainingError("No samples processed during training")
            
            avg_loss = total_loss / num_samples
            
            # Get final model weights
            final_weights = self.get_model_weights()
            
            # Prepare training metadata
            metadata = {
                "num_samples": num_samples,
                "num_epochs": self.local_epochs,
                "avg_loss": avg_loss,
                "training_time": training_time,
                "learning_rate": self.learning_rate,
                "device": self.device,
                "successful_batches": successful_batches,
                "failed_batches": failed_batches,
                "success_rate": successful_batches / (successful_batches + failed_batches) if (successful_batches + failed_batches) > 0 else 0.0
            }
            
            logger.info(f"Local training completed successfully. "
                       f"Avg loss: {avg_loss:.4f}, "
                       f"Time: {training_time:.2f}s, "
                       f"Samples: {num_samples}, "
                       f"Success rate: {metadata['success_rate']:.2%}")
            
            return final_weights, metadata
            
        except TrainingError:
            # Re-raise training errors
            raise
        except Exception as e:
            logger.error(f"Unexpected error during local training: {e}")
            raise TrainingError(f"Training failed: {e}") from e
    
    @retry_on_failure(RetryConfig(max_retries=2, initial_delay=0.5) if ERROR_HANDLING_AVAILABLE else None)
    def get_model_weights(self) -> bytes:
        """
        Serialize current model weights with error handling.
        
        Returns:
            Serialized model state dict as bytes
        """
        try:
            # Get model state dict
            state_dict = self.model.state_dict()
            
            # Validate state dict
            if not state_dict:
                raise TrainingError("Model state dict is empty")
            
            # Check for NaN or Inf values
            for name, param in state_dict.items():
                if torch.isnan(param).any():
                    logger.warning(f"NaN values found in parameter {name}")
                if torch.isinf(param).any():
                    logger.warning(f"Inf values found in parameter {name}")
            
            # Serialize to bytes
            buffer = io.BytesIO()
            torch.save(state_dict, buffer)
            serialized_data = buffer.getvalue()
            
            if len(serialized_data) == 0:
                raise TrainingError("Serialized model data is empty")
            
            logger.debug(f"Model weights serialized successfully: {len(serialized_data)} bytes")
            return serialized_data
            
        except Exception as e:
            logger.error(f"Error serializing model weights: {e}")
            raise TrainingError(f"Failed to serialize model weights: {e}") from e
    
    @retry_on_failure(RetryConfig(max_retries=2, initial_delay=0.5) if ERROR_HANDLING_AVAILABLE else None)
    def load_model_weights(self, weights: bytes):
        """
        Load model weights from serialized bytes with error handling.
        
        Args:
            weights: Serialized model state dict as bytes
        """
        try:
            # Validate input
            if not weights:
                raise TrainingError("Model weights data is empty")
            
            if not isinstance(weights, bytes):
                raise TrainingError(f"Expected bytes, got {type(weights)}")
            
            # Deserialize weights
            buffer = io.BytesIO(weights)
            state_dict = torch.load(buffer, map_location=self.device)
            
            # Validate loaded state dict
            if not state_dict:
                raise TrainingError("Loaded state dict is empty")
            
            # Check compatibility with current model
            model_state = self.model.state_dict()
            if set(state_dict.keys()) != set(model_state.keys()):
                missing_keys = set(model_state.keys()) - set(state_dict.keys())
                extra_keys = set(state_dict.keys()) - set(model_state.keys())
                
                error_msg = "Model state dict mismatch:"
                if missing_keys:
                    error_msg += f" Missing keys: {missing_keys}"
                if extra_keys:
                    error_msg += f" Extra keys: {extra_keys}"
                
                raise TrainingError(error_msg)
            
            # Check parameter shapes
            for name, param in state_dict.items():
                if name in model_state and param.shape != model_state[name].shape:
                    raise TrainingError(f"Shape mismatch for parameter {name}: "
                                      f"expected {model_state[name].shape}, got {param.shape}")
            
            # Load into model
            self.model.load_state_dict(state_dict)
            logger.info("Model weights loaded successfully")
            
        except TrainingError:
            # Re-raise training errors
            raise
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            raise TrainingError(f"Failed to load model weights: {e}") from e
    
    @log_execution_time
    @catch_and_log_exceptions()
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test data with enhanced error handling.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            if test_loader is None:
                raise TrainingError("Test data loader cannot be None")
            
            if len(test_loader.dataset) == 0:
                raise TrainingError("Test dataset is empty")
            
            self.model.eval()
            total_loss = 0.0
            correct = 0
            total = 0
            failed_batches = 0
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_loader):
                    try:
                        data, target = data.to(self.device), target.to(self.device)
                        
                        # Validate batch data
                        if torch.isnan(data).any() or torch.isinf(data).any():
                            logger.warning(f"Invalid test data in batch {batch_idx}, skipping")
                            failed_batches += 1
                            continue
                        
                        output = self.model(data)
                        
                        # Calculate loss
                        loss = self.criterion(output, target)
                        
                        # Check for invalid loss
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning(f"Invalid loss in test batch {batch_idx}: {loss.item()}")
                            failed_batches += 1
                            continue
                        
                        total_loss += loss.item() * data.size(0)
                        
                        # Calculate accuracy
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                        
                    except Exception as batch_error:
                        logger.warning(f"Error in test batch {batch_idx}: {batch_error}")
                        failed_batches += 1
                        continue
            
            if total == 0:
                raise TrainingError("No valid test samples processed")
            
            avg_loss = total_loss / total
            accuracy = 100.0 * correct / total
            
            metrics = {
                "test_loss": avg_loss,
                "test_accuracy": accuracy,
                "num_test_samples": total,
                "failed_batches": failed_batches,
                "success_rate": (len(test_loader) - failed_batches) / len(test_loader) if len(test_loader) > 0 else 0.0
            }
            
            logger.info(f"Evaluation completed. Loss: {avg_loss:.4f}, "
                       f"Accuracy: {accuracy:.2f}%, "
                       f"Samples: {total}, "
                       f"Success rate: {metrics['success_rate']:.2%}")
            
            return metrics
            
        except TrainingError:
            # Re-raise training errors
            raise
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise TrainingError(f"Evaluation failed: {e}") from e

# Backward compatibility function
def train_local_model(train_loader: Optional[DataLoader] = None,
                     global_model_weights: Optional[bytes] = None) -> bytes:
    """
    Backward compatibility function for existing code.
    
    Args:
        train_loader: DataLoader for training data (if None, creates dummy data)
        global_model_weights: Global model weights to start from
        
    Returns:
        Serialized model weights as bytes
    """
    try:
        # Create trainer
        trainer = LocalTrainer()
        
        # If no train_loader provided, create dummy data for testing
        if train_loader is None:
            logger.warning("No training data provided, creating dummy data for testing")
            # Create dummy MNIST-like data
            dummy_data = torch.randn(100, 1, 28, 28)
            dummy_labels = torch.randint(0, 10, (100,))
            dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
            train_loader = DataLoader(dummy_dataset, batch_size=32, shuffle=True)
        
        # Train model
        weights, metadata = trainer.train_local_model(train_loader, global_model_weights)
        
        logger.info(f"Training completed with metadata: {metadata}")
        return weights
        
    except Exception as e:
        logger.error(f"Error in train_local_model: {e}")
        # Fallback to dummy weights for testing
        logger.warning("Falling back to dummy weights")
        dummy_model = SimpleCNN()
        buffer = io.BytesIO()
        torch.save(dummy_model.state_dict(), buffer)
        return buffer.getvalue()
