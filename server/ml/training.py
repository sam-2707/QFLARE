"""
QFLARE Real Training Engine

This module implements actual PyTorch model training for federated learning.
Replaces the mock training system with real ML training loops.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np
import time
import logging
from pathlib import Path
import json
import io

from .models import create_model, serialize_model_weights, deserialize_model_weights

logger = logging.getLogger(__name__)


class FederatedDataset:
    """
    Manages dataset partitioning for federated learning simulation.
    Creates non-IID data distributions across clients.
    """
    
    def __init__(self, dataset_name: str = "mnist", data_dir: str = "./data"):
        self.dataset_name = dataset_name.lower()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.train_dataset = None
        self.test_dataset = None
        self.num_classes = 10
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the appropriate dataset."""
        if self.dataset_name == "mnist":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            self.train_dataset = torchvision.datasets.MNIST(
                root=str(self.data_dir), train=True, download=True, transform=transform
            )
            self.test_dataset = torchvision.datasets.MNIST(
                root=str(self.data_dir), train=False, download=True, transform=transform
            )
            self.num_classes = 10
            
        elif self.dataset_name == "cifar10":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
            self.train_dataset = torchvision.datasets.CIFAR10(
                root=str(self.data_dir), train=True, download=True, transform=transform_train
            )
            self.test_dataset = torchvision.datasets.CIFAR10(
                root=str(self.data_dir), train=False, download=True, transform=transform_test
            )
            self.num_classes = 10
            
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def create_federated_split(self, num_clients: int, alpha: float = 0.5) -> List[Subset]:
        """
        Create federated data splits using Dirichlet distribution for non-IID data.
        
        Args:
            num_clients: Number of federated clients
            alpha: Dirichlet concentration parameter (lower = more non-IID)
            
        Returns:
            List of dataset subsets for each client
        """
        if self.train_dataset is None:
            raise ValueError("Dataset not loaded")
        
        # Get labels
        if hasattr(self.train_dataset, 'targets'):
            labels = np.array(self.train_dataset.targets)
        elif hasattr(self.train_dataset, 'labels'):
            labels = np.array(self.train_dataset.labels)
        else:
            # Extract labels manually
            labels = np.array([self.train_dataset[i][1] for i in range(len(self.train_dataset))])
        
        num_samples = len(labels)
        client_datasets = []
        
        # Create Dirichlet distribution for each class
        class_indices = [np.where(labels == i)[0] for i in range(self.num_classes)]
        
        client_indices = [[] for _ in range(num_clients)]
        
        for class_idx, indices in enumerate(class_indices):
            # Generate Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.cumsum(proportions)
            
            # Shuffle indices for this class
            np.random.shuffle(indices)
            
            # Distribute indices according to proportions
            prev_prop = 0
            for client_idx in range(num_clients):
                start_idx = int(prev_prop * len(indices))
                end_idx = int(proportions[client_idx] * len(indices))
                client_indices[client_idx].extend(indices[start_idx:end_idx])
                prev_prop = proportions[client_idx]
        
        # Create subsets
        for indices in client_indices:
            if len(indices) > 0:
                client_datasets.append(Subset(self.train_dataset, indices))
            else:
                # Ensure each client has at least some data
                client_datasets.append(Subset(self.train_dataset, [0]))
        
        logger.info(f"Created {num_clients} federated splits with alpha={alpha}")
        for i, dataset in enumerate(client_datasets):
            logger.info(f"Client {i}: {len(dataset)} samples")
        
        return client_datasets
    
    def get_test_loader(self, batch_size: int = 64) -> DataLoader:
        """Get test data loader."""
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)


class FederatedTrainer:
    """
    Real federated learning trainer that performs actual model training.
    Replaces the mock training system with PyTorch-based training.
    """
    
    def __init__(self, 
                 dataset_name: str = "mnist",
                 model_name: str = "mnist",
                 data_dir: str = "./data",
                 device: str = "auto"):
        """
        Initialize the federated trainer.
        
        Args:
            dataset_name: Name of dataset to use
            model_name: Name of model architecture to use
            data_dir: Directory for dataset storage
            device: Device to use ("auto", "cpu", "cuda")
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.data_dir = data_dir
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize dataset
        self.federated_dataset = FederatedDataset(dataset_name, data_dir)
        
        # Create global model
        self.global_model = create_model(model_name)
        self.global_model.to(self.device)
        
        # Training history
        self.training_history = {
            "rounds": [],
            "global_accuracy": [],
            "global_loss": [],
            "client_metrics": []
        }
    
    def train_client(self, 
                    client_dataset: Dataset,
                    global_weights: bytes,
                    epochs: int = 5,
                    learning_rate: float = 0.01,
                    batch_size: int = 32) -> Tuple[bytes, Dict[str, float]]:
        """
        Train a client model with real PyTorch training.
        
        Args:
            client_dataset: Client's dataset subset
            global_weights: Serialized global model weights
            epochs: Number of local training epochs
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            
        Returns:
            Tuple of (serialized_weights, training_metrics)
        """
        # Create local model and load global weights
        local_model = create_model(self.model_name)
        local_model = deserialize_model_weights(local_model, global_weights)
        local_model.to(self.device)
        local_model.train()
        
        # Create data loader
        train_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize optimizer and loss function
        optimizer = optim.SGD(local_model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # Training metrics
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                output = local_model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_total += target.size(0)
            
            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total
        
        training_time = time.time() - start_time
        
        # Calculate final metrics
        avg_loss = total_loss / (epochs * len(train_loader))
        accuracy = 100.0 * correct / total
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "training_time": training_time,
            "epochs": epochs,
            "samples": len(client_dataset),
            "batches_per_epoch": len(train_loader)
        }
        
        # Serialize and return weights
        weights_bytes = serialize_model_weights(local_model)
        
        logger.debug(f"Client training completed: {accuracy:.2f}% accuracy, {avg_loss:.4f} loss")
        
        return weights_bytes, metrics
    
    def aggregate_weights(self, 
                         client_weights: List[bytes], 
                         client_sizes: List[int]) -> bytes:
        """
        Aggregate client model weights using FedAvg algorithm.
        
        Args:
            client_weights: List of serialized client weights
            client_sizes: List of client dataset sizes for weighted averaging
            
        Returns:
            Serialized aggregated weights
        """
        if not client_weights:
            raise ValueError("No client weights provided for aggregation")
        
        # Load first model to get structure
        temp_model = create_model(self.model_name)
        temp_model = deserialize_model_weights(temp_model, client_weights[0])
        
        # Initialize aggregated state dict
        aggregated_state = {}
        total_samples = sum(client_sizes)
        
        for name, param in temp_model.state_dict().items():
            aggregated_state[name] = torch.zeros_like(param)
        
        # Weighted averaging
        for i, (weights_bytes, num_samples) in enumerate(zip(client_weights, client_sizes)):
            # Load client model
            client_model = create_model(self.model_name)
            client_model = deserialize_model_weights(client_model, weights_bytes)
            
            # Weight by number of samples
            weight = num_samples / total_samples
            
            # Add weighted parameters
            for name, param in client_model.state_dict().items():
                aggregated_state[name] += weight * param
        
        # Load aggregated weights into a model and serialize
        aggregated_model = create_model(self.model_name)
        aggregated_model.load_state_dict(aggregated_state)
        
        return serialize_model_weights(aggregated_model)
    
    def evaluate_global_model(self, model_weights: bytes) -> Dict[str, float]:
        """
        Evaluate the global model on test dataset.
        
        Args:
            model_weights: Serialized global model weights
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Load model with weights
        model = create_model(self.model_name)
        model = deserialize_model_weights(model, model_weights)
        model.to(self.device)
        model.eval()
        
        # Get test data loader
        test_loader = self.federated_dataset.get_test_loader()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                
                # Calculate loss
                loss = criterion(output, target)
                total_loss += loss.item()
                
                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total
        
        return {
            "test_loss": avg_loss,
            "test_accuracy": accuracy,
            "total_samples": total,
            "correct_predictions": correct
        }
    
    def run_federated_round(self,
                           num_clients: int = 10,
                           participation_rate: float = 0.3,
                           epochs: int = 5,
                           learning_rate: float = 0.01,
                           batch_size: int = 32) -> Dict[str, Any]:
        """
        Run one complete federated learning round.
        
        Args:
            num_clients: Total number of clients
            participation_rate: Fraction of clients participating per round
            epochs: Local training epochs per client
            learning_rate: Learning rate for local training
            batch_size: Batch size for local training
            
        Returns:
            Round results with metrics
        """
        # Create federated data splits (only once)
        if not hasattr(self, 'client_datasets'):
            self.client_datasets = self.federated_dataset.create_federated_split(num_clients)
        
        # Select participating clients
        num_participating = max(1, int(num_clients * participation_rate))
        participating_clients = np.random.choice(num_clients, num_participating, replace=False)
        
        logger.info(f"Round starting with {num_participating} clients: {participating_clients}")
        
        # Get current global weights
        global_weights = serialize_model_weights(self.global_model)
        
        # Train clients
        client_weights = []
        client_sizes = []
        client_metrics = []
        
        for client_id in participating_clients:
            client_dataset = self.client_datasets[client_id]
            
            # Train client
            weights, metrics = self.train_client(
                client_dataset=client_dataset,
                global_weights=global_weights,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size
            )
            
            client_weights.append(weights)
            client_sizes.append(len(client_dataset))
            client_metrics.append({**metrics, "client_id": int(client_id)})
        
        # Aggregate weights
        aggregated_weights = self.aggregate_weights(client_weights, client_sizes)
        
        # Update global model
        self.global_model = deserialize_model_weights(self.global_model, aggregated_weights)
        
        # Evaluate global model
        global_metrics = self.evaluate_global_model(aggregated_weights)
        
        # Record history
        round_data = {
            "participating_clients": participating_clients.tolist(),
            "num_participating": num_participating,
            "client_metrics": client_metrics,
            "global_metrics": global_metrics,
            "aggregated_samples": sum(client_sizes)
        }
        
        self.training_history["rounds"].append(len(self.training_history["rounds"]) + 1)
        self.training_history["global_accuracy"].append(global_metrics["test_accuracy"])
        self.training_history["global_loss"].append(global_metrics["test_loss"])
        self.training_history["client_metrics"].append(client_metrics)
        
        logger.info(f"Round completed - Global accuracy: {global_metrics['test_accuracy']:.2f}%")
        
        return round_data
    
    def get_model_weights(self) -> bytes:
        """Get current global model weights."""
        return serialize_model_weights(self.global_model)
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get complete training history."""
        return self.training_history.copy()


def create_real_fl_trainer(config: Dict[str, Any]) -> FederatedTrainer:
    """
    Factory function to create a federated trainer from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured FederatedTrainer instance
    """
    return FederatedTrainer(
        dataset_name=config.get("dataset", "mnist"),
        model_name=config.get("model", "mnist"),
        data_dir=config.get("data_dir", "./data"),
        device=config.get("device", "auto")
    )


if __name__ == "__main__":
    # Test the real training system
    print("Testing QFLARE Real Training Engine...")
    
    # Create trainer
    trainer = FederatedTrainer(dataset_name="mnist", model_name="mnist")
    
    # Run one federated round
    results = trainer.run_federated_round(
        num_clients=5,
        participation_rate=0.6,
        epochs=1,  # Quick test
        learning_rate=0.01,
        batch_size=64
    )
    
    print(f"Round completed!")
    print(f"Global accuracy: {results['global_metrics']['test_accuracy']:.2f}%")
    print(f"Participating clients: {results['num_participating']}")
    
    print("Real training engine test passed!")