"""
Data loading logic for the QFLARE edge node.
This module handles loading and partitioning datasets for federated learning.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import logging
import os
from typing import Dict, Any, Tuple, Optional, List
import hashlib

logger = logging.getLogger(__name__)

class FederatedDataLoader:
    """
    Federated data loader that simulates data distribution across edge devices.
    Supports MNIST, CIFAR-10, and custom datasets.
    """
    
    def __init__(self, 
                 dataset_name: str = "MNIST",
                 data_dir: str = "./data",
                 device_id: str = "edge_device_001",
                 num_devices: int = 10,
                 iid: bool = True,
                 alpha: float = 0.5):
        """
        Initialize federated data loader.
        
        Args:
            dataset_name: Name of dataset ("MNIST", "CIFAR10")
            data_dir: Directory to store dataset
            device_id: Unique identifier for this device
            num_devices: Total number of devices in federation
            iid: Whether to use IID (independent and identically distributed) data
            alpha: Dirichlet distribution parameter for non-IID data (lower = more non-IID)
        """
        self.dataset_name = dataset_name.upper()
        self.data_dir = data_dir
        self.device_id = device_id
        self.num_devices = num_devices
        self.iid = iid
        self.alpha = alpha
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Load dataset
        self.train_dataset, self.test_dataset = self._load_dataset()
        
        # Partition data for this device
        self.local_train_indices = self._partition_data()
        
        logger.info(f"Federated data loader initialized for {dataset_name}")
        logger.info(f"Device {device_id} has {len(self.local_train_indices)} training samples")
    
    def _load_dataset(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """Load the specified dataset."""
        try:
            if self.dataset_name == "MNIST":
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                
                train_dataset = torchvision.datasets.MNIST(
                    root=self.data_dir,
                    train=True,
                    download=True,
                    transform=transform
                )
                
                test_dataset = torchvision.datasets.MNIST(
                    root=self.data_dir,
                    train=False,
                    download=True,
                    transform=transform
                )
                
            elif self.dataset_name == "CIFAR10":
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
                
                train_dataset = torchvision.datasets.CIFAR10(
                    root=self.data_dir,
                    train=True,
                    download=True,
                    transform=transform
                )
                
                test_dataset = torchvision.datasets.CIFAR10(
                    root=self.data_dir,
                    train=False,
                    download=True,
                    transform=transform
                )
            else:
                raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
            logger.info(f"Dataset {self.dataset_name} loaded successfully")
            logger.info(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
            
            return train_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset {self.dataset_name}: {e}")
            raise
    
    def _partition_data(self) -> List[int]:
        """
        Partition training data for this specific device.
        
        Returns:
            List of indices for this device's training data
        """
        try:
            # Get device index from device_id (deterministic)
            device_hash = hashlib.md5(self.device_id.encode()).hexdigest()
            device_idx = int(device_hash, 16) % self.num_devices
            
            if self.iid:
                # IID partitioning: random split
                indices = list(range(len(self.train_dataset)))
                np.random.seed(42)  # Deterministic split
                np.random.shuffle(indices)
                
                # Split equally among devices
                split_size = len(indices) // self.num_devices
                start_idx = device_idx * split_size
                end_idx = start_idx + split_size
                
                if device_idx == self.num_devices - 1:  # Last device gets remaining samples
                    end_idx = len(indices)
                
                device_indices = indices[start_idx:end_idx]
                
            else:
                # Non-IID partitioning: Dirichlet distribution
                device_indices = self._non_iid_partition(device_idx)
            
            logger.info(f"Device {self.device_id} (idx: {device_idx}) allocated {len(device_indices)} samples")
            return device_indices
            
        except Exception as e:
            logger.error(f"Error partitioning data: {e}")
            raise
    
    def _non_iid_partition(self, device_idx: int) -> List[int]:
        """
        Create non-IID data partition using Dirichlet distribution.
        
        Args:
            device_idx: Index of this device
            
        Returns:
            List of indices for this device
        """
        try:
            # Get labels
            if hasattr(self.train_dataset, 'targets'):
                labels = np.array(self.train_dataset.targets)
            elif hasattr(self.train_dataset, 'labels'):
                labels = np.array(self.train_dataset.labels)
            else:
                # Fallback: extract labels manually
                labels = np.array([self.train_dataset[i][1] for i in range(len(self.train_dataset))])
            
            num_classes = len(np.unique(labels))
            
            # Create Dirichlet distribution for each class
            np.random.seed(42)  # Deterministic partitioning
            
            # For each class, sample proportions for each device
            class_distributions = np.random.dirichlet([self.alpha] * self.num_devices, num_classes)
            
            device_indices = []
            
            for class_id in range(num_classes):
                # Get all indices for this class
                class_indices = np.where(labels == class_id)[0]
                
                # Calculate how many samples this device gets for this class
                proportion = class_distributions[class_id][device_idx]
                num_samples = int(proportion * len(class_indices))
                
                # Sample indices for this device
                if num_samples > 0:
                    start_idx = sum([int(class_distributions[class_id][i] * len(class_indices)) 
                                   for i in range(device_idx)])
                    end_idx = start_idx + num_samples
                    
                    device_class_indices = class_indices[start_idx:end_idx]
                    device_indices.extend(device_class_indices)
            
            return device_indices
            
        except Exception as e:
            logger.error(f"Error creating non-IID partition: {e}")
            # Fallback to IID if non-IID fails
            logger.warning("Falling back to IID partitioning")
            return self._iid_partition(device_idx)
    
    def get_train_loader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """
        Get DataLoader for local training data.
        
        Args:
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader for local training data
        """
        try:
            # Create subset with local indices
            local_dataset = Subset(self.train_dataset, self.local_train_indices)
            
            # Create DataLoader
            train_loader = DataLoader(
                local_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,  # Set to 0 for Windows compatibility
                pin_memory=False
            )
            
            logger.info(f"Created train loader with {len(local_dataset)} samples, batch_size={batch_size}")
            return train_loader
            
        except Exception as e:
            logger.error(f"Error creating train loader: {e}")
            raise
    
    def get_test_loader(self, batch_size: int = 32) -> DataLoader:
        """
        Get DataLoader for test data.
        
        Args:
            batch_size: Batch size for testing
            
        Returns:
            DataLoader for test data
        """
        try:
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # Set to 0 for Windows compatibility
                pin_memory=False
            )
            
            logger.info(f"Created test loader with {len(self.test_dataset)} samples, batch_size={batch_size}")
            return test_loader
            
        except Exception as e:
            logger.error(f"Error creating test loader: {e}")
            raise
    
    def get_data_stats(self) -> Dict[str, Any]:
        """
        Get statistics about local data distribution.
        
        Returns:
            Dictionary with data statistics
        """
        try:
            # Get labels for local data
            local_labels = []
            for idx in self.local_train_indices:
                _, label = self.train_dataset[idx]
                local_labels.append(label)
            
            local_labels = np.array(local_labels)
            unique, counts = np.unique(local_labels, return_counts=True)
            
            stats = {
                "dataset": self.dataset_name,
                "device_id": self.device_id,
                "total_samples": len(self.local_train_indices),
                "num_classes": len(unique),
                "class_distribution": dict(zip(unique.tolist(), counts.tolist())),
                "is_iid": self.iid
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting data stats: {e}")
            return {}

# Backward compatibility functions
def load_local_data(device_id: str = "edge_device_001", 
                   dataset: str = "MNIST") -> Tuple[DataLoader, DataLoader]:
    """
    Backward compatibility function for existing code.
    
    Args:
        device_id: Device identifier
        dataset: Dataset name ("MNIST" or "CIFAR10")
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    try:
        # Create federated data loader
        fed_loader = FederatedDataLoader(
            dataset_name=dataset,
            device_id=device_id,
            num_devices=10,  # Default to 10 devices
            iid=True  # Default to IID for simplicity
        )
        
        # Get data loaders
        train_loader = fed_loader.get_train_loader(batch_size=32)
        test_loader = fed_loader.get_test_loader(batch_size=32)
        
        # Log data statistics
        stats = fed_loader.get_data_stats()
        logger.info(f"Data statistics: {stats}")
        
        return train_loader, test_loader
        
    except Exception as e:
        logger.error(f"Error in load_local_data: {e}")
        raise

def get_sample_data() -> Tuple[DataLoader, DataLoader]:
    """
    Get sample data for testing purposes.
    
    Returns:
        Tuple of (train_loader, test_loader) with dummy data
    """
    try:
        logger.warning("Creating dummy data for testing")
        
        # Create dummy MNIST-like data
        train_data = torch.randn(1000, 1, 28, 28)
        train_labels = torch.randint(0, 10, (1000,))
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        
        test_data = torch.randn(200, 1, 28, 28)
        test_labels = torch.randint(0, 10, (200,))
        test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        return train_loader, test_loader
        
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        raise