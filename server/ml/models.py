"""
QFLARE Real Machine Learning Models

This module contains actual PyTorch models for federated learning training.
Supports MNIST, CIFAR-10, and other datasets mentioned in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MNISTNet(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.
    Based on LeNet-5 architecture with modern improvements.
    """
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Calculate the size for the first linear layer
        # After conv1 + pool: 28x28 -> 14x14
        # After conv2 + pool: 14x14 -> 7x7
        # With 64 channels: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CIFAR10Net(nn.Module):
    """
    Convolutional Neural Network for CIFAR-10 image classification.
    More complex architecture for 32x32 color images.
    """
    
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        
        # After 3 pooling operations: 32x32 -> 16x16 -> 8x8 -> 4x4
        # With 256 channels: 256 * 4 * 4 = 4096
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        # First block
        x = F.relu(self.conv1(x))
        x = self.batch_norm1(F.relu(self.conv2(x)))
        x = self.pool(x)
        
        # Second block
        x = F.relu(self.conv3(x))
        x = self.batch_norm2(F.relu(self.conv4(x)))
        x = self.pool(x)
        
        # Third block
        x = F.relu(self.conv5(x))
        x = self.batch_norm3(F.relu(self.conv6(x)))
        x = self.pool(x)
        
        # Classifier
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class SimpleMLPNet(nn.Module):
    """
    Simple Multi-Layer Perceptron for basic federated learning testing.
    Can be used with various datasets by adjusting input/output dimensions.
    """
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, output_dim: int = 10):
        super(SimpleMLPNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def create_model(dataset: str = "mnist", **kwargs) -> nn.Module:
    """
    Factory function to create appropriate model for the dataset.
    
    Args:
        dataset: Dataset name ("mnist", "cifar10", "simple")
        **kwargs: Additional arguments for model creation
        
    Returns:
        PyTorch model instance
    """
    dataset = dataset.lower()
    
    if dataset == "mnist":
        return MNISTNet()
    elif dataset == "cifar10":
        return CIFAR10Net()
    elif dataset == "simple" or dataset == "mlp":
        input_dim = kwargs.get('input_dim', 784)
        hidden_dim = kwargs.get('hidden_dim', 128)
        output_dim = kwargs.get('output_dim', 10)
        return SimpleMLPNet(input_dim, hidden_dim, output_dim)
    else:
        logger.warning(f"Unknown dataset {dataset}, using MNIST model")
        return MNISTNet()


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_class": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        "architecture": str(model)
    }


def serialize_model_weights(model: nn.Module) -> bytes:
    """
    Serialize PyTorch model weights to bytes for transmission.
    
    Args:
        model: PyTorch model
        
    Returns:
        Serialized model weights as bytes
    """
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def deserialize_model_weights(model: nn.Module, weights_bytes: bytes) -> nn.Module:
    """
    Deserialize model weights from bytes and load into model.
    
    Args:
        model: PyTorch model instance
        weights_bytes: Serialized weights as bytes
        
    Returns:
        Model with loaded weights
    """
    import io
    buffer = io.BytesIO(weights_bytes)
    state_dict = torch.load(buffer, map_location='cpu')
    model.load_state_dict(state_dict)
    return model


def calculate_model_similarity(model1: nn.Module, model2: nn.Module) -> float:
    """
    Calculate cosine similarity between two models' parameters.
    Useful for Byzantine fault detection.
    
    Args:
        model1: First model
        model2: Second model
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    params1 = torch.cat([p.flatten() for p in model1.parameters()])
    params2 = torch.cat([p.flatten() for p in model2.parameters()])
    
    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(params1.unsqueeze(0), params2.unsqueeze(0))
    return cos_sim.item()


if __name__ == "__main__":
    # Test model creation
    print("Testing QFLARE ML Models...")
    
    # Test MNIST model
    mnist_model = create_model("mnist")
    mnist_info = get_model_info(mnist_model)
    print(f"MNIST Model: {mnist_info['total_parameters']} parameters")
    
    # Test CIFAR-10 model
    cifar_model = create_model("cifar10")
    cifar_info = get_model_info(cifar_model)
    print(f"CIFAR-10 Model: {cifar_info['total_parameters']} parameters")
    
    # Test serialization
    weights_bytes = serialize_model_weights(mnist_model)
    print(f"Serialized weights size: {len(weights_bytes)} bytes")
    
    # Test deserialization
    new_model = create_model("mnist")
    deserialize_model_weights(new_model, weights_bytes)
    
    # Test similarity
    similarity = calculate_model_similarity(mnist_model, new_model)
    print(f"Model similarity: {similarity:.4f}")
    
    print("All tests passed!")