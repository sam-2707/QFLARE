"""
Data loading logic for the QFLARE edge node.
This module simulates loading local data for model training.
"""
import numpy as np
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def load_local_data() -> Dict[str, np.ndarray]:
    """
    Simulates loading a small, unique dataset on the edge device.
    
    In a real-world scenario, this function would read from local sensors,
    a local database, or files on the device.
    
    Returns:
        A dictionary containing training data ('x_train') and labels ('y_train').
    """
    try:
        logger.info("Loading local dataset...")
        
        # Simulate a simple dataset for a linear model (y = 2x + 1)
        # Each device will have a slightly different slice of the data.
        num_samples = np.random.randint(50, 150)
        x_train = np.random.rand(num_samples, 1) * 10
        
        # Add some noise to make it more realistic
        noise = np.random.randn(num_samples, 1) * 0.5
        y_train = 2 * x_train + 1 + noise
        
        logger.info(f"Loaded {num_samples} local data samples.")
        
        return {"x_train": x_train, "y_train": y_train}
        
    except Exception as e:
        logger.error(f"Failed to load local data: {e}")
        return {}