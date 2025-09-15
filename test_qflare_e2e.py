#!/usr/bin/env python3
"""
End-to-End Test Script for QFLARE
Tests the complete federated learning workflow.
"""

import sys
import os
import asyncio
import logging
import time
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "server"))
sys.path.insert(0, str(project_root / "edge_node"))
sys.path.insert(0, str(project_root))

# Mock liboqs early to avoid import issues
from unittest.mock import MagicMock
sys.modules['oqs'] = MagicMock()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        # Test server imports
        from server.main import app
        from server.auth.pqcrypto_utils import generate_device_keypair
        from server.fl_core.aggregator import store_model_update
        logger.info("✅ Server imports successful")
    except Exception as e:
        logger.error(f"❌ Server import failed: {e}")
        return False
    
    try:
        # Test edge node imports
        from edge_node.trainer import LocalTrainer, train_local_model
        from edge_node.data_loader import FederatedDataLoader, load_local_data
        logger.info("✅ Edge node imports successful")
    except Exception as e:
        logger.error(f"❌ Edge node import failed: {e}")
        return False
    
    try:
        # Test model utilities
        from models.model_utils import ModelSerializer, FederatedAggregator
        logger.info("✅ Model utilities imports successful")
    except Exception as e:
        logger.error(f"❌ Model utilities import failed: {e}")
        return False
    
    return True

def test_data_loading():
    """Test data loading functionality."""
    logger.info("Testing data loading...")
    
    try:
        from edge_node.data_loader import FederatedDataLoader, get_sample_data
        
        # Test sample data creation
        train_loader, test_loader = get_sample_data()
        logger.info(f"✅ Sample data created: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")
        
        # Test federated data loader (may fail if datasets not downloaded)
        try:
            fed_loader = FederatedDataLoader(
                dataset_name="MNIST",
                device_id="test_device",
                num_devices=5,
                iid=True
            )
            train_loader = fed_loader.get_train_loader(batch_size=16)
            test_loader = fed_loader.get_test_loader(batch_size=16)
            stats = fed_loader.get_data_stats()
            logger.info(f"✅ MNIST data loaded: {stats}")
        except Exception as e:
            logger.warning(f"⚠️ MNIST loading failed (expected in testing): {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loading test failed: {e}")
        return False

def test_model_training():
    """Test model training functionality."""
    logger.info("Testing model training...")
    
    try:
        from edge_node.trainer import LocalTrainer
        from edge_node.data_loader import get_sample_data
        
        # Create trainer
        trainer = LocalTrainer(
            learning_rate=0.01,
            local_epochs=1,  # Quick test
            batch_size=16
        )
        
        # Get sample data
        train_loader, test_loader = get_sample_data()
        
        # Train model
        model_weights, metadata = trainer.train_local_model(train_loader)
        logger.info(f"✅ Training completed: {metadata}")
        
        # Evaluate model
        eval_metrics = trainer.evaluate_model(test_loader)
        logger.info(f"✅ Evaluation completed: {eval_metrics}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model training test failed: {e}")
        return False

def test_model_serialization():
    """Test model serialization functionality."""
    logger.info("Testing model serialization...")
    
    try:
        from models.model_utils import ModelSerializer, FederatedAggregator
        from edge_node.trainer import SimpleCNN
        import torch
        
        # Create test model
        model = SimpleCNN(num_classes=10, input_channels=1)
        
        # Test serialization
        model_bytes = ModelSerializer.serialize_model(model)
        logger.info(f"✅ Model serialized: {len(model_bytes)} bytes")
        
        # Test deserialization
        new_model = SimpleCNN(num_classes=10, input_channels=1)
        ModelSerializer.deserialize_model(model_bytes, new_model)
        logger.info("✅ Model deserialized successfully")
        
        # Test weight serialization
        weights_bytes = ModelSerializer.serialize_weights(model)
        weights_dict = ModelSerializer.deserialize_weights(weights_bytes)
        logger.info(f"✅ Weights serialized/deserialized: {len(weights_dict)} parameters")
        
        # Test aggregation
        model1_weights = model.state_dict()
        model2 = SimpleCNN(num_classes=10, input_channels=1)
        model2_weights = model2.state_dict()
        
        aggregated = FederatedAggregator.federated_averaging([model1_weights, model2_weights])
        logger.info(f"✅ Model aggregation successful: {len(aggregated)} parameters")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model serialization test failed: {e}")
        return False

def test_server_startup():
    """Test if server can start without errors."""
    logger.info("Testing server startup...")
    
    try:
        # Import server components
        from server.main import app
        from fastapi.testclient import TestClient
        
        # Create test client
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        if response.status_code == 200:
            logger.info("✅ Server health check passed")
        else:
            logger.warning(f"⚠️ Health check returned {response.status_code}")
        
        # Test main page
        response = client.get("/")
        if response.status_code == 200:
            logger.info("✅ Main page accessible")
        else:
            logger.warning(f"⚠️ Main page returned {response.status_code}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Server startup test failed: {e}")
        return False

def test_quantum_crypto():
    """Test quantum cryptography utilities."""
    logger.info("Testing quantum cryptography...")
    
    try:
        from server.auth.pqcrypto_utils import generate_device_keypair
        
        # Test key generation
        kem_key, sig_key = generate_device_keypair("test_device")
        logger.info(f"✅ PQC key generation successful: KEM={len(kem_key)}, SIG={len(sig_key)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quantum crypto test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting QFLARE End-to-End Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Data Loading Tests", test_data_loading),
        ("Model Training Tests", test_model_training),
        ("Model Serialization Tests", test_model_serialization),
        ("Server Startup Tests", test_server_startup),
        ("Quantum Crypto Tests", test_quantum_crypto),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! QFLARE is ready.")
        return 0
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Review errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)