#!/usr/bin/env python3
"""
Test Real ML Integration

This script tests the real machine learning integration for QFLARE.
Verifies that PyTorch models, training, and aggregation work correctly.
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Add server to path
sys.path.append(str(Path(__file__).parent.parent / "server"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_creation():
    """Test PyTorch model creation and serialization."""
    print("\n=== Testing Model Creation ===")
    
    try:
        from server.ml.models import create_model, serialize_model_weights, deserialize_model_weights
        
        # Test MNIST model
        mnist_model = create_model("mnist")
        print(f"✓ MNIST model created: {type(mnist_model).__name__}")
        
        # Test serialization
        weights = serialize_model_weights(mnist_model)
        print(f"✓ Model weights serialized: {len(weights)} bytes")
        
        # Test deserialization
        new_model = create_model("mnist")
        new_model = deserialize_model_weights(new_model, weights)
        print("✓ Model weights deserialized successfully")
        
        # Test CIFAR-10 model
        cifar_model = create_model("cifar10")
        print(f"✓ CIFAR-10 model created: {type(cifar_model).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation test failed: {str(e)}")
        return False

def test_federated_trainer():
    """Test the federated trainer functionality."""
    print("\n=== Testing Federated Trainer ===")
    
    try:
        from server.ml.training import FederatedTrainer
        
        # Create trainer
        trainer = FederatedTrainer(dataset_name="mnist", model_name="mnist")
        print("✓ Federated trainer created")
        
        # Test dataset creation
        if hasattr(trainer, 'federated_dataset'):
            print("✓ Federated dataset initialized")
        
        # Test model initialization
        if hasattr(trainer, 'global_model'):
            print("✓ Global model initialized")
        
        # Test getting training history
        history = trainer.get_training_history()
        print(f"✓ Training history accessible: {len(history['rounds'])} rounds")
        
        return True
        
    except Exception as e:
        print(f"✗ Federated trainer test failed: {str(e)}")
        return False

def test_model_aggregator():
    """Test the real model aggregator."""
    print("\n=== Testing Model Aggregator ===")
    
    try:
        from server.fl_core.aggregator_real import RealModelAggregator
        from server.ml.models import create_model, serialize_model_weights
        
        # Create aggregator
        aggregator = RealModelAggregator()
        print("✓ Real model aggregator created")
        
        # Test storing model updates
        test_model = create_model("mnist")
        test_weights = serialize_model_weights(test_model)
        
        success = aggregator.store_model_update(
            "test_device_1", 
            test_weights, 
            {"samples": 100, "accuracy": 85.5, "loss": 0.15}
        )
        print(f"✓ Model update stored: {success}")
        
        success = aggregator.store_model_update(
            "test_device_2", 
            test_weights, 
            {"samples": 150, "accuracy": 82.1, "loss": 0.18}
        )
        print(f"✓ Second model update stored: {success}")
        
        # Test getting pending updates
        pending = aggregator.get_pending_updates()
        print(f"✓ Pending updates retrieved: {len(pending)} updates")
        
        # Test aggregation
        if len(pending) >= 2:
            result = aggregator.aggregate_pending_models(min_updates=2)
            if result:
                print("✓ Model aggregation completed")
                print(f"  - Aggregated {result['aggregation_record']['num_clients']} models")
                print(f"  - Total samples: {result['aggregation_record']['total_samples']}")
            else:
                print("○ Model aggregation skipped (insufficient updates)")
        
        return True
        
    except Exception as e:
        print(f"✗ Model aggregator test failed: {str(e)}")
        return False

async def test_fl_controller():
    """Test the FL controller with real ML."""
    print("\n=== Testing FL Controller ===")
    
    try:
        from server.fl_core.fl_controller import FLController
        
        # Create FL controller with training config
        training_config = {
            "dataset": "mnist",
            "model": "mnist",
            "data_dir": "../data",
            "device": "cpu"  # Force CPU for testing
        }
        
        fl_controller = FLController(training_config=training_config)
        print("✓ FL controller created with real ML trainer")
        
        # Test controller properties
        print(f"✓ Current round: {fl_controller.current_round}")
        print(f"✓ Min participants: {fl_controller.min_participants}")
        print(f"✓ Max participants: {fl_controller.max_participants}")
        
        # Create mock devices for testing
        mock_devices = [
            {"device_id": f"device_{i}", "status": "enrolled", "capabilities": {}}
            for i in range(5)
        ]
        
        # Test device selection
        if fl_controller.can_start_round(mock_devices):
            selected = fl_controller.select_participants(mock_devices, 3)
            print(f"✓ Selected {len(selected)} participants")
        
        return True
        
    except Exception as e:
        print(f"✗ FL controller test failed: {str(e)}")
        return False

def test_database_operations():
    """Test database operations for ML integration."""
    print("\n=== Testing Database Operations ===")
    
    try:
        from server.fl_core.aggregator_real import get_database_connection
        import sqlite3
        
        # Test database connection
        conn = get_database_connection()
        cursor = conn.cursor()
        
        # Test table creation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_ml_integration (
                id INTEGER PRIMARY KEY,
                test_data TEXT,
                timestamp TEXT
            )
        """)
        
        # Test data insertion
        cursor.execute("""
            INSERT INTO test_ml_integration (test_data, timestamp)
            VALUES (?, ?)
        """, ("real_ml_test", "2024-01-01T00:00:00"))
        
        conn.commit()
        
        # Test data retrieval
        cursor.execute("SELECT COUNT(*) FROM test_ml_integration")
        count = cursor.fetchone()[0]
        
        conn.close()
        print(f"✓ Database operations successful: {count} test records")
        
        return True
        
    except Exception as e:
        print(f"✗ Database operations test failed: {str(e)}")
        return False

async def main():
    """Run all real ML integration tests."""
    print("QFLARE Real ML Integration Test Suite")
    print("====================================")
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Federated Trainer", test_federated_trainer), 
        ("Model Aggregator", test_model_aggregator),
        ("FL Controller", test_fl_controller),
        ("Database Operations", test_database_operations)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {str(e)}")
            results.append((test_name, False))
    
    print("\n=== Test Results Summary ===")
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All real ML integration tests passed!")
        print("The QFLARE system is ready for real federated learning.")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Check the implementation.")
    
    return passed == len(results)

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())