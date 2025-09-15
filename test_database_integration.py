"""
Database Integration Test for QFLARE

This script tests the new persistent database functionality including:
- Device registration and management
- Model updates and aggregation
- Audit logging
- Database connection health
"""

import sys
import os
import time
import logging
from datetime import datetime

# Add server directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from server.database import (
    initialize_database, get_database, cleanup_database,
    DeviceService, ModelService, AuditService, TrainingService
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_database_initialization():
    """Test database initialization and connection."""
    print("🔍 Testing Database Initialization...")
    
    try:
        # Clean up any existing test database
        import os
        test_db_path = "test_qflare.db"
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
            print("🧹 Cleaned up existing test database")
        
        # Initialize with SQLite for testing
        config = {
            "database_type": "sqlite",
            "sqlite_path": test_db_path
        }
        
        db_manager = initialize_database(config)
        print("✅ Database initialized successfully")
        
        # Test health check
        if db_manager.health_check():
            print("✅ Database health check passed")
        else:
            print("❌ Database health check failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False


def test_device_management():
    """Test device registration and management."""
    print("\n🔍 Testing Device Management...")
    
    try:
        # Test device registration
        device_id = "test_device_001"
        device_info = {
            "device_type": "edge_node",
            "hardware_info": {"cpu": "ARM", "memory": "4GB"},
            "network_info": {"ip": "192.168.1.100"},
            "capabilities": {"training": True, "inference": True},
            "local_epochs": 5,
            "batch_size": 64,
            "learning_rate": 0.001
        }
        
        success = DeviceService.register_device(device_id, device_info)
        if success:
            print("✅ Device registration successful")
        else:
            print("❌ Device registration failed")
            return False
        
        # Test device retrieval
        retrieved_device = DeviceService.get_device(device_id)
        if retrieved_device and retrieved_device["device_id"] == device_id:
            print("✅ Device retrieval successful")
        else:
            print("❌ Device retrieval failed")
            return False
        
        # Test status update
        success = DeviceService.update_device_status(device_id, "training")
        if success:
            print("✅ Device status update successful")
        else:
            print("❌ Device status update failed")
            return False
        
        # Test listing active devices
        active_devices = DeviceService.list_active_devices()
        if len(active_devices) > 0:
            print(f"✅ Found {len(active_devices)} active devices")
        else:
            print("❌ No active devices found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Device management test failed: {e}")
        return False


def test_model_management():
    """Test model updates and aggregation."""
    print("\n🔍 Testing Model Management...")
    
    try:
        # Create test model weights
        test_weights = b"fake_model_weights_data_for_testing"
        
        # Test model update storage
        metadata = {
            "local_loss": 0.25,
            "local_accuracy": 0.85,
            "local_epochs": 5,
            "samples_count": 1000,
            "training_time": 120.5
        }
        
        device_ids = ["test_device_001", "test_device_002"]
        
        # Register second device for aggregation test
        DeviceService.register_device(device_ids[1], {
            "device_type": "edge_node",
            "capabilities": {"training": True}
        })
        
        # Store model updates from both devices
        for i, device_id in enumerate(device_ids):
            success = ModelService.store_model_update(
                device_id=device_id,
                model_weights=test_weights + str(i).encode(),
                signature=b"test_signature",
                metadata={**metadata, "local_accuracy": 0.8 + i * 0.05}
            )
            
            if success:
                print(f"✅ Model update stored for {device_id}")
            else:
                print(f"❌ Model update storage failed for {device_id}")
                return False
        
        # Test pending updates retrieval
        pending_updates = ModelService.get_pending_updates()
        if len(pending_updates) >= 2:
            print(f"✅ Found {len(pending_updates)} pending model updates")
        else:
            print("❌ Insufficient pending updates for aggregation")
            return False
        
        # Test global model storage
        aggregated_weights = b"aggregated_model_weights_data"
        model_metadata = {
            "model_type": "CNN",
            "aggregation_method": "fedavg",
            "accuracy": 0.875,
            "loss": 0.15
        }
        
        success = ModelService.store_global_model(
            round_number=1,
            model_weights=aggregated_weights,
            model_metadata=model_metadata,
            participating_devices=device_ids
        )
        
        if success:
            print("✅ Global model storage successful")
        else:
            print("❌ Global model storage failed")
            return False
        
        # Test latest global model retrieval
        latest_model = ModelService.get_latest_global_model()
        if latest_model and latest_model["round_number"] == 1:
            print("✅ Latest global model retrieval successful")
        else:
            print("❌ Latest global model retrieval failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Model management test failed: {e}")
        return False


def test_training_sessions():
    """Test training session management."""
    print("\n🔍 Testing Training Session Management...")
    
    try:
        # Create training session
        session_id = "training_session_001"
        device_id = "test_device_001"
        config = {
            "dataset_name": "MNIST",
            "model_type": "CNN",
            "hyperparameters": {"lr": 0.001, "epochs": 10},
            "total_rounds": 20
        }
        
        success = TrainingService.create_training_session(session_id, device_id, config)
        if success:
            print("✅ Training session creation successful")
        else:
            print("❌ Training session creation failed")
            return False
        
        # Update training progress
        metrics = {"loss": 0.25, "accuracy": 0.85}
        success = TrainingService.update_training_progress(session_id, 5, metrics)
        if success:
            print("✅ Training progress update successful")
        else:
            print("❌ Training progress update failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Training session test failed: {e}")
        return False


def test_audit_logging():
    """Test audit logging functionality."""
    print("\n🔍 Testing Audit Logging...")
    
    try:
        # Get recent audit events
        events = AuditService.get_recent_events(hours=1, limit=50)
        
        if len(events) > 0:
            print(f"✅ Found {len(events)} audit events")
            
            # Display some event details
            for event in events[:3]:  # Show first 3 events
                print(f"   - {event['event_type']}: {event['event_description']}")
        else:
            print("⚠️  No audit events found (this may be normal for a fresh database)")
        
        return True
        
    except Exception as e:
        print(f"❌ Audit logging test failed: {e}")
        return False


def test_database_performance():
    """Test database performance with multiple operations."""
    print("\n🔍 Testing Database Performance...")
    
    try:
        start_time = time.time()
        
        # Perform multiple operations
        operations = 0
        
        # Register multiple devices
        for i in range(10):
            device_id = f"perf_test_device_{i:03d}"
            DeviceService.register_device(device_id, {"device_type": "test"})
            operations += 1
        
        # Store multiple model updates
        for i in range(20):
            device_id = f"perf_test_device_{i % 10:03d}"
            ModelService.store_model_update(
                device_id=device_id,
                model_weights=b"test_weights_" + str(i).encode(),
                signature=b"test_sig",
                metadata={"local_loss": 0.1 + i * 0.01}
            )
            operations += 1
        
        end_time = time.time()
        duration = end_time - start_time
        ops_per_second = operations / duration if duration > 0 else 0
        
        print(f"✅ Performance test completed:")
        print(f"   - {operations} operations in {duration:.2f} seconds")
        print(f"   - {ops_per_second:.1f} operations per second")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def run_all_tests():
    """Run all database integration tests."""
    print("🚀 Starting QFLARE Database Integration Tests")
    print("=" * 50)
    
    test_results = []
    
    # Run tests
    test_results.append(("Database Initialization", test_database_initialization()))
    test_results.append(("Device Management", test_device_management()))
    test_results.append(("Model Management", test_model_management()))
    test_results.append(("Training Sessions", test_training_sessions()))
    test_results.append(("Audit Logging", test_audit_logging()))
    test_results.append(("Database Performance", test_database_performance()))
    
    # Print summary
    print("\n" + "=" * 50)
    print("🎯 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} | {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All database integration tests passed!")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
    
    # Cleanup
    try:
        cleanup_database()
        print("🧹 Database connections cleaned up")
    except Exception as e:
        print(f"⚠️  Error during cleanup: {e}")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)