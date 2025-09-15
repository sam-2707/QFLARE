#!/usr/bin/env python3
"""
Simple FL Test - Test core federated learning functionality only
"""

import sys
import os
import logging
from pathlib import Path
from unittest.mock import MagicMock

# Mock oqs early
sys.modules['oqs'] = MagicMock()

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "edge_node"))
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_federated_learning_workflow():
    """Test the core FL workflow: data loading -> training -> aggregation"""
    
    logger.info("🧪 Testing Federated Learning Workflow")
    
    try:
        # 1. Test data loading
        logger.info("📊 Step 1: Testing data loading...")
        from edge_node.data_loader import get_sample_data
        train_loader, test_loader = get_sample_data()
        logger.info(f"✅ Data loaded: {len(train_loader.dataset)} train samples")
        
        # 2. Test model training
        logger.info("🏋️ Step 2: Testing model training...")
        from edge_node.trainer import LocalTrainer
        
        trainer = LocalTrainer(
            learning_rate=0.01,
            local_epochs=1,  # Quick test
            batch_size=32
        )
        
        # Train model
        weights1, metadata1 = trainer.train_local_model(train_loader)
        logger.info(f"✅ Device 1 training completed: {metadata1['num_samples']} samples")
        
        # Simulate second device
        trainer2 = LocalTrainer(
            learning_rate=0.01,
            local_epochs=1,
            batch_size=32
        )
        weights2, metadata2 = trainer2.train_local_model(train_loader)
        logger.info(f"✅ Device 2 training completed: {metadata2['num_samples']} samples")
        
        # 3. Test model aggregation
        logger.info("🔄 Step 3: Testing model aggregation...")
        from models.model_utils import ModelSerializer, FederatedAggregator
        
        # Deserialize weights
        weights_dict1 = ModelSerializer.deserialize_weights(weights1)
        weights_dict2 = ModelSerializer.deserialize_weights(weights2)
        
        # Aggregate models
        aggregated_weights = FederatedAggregator.federated_averaging([weights_dict1, weights_dict2])
        logger.info(f"✅ Model aggregation completed: {len(aggregated_weights)} parameters")
        
        # 4. Test model evaluation
        logger.info("📈 Step 4: Testing model evaluation...")
        
        # Load aggregated weights back into trainer
        aggregated_bytes = ModelSerializer.serialize_weights(trainer.model)
        trainer.load_model_weights(aggregated_bytes)
        
        # Evaluate
        eval_metrics = trainer.evaluate_model(test_loader)
        logger.info(f"✅ Model evaluation: {eval_metrics['test_accuracy']:.2f}% accuracy")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ FL workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_workflow_simulation():
    """Simulate multiple devices and FL rounds"""
    
    logger.info("🌐 Simulating Multi-Device FL Training")
    
    try:
        from edge_node.data_loader import get_sample_data, FederatedDataLoader
        from edge_node.trainer import LocalTrainer
        from models.model_utils import ModelSerializer, FederatedAggregator
        
        # Initialize global model
        global_trainer = LocalTrainer()
        global_weights = global_trainer.get_model_weights()
        
        num_devices = 3
        num_rounds = 2
        
        for round_num in range(num_rounds):
            logger.info(f"🏁 FL Round {round_num + 1}/{num_rounds}")
            
            device_updates = []
            device_metadata = []
            
            # Simulate training on multiple devices
            for device_id in range(num_devices):
                logger.info(f"📱 Training on device {device_id + 1}")
                
                # Create device-specific data
                train_loader, test_loader = get_sample_data()
                
                # Create trainer and load global weights
                trainer = LocalTrainer(local_epochs=1, batch_size=16)
                trainer.load_model_weights(global_weights)
                
                # Train locally
                weights, metadata = trainer.train_local_model(train_loader)
                
                device_updates.append(ModelSerializer.deserialize_weights(weights))
                device_metadata.append(metadata)
                
                logger.info(f"   Device {device_id + 1}: {metadata['num_samples']} samples, "
                          f"loss: {metadata['avg_loss']:.4f}")
            
            # Aggregate updates
            logger.info("🔄 Aggregating models...")
            global_weights_dict = FederatedAggregator.federated_averaging(device_updates)
            
            # Serialize back to bytes
            temp_trainer = LocalTrainer()
            temp_trainer.model.load_state_dict(global_weights_dict)
            global_weights = temp_trainer.get_model_weights()
            
            # Evaluate global model
            temp_trainer.load_model_weights(global_weights)
            test_loader = get_sample_data()[1]
            eval_metrics = temp_trainer.evaluate_model(test_loader)
            
            logger.info(f"🎯 Round {round_num + 1} completed - Global accuracy: "
                       f"{eval_metrics['test_accuracy']:.2f}%")
        
        logger.info("✅ Multi-device FL simulation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ FL simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run FL tests"""
    
    logger.info("🚀 QFLARE Federated Learning Core Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Core FL Workflow", test_federated_learning_workflow),
        ("Multi-Device Simulation", test_workflow_simulation)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        logger.info(f"\n📋 {test_name}")
        if test_func():
            passed += 1
            logger.info(f"✅ {test_name} PASSED\n")
        else:
            logger.error(f"❌ {test_name} FAILED\n")
    
    logger.info("=" * 60)
    logger.info(f"🎯 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🎉 All core FL functionality is working!")
        return 0
    else:
        logger.error("⚠️ Some tests failed - review errors above")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)