"""
Test Privacy-Aware Federated Learning Integration
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from server.privacy.private_trainer import PrivateFederatedTrainer
from server.privacy.private_fl_controller import PrivateFLController

def test_private_trainer():
    """Test private federated trainer."""
    print("Testing Private Federated Trainer...")
    
    try:
        # Test trainer initialization
        trainer = PrivateFederatedTrainer(model_type="mnist", privacy_level="strong")
        print("✓ Private trainer initialized")
        
        # Test privacy status
        status = trainer.get_privacy_status()
        print(f"✓ Privacy status: Level={status['privacy_level']}, Budget valid={status['budget_valid']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Private trainer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_private_controller():
    """Test private FL controller."""
    print("Testing Private FL Controller...")
    
    try:
        # Test controller initialization
        controller = PrivateFLController(websocket_manager=None, privacy_level="moderate")
        print("✓ Private FL controller initialized")
        
        # Test dashboard data
        dashboard_data = await controller.get_privacy_dashboard_data()
        print(f"✓ Dashboard data: {dashboard_data['privacy_overview']['privacy_level']}")
        
        # Test privacy level change
        result = await controller.set_privacy_level("weak")
        print(f"✓ Privacy level changed: {result['success']}")
        
        # Test parameter validation
        validation = await controller.validate_privacy_parameters(epsilon=1.0, delta=1e-5)
        print(f"✓ Parameter validation: {validation['valid']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Private controller test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_integration_tests():
    """Run all integration tests."""
    print("="*50)
    print("RUNNING PRIVACY INTEGRATION TESTS")
    print("="*50)
    
    # Test private trainer
    trainer_success = test_private_trainer()
    
    # Test private controller
    controller_success = asyncio.run(test_private_controller())
    
    if trainer_success and controller_success:
        print("="*50)
        print("✅ ALL PRIVACY INTEGRATION TESTS PASSED!")
        print("="*50)
        return True
    else:
        print("="*50)
        print("❌ SOME PRIVACY INTEGRATION TESTS FAILED!")
        print("="*50)
        return False

if __name__ == "__main__":
    run_integration_tests()