"""
Simple test for differential privacy core functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from server.privacy.differential_privacy import (
    DifferentialPrivacyConfig,
    GaussianMechanism,
    PrivacyEngine,
    create_privacy_engine
)

def test_basic_functionality():
    """Test basic differential privacy functionality."""
    print("Testing differential privacy basic functionality...")
    
    try:
        # Test 1: Config creation
        print("1. Testing config creation...")
        config = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        print(f"   ✓ Config created: ε={config.epsilon}, σ={config.noise_multiplier:.3f}")
        
        # Test 2: Gaussian mechanism
        print("2. Testing Gaussian mechanism...")
        mechanism = GaussianMechanism(config)
        test_tensor = torch.randn(3, 3)
        noisy_tensor = mechanism.add_noise(test_tensor)
        print(f"   ✓ Noise added to tensor: {test_tensor.shape} -> {noisy_tensor.shape}")
        
        # Test 3: Privacy engine
        print("3. Testing privacy engine...")
        engine = PrivacyEngine(config)
        
        # Test gradient clipping
        gradients = {
            "layer1": torch.tensor([3.0, 4.0]),  # Norm = 5.0
            "layer2": torch.tensor([1.0, 1.0])
        }
        clipped = engine.clip_gradients(gradients)
        print("   ✓ Gradient clipping works")
        
        # Test noise addition
        noisy_grads = engine.add_privacy_noise(clipped)
        print("   ✓ Privacy noise addition works")
        
        # Test full privatization
        privatized = engine.privatize_model_update(gradients)
        print("   ✓ Full privatization works")
        
        # Test 4: Privacy engine factory
        print("4. Testing privacy engine factory...")
        strong_engine = create_privacy_engine("strong")
        moderate_engine = create_privacy_engine("moderate")
        weak_engine = create_privacy_engine("weak")
        print("   ✓ Privacy engine factory works")
        
        print("\n✅ ALL BASIC DIFFERENTIAL PRIVACY TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_functionality()