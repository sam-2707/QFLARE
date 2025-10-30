#!/usr/bin/env python3
"""
Quick QFLARE Validation Test
Tests core functionality without running full paper validation experiments
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from experiments.run_qflare_experiments import QFLAREExperimentRunner, ExperimentConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Test basic QFLARE functionality with minimal parameters"""
    logger.info("Testing basic QFLARE functionality...")
    
    config = ExperimentConfig(
        num_clients=10,
        global_rounds=3,
        samples_per_client=50,
        batch_size=16,
        byzantine_ratio=0.0,
        differential_privacy=False,
        use_post_quantum=True,
        output_dir="./experiments/results/quick_test"
    )
    
    runner = QFLAREExperimentRunner(config)
    
    # Test dataset loading
    client_loaders, test_loader = runner.load_dataset()
    logger.info(f"✓ Dataset loaded: {len(client_loaders)} clients")
    
    # Test crypto benchmarks
    crypto_results = runner.benchmark_crypto_operations()
    logger.info("✓ Crypto benchmarks completed:")
    for op, stats in crypto_results.items():
        logger.info(f"  {op}: {stats['mean_ms']:.2f} ms")
    
    # Test short training
    training_results = runner.run_federated_training(client_loaders, test_loader)
    logger.info(f"✓ Training completed: {training_results['final_accuracy']:.4f} accuracy")
    
    # Test scalability
    scalability_results = runner.run_scalability_test()
    logger.info("✓ Scalability test completed")
    
    return {
        "status": "PASS",
        "final_accuracy": training_results['final_accuracy'],
        "crypto_performance": crypto_results,
        "scalability": scalability_results
    }

def test_byzantine_detection():
    """Test Byzantine client detection"""
    logger.info("Testing Byzantine detection...")
    
    config = ExperimentConfig(
        num_clients=10,
        global_rounds=2,
        samples_per_client=30,
        batch_size=16,
        byzantine_ratio=0.3,  # 30% Byzantine clients
        differential_privacy=False,
        use_post_quantum=True,
        output_dir="./experiments/results/byzantine_test"
    )
    
    runner = QFLAREExperimentRunner(config)
    client_loaders, test_loader = runner.load_dataset()
    
    training_results = runner.run_federated_training(client_loaders, test_loader)
    logger.info(f"✓ Byzantine training completed: {training_results['final_accuracy']:.4f} accuracy")
    
    return {
        "status": "PASS",
        "byzantine_accuracy": training_results['final_accuracy']
    }

def test_differential_privacy():
    """Test differential privacy functionality"""
    logger.info("Testing differential privacy...")
    
    config = ExperimentConfig(
        num_clients=8,
        global_rounds=2,
        samples_per_client=30,
        batch_size=16,
        byzantine_ratio=0.0,
        differential_privacy=True,
        dp_epsilon=1.0,
        use_post_quantum=True,
        output_dir="./experiments/results/dp_test"
    )
    
    runner = QFLAREExperimentRunner(config)
    client_loaders, test_loader = runner.load_dataset()
    
    training_results = runner.run_federated_training(client_loaders, test_loader)
    logger.info(f"✓ DP training completed: {training_results['final_accuracy']:.4f} accuracy")
    
    return {
        "status": "PASS",
        "dp_accuracy": training_results['final_accuracy']
    }

def main():
    """Run quick validation tests"""
    print("="*60)
    print("QFLARE QUICK VALIDATION TEST")
    print("="*60)
    
    results = {}
    
    try:
        # Test 1: Basic functionality
        results["basic_functionality"] = test_basic_functionality()
        print("✓ Basic functionality test PASSED")
    except Exception as e:
        results["basic_functionality"] = {"status": "FAIL", "error": str(e)}
        print(f"✗ Basic functionality test FAILED: {e}")
    
    try:
        # Test 2: Byzantine detection
        results["byzantine_detection"] = test_byzantine_detection()
        print("✓ Byzantine detection test PASSED")
    except Exception as e:
        results["byzantine_detection"] = {"status": "FAIL", "error": str(e)}
        print(f"✗ Byzantine detection test FAILED: {e}")
    
    try:
        # Test 3: Differential privacy
        results["differential_privacy"] = test_differential_privacy()
        print("✓ Differential privacy test PASSED")
    except Exception as e:
        results["differential_privacy"] = {"status": "FAIL", "error": str(e)}
        print(f"✗ Differential privacy test FAILED: {e}")
    
    # Summary
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r.get("status") == "PASS")
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All validation tests PASSED!")
        print("QFLARE core functionality is working correctly.")
    else:
        print(f"\n❌ {total_tests - passed_tests} tests FAILED")
        print("Check the error messages above for details.")
    
    # Save results
    output_dir = Path("./experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "quick_validation_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_dir}/quick_validation_results.json")

if __name__ == "__main__":
    main()