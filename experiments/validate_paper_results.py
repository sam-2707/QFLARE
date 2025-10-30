#!/usr/bin/env python3
"""
Validate QFLARE Paper Results
Runs experiments to validate key metrics from the enhanced paper:
- 96.8% accuracy under honest conditions
- 1,247 model updates/second throughput
- Byzantine fault tolerance with <2% accuracy degradation
- Post-quantum crypto overhead measurements
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from experiments.run_qflare_experiments import QFLAREExperimentRunner, ExperimentConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperValidationSuite:
    """Validates experimental results claimed in the QFLARE paper"""
    
    def __init__(self, output_dir: str = "./experiments/paper_validation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validation_results = {}
    
    def validate_honest_accuracy(self) -> Dict:
        """Validate 96.8% accuracy claim under honest conditions"""
        logger.info("Validating honest conditions accuracy (Target: 96.8%)")
        
        config = ExperimentConfig(
            num_clients=100,
            global_rounds=100,
            byzantine_ratio=0.0,  # Honest conditions
            differential_privacy=False,
            use_post_quantum=True,
            iid=False,  # Non-IID as in paper
            alpha=0.5,
            output_dir=str(self.output_dir / "honest_accuracy")
        )
        
        runner = QFLAREExperimentRunner(config)
        results = runner.run_full_experiment()
        
        final_accuracy = results["training_results"]["final_accuracy"]
        target_accuracy = 0.968
        
        validation = {
            "test_name": "Honest Accuracy Validation",
            "target_accuracy": target_accuracy,
            "achieved_accuracy": final_accuracy,
            "difference": final_accuracy - target_accuracy,
            "within_tolerance": abs(final_accuracy - target_accuracy) <= 0.02,
            "status": "PASS" if abs(final_accuracy - target_accuracy) <= 0.02 else "FAIL"
        }
        
        logger.info(f"Honest accuracy: {final_accuracy:.4f} (target: {target_accuracy:.4f}) - {validation['status']}")
        return validation
    
    def validate_byzantine_resilience(self) -> Dict:
        """Validate Byzantine fault tolerance with <2% accuracy degradation"""
        logger.info("Validating Byzantine resilience (Target: <2% degradation)")
        
        # First run honest baseline
        honest_config = ExperimentConfig(
            num_clients=100,
            global_rounds=50,  # Shorter for comparison
            byzantine_ratio=0.0,
            differential_privacy=False,
            use_post_quantum=True,
            output_dir=str(self.output_dir / "byzantine_honest")
        )
        
        honest_runner = QFLAREExperimentRunner(honest_config)
        honest_results = honest_runner.run_full_experiment()
        honest_accuracy = honest_results["training_results"]["final_accuracy"]
        
        # Run with Byzantine clients (20% as in paper)
        byzantine_config = ExperimentConfig(
            num_clients=100,
            global_rounds=50,
            byzantine_ratio=0.2,  # 20% Byzantine clients
            differential_privacy=False,
            use_post_quantum=True,
            output_dir=str(self.output_dir / "byzantine_attack")
        )
        
        byzantine_runner = QFLAREExperimentRunner(byzantine_config)
        byzantine_results = byzantine_runner.run_full_experiment()
        byzantine_accuracy = byzantine_results["training_results"]["final_accuracy"]
        
        accuracy_degradation = honest_accuracy - byzantine_accuracy
        target_degradation = 0.02  # 2% maximum degradation
        
        validation = {
            "test_name": "Byzantine Resilience Validation",
            "honest_accuracy": honest_accuracy,
            "byzantine_accuracy": byzantine_accuracy,
            "accuracy_degradation": accuracy_degradation,
            "target_max_degradation": target_degradation,
            "within_tolerance": accuracy_degradation <= target_degradation,
            "status": "PASS" if accuracy_degradation <= target_degradation else "FAIL"
        }
        
        logger.info(f"Byzantine resilience: {accuracy_degradation:.4f} degradation (target: <{target_degradation:.4f}) - {validation['status']}")
        return validation
    
    def validate_throughput_performance(self) -> Dict:
        """Validate 1,247 model updates/second throughput claim"""
        logger.info("Validating throughput performance (Target: 1,247 updates/sec)")
        
        config = ExperimentConfig(
            num_clients=200,  # Larger scale for throughput test
            global_rounds=1,  # Single round for pure throughput measurement
            use_post_quantum=True,
            output_dir=str(self.output_dir / "throughput")
        )
        
        runner = QFLAREExperimentRunner(config)
        
        # Run scalability test to measure throughput
        scalability_results = runner.run_scalability_test()
        
        # Get peak throughput
        if scalability_results:
            peak_throughput = max(result["throughput_clients_per_s"] for result in scalability_results.values())
        else:
            # Fallback: estimate from crypto benchmarks
            crypto_results = runner.benchmark_crypto_operations()
            avg_crypto_time = (
                crypto_results["keygen"]["mean_ms"] +
                crypto_results["encapsulate"]["mean_ms"] +
                crypto_results["sign"]["mean_ms"] +
                crypto_results["verify"]["mean_ms"]
            ) / 1000  # Convert to seconds
            peak_throughput = 1 / avg_crypto_time if avg_crypto_time > 0 else 0
        
        target_throughput = 1247
        
        validation = {
            "test_name": "Throughput Performance Validation",
            "target_throughput": target_throughput,
            "achieved_throughput": peak_throughput,
            "difference": peak_throughput - target_throughput,
            "percentage_of_target": (peak_throughput / target_throughput) * 100,
            "within_tolerance": peak_throughput >= target_throughput * 0.8,  # 80% of target
            "status": "PASS" if peak_throughput >= target_throughput * 0.8 else "FAIL"
        }
        
        logger.info(f"Throughput: {peak_throughput:.1f} updates/sec (target: {target_throughput}) - {validation['status']}")
        return validation
    
    def validate_crypto_overhead(self) -> Dict:
        """Validate post-quantum cryptographic overhead measurements"""
        logger.info("Validating post-quantum crypto overhead")
        
        config = ExperimentConfig(
            use_post_quantum=True,
            output_dir=str(self.output_dir / "crypto_overhead")
        )
        
        runner = QFLAREExperimentRunner(config)
        crypto_results = runner.benchmark_crypto_operations()
        
        # Expected ranges based on literature and paper claims
        expected_ranges = {
            "keygen": {"min": 1.0, "max": 5.0},    # 1-5 ms
            "encapsulate": {"min": 0.5, "max": 2.0},  # 0.5-2 ms
            "sign": {"min": 2.0, "max": 5.0},     # 2-5 ms
            "verify": {"min": 1.0, "max": 3.0}    # 1-3 ms
        }
        
        validations = {}
        all_pass = True
        
        for operation, stats in crypto_results.items():
            mean_time = stats["mean_ms"]
            expected = expected_ranges[operation]
            
            within_range = expected["min"] <= mean_time <= expected["max"]
            if not within_range:
                all_pass = False
            
            validations[operation] = {
                "operation": operation,
                "measured_time_ms": mean_time,
                "expected_range_ms": expected,
                "within_range": within_range,
                "status": "PASS" if within_range else "FAIL"
            }
        
        validation = {
            "test_name": "Crypto Overhead Validation",
            "operations": validations,
            "all_operations_pass": all_pass,
            "status": "PASS" if all_pass else "FAIL"
        }
        
        logger.info(f"Crypto overhead: {validation['status']}")
        for op, val in validations.items():
            logger.info(f"  {op}: {val['measured_time_ms']:.2f} ms - {val['status']}")
        
        return validation
    
    def validate_differential_privacy(self) -> Dict:
        """Validate differential privacy functionality"""
        logger.info("Validating differential privacy implementation")
        
        # Run without DP
        no_dp_config = ExperimentConfig(
            num_clients=50,
            global_rounds=20,
            differential_privacy=False,
            output_dir=str(self.output_dir / "no_dp")
        )
        
        no_dp_runner = QFLAREExperimentRunner(no_dp_config)
        no_dp_results = no_dp_runner.run_full_experiment()
        no_dp_accuracy = no_dp_results["training_results"]["final_accuracy"]
        
        # Run with DP
        dp_config = ExperimentConfig(
            num_clients=50,
            global_rounds=20,
            differential_privacy=True,
            dp_epsilon=1.0,
            dp_delta=1e-5,
            output_dir=str(self.output_dir / "with_dp")
        )
        
        dp_runner = QFLAREExperimentRunner(dp_config)
        dp_results = dp_runner.run_full_experiment()
        dp_accuracy = dp_results["training_results"]["final_accuracy"]
        
        accuracy_loss = no_dp_accuracy - dp_accuracy
        acceptable_loss = 0.05  # 5% maximum acceptable accuracy loss
        
        validation = {
            "test_name": "Differential Privacy Validation",
            "no_dp_accuracy": no_dp_accuracy,
            "dp_accuracy": dp_accuracy,
            "accuracy_loss": accuracy_loss,
            "acceptable_loss": acceptable_loss,
            "privacy_preserved": accuracy_loss <= acceptable_loss,
            "status": "PASS" if accuracy_loss <= acceptable_loss else "FAIL"
        }
        
        logger.info(f"Differential privacy: {accuracy_loss:.4f} accuracy loss (acceptable: <{acceptable_loss:.4f}) - {validation['status']}")
        return validation
    
    def run_full_validation(self) -> Dict:
        """Run complete paper validation suite"""
        logger.info("Starting QFLARE paper validation suite...")
        
        validations = {}
        
        try:
            # Test 1: Honest accuracy
            validations["honest_accuracy"] = self.validate_honest_accuracy()
        except Exception as e:
            logger.error(f"Honest accuracy validation failed: {e}")
            validations["honest_accuracy"] = {"status": "ERROR", "error": str(e)}
        
        try:
            # Test 2: Byzantine resilience
            validations["byzantine_resilience"] = self.validate_byzantine_resilience()
        except Exception as e:
            logger.error(f"Byzantine resilience validation failed: {e}")
            validations["byzantine_resilience"] = {"status": "ERROR", "error": str(e)}
        
        try:
            # Test 3: Throughput performance
            validations["throughput"] = self.validate_throughput_performance()
        except Exception as e:
            logger.error(f"Throughput validation failed: {e}")
            validations["throughput"] = {"status": "ERROR", "error": str(e)}
        
        try:
            # Test 4: Crypto overhead
            validations["crypto_overhead"] = self.validate_crypto_overhead()
        except Exception as e:
            logger.error(f"Crypto overhead validation failed: {e}")
            validations["crypto_overhead"] = {"status": "ERROR", "error": str(e)}
        
        try:
            # Test 5: Differential privacy
            validations["differential_privacy"] = self.validate_differential_privacy()
        except Exception as e:
            logger.error(f"Differential privacy validation failed: {e}")
            validations["differential_privacy"] = {"status": "ERROR", "error": str(e)}
        
        # Compile overall results
        total_tests = len(validations)
        passed_tests = sum(1 for v in validations.values() if v.get("status") == "PASS")
        failed_tests = sum(1 for v in validations.values() if v.get("status") == "FAIL")
        error_tests = sum(1 for v in validations.values() if v.get("status") == "ERROR")
        
        overall_result = {
            "validation_summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "overall_status": "PASS" if failed_tests == 0 and error_tests == 0 else "FAIL"
            },
            "individual_validations": validations,
            "timestamp": str(Path(__file__).stat().st_mtime)
        }
        
        # Save results
        output_file = self.output_dir / "paper_validation_results.json"
        with open(output_file, 'w') as f:
            json.dump(overall_result, f, indent=2, default=str)
        
        # Print summary
        self._print_validation_summary(overall_result)
        
        logger.info(f"Validation completed. Results saved to {output_file}")
        return overall_result
    
    def _print_validation_summary(self, results: Dict):
        """Print validation summary"""
        print("\n" + "="*70)
        print("QFLARE PAPER VALIDATION SUMMARY")
        print("="*70)
        
        summary = results["validation_summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Errors: {summary['errors']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Overall Status: {summary['overall_status']}")
        
        print("\nIndividual Test Results:")
        print("-" * 50)
        
        for test_name, validation in results["individual_validations"].items():
            status = validation.get("status", "UNKNOWN")
            print(f"{test_name.replace('_', ' ').title():<30} {status}")
            
            if status == "PASS":
                if "achieved_accuracy" in validation:
                    print(f"  → Accuracy: {validation['achieved_accuracy']:.4f}")
                elif "achieved_throughput" in validation:
                    print(f"  → Throughput: {validation['achieved_throughput']:.1f} updates/sec")
            elif status == "FAIL":
                if "difference" in validation:
                    print(f"  → Difference: {validation['difference']:.4f}")
            elif status == "ERROR":
                print(f"  → Error: {validation.get('error', 'Unknown error')}")
        
        print("="*70)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="QFLARE Paper Validation Suite")
    parser.add_argument("--output-dir", type=str, default="./experiments/paper_validation",
                       help="Output directory for validation results")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick validation (reduced parameters)")
    
    args = parser.parse_args()
    
    # Run validation
    validator = PaperValidationSuite(args.output_dir)
    
    if args.quick:
        logger.info("Running quick validation mode...")
        # Override some configs for speed
        # This would require modifying the validation methods
    
    results = validator.run_full_validation()
    
    # Exit with appropriate code
    if results["validation_summary"]["overall_status"] == "PASS":
        logger.info("All validations passed!")
        sys.exit(0)
    else:
        logger.error("Some validations failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()