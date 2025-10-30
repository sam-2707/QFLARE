#!/usr/bin/env python3
"""
Auto-generated script to run all QFLARE experiments
Generated configurations: 13
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def run_experiment(config_name: str, config_file: str):
    """Run a single experiment"""
    print(f"\nRunning experiment: {config_name}")
    print(f"Config file: {config_file}")
    print("-" * 50)
    
    cmd = [
        sys.executable,
        "experiments/run_qflare_experiments.py",
        "--config", config_file
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"* {config_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"X {config_name} failed:")
        print(f"  Return code: {e.returncode}")
        print(f"  Error output: {e.stderr}")
        return False

def main():
    """Run all experiments"""
    config_dir = Path(__file__).parent / "configs"
    
    experiments = [
        ("baseline_honest", config_dir / "baseline_honest.json"),
        ("byzantine_10pct", config_dir / "byzantine_10pct.json"),
        ("byzantine_20pct", config_dir / "byzantine_20pct.json"),
        ("byzantine_30pct", config_dir / "byzantine_30pct.json"),
        ("dp_epsilon_0_1", config_dir / "dp_epsilon_0_1.json"),
        ("dp_epsilon_1_0", config_dir / "dp_epsilon_1_0.json"),
        ("dp_epsilon_10_0", config_dir / "dp_epsilon_10_0.json"),
        ("scale_50_clients", config_dir / "scale_50_clients.json"),
        ("scale_200_clients", config_dir / "scale_200_clients.json"),
        ("scale_500_clients", config_dir / "scale_500_clients.json"),
        ("iid_distribution", config_dir / "iid_distribution.json"),
        ("highly_non_iid", config_dir / "highly_non_iid.json"),
        ("quick_test", config_dir / "quick_test.json"),
    ]
    
    print(f"Starting {len(experiments)} QFLARE experiments...")
    
    passed = 0
    failed = 0
    
    for config_name, config_file in experiments:
        success = run_experiment(config_name, str(config_file))
        if success:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print("EXPERIMENT SUITE SUMMARY")
    print("=" * 60)
    print(f"Total experiments: {len(experiments)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed / len(experiments) * 100:.1f}%")
    
    if failed == 0:
        print("\nAll experiments completed successfully!")
        sys.exit(0)
    else:
        print(f"\n{failed} experiments failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
