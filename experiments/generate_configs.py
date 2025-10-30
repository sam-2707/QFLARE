#!/usr/bin/env python3
"""
QFLARE Experiment Configuration Generator
Creates standardized experiment configurations for reproducible research
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List

def create_baseline_config() -> Dict:
    """Create baseline experimental configuration"""
    return {
        "dataset": "MNIST",
        "data_path": "./data",
        "num_clients": 100,
        "samples_per_client": 600,
        "iid": False,
        "alpha": 0.5,
        
        "model_name": "CNN",
        "num_classes": 10,
        
        "local_epochs": 5,
        "global_rounds": 100,
        "batch_size": 32,
        "learning_rate": 0.01,
        
        "byzantine_ratio": 0.0,
        "differential_privacy": False,
        "dp_epsilon": 1.0,
        "dp_delta": 1e-5,
        
        "use_post_quantum": True,
        "kyber_variant": "Kyber1024",
        "dilithium_variant": "Dilithium5",
        
        "output_dir": "./experiments/results",
        "save_models": False
    }

def create_paper_configs() -> Dict[str, Dict]:
    """Create all configurations used in the paper"""
    configs = {}
    
    # Baseline configuration (honest conditions)
    configs["baseline_honest"] = create_baseline_config()
    
    # Byzantine fault tolerance experiments
    configs["byzantine_10pct"] = create_baseline_config()
    configs["byzantine_10pct"]["byzantine_ratio"] = 0.1
    configs["byzantine_10pct"]["output_dir"] = "./experiments/results/byzantine_10pct"
    
    configs["byzantine_20pct"] = create_baseline_config()
    configs["byzantine_20pct"]["byzantine_ratio"] = 0.2
    configs["byzantine_20pct"]["output_dir"] = "./experiments/results/byzantine_20pct"
    
    configs["byzantine_30pct"] = create_baseline_config()
    configs["byzantine_30pct"]["byzantine_ratio"] = 0.3
    configs["byzantine_30pct"]["output_dir"] = "./experiments/results/byzantine_30pct"
    
    # Differential privacy experiments
    configs["dp_epsilon_0_1"] = create_baseline_config()
    configs["dp_epsilon_0_1"]["differential_privacy"] = True
    configs["dp_epsilon_0_1"]["dp_epsilon"] = 0.1
    configs["dp_epsilon_0_1"]["output_dir"] = "./experiments/results/dp_epsilon_0_1"
    
    configs["dp_epsilon_1_0"] = create_baseline_config()
    configs["dp_epsilon_1_0"]["differential_privacy"] = True
    configs["dp_epsilon_1_0"]["dp_epsilon"] = 1.0
    configs["dp_epsilon_1_0"]["output_dir"] = "./experiments/results/dp_epsilon_1_0"
    
    configs["dp_epsilon_10_0"] = create_baseline_config()
    configs["dp_epsilon_10_0"]["differential_privacy"] = True
    configs["dp_epsilon_10_0"]["dp_epsilon"] = 10.0
    configs["dp_epsilon_10_0"]["output_dir"] = "./experiments/results/dp_epsilon_10_0"
    
    # Scalability experiments
    configs["scale_50_clients"] = create_baseline_config()
    configs["scale_50_clients"]["num_clients"] = 50
    configs["scale_50_clients"]["output_dir"] = "./experiments/results/scale_50_clients"
    
    configs["scale_200_clients"] = create_baseline_config()
    configs["scale_200_clients"]["num_clients"] = 200
    configs["scale_200_clients"]["output_dir"] = "./experiments/results/scale_200_clients"
    
    configs["scale_500_clients"] = create_baseline_config()
    configs["scale_500_clients"]["num_clients"] = 500
    configs["scale_500_clients"]["output_dir"] = "./experiments/results/scale_500_clients"
    
    # Data distribution experiments
    configs["iid_distribution"] = create_baseline_config()
    configs["iid_distribution"]["iid"] = True
    configs["iid_distribution"]["output_dir"] = "./experiments/results/iid_distribution"
    
    configs["highly_non_iid"] = create_baseline_config()
    configs["highly_non_iid"]["alpha"] = 0.1  # More non-IID
    configs["highly_non_iid"]["output_dir"] = "./experiments/results/highly_non_iid"
    
    # Quick test configuration
    configs["quick_test"] = create_baseline_config()
    configs["quick_test"]["num_clients"] = 20
    configs["quick_test"]["global_rounds"] = 10
    configs["quick_test"]["samples_per_client"] = 100
    configs["quick_test"]["output_dir"] = "./experiments/results/quick_test"
    
    return configs

def create_ablation_configs() -> Dict[str, Dict]:
    """Create configurations for ablation studies"""
    configs = {}
    base = create_baseline_config()
    
    # Ablation: No post-quantum crypto
    configs["ablation_no_pqc"] = base.copy()
    configs["ablation_no_pqc"]["use_post_quantum"] = False
    configs["ablation_no_pqc"]["output_dir"] = "./experiments/results/ablation_no_pqc"
    
    # Ablation: Different Kyber variants
    configs["ablation_kyber512"] = base.copy()
    configs["ablation_kyber512"]["kyber_variant"] = "Kyber512"
    configs["ablation_kyber512"]["output_dir"] = "./experiments/results/ablation_kyber512"
    
    configs["ablation_kyber768"] = base.copy()
    configs["ablation_kyber768"]["kyber_variant"] = "Kyber768"
    configs["ablation_kyber768"]["output_dir"] = "./experiments/results/ablation_kyber768"
    
    # Ablation: Different learning rates
    configs["ablation_lr_0_001"] = base.copy()
    configs["ablation_lr_0_001"]["learning_rate"] = 0.001
    configs["ablation_lr_0_001"]["output_dir"] = "./experiments/results/ablation_lr_0_001"
    
    configs["ablation_lr_0_1"] = base.copy()
    configs["ablation_lr_0_1"]["learning_rate"] = 0.1
    configs["ablation_lr_0_1"]["output_dir"] = "./experiments/results/ablation_lr_0_1"
    
    # Ablation: Different batch sizes
    configs["ablation_batch_16"] = base.copy()
    configs["ablation_batch_16"]["batch_size"] = 16
    configs["ablation_batch_16"]["output_dir"] = "./experiments/results/ablation_batch_16"
    
    configs["ablation_batch_64"] = base.copy()
    configs["ablation_batch_64"]["batch_size"] = 64
    configs["ablation_batch_64"]["output_dir"] = "./experiments/results/ablation_batch_64"
    
    return configs

def create_benchmark_configs() -> Dict[str, Dict]:
    """Create configurations for benchmarking against other methods"""
    configs = {}
    base = create_baseline_config()
    
    # Compare against FedAvg (no security)
    configs["benchmark_fedavg"] = base.copy()
    configs["benchmark_fedavg"]["use_post_quantum"] = False
    configs["benchmark_fedavg"]["differential_privacy"] = False
    configs["benchmark_fedavg"]["byzantine_ratio"] = 0.0
    configs["benchmark_fedavg"]["output_dir"] = "./experiments/results/benchmark_fedavg"
    
    # Compare against FedProx
    configs["benchmark_fedprox"] = base.copy()
    configs["benchmark_fedprox"]["use_post_quantum"] = False
    configs["benchmark_fedprox"]["differential_privacy"] = False
    configs["benchmark_fedprox"]["output_dir"] = "./experiments/results/benchmark_fedprox"
    # Note: Would need to implement FedProx aggregation in the experiment runner
    
    # Compare against classical crypto
    configs["benchmark_classical_crypto"] = base.copy()
    configs["benchmark_classical_crypto"]["use_post_quantum"] = False
    configs["benchmark_classical_crypto"]["output_dir"] = "./experiments/results/benchmark_classical_crypto"
    
    return configs

def save_configs(configs: Dict[str, Dict], output_dir: str):
    """Save all configurations to files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save individual config files
    for name, config in configs.items():
        config_file = output_path / f"{name}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved config: {config_file}")
    
    # Save master config file with all configurations
    master_config = {
        "description": "QFLARE Experiment Configurations",
        "total_configs": len(configs),
        "config_names": list(configs.keys()),
        "configs": configs
    }
    
    master_file = output_path / "all_configs.json"
    with open(master_file, 'w') as f:
        json.dump(master_config, f, indent=2)
    
    print(f"Saved master config: {master_file}")

def create_experiment_script(configs: Dict[str, Dict], output_dir: str):
    """Create a script to run all experiments"""
    script_content = f"""#!/usr/bin/env python3
\"\"\"
Auto-generated script to run all QFLARE experiments
Generated configurations: {len(configs)}
\"\"\"

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def run_experiment(config_name: str, config_file: str):
    \"\"\"Run a single experiment\"\"\"
    print(f"\\nRunning experiment: {{config_name}}")
    print(f"Config file: {{config_file}}")
    print("-" * 50)
    
    cmd = [
        sys.executable,
        "experiments/run_qflare_experiments.py",
        "--config", config_file
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"* {{config_name}} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"X {{config_name}} failed:")
        print(f"  Return code: {{e.returncode}}")
        print(f"  Error output: {{e.stderr}}")
        return False

def main():
    \"\"\"Run all experiments\"\"\"
    config_dir = Path(__file__).parent / "configs"
    
    experiments = [
"""

    for name in configs.keys():
        script_content += f'        ("{name}", config_dir / "{name}.json"),\n'

    script_content += f"""    ]
    
    print(f"Starting {{len(experiments)}} QFLARE experiments...")
    
    passed = 0
    failed = 0
    
    for config_name, config_file in experiments:
        success = run_experiment(config_name, str(config_file))
        if success:
            passed += 1
        else:
            failed += 1
    
    print("\\n" + "=" * 60)
    print("EXPERIMENT SUITE SUMMARY")
    print("=" * 60)
    print(f"Total experiments: {{len(experiments)}}")
    print(f"Passed: {{passed}}")
    print(f"Failed: {{failed}}")
    print(f"Success rate: {{passed / len(experiments) * 100:.1f}}%")
    
    if failed == 0:
        print("\\nAll experiments completed successfully!")
        sys.exit(0)
    else:
        print(f"\\n{{failed}} experiments failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""

    script_file = Path(output_dir) / "run_all_experiments.py"
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # Make script executable
    script_file.chmod(0o755)
    print(f"Created experiment runner script: {script_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate QFLARE experiment configurations")
    parser.add_argument("--output-dir", type=str, default="./experiments/configs",
                       help="Output directory for config files")
    parser.add_argument("--config-type", type=str, choices=["paper", "ablation", "benchmark", "all"],
                       default="all", help="Type of configurations to generate")
    
    args = parser.parse_args()
    
    print("Generating QFLARE experiment configurations...")
    
    all_configs = {}
    
    if args.config_type in ["paper", "all"]:
        paper_configs = create_paper_configs()
        all_configs.update(paper_configs)
        print(f"Generated {len(paper_configs)} paper configurations")
    
    if args.config_type in ["ablation", "all"]:
        ablation_configs = create_ablation_configs()
        all_configs.update(ablation_configs)
        print(f"Generated {len(ablation_configs)} ablation configurations")
    
    if args.config_type in ["benchmark", "all"]:
        benchmark_configs = create_benchmark_configs()
        all_configs.update(benchmark_configs)
        print(f"Generated {len(benchmark_configs)} benchmark configurations")
    
    # Save configurations
    save_configs(all_configs, args.output_dir)
    
    # Create experiment runner script
    create_experiment_script(all_configs, Path(args.output_dir).parent)
    
    print(f"\\nTotal configurations generated: {len(all_configs)}")
    print(f"Output directory: {args.output_dir}")
    print("\\nTo run experiments:")
    print(f"  python {Path(args.output_dir).parent}/run_all_experiments.py")

if __name__ == "__main__":
    main()