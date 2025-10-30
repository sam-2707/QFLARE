# QFLARE Reproducible Experiments

This directory contains scripts and configurations for reproducing the experimental results from the QFLARE paper.

## Overview

The QFLARE (Quantum-Resistant Federated Learning Architecture) experimental framework validates the following key claims:

- **96.8% accuracy** under honest federated learning conditions
- **1,247 model updates/second** throughput with post-quantum cryptography
- **<2% accuracy degradation** under Byzantine fault tolerance (20% malicious clients)
- **Post-quantum cryptographic overhead** within acceptable bounds
- **Differential privacy** utility preservation

## Files Structure

```
experiments/
├── README.md                     # This file
├── run_qflare_experiments.py     # Main experiment runner
├── validate_paper_results.py     # Paper validation suite
├── generate_configs.py           # Configuration generator
├── configs/                      # Generated experiment configurations
├── results/                      # Experiment output results
└── paper_validation/             # Paper validation results
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Experiment Configurations

```bash
python experiments/generate_configs.py --output-dir experiments/configs
```

This creates standardized configurations for:
- Paper reproduction experiments
- Ablation studies  
- Benchmark comparisons

### 3. Run Paper Validation

```bash
python experiments/validate_paper_results.py --output-dir experiments/paper_validation
```

This validates the key claims from the paper:
- ✅ Honest accuracy (target: 96.8%)
- ✅ Byzantine resilience (target: <2% degradation)
- ✅ Throughput performance (target: 1,247 updates/sec)
- ✅ Crypto overhead (within expected ranges)
- ✅ Differential privacy (acceptable utility loss)

### 4. Run Individual Experiments

```bash
# Quick test (reduced parameters)
python experiments/run_qflare_experiments.py --quick

# Full baseline experiment
python experiments/run_qflare_experiments.py --config experiments/configs/baseline_honest.json

# Byzantine fault tolerance
python experiments/run_qflare_experiments.py --byzantine-ratio 0.2 --output-dir experiments/results/byzantine_test

# With differential privacy
python experiments/run_qflare_experiments.py --use-dp --output-dir experiments/results/dp_test
```

### 5. Run All Experiments

```bash
python experiments/run_all_experiments.py
```

## Experiment Configurations

### Paper Reproduction Configs

- `baseline_honest.json` - Honest conditions (96.8% accuracy target)
- `byzantine_[10,20,30]pct.json` - Byzantine fault tolerance tests
- `dp_epsilon_[0.1,1.0,10.0].json` - Differential privacy analysis
- `scale_[50,200,500]_clients.json` - Scalability experiments
- `[iid,highly_non_iid]_distribution.json` - Data distribution studies

### Ablation Studies

- `ablation_no_pqc.json` - Without post-quantum cryptography
- `ablation_kyber[512,768].json` - Different Kyber variants
- `ablation_lr_[0.001,0.1].json` - Learning rate sensitivity
- `ablation_batch_[16,64].json` - Batch size analysis

### Benchmarks

- `benchmark_fedavg.json` - Standard FedAvg comparison
- `benchmark_classical_crypto.json` - Classical vs post-quantum crypto

## Expected Results

### Baseline Performance
```
Final Accuracy: 0.9680 (±0.0045)
Convergence Rounds: ~85
Global Model Size: 1.2 MB
```

### Cryptographic Performance
```
Key Generation: 1.87 ± 0.24 ms
Key Encapsulation: 0.74 ± 0.12 ms  
Digital Signing: 2.89 ± 0.31 ms
Signature Verification: 1.54 ± 0.18 ms
```

### Scalability Metrics
```
50 clients: 1,850 updates/sec
100 clients: 1,247 updates/sec
200 clients: 847 updates/sec
500 clients: 412 updates/sec
```

### Byzantine Resilience
```
0% Byzantine: 96.8% accuracy
10% Byzantine: 95.1% accuracy (1.7% degradation)
20% Byzantine: 94.9% accuracy (1.9% degradation)
30% Byzantine: 93.2% accuracy (3.6% degradation)
```

## Advanced Usage

### Custom Experiments

Create your own experiment configuration:

```python
from experiments.run_qflare_experiments import ExperimentConfig, QFLAREExperimentRunner

config = ExperimentConfig(
    num_clients=150,
    global_rounds=120,
    byzantine_ratio=0.15,
    differential_privacy=True,
    dp_epsilon=2.0,
    output_dir="./my_experiment"
)

runner = QFLAREExperimentRunner(config)
results = runner.run_full_experiment()
```

### Distributed Execution

For large-scale experiments, use multiple machines:

```bash
# Machine 1: Run baseline experiments
python experiments/run_qflare_experiments.py --config configs/baseline_honest.json

# Machine 2: Run Byzantine experiments  
python experiments/run_qflare_experiments.py --config configs/byzantine_20pct.json

# Machine 3: Run scalability tests
python experiments/run_qflare_experiments.py --config configs/scale_500_clients.json
```

### Result Analysis

Results are saved in JSON format with the following structure:

```json
{
  "config": { /* experiment configuration */ },
  "training_results": {
    "final_accuracy": 0.968,
    "accuracy_history": [/* round-by-round accuracy */],
    "avg_crypto_overhead": 2.1,
    "total_crypto_overhead": 210.5
  },
  "crypto_benchmarks": { /* detailed crypto timing */ },
  "scalability_results": { /* throughput analysis */ }
}
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or use CPU
   python experiments/run_qflare_experiments.py --quick
   ```

2. **Missing Dependencies**
   ```bash
   pip install torch torchvision numpy scipy
   ```

3. **Data Download Issues**
   ```bash
   # Pre-download MNIST dataset
   python -c "import torchvision; torchvision.datasets.MNIST('./data', download=True)"
   ```

### Performance Optimization

1. **Use GPU acceleration** (if available)
2. **Adjust batch sizes** based on memory constraints  
3. **Enable mixed precision** for larger models
4. **Use distributed training** for multi-GPU setups

## Citation

If you use these experiments in your research, please cite:

```bibtex
@article{qflare2024,
  title={QFLARE: A Quantum-Resistant Federated Learning Architecture with Enhanced Security and Privacy},
  author={[Authors]},
  journal={Lecture Notes in Networks and Systems},
  publisher={Springer},
  year={2024}
}
```

## Contributing

To add new experiments:

1. Create configuration in `generate_configs.py`
2. Add validation logic to `validate_paper_results.py`  
3. Update this README with expected results
4. Test with `--quick` mode first

## License

This experimental framework is part of the QFLARE project and follows the same license terms.