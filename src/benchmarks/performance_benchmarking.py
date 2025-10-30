"""
QFLARE Performance Benchmarking System
Comprehensive benchmarking suite for measuring and validating system performance
"""

import time
import psutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
import logging
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import threading
from collections import defaultdict
import statistics
import os
import pickle

# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None
    pd = None

# Import QFLARE components
try:
    from ..crypto.post_quantum_crypto import QFLARECrypto
    from ..privacy.differential_privacy_engine import DifferentialPrivacyEngine, PrivacyParameters
    from ..security.byzantine_resilience import ByzantineResilienceSystem, AggregationMethod
    from ..federated.fl_training_engine import FederatedServer, FederatedClient, TrainingConfig
except ImportError:
    # Handle relative imports when running as main module
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from crypto.post_quantum_crypto import QFLARECrypto
    from privacy.differential_privacy_engine import DifferentialPrivacyEngine, PrivacyParameters
    from security.byzantine_resilience import ByzantineResilienceSystem, AggregationMethod
    from federated.fl_training_engine import FederatedServer, FederatedClient, TrainingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple neural network for testing
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Synthetic dataset for testing
class SyntheticMNIST(Dataset):
    def __init__(self, num_samples=1000):
        self.data = torch.randn(num_samples, 28, 28)
        self.targets = torch.randint(0, 10, (num_samples,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarks"""
    # Test parameters
    num_clients: int = 100
    num_rounds: int = 50
    data_size_per_client: int = 1000
    model_size: str = "small"  # small, medium, large
    
    # Performance test scenarios
    test_crypto_performance: bool = True
    test_privacy_overhead: bool = True
    test_byzantine_resilience: bool = True
    test_scalability: bool = True
    test_communication_overhead: bool = True
    test_memory_usage: bool = True
    
    # Security configurations to test
    security_configs: List[Dict] = field(default_factory=lambda: [
        {"pqc": False, "dp": False, "byzantine": False},  # Baseline
        {"pqc": True, "dp": False, "byzantine": False},   # PQC only
        {"pqc": False, "dp": True, "byzantine": False},   # DP only
        {"pqc": False, "dp": False, "byzantine": True},   # Byzantine only
        {"pqc": True, "dp": True, "byzantine": True},     # All features
    ])
    
    # Scalability test parameters
    client_scales: List[int] = field(default_factory=lambda: [10, 25, 50, 100, 250, 500])
    model_sizes: List[str] = field(default_factory=lambda: ["small", "medium", "large"])
    
    # Output configuration
    save_results: bool = True
    generate_plots: bool = True
    results_dir: str = "benchmark_results"

@dataclass
class PerformanceMetrics:
    """Performance measurement results"""
    # Timing metrics
    training_time: float = 0.0
    communication_time: float = 0.0
    crypto_time: float = 0.0
    aggregation_time: float = 0.0
    
    # Resource usage metrics
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_received: int = 0
    
    # ML metrics
    final_accuracy: float = 0.0
    convergence_round: int = 0
    loss_reduction: float = 0.0
    
    # Security metrics
    privacy_epsilon_used: float = 0.0
    byzantine_clients_detected: int = 0
    crypto_operations_per_second: float = 0.0
    
    # Scalability metrics
    throughput_ops_per_sec: float = 0.0
    latency_percentiles: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {
            'training_time': self.training_time,
            'communication_time': self.communication_time,
            'crypto_time': self.crypto_time,
            'aggregation_time': self.aggregation_time,
            'peak_memory_mb': self.peak_memory_mb,
            'avg_cpu_percent': self.avg_cpu_percent,
            'network_bytes_sent': self.network_bytes_sent,
            'network_bytes_received': self.network_bytes_received,
            'final_accuracy': self.final_accuracy,
            'convergence_round': self.convergence_round,
            'loss_reduction': self.loss_reduction,
            'privacy_epsilon_used': self.privacy_epsilon_used,
            'byzantine_clients_detected': self.byzantine_clients_detected,
            'crypto_operations_per_second': self.crypto_operations_per_second,
            'throughput_ops_per_sec': self.throughput_ops_per_sec,
            'latency_percentiles': self.latency_percentiles
        }

class SystemMonitor:
    """Real-time system performance monitoring"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = defaultdict(list)
        self.start_time = 0
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start system monitoring in background thread"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.debug("System monitoring started")
    
    def stop_monitoring(self) -> Dict:
        """Stop monitoring and return collected metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        # Calculate summary statistics
        summary = {}
        for metric, values in self.metrics.items():
            if values:
                summary[f'{metric}_avg'] = statistics.mean(values)
                summary[f'{metric}_max'] = max(values)
                summary[f'{metric}_min'] = min(values)
                if len(values) > 1:
                    summary[f'{metric}_std'] = statistics.stdev(values)
        
        logger.debug("System monitoring stopped")
        return summary
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                self.metrics['cpu_percent'].append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics['memory_percent'].append(memory.percent)
                self.metrics['memory_mb'].append(memory.used / 1024 / 1024)
                
                # Network I/O
                net_io = psutil.net_io_counters()
                self.metrics['bytes_sent'].append(net_io.bytes_sent)
                self.metrics['bytes_recv'].append(net_io.bytes_recv)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.metrics['disk_read_mb'].append(disk_io.read_bytes / 1024 / 1024)
                    self.metrics['disk_write_mb'].append(disk_io.write_bytes / 1024 / 1024)
                
                time.sleep(0.1)  # Monitor every 100ms
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                break

class CryptographicBenchmark:
    """Benchmarks for post-quantum cryptography operations"""
    
    def __init__(self):
        self.crypto = QFLARECrypto()
    
    def benchmark_key_generation(self, num_iterations: int = 100) -> Dict:
        """Benchmark key generation performance"""
        results = {}
        
        # Kyber key generation
        kyber_times = []
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            keypair = self.crypto.generate_kyber_keypair(1024)
            end_time = time.perf_counter()
            kyber_times.append(end_time - start_time)
        
        results['kyber_keygen'] = {
            'avg_time': statistics.mean(kyber_times),
            'min_time': min(kyber_times),
            'max_time': max(kyber_times),
            'ops_per_second': num_iterations / sum(kyber_times)
        }
        
        # Dilithium key generation
        dilithium_times = []
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            keypair = self.crypto.generate_dilithium_keypair(2)
            end_time = time.perf_counter()
            dilithium_times.append(end_time - start_time)
        
        results['dilithium_keygen'] = {
            'avg_time': statistics.mean(dilithium_times),
            'min_time': min(dilithium_times),
            'max_time': max(dilithium_times),
            'ops_per_second': num_iterations / sum(dilithium_times)
        }
        
        return results
    
    def benchmark_encryption_decryption(self, num_iterations: int = 100, data_sizes: List[int] = None) -> Dict:
        """Benchmark encryption/decryption performance"""
        if data_sizes is None:
            data_sizes = [1024, 4096, 16384, 65536]  # 1KB, 4KB, 16KB, 64KB
        
        results = {}
        
        # Generate keypair once
        keypair = self.crypto.generate_kyber_keypair(1024)
        
        for data_size in data_sizes:
            data = b'x' * data_size
            
            # Encryption timing
            encrypt_times = []
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                encrypted = self.crypto.hybrid_encrypt(data, keypair.public_key, 1024)
                end_time = time.perf_counter()
                encrypt_times.append(end_time - start_time)
            
            # Decryption timing (use last encrypted data)
            decrypt_times = []
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                decrypted = self.crypto.hybrid_decrypt(encrypted, keypair.private_key)
                end_time = time.perf_counter()
                decrypt_times.append(end_time - start_time)
            
            results[f'encrypt_{data_size}b'] = {
                'avg_time': statistics.mean(encrypt_times),
                'throughput_mbps': (data_size * num_iterations / sum(encrypt_times)) / (1024 * 1024)
            }
            
            results[f'decrypt_{data_size}b'] = {
                'avg_time': statistics.mean(decrypt_times),
                'throughput_mbps': (data_size * num_iterations / sum(decrypt_times)) / (1024 * 1024)
            }
        
        return results
    
    def benchmark_signatures(self, num_iterations: int = 100) -> Dict:
        """Benchmark digital signature performance"""
        results = {}
        
        # Generate keypair
        keypair = self.crypto.generate_dilithium_keypair(2)
        message = b"QFLARE federated learning gradient update"
        
        # Signature generation
        sign_times = []
        signatures = []
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            signature = self.crypto.dilithium_sign(message, keypair.private_key)
            end_time = time.perf_counter()
            sign_times.append(end_time - start_time)
            signatures.append(signature)
        
        # Signature verification
        verify_times = []
        for signature in signatures:
            start_time = time.perf_counter()
            valid = self.crypto.dilithium_verify(message, signature, keypair.public_key)
            end_time = time.perf_counter()
            verify_times.append(end_time - start_time)
        
        results['signature_generation'] = {
            'avg_time': statistics.mean(sign_times),
            'ops_per_second': num_iterations / sum(sign_times)
        }
        
        results['signature_verification'] = {
            'avg_time': statistics.mean(verify_times),
            'ops_per_second': num_iterations / sum(verify_times)
        }
        
        return results

class PrivacyBenchmark:
    """Benchmarks for differential privacy operations"""
    
    def __init__(self):
        privacy_params = PrivacyParameters(epsilon=0.1, delta=1e-6)
        self.dp_engine = DifferentialPrivacyEngine(privacy_params)
    
    def benchmark_noise_addition(self, num_iterations: int = 1000) -> Dict:
        """Benchmark noise addition performance"""
        results = {}
        
        # Test different gradient sizes
        gradient_sizes = [1000, 10000, 100000, 1000000]  # 1K, 10K, 100K, 1M parameters
        
        for size in gradient_sizes:
            # Gaussian noise
            gaussian_times = []
            gradient = torch.randn(size)
            
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                noisy_grad = self.dp_engine.add_gaussian_noise(gradient)
                end_time = time.perf_counter()
                gaussian_times.append(end_time - start_time)
            
            # Laplace noise
            laplace_times = []
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                noisy_grad = self.dp_engine.add_laplace_noise(gradient)
                end_time = time.perf_counter()
                laplace_times.append(end_time - start_time)
            
            results[f'gaussian_noise_{size}'] = {
                'avg_time': statistics.mean(gaussian_times),
                'params_per_second': size * num_iterations / sum(gaussian_times)
            }
            
            results[f'laplace_noise_{size}'] = {
                'avg_time': statistics.mean(laplace_times),
                'params_per_second': size * num_iterations / sum(laplace_times)
            }
        
        return results
    
    def benchmark_gradient_clipping(self, num_iterations: int = 1000) -> Dict:
        """Benchmark gradient clipping performance"""
        results = {}
        
        gradient_sizes = [1000, 10000, 100000]
        
        for size in gradient_sizes:
            clip_times = []
            
            for _ in range(num_iterations):
                gradients = [torch.randn(size) * 10 for _ in range(5)]  # 5 gradient tensors
                
                start_time = time.perf_counter()
                clipped = [self.dp_engine.clip_gradients(grad) for grad in gradients]
                end_time = time.perf_counter()
                
                clip_times.append(end_time - start_time)
            
            results[f'gradient_clipping_{size}'] = {
                'avg_time': statistics.mean(clip_times),
                'params_per_second': size * 5 * num_iterations / sum(clip_times)
            }
        
        return results

class ByzantineBenchmark:
    """Benchmarks for Byzantine resilience operations"""
    
    def __init__(self):
        self.byzantine_system = ByzantineResilienceSystem(
            aggregation_method=AggregationMethod.KRUM,
            byzantine_ratio=0.33
        )
    
    def benchmark_aggregation_methods(self, num_clients_list: List[int] = None) -> Dict:
        """Benchmark different aggregation methods"""
        if num_clients_list is None:
            num_clients_list = [10, 25, 50, 100]
        
        results = {}
        methods = [AggregationMethod.FEDAVG, AggregationMethod.KRUM, 
                  AggregationMethod.TRIMMED_MEAN, AggregationMethod.MEDIAN]
        
        gradient_size = 10000  # 10K parameters
        
        for num_clients in num_clients_list:
            for method in methods:
                aggregator = ByzantineResilienceSystem(method, 0.33).aggregator
                
                # Generate test gradients
                gradients = [torch.randn(gradient_size) for _ in range(num_clients)]
                client_ids = [f"client_{i}" for i in range(num_clients)]
                
                # Add some Byzantine gradients
                num_byzantine = int(num_clients * 0.3)
                for i in range(num_byzantine):
                    gradients[i] = -gradients[i] * 3  # Sign-flip attack
                
                # Benchmark aggregation
                num_iterations = 50
                agg_times = []
                
                for _ in range(num_iterations):
                    start_time = time.perf_counter()
                    agg_grad, info = aggregator.aggregate(gradients, client_ids)
                    end_time = time.perf_counter()
                    agg_times.append(end_time - start_time)
                
                results[f'{method.value}_{num_clients}_clients'] = {
                    'avg_time': statistics.mean(agg_times),
                    'clients_per_second': num_clients * num_iterations / sum(agg_times)
                }
        
        return results
    
    def benchmark_detection_accuracy(self, num_iterations: int = 100) -> Dict:
        """Benchmark Byzantine detection accuracy"""
        results = {}
        
        num_clients = 20
        gradient_size = 5000
        
        detection_rates = {
            'statistical_outliers': [],
            'sign_flippers': [],
            'combined_detection': []
        }
        
        for _ in range(num_iterations):
            # Generate honest gradients
            honest_gradients = [torch.randn(gradient_size) * 0.1 for _ in range(15)]
            
            # Add Byzantine gradients
            byzantine_gradients = [
                -torch.randn(gradient_size) * 0.5,  # Sign-flip
                torch.randn(gradient_size) * 3.0,   # High variance
                torch.randn(gradient_size) * 3.0,   # High variance
                -torch.randn(gradient_size) * 2.0,  # Sign-flip
                torch.randn(gradient_size) * 5.0    # Very high variance
            ]
            
            all_gradients = honest_gradients + byzantine_gradients
            client_ids = [f"client_{i}" for i in range(20)]
            
            # Test detection
            detector = self.byzantine_system.detector
            
            outliers = detector.detect_statistical_outliers(all_gradients, client_ids)
            sign_flips = detector.detect_sign_flip_attack(all_gradients, client_ids)
            
            # Calculate detection rates
            outlier_detection_rate = sum(outliers[15:]) / 5  # Byzantine clients are indices 15-19
            sign_flip_detection_rate = sum(sign_flips[15:]) / 5
            combined_rate = sum(1 for i in range(15, 20) if outliers[i] or sign_flips[i]) / 5
            
            detection_rates['statistical_outliers'].append(outlier_detection_rate)
            detection_rates['sign_flippers'].append(sign_flip_detection_rate)
            detection_rates['combined_detection'].append(combined_rate)
        
        for detection_type, rates in detection_rates.items():
            results[detection_type] = {
                'avg_detection_rate': statistics.mean(rates),
                'std_detection_rate': statistics.stdev(rates) if len(rates) > 1 else 0.0,
                'min_detection_rate': min(rates),
                'max_detection_rate': max(rates)
            }
        
        return results

class FederatedLearningBenchmark:
    """End-to-end federated learning benchmarks"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
    
    def benchmark_training_scenarios(self) -> Dict:
        """Benchmark different training scenarios"""
        results = {}
        
        for i, security_config in enumerate(self.config.security_configs):
            scenario_name = self._get_scenario_name(security_config)
            print(f"Benchmarking scenario: {scenario_name}")
            
            # Create training configuration
            training_config = TrainingConfig(
                local_epochs=3,
                num_rounds=10,
                clients_per_round=min(20, self.config.num_clients),
                enable_post_quantum_crypto=security_config["pqc"],
                enable_differential_privacy=security_config["dp"],
                enable_byzantine_resilience=security_config["byzantine"]
            )
            
            # Run benchmark
            metrics = self._benchmark_single_scenario(training_config)
            results[scenario_name] = metrics
            
            print(f"  Completed: {metrics.final_accuracy:.3f} accuracy, "
                  f"{metrics.training_time:.2f}s total time")
        
        return results
    
    def benchmark_scalability(self) -> Dict:
        """Benchmark system scalability"""
        results = {}
        
        for num_clients in self.config.client_scales:
            if num_clients > self.config.num_clients:
                continue
            
            print(f"Benchmarking scalability: {num_clients} clients")
            
            training_config = TrainingConfig(
                num_rounds=5,
                clients_per_round=min(num_clients // 2, 50),
                enable_post_quantum_crypto=True,
                enable_differential_privacy=True,
                enable_byzantine_resilience=True
            )
            
            metrics = self._benchmark_single_scenario(training_config, num_clients)
            results[f'{num_clients}_clients'] = metrics
            
            print(f"  {num_clients} clients: {metrics.throughput_ops_per_sec:.2f} ops/sec")
        
        return results
    
    def _benchmark_single_scenario(self, config: TrainingConfig, num_clients: int = None) -> PerformanceMetrics:
        """Benchmark a single training scenario"""
        if num_clients is None:
            num_clients = 20
        
        # Initialize monitoring
        monitor = SystemMonitor()
        monitor.start_monitoring()
        
        start_time = time.perf_counter()
        
        try:
            # Create model and server
            model = self._create_model(self.config.model_size)
            server = FederatedServer(model, config)
            
            # Create clients
            clients = []
            for i in range(num_clients):
                dataset = SyntheticMNIST(num_samples=self.config.data_size_per_client)
                client = FederatedClient(f"client_{i:03d}", model, dataset, config)
                clients.append(client)
                server.register_client(client)
            
            # Create test dataset
            test_dataset = SyntheticMNIST(num_samples=1000)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            
            # Training metrics
            total_comm_time = 0.0
            total_crypto_time = 0.0
            total_agg_time = 0.0
            
            # Execute training rounds
            for round_num in range(config.num_rounds):
                round_start = time.perf_counter()
                
                # Execute round
                results = server.federated_round(clients, test_loader)
                
                round_end = time.perf_counter()
                
                if 'error' not in results:
                    total_comm_time += results.get('round_time', 0.0)
                    if 'aggregation_info' in results:
                        total_agg_time += results['aggregation_info'].get('aggregation_time', 0.0)
            
            # Final evaluation
            final_eval = server.evaluate_global_model(test_loader)
            
        except Exception as e:
            logger.error(f"Benchmark scenario failed: {e}")
            return PerformanceMetrics()
        
        finally:
            total_time = time.perf_counter() - start_time
            system_metrics = monitor.stop_monitoring()
        
        # Compile metrics
        metrics = PerformanceMetrics()
        metrics.training_time = total_time
        metrics.communication_time = total_comm_time
        metrics.aggregation_time = total_agg_time
        metrics.final_accuracy = final_eval.get('accuracy', 0.0)
        metrics.peak_memory_mb = system_metrics.get('memory_mb_max', 0.0)
        metrics.avg_cpu_percent = system_metrics.get('cpu_percent_avg', 0.0)
        metrics.throughput_ops_per_sec = (config.num_rounds * num_clients) / total_time
        
        # Privacy metrics
        if config.enable_differential_privacy and hasattr(server, 'dp_engine'):
            metrics.privacy_epsilon_used = server.dp_engine.ledger.total_epsilon
        
        # Byzantine metrics
        if config.enable_byzantine_resilience and hasattr(server, 'byzantine_system'):
            byzantine_stats = server.byzantine_system.get_defense_effectiveness(0.3)
            metrics.byzantine_clients_detected = byzantine_stats.get('total_clients', 0)
        
        return metrics
    
    def _create_model(self, size: str) -> nn.Module:
        """Create model of specified size"""
        if size == "small":
            return SimpleNet()
        elif size == "medium":
            class MediumNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(28*28, 512)
                    self.fc2 = nn.Linear(512, 256)
                    self.fc3 = nn.Linear(256, 128)
                    self.fc4 = nn.Linear(128, 10)
                    self.dropout = nn.Dropout(0.3)
                
                def forward(self, x):
                    x = x.view(-1, 28*28)
                    x = torch.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = torch.relu(self.fc2(x))
                    x = self.dropout(x)
                    x = torch.relu(self.fc3(x))
                    x = self.fc4(x)
                    return x
            return MediumNet()
        else:  # large
            class LargeNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(28*28, 1024)
                    self.fc2 = nn.Linear(1024, 512)
                    self.fc3 = nn.Linear(512, 256)
                    self.fc4 = nn.Linear(256, 128)
                    self.fc5 = nn.Linear(128, 64)
                    self.fc6 = nn.Linear(64, 10)
                    self.dropout = nn.Dropout(0.4)
                
                def forward(self, x):
                    x = x.view(-1, 28*28)
                    x = torch.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = torch.relu(self.fc2(x))
                    x = self.dropout(x)
                    x = torch.relu(self.fc3(x))
                    x = self.dropout(x)
                    x = torch.relu(self.fc4(x))
                    x = self.dropout(x)
                    x = torch.relu(self.fc5(x))
                    x = self.fc6(x)
                    return x
            return LargeNet()
    
    def _get_scenario_name(self, config: Dict) -> str:
        """Generate scenario name from configuration"""
        features = []
        if config["pqc"]:
            features.append("PQC")
        if config["dp"]:
            features.append("DP")
        if config["byzantine"]:
            features.append("BYZ")
        
        if not features:
            return "baseline"
        return "_".join(features).lower()

class QFLAREBenchmarkSuite:
    """Complete QFLARE performance benchmarking suite"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
        
        # Create results directory
        if self.config.save_results:
            os.makedirs(self.config.results_dir, exist_ok=True)
    
    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite"""
        print("🔬 QFLARE Performance Benchmarking Suite")
        print("=" * 60)
        
        total_start_time = time.perf_counter()
        
        # 1. Cryptographic benchmarks
        if self.config.test_crypto_performance:
            print("\n1. Cryptographic Performance Benchmarks")
            print("-" * 40)
            crypto_bench = CryptographicBenchmark()
            
            print("  Testing key generation...")
            self.results['crypto_keygen'] = crypto_bench.benchmark_key_generation()
            
            print("  Testing encryption/decryption...")
            self.results['crypto_encryption'] = crypto_bench.benchmark_encryption_decryption()
            
            print("  Testing signatures...")
            self.results['crypto_signatures'] = crypto_bench.benchmark_signatures()
        
        # 2. Privacy benchmarks
        if self.config.test_privacy_overhead:
            print("\n2. Differential Privacy Overhead Benchmarks")
            print("-" * 40)
            privacy_bench = PrivacyBenchmark()
            
            print("  Testing noise addition...")
            self.results['privacy_noise'] = privacy_bench.benchmark_noise_addition()
            
            print("  Testing gradient clipping...")
            self.results['privacy_clipping'] = privacy_bench.benchmark_gradient_clipping()
        
        # 3. Byzantine resilience benchmarks
        if self.config.test_byzantine_resilience:
            print("\n3. Byzantine Resilience Benchmarks")
            print("-" * 40)
            byzantine_bench = ByzantineBenchmark()
            
            print("  Testing aggregation methods...")
            self.results['byzantine_aggregation'] = byzantine_bench.benchmark_aggregation_methods()
            
            print("  Testing detection accuracy...")
            self.results['byzantine_detection'] = byzantine_bench.benchmark_detection_accuracy()
        
        # 4. Federated learning benchmarks
        print("\n4. Federated Learning End-to-End Benchmarks")
        print("-" * 40)
        fl_bench = FederatedLearningBenchmark(self.config)
        
        print("  Testing security configurations...")
        self.results['fl_scenarios'] = fl_bench.benchmark_training_scenarios()
        
        # 5. Scalability benchmarks
        if self.config.test_scalability:
            print("\n5. Scalability Benchmarks")
            print("-" * 40)
            print("  Testing client scaling...")
            self.results['scalability'] = fl_bench.benchmark_scalability()
        
        total_time = time.perf_counter() - total_start_time
        self.results['benchmark_metadata'] = {
            'total_benchmark_time': total_time,
            'config': self.config.__dict__,
            'system_info': {
                'cpu_count': multiprocessing.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'platform': os.name
            }
        }
        
        print(f"\n✅ Benchmark suite completed in {total_time:.2f} seconds")
        
        # Save and visualize results
        if self.config.save_results:
            self._save_results()
        
        if self.config.generate_plots:
            self._generate_visualizations()
        
        return self.results
    
    def _save_results(self):
        """Save benchmark results to files"""
        # Save raw results as JSON
        results_file = os.path.join(self.config.results_dir, 'benchmark_results.json')
        with open(results_file, 'w') as f:
            # Convert any numpy arrays to lists for JSON serialization
            serializable_results = json.loads(json.dumps(self.results, default=str))
            json.dump(serializable_results, f, indent=2)
        
        # Save as pickle for Python analysis
        pickle_file = os.path.join(self.config.results_dir, 'benchmark_results.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"Results saved to {self.config.results_dir}")
    
    def _generate_visualizations(self):
        """Generate performance visualization plots"""
        if not PLOTTING_AVAILABLE:
            print("Matplotlib/seaborn not available, skipping plot generation")
            return
        
        try:
            plt.style.use('default')  # Use default style instead of seaborn
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('QFLARE Performance Benchmark Results', fontsize=16)
            
            # 1. Crypto performance
            if 'crypto_keygen' in self.results:
                ax = axes[0, 0]
                crypto_data = self.results['crypto_keygen']
                
                operations = []
                ops_per_sec = []
                for op, metrics in crypto_data.items():
                    operations.append(op.replace('_', ' ').title())
                    ops_per_sec.append(metrics['ops_per_second'])
                
                ax.bar(operations, ops_per_sec)
                ax.set_title('Cryptographic Operations Performance')
                ax.set_ylabel('Operations/Second')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # 2. FL scenarios comparison
            if 'fl_scenarios' in self.results:
                ax = axes[0, 1]
                scenarios = []
                accuracies = []
                times = []
                
                for scenario, metrics in self.results['fl_scenarios'].items():
                    scenarios.append(scenario.upper())
                    if hasattr(metrics, 'final_accuracy'):
                        accuracies.append(metrics.final_accuracy)
                        times.append(metrics.training_time)
                
                if accuracies:
                    x_pos = np.arange(len(scenarios))
                    ax2 = ax.twinx()
                    
                    bars1 = ax.bar(x_pos - 0.2, accuracies, 0.4, label='Accuracy', alpha=0.7)
                    bars2 = ax2.bar(x_pos + 0.2, times, 0.4, label='Time (s)', alpha=0.7, color='orange')
                    
                    ax.set_title('FL Scenarios: Accuracy vs Time')
                    ax.set_xlabel('Security Configuration')
                    ax.set_ylabel('Accuracy')
                    ax2.set_ylabel('Training Time (s)')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(scenarios, rotation=45, ha='right')
            
            # 3. Scalability
            if 'scalability' in self.results:
                ax = axes[0, 2]
                client_counts = []
                throughputs = []
                
                for scenario, metrics in self.results['scalability'].items():
                    if 'clients' in scenario:
                        num_clients = int(scenario.split('_')[0])
                        client_counts.append(num_clients)
                        if hasattr(metrics, 'throughput_ops_per_sec'):
                            throughputs.append(metrics.throughput_ops_per_sec)
                
                if throughputs:
                    ax.plot(client_counts, throughputs, 'o-', linewidth=2, markersize=6)
                    ax.set_title('Scalability: Throughput vs Client Count')
                    ax.set_xlabel('Number of Clients')
                    ax.set_ylabel('Throughput (ops/sec)')
                    ax.grid(True, alpha=0.3)
            
            # 4. Privacy overhead
            if 'privacy_noise' in self.results:
                ax = axes[1, 0]
                gradient_sizes = []
                gaussian_perf = []
                laplace_perf = []
                
                for key, metrics in self.results['privacy_noise'].items():
                    if 'gaussian_noise' in key:
                        size = int(key.split('_')[-1])
                        gradient_sizes.append(size)
                        gaussian_perf.append(metrics['params_per_second'])
                    elif 'laplace_noise' in key:
                        laplace_perf.append(metrics['params_per_second'])
                
                if gradient_sizes:
                    ax.loglog(gradient_sizes, gaussian_perf, 'o-', label='Gaussian', linewidth=2)
                    if len(laplace_perf) == len(gradient_sizes):
                        ax.loglog(gradient_sizes, laplace_perf, 's-', label='Laplace', linewidth=2)
                    
                    ax.set_title('Privacy Noise Addition Performance')
                    ax.set_xlabel('Gradient Size (parameters)')
                    ax.set_ylabel('Parameters/Second')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            # 5. Byzantine detection
            if 'byzantine_detection' in self.results:
                ax = axes[1, 1]
                detection_types = []
                detection_rates = []
                
                for det_type, metrics in self.results['byzantine_detection'].items():
                    detection_types.append(det_type.replace('_', ' ').title())
                    detection_rates.append(metrics['avg_detection_rate'] * 100)
                
                bars = ax.bar(detection_types, detection_rates)
                ax.set_title('Byzantine Detection Accuracy')
                ax.set_ylabel('Detection Rate (%)')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, rate in zip(bars, detection_rates):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{rate:.1f}%', ha='center', va='bottom')
            
            # 6. Memory usage comparison
            if 'fl_scenarios' in self.results:
                ax = axes[1, 2]
                scenarios = []
                memory_usage = []
                
                for scenario, metrics in self.results['fl_scenarios'].items():
                    if hasattr(metrics, 'peak_memory_mb') and metrics.peak_memory_mb > 0:
                        scenarios.append(scenario.upper())
                        memory_usage.append(metrics.peak_memory_mb)
                
                if memory_usage:
                    bars = ax.bar(scenarios, memory_usage)
                    ax.set_title('Peak Memory Usage by Scenario')
                    ax.set_ylabel('Peak Memory (MB)')
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            plot_file = os.path.join(self.config.results_dir, 'benchmark_plots.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {plot_file}")
            
        except Exception as e:
            print(f"Plot generation failed: {e}")
            logger.error(f"Visualization generation error: {e}")
    
    def generate_report(self) -> str:
        """Generate human-readable benchmark report"""
        report = []
        report.append("QFLARE Performance Benchmark Report")
        report.append("=" * 50)
        
        if 'benchmark_metadata' in self.results:
            metadata = self.results['benchmark_metadata']
            report.append(f"\nBenchmark completed in {metadata['total_benchmark_time']:.2f} seconds")
            report.append(f"System: {metadata['system_info']['cpu_count']} CPUs, "
                         f"{metadata['system_info']['memory_gb']:.1f} GB RAM")
        
        # Crypto performance summary
        if 'crypto_keygen' in self.results:
            report.append("\n--- Cryptographic Performance ---")
            for op, metrics in self.results['crypto_keygen'].items():
                report.append(f"{op.replace('_', ' ').title()}: {metrics['ops_per_second']:.2f} ops/sec")
        
        # FL scenarios summary
        if 'fl_scenarios' in self.results:
            report.append("\n--- Federated Learning Scenarios ---")
            for scenario, metrics in self.results['fl_scenarios'].items():
                if hasattr(metrics, 'final_accuracy'):
                    report.append(f"{scenario.upper()}: {metrics.final_accuracy:.3f} accuracy, "
                                f"{metrics.training_time:.2f}s training time")
        
        # Scalability summary
        if 'scalability' in self.results:
            report.append("\n--- Scalability Results ---")
            for scenario, metrics in self.results['scalability'].items():
                if hasattr(metrics, 'throughput_ops_per_sec'):
                    clients = scenario.split('_')[0]
                    report.append(f"{clients} clients: {metrics.throughput_ops_per_sec:.2f} ops/sec throughput")
        
        # Byzantine detection summary
        if 'byzantine_detection' in self.results:
            report.append("\n--- Byzantine Detection Accuracy ---")
            for det_type, metrics in self.results['byzantine_detection'].items():
                rate = metrics['avg_detection_rate'] * 100
                report.append(f"{det_type.replace('_', ' ').title()}: {rate:.1f}% detection rate")
        
        report_text = '\n'.join(report)
        
        if self.config.save_results:
            report_file = os.path.join(self.config.results_dir, 'benchmark_report.txt')
            with open(report_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {report_file}")
        
        return report_text

def demo_performance_benchmarking():
    """Demonstration of QFLARE performance benchmarking"""
    print("🔬 QFLARE Performance Benchmarking Demo")
    print("=" * 50)
    
    # Configure benchmark (reduced scale for demo)
    config = BenchmarkConfig(
        num_clients=50,
        num_rounds=10,
        data_size_per_client=500,
        model_size="small",
        client_scales=[10, 20, 30],
        security_configs=[
            {"pqc": False, "dp": False, "byzantine": False},  # Baseline
            {"pqc": True, "dp": False, "byzantine": False},   # PQC only
            {"pqc": True, "dp": True, "byzantine": True},     # All features
        ]
    )
    
    # Run benchmark suite
    suite = QFLAREBenchmarkSuite(config)
    results = suite.run_full_benchmark()
    
    # Generate and display report
    report = suite.generate_report()
    print("\n" + report)
    
    print(f"\n🔬 Performance benchmarking completed!")
    print(f"   Results demonstrate QFLARE's security-performance tradeoffs")

if __name__ == "__main__":
    demo_performance_benchmarking()