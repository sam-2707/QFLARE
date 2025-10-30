#!/usr/bin/env python3
"""
QFLARE Benchmark Harness - Simulate federated learning clients and measure performance
Usage: python benchmark_harness.py --clients 100 --rounds 5 --secure-agg
"""

import argparse
import asyncio
import json
import multiprocessing as mp
import time
import psutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests"""
    num_clients: int = 100
    num_rounds: int = 5
    model_size_mb: float = 10.0  # Simulated model size
    secure_aggregation: bool = False
    differential_privacy: bool = False
    pqc_enabled: bool = False
    edge_nodes: int = 1
    network_latency_ms: int = 50
    bandwidth_mbps: float = 10.0

@dataclass
class PerformanceMetrics:
    """Performance measurement results"""
    round_duration_seconds: float
    cpu_usage_percent: float
    memory_usage_mb: float
    network_bytes_sent: int
    network_bytes_received: int
    handshake_time_ms: float
    aggregation_time_ms: float
    client_training_time_ms: float
    accuracy_drop_percent: float = 0.0

class SimulatedClient:
    """Simulates a federated learning client device"""
    
    def __init__(self, client_id: int, config: BenchmarkConfig):
        self.client_id = client_id
        self.config = config
        self.model_weights = np.random.randn(int(config.model_size_mb * 1024 * 100))  # Simulate model
        
    async def perform_handshake(self) -> float:
        """Simulate cryptographic handshake (PQC if enabled)"""
        start_time = time.time()
        
        if self.config.pqc_enabled:
            # Simulate CRYSTALS-Kyber key exchange overhead
            await asyncio.sleep(0.1 + np.random.exponential(0.05))  # 100ms + exponential noise
        else:
            # Classical TLS handshake
            await asyncio.sleep(0.02 + np.random.exponential(0.01))  # 20ms + noise
            
        handshake_duration = (time.time() - start_time) * 1000
        logger.debug(f"Client {self.client_id}: Handshake completed in {handshake_duration:.1f}ms")
        return handshake_duration
    
    async def local_training(self) -> np.ndarray:
        """Simulate local model training and return gradients"""
        start_time = time.time()
        
        # Simulate training computation (CPU intensive)
        training_time = 0.5 + np.random.exponential(0.2)  # 500ms + noise
        await asyncio.sleep(training_time)
        
        # Generate synthetic gradients
        gradients = np.random.randn(len(self.model_weights)) * 0.01
        
        if self.config.differential_privacy:
            # Add DP noise
            noise_scale = 0.001  # Simplified DP noise
            gradients += np.random.normal(0, noise_scale, gradients.shape)
        
        training_duration = (time.time() - start_time) * 1000
        logger.debug(f"Client {self.client_id}: Training completed in {training_duration:.1f}ms")
        return gradients, training_duration
    
    async def upload_gradients(self, gradients: np.ndarray) -> int:
        """Simulate uploading encrypted gradients to edge node"""
        # Simulate network transmission time based on bandwidth
        data_size_bytes = gradients.nbytes
        if self.config.secure_aggregation:
            data_size_bytes *= 1.5  # Encryption overhead
            
        transmission_time = (data_size_bytes * 8) / (self.config.bandwidth_mbps * 1024 * 1024)
        transmission_time += self.config.network_latency_ms / 1000.0  # Add latency
        
        await asyncio.sleep(transmission_time)
        logger.debug(f"Client {self.client_id}: Uploaded {data_size_bytes} bytes")
        return data_size_bytes

class EdgeAggregator:
    """Simulates edge node aggregation"""
    
    def __init__(self, edge_id: int, config: BenchmarkConfig):
        self.edge_id = edge_id
        self.config = config
        self.client_gradients: List[np.ndarray] = []
        
    async def aggregate_gradients(self, gradients_list: List[np.ndarray]) -> float:
        """Perform secure aggregation of client gradients"""
        start_time = time.time()
        
        if self.config.secure_aggregation:
            # Simulate secure multi-party computation overhead
            computation_time = len(gradients_list) * 0.01 + 0.1  # Linear in clients + base overhead
        else:
            # Simple averaging
            computation_time = len(gradients_list) * 0.001  # Much faster
            
        await asyncio.sleep(computation_time)
        
        # Perform aggregation (simple average)
        aggregated = np.mean(gradients_list, axis=0)
        
        aggregation_duration = (time.time() - start_time) * 1000
        logger.debug(f"Edge {self.edge_id}: Aggregated {len(gradients_list)} gradients in {aggregation_duration:.1f}ms")
        return aggregation_duration

class BenchmarkHarness:
    """Main benchmark orchestrator"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.clients = [SimulatedClient(i, config) for i in range(config.num_clients)]
        self.edge_nodes = [EdgeAggregator(i, config) for i in range(config.edge_nodes)]
        self.results: List[PerformanceMetrics] = []
        
    async def run_training_round(self, round_num: int) -> PerformanceMetrics:
        """Execute one complete federated learning round"""
        logger.info(f"Starting round {round_num + 1}/{self.config.num_rounds}")
        round_start_time = time.time()
        
        # Monitor system resources
        process = psutil.Process()
        initial_cpu = process.cpu_percent()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        total_bytes_sent = 0
        total_bytes_received = 0
        handshake_times = []
        training_times = []
        
        # Distribute clients across edge nodes
        clients_per_edge = self.config.num_clients // self.config.edge_nodes
        
        tasks = []
        for edge_idx, edge_node in enumerate(self.edge_nodes):
            start_client = edge_idx * clients_per_edge
            end_client = start_client + clients_per_edge
            if edge_idx == len(self.edge_nodes) - 1:  # Last edge gets remaining clients
                end_client = self.config.num_clients
                
            edge_clients = self.clients[start_client:end_client]
            task = asyncio.create_task(self._process_edge_clients(edge_node, edge_clients))
            tasks.append(task)
        
        # Wait for all edge nodes to complete
        edge_results = await asyncio.gather(*tasks)
        
        # Collect metrics from all edge nodes
        for result in edge_results:
            total_bytes_sent += result['bytes_sent']
            total_bytes_received += result['bytes_received']
            handshake_times.extend(result['handshake_times'])
            training_times.extend(result['training_times'])
        
        # Calculate final metrics
        round_duration = time.time() - round_start_time
        final_cpu = process.cpu_percent()
        final_memory = process.memory_info().rss / 1024 / 1024
        
        # Simulate aggregation time across edge nodes
        global_aggregation_time = max([r['aggregation_time'] for r in edge_results])
        
        metrics = PerformanceMetrics(
            round_duration_seconds=round_duration,
            cpu_usage_percent=max(final_cpu, initial_cpu),
            memory_usage_mb=final_memory,
            network_bytes_sent=total_bytes_sent,
            network_bytes_received=total_bytes_received,
            handshake_time_ms=np.mean(handshake_times),
            aggregation_time_ms=global_aggregation_time,
            client_training_time_ms=np.mean(training_times)
        )
        
        logger.info(f"Round {round_num + 1} completed in {round_duration:.2f}s")
        return metrics
    
    async def _process_edge_clients(self, edge_node: EdgeAggregator, clients: List[SimulatedClient]) -> Dict:
        """Process all clients connected to a single edge node"""
        gradients_list = []
        bytes_sent = 0
        bytes_received = 0
        handshake_times = []
        training_times = []
        
        # Process clients in parallel (limited concurrency to simulate real constraints)
        semaphore = asyncio.Semaphore(min(50, len(clients)))  # Max 50 concurrent clients per edge
        
        async def process_client(client):
            async with semaphore:
                # Handshake
                handshake_time = await client.perform_handshake()
                handshake_times.append(handshake_time)
                
                # Local training
                gradients, training_time = await client.local_training()
                training_times.append(training_time)
                
                # Upload gradients
                upload_bytes = await client.upload_gradients(gradients)
                
                return gradients, upload_bytes
        
        # Execute all client tasks
        client_tasks = [process_client(client) for client in clients]
        client_results = await asyncio.gather(*client_tasks)
        
        # Collect results
        for gradients, upload_bytes in client_results:
            gradients_list.append(gradients)
            bytes_sent += upload_bytes
            bytes_received += gradients.nbytes  # Simplified
        
        # Perform edge aggregation
        aggregation_time = await edge_node.aggregate_gradients(gradients_list)
        
        return {
            'bytes_sent': bytes_sent,
            'bytes_received': bytes_received,
            'handshake_times': handshake_times,
            'training_times': training_times,
            'aggregation_time': aggregation_time
        }
    
    async def run_benchmark(self) -> List[PerformanceMetrics]:
        """Run complete benchmark with multiple rounds"""
        logger.info(f"Starting benchmark: {self.config.num_clients} clients, {self.config.num_rounds} rounds")
        logger.info(f"Configuration: Secure Aggregation={self.config.secure_aggregation}, "
                   f"DP={self.config.differential_privacy}, PQC={self.config.pqc_enabled}")
        
        for round_num in range(self.config.num_rounds):
            metrics = await self.run_training_round(round_num)
            self.results.append(metrics)
        
        return self.results
    
    def generate_report(self) -> Dict:
        """Generate comprehensive benchmark report"""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        # Calculate statistics across all rounds
        round_durations = [r.round_duration_seconds for r in self.results]
        handshake_times = [r.handshake_time_ms for r in self.results]
        aggregation_times = [r.aggregation_time_ms for r in self.results]
        cpu_usage = [r.cpu_usage_percent for r in self.results]
        memory_usage = [r.memory_usage_mb for r in self.results]
        
        report = {
            "benchmark_config": asdict(self.config),
            "summary_statistics": {
                "total_rounds": len(self.results),
                "avg_round_duration_seconds": np.mean(round_durations),
                "max_round_duration_seconds": np.max(round_durations),
                "avg_handshake_time_ms": np.mean(handshake_times),
                "avg_aggregation_time_ms": np.mean(aggregation_times),
                "avg_cpu_usage_percent": np.mean(cpu_usage),
                "peak_memory_usage_mb": np.max(memory_usage),
                "total_network_bytes": sum(r.network_bytes_sent + r.network_bytes_received for r in self.results)
            },
            "performance_targets": {
                "round_duration_target_seconds": 30.0,
                "handshake_target_ms": 500.0,
                "cpu_usage_target_percent": 80.0,
                "memory_target_mb": 512.0
            },
            "pass_fail_results": {},
            "detailed_results": [asdict(r) for r in self.results]
        }
        
        # Evaluate pass/fail criteria
        targets = report["performance_targets"]
        stats = report["summary_statistics"]
        
        report["pass_fail_results"] = {
            "round_duration_pass": stats["avg_round_duration_seconds"] <= targets["round_duration_target_seconds"],
            "handshake_time_pass": stats["avg_handshake_time_ms"] <= targets["handshake_target_ms"],
            "cpu_usage_pass": stats["avg_cpu_usage_percent"] <= targets["cpu_usage_target_percent"],
            "memory_usage_pass": stats["peak_memory_usage_mb"] <= targets["memory_target_mb"],
            "overall_pass": all([
                stats["avg_round_duration_seconds"] <= targets["round_duration_target_seconds"],
                stats["avg_handshake_time_ms"] <= targets["handshake_target_ms"],
                stats["avg_cpu_usage_percent"] <= targets["cpu_usage_target_percent"],
                stats["peak_memory_usage_mb"] <= targets["memory_target_mb"]
            ])
        }
        
        return report

def create_benchmark_configs() -> List[BenchmarkConfig]:
    """Create standard benchmark configurations for different scenarios"""
    configs = [
        # Baseline configuration
        BenchmarkConfig(
            num_clients=100, 
            num_rounds=3,
            secure_aggregation=False,
            differential_privacy=False,
            pqc_enabled=False
        ),
        # QFLARE full configuration
        BenchmarkConfig(
            num_clients=100,
            num_rounds=3,
            secure_aggregation=True,
            differential_privacy=True,
            pqc_enabled=True
        ),
        # Scalability test
        BenchmarkConfig(
            num_clients=1000,
            num_rounds=2,
            edge_nodes=10,
            secure_aggregation=True,
            pqc_enabled=True
        )
    ]
    return configs

async def main():
    """Main benchmark execution"""
    parser = argparse.ArgumentParser(description='QFLARE Benchmark Harness')
    parser.add_argument('--clients', type=int, default=100, help='Number of simulated clients')
    parser.add_argument('--rounds', type=int, default=3, help='Number of training rounds')
    parser.add_argument('--secure-agg', action='store_true', help='Enable secure aggregation')
    parser.add_argument('--differential-privacy', action='store_true', help='Enable differential privacy')
    parser.add_argument('--pqc', action='store_true', help='Enable post-quantum cryptography')
    parser.add_argument('--edge-nodes', type=int, default=1, help='Number of edge nodes')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    parser.add_argument('--preset', choices=['baseline', 'qflare', 'scale'], help='Use preset configuration')
    
    args = parser.parse_args()
    
    # Create configuration
    if args.preset:
        configs = create_benchmark_configs()
        if args.preset == 'baseline':
            config = configs[0]
        elif args.preset == 'qflare':
            config = configs[1]
        elif args.preset == 'scale':
            config = configs[2]
    else:
        config = BenchmarkConfig(
            num_clients=args.clients,
            num_rounds=args.rounds,
            secure_aggregation=args.secure_agg,
            differential_privacy=args.differential_privacy,
            pqc_enabled=args.pqc,
            edge_nodes=args.edge_nodes
        )
    
    # Run benchmark
    harness = BenchmarkHarness(config)
    start_time = time.time()
    
    try:
        await harness.run_benchmark()
        total_time = time.time() - start_time
        
        # Generate and display report
        report = harness.generate_report()
        report["total_benchmark_time_seconds"] = total_time
        
        print("\n" + "="*60)
        print("QFLARE BENCHMARK RESULTS")
        print("="*60)
        print(f"Configuration: {args.clients} clients, {args.rounds} rounds")
        print(f"Features: Secure Aggregation={config.secure_aggregation}, "
              f"DP={config.differential_privacy}, PQC={config.pqc_enabled}")
        print(f"Total Benchmark Time: {total_time:.2f} seconds")
        print()
        
        stats = report["summary_statistics"]
        targets = report["performance_targets"]
        results = report["pass_fail_results"]
        
        print("Performance Metrics:")
        print(f"  Average Round Duration: {stats['avg_round_duration_seconds']:.2f}s "
              f"(Target: ≤{targets['round_duration_target_seconds']}s) "
              f"{'✅' if results['round_duration_pass'] else '❌'}")
        print(f"  Average Handshake Time: {stats['avg_handshake_time_ms']:.1f}ms "
              f"(Target: ≤{targets['handshake_target_ms']}ms) "
              f"{'✅' if results['handshake_time_pass'] else '❌'}")
        print(f"  Average CPU Usage: {stats['avg_cpu_usage_percent']:.1f}% "
              f"(Target: ≤{targets['cpu_usage_target_percent']}%) "
              f"{'✅' if results['cpu_usage_pass'] else '❌'}")
        print(f"  Peak Memory Usage: {stats['peak_memory_usage_mb']:.1f}MB "
              f"(Target: ≤{targets['memory_target_mb']}MB) "
              f"{'✅' if results['memory_usage_pass'] else '❌'}")
        print(f"  Total Network Traffic: {stats['total_network_bytes'] / 1024 / 1024:.2f}MB")
        print()
        print(f"Overall Result: {'✅ PASS' if results['overall_pass'] else '❌ FAIL'}")
        
        # Save detailed results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nDetailed results saved to: {args.output}")
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())