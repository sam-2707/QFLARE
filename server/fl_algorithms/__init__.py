"""
FL Algorithms Module Initialization for QFLARE

This module provides advanced federated learning algorithms including:
- FedProx: Handles statistical heterogeneity with proximal terms
- FedBN: Manages batch normalization in non-IID settings
- Personalized FL: Various personalization strategies
- Adaptive Aggregation: Advanced server-side optimization

Key Features:
- Algorithm factory for easy instantiation
- Unified interface for all FL algorithms
- Configuration management and validation
- Integration with QFLARE monitoring and security systems
- Performance benchmarking and comparison tools
"""

import logging
from typing import Dict, List, Any, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum

# Import algorithm implementations
from .fedprox import FedProxAlgorithm, FedProxConfig
from .fedbn import FedBNAlgorithm, FedBNConfig
from .personalized_fl import PersonalizedFederatedLearning, PersonalizationConfig
from .adaptive_aggregation import AdaptiveAggregationOrchestrator, AdaptiveAggregationConfig, AggregationStrategy

logger = logging.getLogger(__name__)


class FLAlgorithmType(Enum):
    """Available federated learning algorithm types."""
    FEDPROX = "fedprox"
    FEDBN = "fedbn"
    PERSONALIZED_FL = "personalized_fl"
    ADAPTIVE_AGGREGATION = "adaptive_aggregation"


@dataclass
class AlgorithmMetrics:
    """Common metrics structure for all FL algorithms."""
    algorithm_type: str
    round_number: int
    global_loss: float
    participating_clients: int
    convergence_rate: float
    training_time: float
    communication_overhead: int  # bytes
    memory_usage: float  # MB
    
    # Algorithm-specific metrics
    algorithm_specific: Dict[str, Any]


@dataclass
class BenchmarkResult:
    """Results from algorithm benchmarking."""
    algorithm_type: str
    total_rounds: int
    final_accuracy: float
    convergence_rounds: int
    avg_round_time: float
    total_communication: int  # bytes
    client_fairness_score: float
    robustness_score: float
    
    # Detailed metrics per round
    round_metrics: List[AlgorithmMetrics]


class FLAlgorithmFactory:
    """Factory for creating FL algorithms with appropriate configurations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._algorithm_registry = {
            FLAlgorithmType.FEDPROX: (FedProxAlgorithm, FedProxConfig),
            FLAlgorithmType.FEDBN: (FedBNAlgorithm, FedBNConfig),
            FLAlgorithmType.PERSONALIZED_FL: (PersonalizedFederatedLearning, PersonalizationConfig),
            FLAlgorithmType.ADAPTIVE_AGGREGATION: (AdaptiveAggregationOrchestrator, AdaptiveAggregationConfig)
        }
    
    def create_algorithm(self, algorithm_type: Union[FLAlgorithmType, str], 
                        config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create an FL algorithm instance with the specified configuration.
        
        Args:
            algorithm_type: Type of algorithm to create
            config: Algorithm-specific configuration parameters
            
        Returns:
            Configured algorithm instance
        """
        if isinstance(algorithm_type, str):
            try:
                algorithm_type = FLAlgorithmType(algorithm_type.lower())
            except ValueError:
                raise ValueError(f"Unknown algorithm type: {algorithm_type}")
        
        if algorithm_type not in self._algorithm_registry:
            raise ValueError(f"Algorithm type {algorithm_type} not registered")
        
        algorithm_class, config_class = self._algorithm_registry[algorithm_type]
        
        # Create configuration
        if config is None:
            config = {}
        
        try:
            algorithm_config = config_class(**config)
        except TypeError as e:
            self.logger.error(f"Invalid configuration for {algorithm_type}: {e}")
            # Create with default configuration
            algorithm_config = config_class()
        
        # Create algorithm instance
        algorithm_instance = algorithm_class(algorithm_config)
        
        self.logger.info(f"Created {algorithm_type.value} algorithm with config: {algorithm_config}")
        
        return algorithm_instance
    
    def get_available_algorithms(self) -> List[str]:
        """Get list of available algorithm types."""
        return [alg_type.value for alg_type in self._algorithm_registry.keys()]
    
    def get_default_config(self, algorithm_type: Union[FLAlgorithmType, str]) -> Dict[str, Any]:
        """Get default configuration for an algorithm type."""
        if isinstance(algorithm_type, str):
            algorithm_type = FLAlgorithmType(algorithm_type.lower())
        
        if algorithm_type not in self._algorithm_registry:
            raise ValueError(f"Algorithm type {algorithm_type} not registered")
        
        _, config_class = self._algorithm_registry[algorithm_type]
        default_config = config_class()
        
        # Convert to dictionary (simplified)
        config_dict = {}
        for field_name in default_config.__dataclass_fields__:
            config_dict[field_name] = getattr(default_config, field_name)
        
        return config_dict


class FLAlgorithmCoordinator:
    """Coordinates multiple FL algorithms and provides unified interface."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.factory = FLAlgorithmFactory()
        self.active_algorithms = {}
        self.algorithm_metrics = {}
    
    def register_algorithm(self, algorithm_id: str, algorithm_type: str, 
                          config: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a new algorithm instance.
        
        Args:
            algorithm_id: Unique identifier for this algorithm instance
            algorithm_type: Type of algorithm
            config: Algorithm configuration
            
        Returns:
            Algorithm ID for reference
        """
        if algorithm_id in self.active_algorithms:
            raise ValueError(f"Algorithm ID {algorithm_id} already exists")
        
        # Create algorithm instance
        algorithm = self.factory.create_algorithm(algorithm_type, config)
        
        # Register
        self.active_algorithms[algorithm_id] = {
            'algorithm': algorithm,
            'type': algorithm_type,
            'config': config or {},
            'created_at': __import__('time').time()
        }
        
        self.algorithm_metrics[algorithm_id] = []
        
        self.logger.info(f"Registered algorithm {algorithm_id} of type {algorithm_type}")
        return algorithm_id
    
    def get_algorithm(self, algorithm_id: str) -> Any:
        """Get algorithm instance by ID."""
        if algorithm_id not in self.active_algorithms:
            raise ValueError(f"Algorithm ID {algorithm_id} not found")
        
        return self.active_algorithms[algorithm_id]['algorithm']
    
    def remove_algorithm(self, algorithm_id: str):
        """Remove algorithm instance."""
        if algorithm_id in self.active_algorithms:
            del self.active_algorithms[algorithm_id]
            if algorithm_id in self.algorithm_metrics:
                del self.algorithm_metrics[algorithm_id]
            self.logger.info(f"Removed algorithm {algorithm_id}")
    
    def record_metrics(self, algorithm_id: str, metrics: AlgorithmMetrics):
        """Record metrics for an algorithm."""
        if algorithm_id in self.algorithm_metrics:
            self.algorithm_metrics[algorithm_id].append(metrics)
    
    def get_algorithm_stats(self, algorithm_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for an algorithm."""
        if algorithm_id not in self.active_algorithms:
            raise ValueError(f"Algorithm ID {algorithm_id} not found")
        
        algorithm_info = self.active_algorithms[algorithm_id]
        algorithm = algorithm_info['algorithm']
        
        # Base stats
        stats = {
            'algorithm_id': algorithm_id,
            'algorithm_type': algorithm_info['type'],
            'created_at': algorithm_info['created_at'],
            'config': algorithm_info['config']
        }
        
        # Get algorithm-specific stats
        if hasattr(algorithm, 'get_algorithm_stats'):
            algorithm_stats = algorithm.get_algorithm_stats()
            stats.update(algorithm_stats)
        
        # Metrics history
        if algorithm_id in self.algorithm_metrics:
            metrics_history = self.algorithm_metrics[algorithm_id]
            if metrics_history:
                stats['total_rounds'] = len(metrics_history)
                stats['latest_loss'] = metrics_history[-1].global_loss
                stats['avg_round_time'] = sum(m.training_time for m in metrics_history) / len(metrics_history)
                
                # Convergence analysis
                if len(metrics_history) >= 2:
                    loss_improvement = metrics_history[0].global_loss - metrics_history[-1].global_loss
                    stats['total_loss_improvement'] = loss_improvement
        
        return stats
    
    def compare_algorithms(self, algorithm_ids: List[str]) -> Dict[str, Any]:
        """Compare performance of multiple algorithms."""
        comparison = {
            'algorithms': algorithm_ids,
            'comparison_metrics': {},
            'recommendations': []
        }
        
        for alg_id in algorithm_ids:
            if alg_id in self.algorithm_metrics:
                metrics = self.algorithm_metrics[alg_id]
                if metrics:
                    comparison['comparison_metrics'][alg_id] = {
                        'final_loss': metrics[-1].global_loss,
                        'convergence_speed': len(metrics),
                        'avg_round_time': sum(m.training_time for m in metrics) / len(metrics),
                        'communication_efficiency': sum(m.communication_overhead for m in metrics)
                    }
        
        # Simple recommendations based on metrics
        if comparison['comparison_metrics']:
            best_loss = min(comparison['comparison_metrics'].items(), 
                          key=lambda x: x[1]['final_loss'])
            comparison['recommendations'].append(f"Best final loss: {best_loss[0]}")
            
            fastest_convergence = min(comparison['comparison_metrics'].items(), 
                                    key=lambda x: x[1]['convergence_speed'])
            comparison['recommendations'].append(f"Fastest convergence: {fastest_convergence[0]}")
        
        return comparison


class FLAlgorithmBenchmark:
    """Benchmarking suite for FL algorithms."""
    
    def __init__(self, coordinator: FLAlgorithmCoordinator):
        self.coordinator = coordinator
        self.logger = logging.getLogger(__name__)
    
    def run_benchmark(self, algorithm_configs: List[Dict[str, Any]], 
                     benchmark_config: Dict[str, Any]) -> Dict[str, BenchmarkResult]:
        """
        Run benchmark comparing multiple algorithms.
        
        Args:
            algorithm_configs: List of algorithm configurations to test
            benchmark_config: Benchmark parameters (rounds, data, etc.)
            
        Returns:
            Benchmark results for each algorithm
        """
        results = {}
        
        self.logger.info(f"Starting benchmark with {len(algorithm_configs)} algorithms")
        
        for i, alg_config in enumerate(algorithm_configs):
            algorithm_id = f"benchmark_alg_{i}"
            algorithm_type = alg_config.get('type', 'fedprox')
            config = alg_config.get('config', {})
            
            try:
                # Register algorithm
                self.coordinator.register_algorithm(algorithm_id, algorithm_type, config)
                
                # Run benchmark (simplified - would need actual training loop)
                result = self._run_single_benchmark(algorithm_id, benchmark_config)
                results[algorithm_type] = result
                
                self.logger.info(f"Completed benchmark for {algorithm_type}")
                
            except Exception as e:
                self.logger.error(f"Benchmark failed for {algorithm_type}: {e}")
            finally:
                # Cleanup
                if algorithm_id in self.coordinator.active_algorithms:
                    self.coordinator.remove_algorithm(algorithm_id)
        
        return results
    
    def _run_single_benchmark(self, algorithm_id: str, 
                            benchmark_config: Dict[str, Any]) -> BenchmarkResult:
        """Run benchmark for a single algorithm."""
        # This is a simplified implementation
        # In practice, would need actual training data and simulation
        
        algorithm_info = self.coordinator.active_algorithms[algorithm_id]
        algorithm_type = algorithm_info['type']
        
        # Simulate benchmark results
        num_rounds = benchmark_config.get('rounds', 100)
        round_metrics = []
        
        for round_num in range(num_rounds):
            # Simulate round metrics
            metrics = AlgorithmMetrics(
                algorithm_type=algorithm_type,
                round_number=round_num,
                global_loss=1.0 - (round_num * 0.01),  # Decreasing loss
                participating_clients=benchmark_config.get('num_clients', 10),
                convergence_rate=0.01,
                training_time=1.0 + __import__('random').random(),
                communication_overhead=1024 * (round_num + 1),
                memory_usage=128.0,
                algorithm_specific={}
            )
            round_metrics.append(metrics)
            self.coordinator.record_metrics(algorithm_id, metrics)
        
        # Create benchmark result
        result = BenchmarkResult(
            algorithm_type=algorithm_type,
            total_rounds=num_rounds,
            final_accuracy=0.95,  # Mock accuracy
            convergence_rounds=num_rounds // 2,
            avg_round_time=1.5,
            total_communication=sum(m.communication_overhead for m in round_metrics),
            client_fairness_score=0.8,
            robustness_score=0.9,
            round_metrics=round_metrics
        )
        
        return result


# Create global instances
algorithm_factory = FLAlgorithmFactory()
algorithm_coordinator = FLAlgorithmCoordinator()


def create_algorithm(algorithm_type: str, config: Optional[Dict[str, Any]] = None):
    """Convenience function to create an algorithm."""
    return algorithm_factory.create_algorithm(algorithm_type, config)


def get_available_algorithms() -> List[str]:
    """Get list of available algorithm types."""
    return algorithm_factory.get_available_algorithms()


def get_default_config(algorithm_type: str) -> Dict[str, Any]:
    """Get default configuration for an algorithm type."""
    return algorithm_factory.get_default_config(algorithm_type)


def register_algorithm(algorithm_id: str, algorithm_type: str, 
                      config: Optional[Dict[str, Any]] = None) -> str:
    """Register an algorithm instance."""
    return algorithm_coordinator.register_algorithm(algorithm_id, algorithm_type, config)


def get_algorithm(algorithm_id: str):
    """Get algorithm instance by ID."""
    return algorithm_coordinator.get_algorithm(algorithm_id)


def create_benchmark(benchmark_config: Dict[str, Any]) -> FLAlgorithmBenchmark:
    """Create a benchmark instance."""
    return FLAlgorithmBenchmark(algorithm_coordinator)


# Module exports
__all__ = [
    # Core classes
    'FLAlgorithmFactory',
    'FLAlgorithmCoordinator', 
    'FLAlgorithmBenchmark',
    
    # Enums
    'FLAlgorithmType',
    'AggregationStrategy',
    
    # Data classes
    'AlgorithmMetrics',
    'BenchmarkResult',
    
    # Algorithm classes
    'FedProxAlgorithm',
    'FedBNAlgorithm', 
    'PersonalizedFederatedLearning',
    'AdaptiveAggregationOrchestrator',
    
    # Config classes
    'FedProxConfig',
    'FedBNConfig',
    'PersonalizationConfig', 
    'AdaptiveAggregationConfig',
    
    # Global instances and convenience functions
    'algorithm_factory',
    'algorithm_coordinator',
    'create_algorithm',
    'get_available_algorithms',
    'get_default_config',
    'register_algorithm',
    'get_algorithm',
    'create_benchmark'
]


# Module initialization
logger.info("QFLARE FL Algorithms module initialized")
logger.info(f"Available algorithms: {get_available_algorithms()}")