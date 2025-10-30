#!/usr/bin/env python3
"""
QFLARE Performance Monitoring System

This module provides comprehensive performance monitoring for QFLARE including:
- System resource monitoring (CPU, Memory, Disk, Network)
- ML model performance tracking (training time, accuracy, convergence)
- Federated learning metrics (aggregation time, client participation)
- Post-quantum cryptography performance (encryption/decryption latency)
- API performance metrics (request latency, throughput)
- Database performance monitoring
- Custom QFLARE-specific KPIs

Integration with Prometheus for metrics collection and Grafana for visualization.
"""

import psutil
import time
import threading
import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import numpy as np
from prometheus_client import (
    CollectorRegistry, Gauge, Counter, Histogram, 
    Summary, generate_latest, CONTENT_TYPE_LATEST
)
from contextlib import contextmanager
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float
    load_avg_1m: float
    load_avg_5m: float
    load_avg_15m: float

@dataclass 
class MLPerformanceMetrics:
    """Machine Learning performance metrics"""
    timestamp: float
    model_name: str
    training_time_seconds: float
    inference_time_ms: float
    accuracy: float
    loss: float
    memory_usage_mb: float
    gpu_utilization_percent: Optional[float]
    batch_size: int
    epoch: int

@dataclass
class FederatedLearningMetrics:
    """Federated learning specific metrics"""
    timestamp: float
    round_number: int
    num_clients: int
    participating_clients: int
    aggregation_time_seconds: float
    communication_overhead_mb: float
    convergence_rate: float
    global_accuracy: float
    client_dropout_rate: float

@dataclass
class CryptographyMetrics:
    """Post-quantum cryptography performance metrics"""
    timestamp: float
    algorithm: str
    operation: str  # encrypt, decrypt, sign, verify
    duration_ms: float
    key_size_bytes: int
    data_size_bytes: int
    throughput_mbps: float

@dataclass
class APIMetrics:
    """API performance metrics"""
    timestamp: float
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    request_size_bytes: int
    response_size_bytes: int
    concurrent_requests: int

class PrometheusMetricsCollector:
    """Collects and exposes metrics in Prometheus format"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self._setup_metrics()
        
    def _setup_metrics(self):
        """Initialize Prometheus metrics"""
        # System Metrics
        self.cpu_usage = Gauge('qflare_cpu_usage_percent', 'CPU usage percentage', registry=self.registry)
        self.memory_usage = Gauge('qflare_memory_usage_percent', 'Memory usage percentage', registry=self.registry)
        self.memory_available = Gauge('qflare_memory_available_gb', 'Available memory in GB', registry=self.registry)
        self.disk_usage = Gauge('qflare_disk_usage_percent', 'Disk usage percentage', registry=self.registry)
        self.disk_free = Gauge('qflare_disk_free_gb', 'Free disk space in GB', registry=self.registry)
        self.network_sent = Counter('qflare_network_sent_bytes_total', 'Network bytes sent', registry=self.registry)
        self.network_recv = Counter('qflare_network_recv_bytes_total', 'Network bytes received', registry=self.registry)
        self.load_avg = Gauge('qflare_load_average', 'System load average', ['period'], registry=self.registry)
        
        # ML Performance Metrics
        self.training_time = Histogram('qflare_ml_training_duration_seconds', 'Model training time', 
                                     ['model_name'], registry=self.registry)
        self.inference_time = Histogram('qflare_ml_inference_duration_seconds', 'Model inference time',
                                      ['model_name'], registry=self.registry)
        self.model_accuracy = Gauge('qflare_ml_model_accuracy', 'Model accuracy', ['model_name'], registry=self.registry)
        self.model_loss = Gauge('qflare_ml_model_loss', 'Model loss', ['model_name'], registry=self.registry)
        self.ml_memory_usage = Gauge('qflare_ml_memory_usage_mb', 'ML memory usage in MB', 
                                   ['model_name'], registry=self.registry)
        self.gpu_utilization = Gauge('qflare_gpu_utilization_percent', 'GPU utilization percentage', registry=self.registry)
        
        # Federated Learning Metrics
        self.fl_round = Gauge('qflare_fl_current_round', 'Current federated learning round', registry=self.registry)
        self.fl_clients_total = Gauge('qflare_fl_clients_total', 'Total number of FL clients', registry=self.registry)
        self.fl_clients_participating = Gauge('qflare_fl_clients_participating', 'Participating FL clients', registry=self.registry)
        self.fl_aggregation_time = Histogram('qflare_fl_aggregation_duration_seconds', 'FL aggregation time', registry=self.registry)
        self.fl_communication_overhead = Gauge('qflare_fl_communication_overhead_mb', 'FL communication overhead in MB', registry=self.registry)
        self.fl_global_accuracy = Gauge('qflare_fl_global_accuracy', 'Global model accuracy', registry=self.registry)
        self.fl_dropout_rate = Gauge('qflare_fl_client_dropout_rate', 'Client dropout rate', registry=self.registry)
        
        # Cryptography Metrics
        self.crypto_duration = Histogram('qflare_crypto_operation_duration_seconds', 'Cryptographic operation time',
                                       ['algorithm', 'operation'], registry=self.registry)
        self.crypto_throughput = Gauge('qflare_crypto_throughput_mbps', 'Cryptographic throughput in Mbps',
                                     ['algorithm', 'operation'], registry=self.registry)
        
        # API Metrics
        self.api_requests_total = Counter('qflare_api_requests_total', 'Total API requests',
                                        ['endpoint', 'method', 'status'], registry=self.registry)
        self.api_request_duration = Histogram('qflare_api_request_duration_seconds', 'API request duration',
                                            ['endpoint', 'method'], registry=self.registry)
        self.api_concurrent_requests = Gauge('qflare_api_concurrent_requests', 'Concurrent API requests', registry=self.registry)
        
    def update_system_metrics(self, metrics: SystemMetrics):
        """Update system metrics"""
        self.cpu_usage.set(metrics.cpu_percent)
        self.memory_usage.set(metrics.memory_percent)
        self.memory_available.set(metrics.memory_available_gb)
        self.disk_usage.set(metrics.disk_usage_percent)
        self.disk_free.set(metrics.disk_free_gb)
        self.network_sent.inc(metrics.network_sent_mb * 1024 * 1024)  # Convert to bytes
        self.network_recv.inc(metrics.network_recv_mb * 1024 * 1024)
        self.load_avg.labels(period='1m').set(metrics.load_avg_1m)
        self.load_avg.labels(period='5m').set(metrics.load_avg_5m)
        self.load_avg.labels(period='15m').set(metrics.load_avg_15m)
        
    def update_ml_metrics(self, metrics: MLPerformanceMetrics):
        """Update ML performance metrics"""
        self.training_time.labels(model_name=metrics.model_name).observe(metrics.training_time_seconds)
        self.inference_time.labels(model_name=metrics.model_name).observe(metrics.inference_time_ms / 1000)
        self.model_accuracy.labels(model_name=metrics.model_name).set(metrics.accuracy)
        self.model_loss.labels(model_name=metrics.model_name).set(metrics.loss)
        self.ml_memory_usage.labels(model_name=metrics.model_name).set(metrics.memory_usage_mb)
        if metrics.gpu_utilization_percent is not None:
            self.gpu_utilization.set(metrics.gpu_utilization_percent)
            
    def update_fl_metrics(self, metrics: FederatedLearningMetrics):
        """Update federated learning metrics"""
        self.fl_round.set(metrics.round_number)
        self.fl_clients_total.set(metrics.num_clients)
        self.fl_clients_participating.set(metrics.participating_clients)
        self.fl_aggregation_time.observe(metrics.aggregation_time_seconds)
        self.fl_communication_overhead.set(metrics.communication_overhead_mb)
        self.fl_global_accuracy.set(metrics.global_accuracy)
        self.fl_dropout_rate.set(metrics.client_dropout_rate)
        
    def update_crypto_metrics(self, metrics: CryptographyMetrics):
        """Update cryptography metrics"""
        duration_seconds = metrics.duration_ms / 1000
        self.crypto_duration.labels(algorithm=metrics.algorithm, operation=metrics.operation).observe(duration_seconds)
        self.crypto_throughput.labels(algorithm=metrics.algorithm, operation=metrics.operation).set(metrics.throughput_mbps)
        
    def update_api_metrics(self, metrics: APIMetrics):
        """Update API metrics"""
        self.api_requests_total.labels(
            endpoint=metrics.endpoint,
            method=metrics.method,
            status=str(metrics.status_code)
        ).inc()
        self.api_request_duration.labels(
            endpoint=metrics.endpoint,
            method=metrics.method
        ).observe(metrics.response_time_ms / 1000)
        self.api_concurrent_requests.set(metrics.concurrent_requests)
        
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry)

class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self):
        self.network_io_baseline = psutil.net_io_counters()
        
    def get_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network IO
        network_io = psutil.net_io_counters()
        network_sent_mb = (network_io.bytes_sent - self.network_io_baseline.bytes_sent) / (1024 * 1024)
        network_recv_mb = (network_io.bytes_recv - self.network_io_baseline.bytes_recv) / (1024 * 1024)
        
        # Load average (Unix-like systems)
        try:
            load_avg = psutil.getloadavg()
        except AttributeError:
            # Windows doesn't have load average
            load_avg = (0.0, 0.0, 0.0)
            
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk.percent,
            disk_free_gb=disk.free / (1024**3),
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            load_avg_1m=load_avg[0],
            load_avg_5m=load_avg[1],
            load_avg_15m=load_avg[2]
        )

class MLPerformanceTracker:
    """ML model performance tracking"""
    
    @contextmanager
    def track_training(self, model_name: str):
        """Context manager for tracking training performance"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        yield
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        training_time = end_time - start_time
        memory_usage = max(end_memory - start_memory, 0)
        
        logger.info(f"Training completed for {model_name}: {training_time:.2f}s, {memory_usage:.2f}MB")
        
    @contextmanager
    def track_inference(self, model_name: str):
        """Context manager for tracking inference performance"""
        start_time = time.time()
        
        yield
        
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        logger.info(f"Inference completed for {model_name}: {inference_time:.2f}ms")
        
    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
        
    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization if available"""
        try:
            if torch.cuda.is_available():
                # This is a simplified version - in practice you'd use nvidia-ml-py
                return float(torch.cuda.utilization())
        except Exception:
            pass
        return None

class PerformanceDatabase:
    """SQLite database for storing performance metrics"""
    
    def __init__(self, db_path: str = "data/performance_metrics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_available_gb REAL,
                    disk_usage_percent REAL,
                    disk_free_gb REAL,
                    network_sent_mb REAL,
                    network_recv_mb REAL,
                    load_avg_1m REAL,
                    load_avg_5m REAL,
                    load_avg_15m REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    model_name TEXT,
                    training_time_seconds REAL,
                    inference_time_ms REAL,
                    accuracy REAL,
                    loss REAL,
                    memory_usage_mb REAL,
                    gpu_utilization_percent REAL,
                    batch_size INTEGER,
                    epoch INTEGER
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fl_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    round_number INTEGER,
                    num_clients INTEGER,
                    participating_clients INTEGER,
                    aggregation_time_seconds REAL,
                    communication_overhead_mb REAL,
                    convergence_rate REAL,
                    global_accuracy REAL,
                    client_dropout_rate REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS crypto_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    algorithm TEXT,
                    operation TEXT,
                    duration_ms REAL,
                    key_size_bytes INTEGER,
                    data_size_bytes INTEGER,
                    throughput_mbps REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    endpoint TEXT,
                    method TEXT,
                    status_code INTEGER,
                    response_time_ms REAL,
                    request_size_bytes INTEGER,
                    response_size_bytes INTEGER,
                    concurrent_requests INTEGER
                )
            """)
            
    def store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO system_metrics 
                (timestamp, cpu_percent, memory_percent, memory_available_gb, 
                 disk_usage_percent, disk_free_gb, network_sent_mb, network_recv_mb,
                 load_avg_1m, load_avg_5m, load_avg_15m)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp, metrics.cpu_percent, metrics.memory_percent,
                metrics.memory_available_gb, metrics.disk_usage_percent, metrics.disk_free_gb,
                metrics.network_sent_mb, metrics.network_recv_mb,
                metrics.load_avg_1m, metrics.load_avg_5m, metrics.load_avg_15m
            ))
            
    def store_ml_metrics(self, metrics: MLPerformanceMetrics):
        """Store ML performance metrics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO ml_metrics
                (timestamp, model_name, training_time_seconds, inference_time_ms,
                 accuracy, loss, memory_usage_mb, gpu_utilization_percent, batch_size, epoch)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp, metrics.model_name, metrics.training_time_seconds,
                metrics.inference_time_ms, metrics.accuracy, metrics.loss,
                metrics.memory_usage_mb, metrics.gpu_utilization_percent,
                metrics.batch_size, metrics.epoch
            ))
            
    def store_fl_metrics(self, metrics: FederatedLearningMetrics):
        """Store federated learning metrics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO fl_metrics
                (timestamp, round_number, num_clients, participating_clients,
                 aggregation_time_seconds, communication_overhead_mb, convergence_rate,
                 global_accuracy, client_dropout_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp, metrics.round_number, metrics.num_clients,
                metrics.participating_clients, metrics.aggregation_time_seconds,
                metrics.communication_overhead_mb, metrics.convergence_rate,
                metrics.global_accuracy, metrics.client_dropout_rate
            ))
            
    def store_crypto_metrics(self, metrics: CryptographyMetrics):
        """Store cryptography metrics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO crypto_metrics
                (timestamp, algorithm, operation, duration_ms, key_size_bytes,
                 data_size_bytes, throughput_mbps)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp, metrics.algorithm, metrics.operation,
                metrics.duration_ms, metrics.key_size_bytes, metrics.data_size_bytes,
                metrics.throughput_mbps
            ))
            
    def store_api_metrics(self, metrics: APIMetrics):
        """Store API metrics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO api_metrics
                (timestamp, endpoint, method, status_code, response_time_ms,
                 request_size_bytes, response_size_bytes, concurrent_requests)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp, metrics.endpoint, metrics.method,
                metrics.status_code, metrics.response_time_ms,
                metrics.request_size_bytes, metrics.response_size_bytes,
                metrics.concurrent_requests
            ))
            
    def get_recent_metrics(self, metric_type: str, hours: int = 24) -> List[Dict]:
        """Get recent metrics of specified type"""
        since_timestamp = time.time() - (hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(f"""
                SELECT * FROM {metric_type}_metrics 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            """, (since_timestamp,))
            
            return [dict(row) for row in cursor.fetchall()]

class QFLAREPerformanceMonitor:
    """Main performance monitoring coordinator"""
    
    def __init__(self, monitoring_interval: int = 30):
        self.monitoring_interval = monitoring_interval
        self.running = False
        
        # Initialize components
        self.system_monitor = SystemMonitor()
        self.ml_tracker = MLPerformanceTracker()
        self.prometheus_collector = PrometheusMetricsCollector()
        self.database = PerformanceDatabase()
        
        # Monitoring thread
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start continuous monitoring"""
        if self.running:
            logger.warning("Monitoring already running")
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Performance monitoring started (interval: {self.monitoring_interval}s)")
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect system metrics
                system_metrics = self.system_monitor.get_system_metrics()
                self.prometheus_collector.update_system_metrics(system_metrics)
                self.database.store_system_metrics(system_metrics)
                
                logger.debug(f"Collected system metrics: CPU {system_metrics.cpu_percent:.1f}%, "
                           f"Memory {system_metrics.memory_percent:.1f}%")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                
            time.sleep(self.monitoring_interval)
            
    def record_ml_metrics(self, metrics: MLPerformanceMetrics):
        """Record ML performance metrics"""
        self.prometheus_collector.update_ml_metrics(metrics)
        self.database.store_ml_metrics(metrics)
        logger.info(f"Recorded ML metrics for {metrics.model_name}: "
                   f"accuracy={metrics.accuracy:.3f}, loss={metrics.loss:.3f}")
        
    def record_fl_metrics(self, metrics: FederatedLearningMetrics):
        """Record federated learning metrics"""
        self.prometheus_collector.update_fl_metrics(metrics)
        self.database.store_fl_metrics(metrics)
        logger.info(f"Recorded FL metrics: round={metrics.round_number}, "
                   f"clients={metrics.participating_clients}/{metrics.num_clients}")
        
    def record_crypto_metrics(self, metrics: CryptographyMetrics):
        """Record cryptography metrics"""
        self.prometheus_collector.update_crypto_metrics(metrics)
        self.database.store_crypto_metrics(metrics)
        logger.info(f"Recorded crypto metrics: {metrics.algorithm} {metrics.operation} "
                   f"took {metrics.duration_ms:.2f}ms")
        
    def record_api_metrics(self, metrics: APIMetrics):
        """Record API metrics"""
        self.prometheus_collector.update_api_metrics(metrics)
        self.database.store_api_metrics(metrics)
        
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return self.prometheus_collector.get_metrics()
        
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'period_hours': hours,
            'system_metrics': self.database.get_recent_metrics('system', hours),
            'ml_metrics': self.database.get_recent_metrics('ml', hours),
            'fl_metrics': self.database.get_recent_metrics('fl', hours),
            'crypto_metrics': self.database.get_recent_metrics('crypto', hours),
            'api_metrics': self.database.get_recent_metrics('api', hours)
        }
        
        return summary
        
    def generate_performance_report(self, output_file: str = "performance_report.json"):
        """Generate comprehensive performance report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'monitoring_status': 'active' if self.running else 'inactive',
            'monitoring_interval_seconds': self.monitoring_interval,
            'summary_24h': self.get_performance_summary(24),
            'summary_7d': self.get_performance_summary(24 * 7)
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Performance report generated: {output_file}")
        return report

# Global monitor instance
monitor = QFLAREPerformanceMonitor()

def get_monitor() -> QFLAREPerformanceMonitor:
    """Get the global performance monitor instance"""
    return monitor

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QFLARE Performance Monitoring System")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    parser.add_argument("--report", action="store_true", help="Generate performance report and exit")
    parser.add_argument("--output", type=str, default="performance_report.json", help="Output file for report")
    
    args = parser.parse_args()
    
    if args.report:
        # Generate report only
        monitor = QFLAREPerformanceMonitor()
        monitor.generate_performance_report(args.output)
    else:
        # Start continuous monitoring
        monitor = QFLAREPerformanceMonitor(monitoring_interval=args.interval)
        try:
            monitor.start_monitoring()
            print(f"Performance monitoring started. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping performance monitoring...")
            monitor.stop_monitoring()