#!/usr/bin/env python3
"""
QFLARE Performance Dashboard

A real-time performance monitoring dashboard for QFLARE that displays:
- System resource utilization
- ML model performance metrics
- Federated learning progress
- Post-quantum cryptography performance
- API performance statistics

Usage:
    python performance_dashboard.py
    python performance_dashboard.py --port 8080
    python performance_dashboard.py --refresh-interval 5
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
except ImportError:
    print("FastAPI not installed. Install with: pip install fastapi uvicorn")
    exit(1)

from performance_monitor import QFLAREPerformanceMonitor, get_monitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceDashboard:
    """Real-time performance dashboard"""
    
    def __init__(self, port: int = 8080, refresh_interval: int = 5):
        self.port = port
        self.refresh_interval = refresh_interval
        self.app = FastAPI(title="QFLARE Performance Dashboard")
        self.monitor = get_monitor()
        self.connected_clients: List[WebSocket] = []
        
        # Start monitoring
        if not self.monitor.running:
            self.monitor.start_monitoring()
            
        self._setup_routes()
        self._start_background_tasks()
        
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            """Main dashboard page"""
            return HTMLResponse(self._get_dashboard_html())
            
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get current performance metrics"""
            try:
                summary = self.monitor.get_performance_summary(hours=1)
                return JSONResponse(summary)
            except Exception as e:
                logger.error(f"Error getting metrics: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)
                
        @self.app.get("/api/prometheus")
        async def get_prometheus_metrics():
            """Get Prometheus formatted metrics"""
            try:
                metrics = self.monitor.get_prometheus_metrics()
                return Response(content=metrics, media_type="text/plain")
            except Exception as e:
                logger.error(f"Error getting Prometheus metrics: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)
                
        @self.app.get("/api/status")
        async def get_status():
            """Get monitoring status"""
            return JSONResponse({
                "status": "active" if self.monitor.running else "inactive",
                "monitoring_interval": self.monitor.monitoring_interval,
                "connected_clients": len(self.connected_clients),
                "timestamp": datetime.now().isoformat()
            })
            
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.connected_clients.append(websocket)
            logger.info(f"Client connected. Total clients: {len(self.connected_clients)}")
            
            try:
                while True:
                    # Send periodic updates
                    await asyncio.sleep(self.refresh_interval)
                    summary = self.monitor.get_performance_summary(hours=1)
                    await websocket.send_json({
                        "type": "metrics_update",
                        "data": summary,
                        "timestamp": datetime.now().isoformat()
                    })
            except WebSocketDisconnect:
                self.connected_clients.remove(websocket)
                logger.info(f"Client disconnected. Total clients: {len(self.connected_clients)}")
                
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        def monitor_loop():
            while True:
                try:
                    # Generate sample metrics for demonstration
                    self._generate_sample_metrics()
                    time.sleep(self.refresh_interval)
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(5)
                    
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        
    def _generate_sample_metrics(self):
        """Generate sample metrics for demonstration"""
        import random
        from performance_monitor import (
            MLPerformanceMetrics, FederatedLearningMetrics, 
            CryptographyMetrics, APIMetrics
        )
        
        current_time = time.time()
        
        # Sample ML metrics
        ml_metrics = MLPerformanceMetrics(
            timestamp=current_time,
            model_name="QFLARE-CNN",
            training_time_seconds=random.uniform(10, 60),
            inference_time_ms=random.uniform(5, 50),
            accuracy=random.uniform(0.85, 0.95),
            loss=random.uniform(0.05, 0.3),
            memory_usage_mb=random.uniform(512, 2048),
            gpu_utilization_percent=random.uniform(70, 95),
            batch_size=32,
            epoch=random.randint(1, 100)
        )
        self.monitor.record_ml_metrics(ml_metrics)
        
        # Sample FL metrics
        fl_metrics = FederatedLearningMetrics(
            timestamp=current_time,
            round_number=random.randint(1, 50),
            num_clients=random.randint(10, 100),
            participating_clients=random.randint(8, 95),
            aggregation_time_seconds=random.uniform(5, 30),
            communication_overhead_mb=random.uniform(10, 100),
            convergence_rate=random.uniform(0.01, 0.1),
            global_accuracy=random.uniform(0.8, 0.92),
            client_dropout_rate=random.uniform(0.05, 0.2)
        )
        self.monitor.record_fl_metrics(fl_metrics)
        
        # Sample crypto metrics
        crypto_metrics = CryptographyMetrics(
            timestamp=current_time,
            algorithm="CRYSTALS-Kyber-1024",
            operation="encrypt",
            duration_ms=random.uniform(1, 10),
            key_size_bytes=1568,
            data_size_bytes=random.randint(1024, 8192),
            throughput_mbps=random.uniform(50, 200)
        )
        self.monitor.record_crypto_metrics(crypto_metrics)
        
        # Sample API metrics
        api_metrics = APIMetrics(
            timestamp=current_time,
            endpoint="/api/v1/federated/train",
            method="POST",
            status_code=200,
            response_time_ms=random.uniform(100, 1000),
            request_size_bytes=random.randint(1024, 10240),
            response_size_bytes=random.randint(512, 5120),
            concurrent_requests=random.randint(1, 20)
        )
        self.monitor.record_api_metrics(api_metrics)
        
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QFLARE Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }
        
        .header {
            background: rgba(0,0,0,0.2);
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        
        .header h1 {
            margin: 0;
            color: #00d4ff;
            font-size: 2.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .status {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            background: #00ff88;
            color: #000;
            font-weight: bold;
            margin-left: 1rem;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card h3 {
            margin: 0 0 1rem 0;
            color: #00d4ff;
            font-size: 1.3rem;
            border-bottom: 2px solid #00d4ff;
            padding-bottom: 0.5rem;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 0.8rem 0;
            padding: 0.5rem;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
        }
        
        .metric-label {
            font-weight: 500;
        }
        
        .metric-value {
            font-weight: bold;
            font-size: 1.1rem;
        }
        
        .metric-value.good { color: #00ff88; }
        .metric-value.warning { color: #ffaa00; }
        .metric-value.critical { color: #ff4444; }
        
        .chart-container {
            height: 300px;
            margin-top: 1rem;
        }
        
        .last-updated {
            text-align: center;
            padding: 1rem;
            color: rgba(255,255,255,0.7);
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 QFLARE Performance Dashboard</h1>
        <span class="status" id="status">MONITORING ACTIVE</span>
    </div>
    
    <div class="dashboard">
        <!-- System Metrics -->
        <div class="card">
            <h3>🖥️ System Resources</h3>
            <div class="metric">
                <span class="metric-label">CPU Usage</span>
                <span class="metric-value" id="cpu-usage">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">Memory Usage</span>
                <span class="metric-value" id="memory-usage">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">Disk Usage</span>
                <span class="metric-value" id="disk-usage">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">Load Average</span>
                <span class="metric-value" id="load-avg">--</span>
            </div>
        </div>
        
        <!-- ML Performance -->
        <div class="card">
            <h3>🧠 ML Performance</h3>
            <div class="metric">
                <span class="metric-label">Model Accuracy</span>
                <span class="metric-value" id="model-accuracy">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">Training Time</span>
                <span class="metric-value" id="training-time">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">Inference Time</span>
                <span class="metric-value" id="inference-time">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">GPU Utilization</span>
                <span class="metric-value" id="gpu-usage">--</span>
            </div>
        </div>
        
        <!-- Federated Learning -->
        <div class="card">
            <h3>🌐 Federated Learning</h3>
            <div class="metric">
                <span class="metric-label">Current Round</span>
                <span class="metric-value" id="fl-round">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">Active Clients</span>
                <span class="metric-value" id="fl-clients">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">Global Accuracy</span>
                <span class="metric-value" id="fl-accuracy">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">Dropout Rate</span>
                <span class="metric-value" id="fl-dropout">--</span>
            </div>
        </div>
        
        <!-- Cryptography Performance -->
        <div class="card">
            <h3>🔐 Post-Quantum Crypto</h3>
            <div class="metric">
                <span class="metric-label">Kyber Encryption</span>
                <span class="metric-value" id="kyber-time">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">Dilithium Signing</span>
                <span class="metric-value" id="dilithium-time">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">Crypto Throughput</span>
                <span class="metric-value" id="crypto-throughput">--</span>
            </div>
        </div>
        
        <!-- Performance Chart -->
        <div class="card" style="grid-column: 1 / -1;">
            <h3>📊 Performance Trends</h3>
            <div class="chart-container">
                <canvas id="performanceChart"></canvas>
            </div>
        </div>
    </div>
    
    <div class="last-updated">
        Last updated: <span id="last-updated">--</span>
    </div>
    
    <script>
        // WebSocket connection
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        // Chart setup
        const ctx = document.getElementById('performanceChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'CPU %',
                        data: [],
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0, 212, 255, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Memory %',
                        data: [],
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'ML Accuracy',
                        data: [],
                        borderColor: '#ffaa00',
                        backgroundColor: 'rgba(255, 170, 0, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: 'white'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: 'white' },
                        grid: { color: 'rgba(255,255,255,0.1)' }
                    },
                    y: {
                        ticks: { color: 'white' },
                        grid: { color: 'rgba(255,255,255,0.1)' }
                    }
                }
            }
        });
        
        function updateMetrics(data) {
            const now = new Date().toLocaleTimeString();
            
            // System metrics
            if (data.system_metrics && data.system_metrics.length > 0) {
                const latest = data.system_metrics[0];
                updateElement('cpu-usage', `${latest.cpu_percent.toFixed(1)}%`, latest.cpu_percent);
                updateElement('memory-usage', `${latest.memory_percent.toFixed(1)}%`, latest.memory_percent);
                updateElement('disk-usage', `${latest.disk_usage_percent.toFixed(1)}%`, latest.disk_usage_percent);
                updateElement('load-avg', latest.load_avg_1m.toFixed(2), latest.load_avg_1m);
            }
            
            // ML metrics
            if (data.ml_metrics && data.ml_metrics.length > 0) {
                const latest = data.ml_metrics[0];
                updateElement('model-accuracy', `${(latest.accuracy * 100).toFixed(1)}%`, latest.accuracy * 100);
                updateElement('training-time', `${latest.training_time_seconds.toFixed(1)}s`, latest.training_time_seconds);
                updateElement('inference-time', `${latest.inference_time_ms.toFixed(1)}ms`, latest.inference_time_ms);
                if (latest.gpu_utilization_percent) {
                    updateElement('gpu-usage', `${latest.gpu_utilization_percent.toFixed(1)}%`, latest.gpu_utilization_percent);
                }
            }
            
            // FL metrics
            if (data.fl_metrics && data.fl_metrics.length > 0) {
                const latest = data.fl_metrics[0];
                document.getElementById('fl-round').textContent = latest.round_number;
                document.getElementById('fl-clients').textContent = `${latest.participating_clients}/${latest.num_clients}`;
                updateElement('fl-accuracy', `${(latest.global_accuracy * 100).toFixed(1)}%`, latest.global_accuracy * 100);
                updateElement('fl-dropout', `${(latest.client_dropout_rate * 100).toFixed(1)}%`, latest.client_dropout_rate * 100);
            }
            
            // Crypto metrics
            if (data.crypto_metrics && data.crypto_metrics.length > 0) {
                const kyber = data.crypto_metrics.find(m => m.algorithm.includes('Kyber'));
                const dilithium = data.crypto_metrics.find(m => m.algorithm.includes('Dilithium'));
                
                if (kyber) {
                    updateElement('kyber-time', `${kyber.duration_ms.toFixed(1)}ms`, kyber.duration_ms);
                }
                if (dilithium) {
                    updateElement('dilithium-time', `${dilithium.duration_ms.toFixed(1)}ms`, dilithium.duration_ms);
                }
                
                const latest = data.crypto_metrics[0];
                updateElement('crypto-throughput', `${latest.throughput_mbps.toFixed(1)} Mbps`, latest.throughput_mbps);
            }
            
            // Update chart
            updateChart(data);
            
            document.getElementById('last-updated').textContent = now;
        }
        
        function updateElement(id, text, value) {
            const element = document.getElementById(id);
            element.textContent = text;
            
            // Color coding
            element.className = 'metric-value';
            if (id.includes('usage') || id.includes('time') || id.includes('dropout')) {
                if (value < 50) element.classList.add('good');
                else if (value < 80) element.classList.add('warning');
                else element.classList.add('critical');
            } else if (id.includes('accuracy')) {
                if (value > 85) element.classList.add('good');
                else if (value > 70) element.classList.add('warning');
                else element.classList.add('critical');
            }
        }
        
        function updateChart(data) {
            const now = new Date().toLocaleTimeString();
            
            // Limit chart data points
            if (chart.data.labels.length > 20) {
                chart.data.labels.shift();
                chart.data.datasets.forEach(dataset => dataset.data.shift());
            }
            
            chart.data.labels.push(now);
            
            // Add data points
            if (data.system_metrics && data.system_metrics.length > 0) {
                const latest = data.system_metrics[0];
                chart.data.datasets[0].data.push(latest.cpu_percent);
                chart.data.datasets[1].data.push(latest.memory_percent);
            }
            
            if (data.ml_metrics && data.ml_metrics.length > 0) {
                const latest = data.ml_metrics[0];
                chart.data.datasets[2].data.push(latest.accuracy * 100);
            }
            
            chart.update();
        }
        
        // WebSocket event handlers
        ws.onmessage = function(event) {
            const message = JSON.parse(event.data);
            if (message.type === 'metrics_update') {
                updateMetrics(message.data);
            }
        };
        
        ws.onopen = function() {
            console.log('Connected to QFLARE Performance Dashboard');
            document.getElementById('status').textContent = 'MONITORING ACTIVE';
        };
        
        ws.onclose = function() {
            console.log('Disconnected from dashboard');
            document.getElementById('status').textContent = 'DISCONNECTED';
            document.getElementById('status').style.backgroundColor = '#ff4444';
        };
        
        // Initial data load
        fetch('/api/metrics')
            .then(response => response.json())
            .then(data => updateMetrics(data))
            .catch(error => console.error('Error loading initial data:', error));
    </script>
</body>
</html>
        """
        
    def run(self):
        """Run the dashboard server"""
        logger.info(f"Starting QFLARE Performance Dashboard on port {self.port}")
        uvicorn.run(self.app, host="0.0.0.0", port=self.port)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="QFLARE Performance Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Dashboard port")
    parser.add_argument("--refresh-interval", type=int, default=5, help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    dashboard = PerformanceDashboard(port=args.port, refresh_interval=args.refresh_interval)
    dashboard.run()

if __name__ == "__main__":
    main()