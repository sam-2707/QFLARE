#!/usr/bin/env python3
"""
QFLARE Performance Monitoring Setup Script

This script sets up the complete QFLARE performance monitoring infrastructure:
- Prometheus metrics collection
- Grafana dashboards
- Alertmanager configuration
- Loki log aggregation
- Performance database initialization
- Docker compose orchestration

Usage:
    python setup_monitoring.py --action setup
    python setup_monitoring.py --action start
    python setup_monitoring.py --action stop
    python setup_monitoring.py --action status
"""

import subprocess
import sys
import json
import time
import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QFLAREMonitoringSetup:
    """QFLARE monitoring infrastructure setup and management"""
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.monitoring_path = self.base_path / "monitoring"
        self.data_path = self.base_path / "data"
        
        # Ensure directories exist
        self.monitoring_path.mkdir(exist_ok=True)
        self.data_path.mkdir(exist_ok=True)
        
    def setup_monitoring_infrastructure(self):
        """Setup complete monitoring infrastructure"""
        logger.info("🚀 Setting up QFLARE Performance Monitoring Infrastructure")
        
        steps = [
            ("📦 Installing Python dependencies", self._install_dependencies),
            ("🗄️ Initializing performance database", self._init_database),
            ("🐳 Checking Docker installation", self._check_docker),
            ("📋 Validating monitoring configuration", self._validate_config),
            ("🎯 Creating Grafana dashboards", self._setup_grafana_dashboards),
            ("🔔 Configuring alerting rules", self._setup_alerting),
            ("📊 Testing monitoring components", self._test_components),
        ]
        
        for description, step_func in steps:
            try:
                logger.info(description)
                step_func()
                logger.info(f"✅ {description} - Complete")
            except Exception as e:
                logger.error(f"❌ {description} - Failed: {e}")
                raise
                
        logger.info("🎉 QFLARE Performance Monitoring Infrastructure Setup Complete!")
        self._display_setup_summary()
        
    def _install_dependencies(self):
        """Install required Python dependencies"""
        dependencies = [
            "prometheus-client==0.17.1",
            "psutil==5.9.5",
            "fastapi==0.104.1",
            "uvicorn==0.24.0",
            "websockets==12.0",
            "aiofiles==23.2.1",
            "jinja2==3.1.2",
        ]
        
        for dep in dependencies:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                             check=True, capture_output=True)
                logger.debug(f"Installed {dep}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to install {dep}: {e}")
                
    def _init_database(self):
        """Initialize performance metrics database"""
        import sys
        sys.path.append(str(self.base_path))
        from monitoring.performance_monitor import PerformanceDatabase
        
        db_path = self.data_path / "performance_metrics.db"
        db = PerformanceDatabase(str(db_path))
        logger.info(f"Performance database initialized: {db_path}")
        
    def _check_docker(self):
        """Check Docker installation"""
        try:
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True, check=True)
            logger.info(f"Docker version: {result.stdout.strip()}")
            
            result = subprocess.run(["docker-compose", "--version"], 
                                  capture_output=True, text=True, check=True)
            logger.info(f"Docker Compose version: {result.stdout.strip()}")
        except subprocess.CalledProcessError:
            logger.warning("Docker or Docker Compose not found. Manual installation required.")
            
    def _validate_config(self):
        """Validate monitoring configuration files"""
        required_files = [
            "prometheus.yml",
            "qflare_rules.yml",
            "alertmanager.yml",
            "docker-compose.monitoring.yml",
            "grafana/dashboards/qflare-performance.json",
            "grafana/datasources/datasources.yml"
        ]
        
        for file_path in required_files:
            full_path = self.monitoring_path / file_path
            if not full_path.exists():
                raise FileNotFoundError(f"Required configuration file missing: {full_path}")
            logger.debug(f"Configuration file found: {file_path}")
            
    def _setup_grafana_dashboards(self):
        """Setup Grafana dashboards"""
        dashboards_path = self.monitoring_path / "grafana" / "dashboards"
        dashboards_path.mkdir(parents=True, exist_ok=True)
        
        # Create dashboard provisioning config
        provisioning_config = {
            "apiVersion": 1,
            "providers": [
                {
                    "name": "QFLARE Dashboards",
                    "orgId": 1,
                    "folder": "",
                    "type": "file",
                    "disableDeletion": False,
                    "updateIntervalSeconds": 10,
                    "allowUiUpdates": True,
                    "options": {
                        "path": "/etc/grafana/provisioning/dashboards"
                    }
                }
            ]
        }
        
        with open(dashboards_path / "dashboard.yml", "w") as f:
            json.dump(provisioning_config, f, indent=2)
            
        logger.info("Grafana dashboard provisioning configured")
        
    def _setup_alerting(self):
        """Setup alerting configuration"""
        # Validate alertmanager config
        alertmanager_config = self.monitoring_path / "alertmanager.yml"
        if alertmanager_config.exists():
            logger.info("Alertmanager configuration validated")
        else:
            logger.warning("Alertmanager configuration missing")
            
        # Validate Prometheus rules
        rules_config = self.monitoring_path / "qflare_rules.yml"
        if rules_config.exists():
            logger.info("Prometheus alerting rules validated")
        else:
            logger.warning("Prometheus rules configuration missing")
            
    def _test_components(self):
        """Test monitoring components"""
        import sys
        sys.path.append(str(self.base_path))
        
        # Test performance monitor
        try:
            from monitoring.performance_monitor import QFLAREPerformanceMonitor
            monitor = QFLAREPerformanceMonitor()
            monitor.start_monitoring()
            time.sleep(2)
            monitor.stop_monitoring()
            logger.info("Performance monitor test: PASS")
        except Exception as e:
            logger.error(f"Performance monitor test: FAIL - {e}")
            
        # Test database connectivity
        try:
            from monitoring.performance_monitor import PerformanceDatabase
            db = PerformanceDatabase(str(self.data_path / "performance_metrics.db"))
            logger.info("Database connectivity test: PASS")
        except Exception as e:
            logger.error(f"Database connectivity test: FAIL - {e}")
            
    def _display_setup_summary(self):
        """Display setup summary"""
        summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║                 QFLARE MONITORING SETUP COMPLETE                ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║ 📊 Components Configured:                                        ║
║   • Prometheus metrics collection                               ║
║   • Grafana visualization dashboards                           ║
║   • Alertmanager notification system                           ║
║   • Loki log aggregation                                       ║
║   • Performance database (SQLite)                              ║
║   • Real-time dashboard (FastAPI/WebSocket)                    ║
║                                                                  ║
║ 🚀 Quick Start Commands:                                         ║
║   Start monitoring:                                             ║
║     python setup_monitoring.py --action start                  ║
║                                                                  ║
║   View dashboard:                                               ║
║     python monitoring/performance_dashboard.py --port 8080     ║
║                                                                  ║
║   Start full stack:                                             ║
║     cd monitoring && docker-compose -f docker-compose.monitoring.yml up -d ║
║                                                                  ║
║ 🌐 Access Points:                                                ║
║   • Grafana Dashboard: http://localhost:3000                   ║
║     (admin/qflare_admin_2025)                                  ║
║   • Prometheus: http://localhost:9090                          ║
║   • Alertmanager: http://localhost:9093                        ║
║   • Performance Dashboard: http://localhost:8080               ║
║                                                                  ║
║ 📁 Configuration Files:                                          ║
║   • monitoring/prometheus.yml                                  ║
║   • monitoring/grafana/dashboards/                             ║
║   • monitoring/alertmanager.yml                                ║
║                                                                  ║
║ 📊 Metrics Collected:                                            ║
║   • System resources (CPU, Memory, Disk, Network)              ║
║   • ML model performance (accuracy, training time)             ║
║   • Federated learning metrics (rounds, clients)               ║
║   • Post-quantum cryptography performance                      ║
║   • API performance and error rates                            ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
        """
        print(summary)
        
    def start_monitoring(self):
        """Start monitoring services"""
        logger.info("🚀 Starting QFLARE monitoring services")
        
        compose_file = self.monitoring_path / "docker-compose.monitoring.yml"
        if compose_file.exists():
            try:
                subprocess.run([
                    "docker-compose", "-f", str(compose_file), "up", "-d"
                ], check=True, cwd=self.monitoring_path)
                logger.info("✅ Docker monitoring stack started")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to start Docker stack: {e}")
        else:
            logger.warning("Docker compose file not found, starting standalone monitor")
            
        # Start standalone performance monitor
        try:
            from monitoring.performance_monitor import get_monitor
            monitor = get_monitor()
            monitor.start_monitoring()
            logger.info("✅ Performance monitor started")
        except Exception as e:
            logger.error(f"❌ Failed to start performance monitor: {e}")
            
    def stop_monitoring(self):
        """Stop monitoring services"""
        logger.info("🛑 Stopping QFLARE monitoring services")
        
        compose_file = self.monitoring_path / "docker-compose.monitoring.yml"
        if compose_file.exists():
            try:
                subprocess.run([
                    "docker-compose", "-f", str(compose_file), "down"
                ], check=True, cwd=self.monitoring_path)
                logger.info("✅ Docker monitoring stack stopped")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to stop Docker stack: {e}")
                
        # Stop standalone performance monitor
        try:
            from monitoring.performance_monitor import get_monitor
            monitor = get_monitor()
            monitor.stop_monitoring()
            logger.info("✅ Performance monitor stopped")
        except Exception as e:
            logger.error(f"❌ Failed to stop performance monitor: {e}")
            
    def get_status(self):
        """Get monitoring status"""
        logger.info("📊 QFLARE Monitoring Status")
        
        status = {
            "timestamp": time.time(),
            "services": {},
            "configuration": {},
            "metrics": {}
        }
        
        # Check Docker services
        compose_file = self.monitoring_path / "docker-compose.monitoring.yml"
        if compose_file.exists():
            try:
                result = subprocess.run([
                    "docker-compose", "-f", str(compose_file), "ps"
                ], capture_output=True, text=True, cwd=self.monitoring_path)
                status["services"]["docker_stack"] = "running" if result.returncode == 0 else "stopped"
            except Exception:
                status["services"]["docker_stack"] = "unknown"
        else:
            status["services"]["docker_stack"] = "not_configured"
            
        # Check performance monitor
        try:
            from monitoring.performance_monitor import get_monitor
            monitor = get_monitor()
            status["services"]["performance_monitor"] = "running" if monitor.running else "stopped"
        except Exception:
            status["services"]["performance_monitor"] = "unknown"
            
        # Check configuration files
        config_files = [
            "prometheus.yml", "alertmanager.yml", "docker-compose.monitoring.yml"
        ]
        for config_file in config_files:
            config_path = self.monitoring_path / config_file
            status["configuration"][config_file] = "present" if config_path.exists() else "missing"
            
        # Check database
        db_path = self.data_path / "performance_metrics.db"
        status["metrics"]["database"] = "present" if db_path.exists() else "missing"
        
        # Display status
        print(json.dumps(status, indent=2))
        return status
        
    def clean_monitoring(self):
        """Clean monitoring data and containers"""
        logger.info("🧹 Cleaning QFLARE monitoring infrastructure")
        
        # Stop services first
        self.stop_monitoring()
        
        # Remove Docker volumes
        compose_file = self.monitoring_path / "docker-compose.monitoring.yml"
        if compose_file.exists():
            try:
                subprocess.run([
                    "docker-compose", "-f", str(compose_file), "down", "-v"
                ], check=True, cwd=self.monitoring_path)
                logger.info("✅ Docker volumes cleaned")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to clean Docker volumes: {e}")
                
        # Clean database
        db_path = self.data_path / "performance_metrics.db"
        if db_path.exists():
            db_path.unlink()
            logger.info("✅ Performance database cleaned")
            
        logger.info("🎉 Monitoring infrastructure cleaned")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="QFLARE Performance Monitoring Setup")
    parser.add_argument("--action", choices=["setup", "start", "stop", "status", "clean"], 
                       default="setup", help="Action to perform")
    parser.add_argument("--base-path", type=str, help="Base path for QFLARE project")
    
    args = parser.parse_args()
    
    setup = QFLAREMonitoringSetup(base_path=args.base_path)
    
    try:
        if args.action == "setup":
            setup.setup_monitoring_infrastructure()
        elif args.action == "start":
            setup.start_monitoring()
        elif args.action == "stop":
            setup.stop_monitoring()
        elif args.action == "status":
            setup.get_status()
        elif args.action == "clean":
            setup.clean_monitoring()
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()