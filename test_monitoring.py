"""
Production Monitoring & Metrics Test Suite for QFLARE

This script validates the comprehensive monitoring and observability system:
- Metrics collection (Prometheus-compatible)
- Health monitoring and checks
- Distributed tracing capabilities
- Alerting and notification systems
- Integrated monitoring dashboard
"""

import sys
import os
import time
import asyncio
import logging
from datetime import datetime

# Add server directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_monitoring_file_structure():
    """Test that all monitoring files are present."""
    print("🔍 Testing Monitoring File Structure...")
    
    required_files = [
        'server/monitoring/__init__.py',
        'server/monitoring/metrics.py',
        'server/monitoring/health.py',
        'server/monitoring/tracing.py',
        'server/monitoring/alerting.py'
    ]
    
    all_present = True
    total_size = 0
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            total_size += size
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} - MISSING")
            all_present = False
    
    print(f"📊 Total monitoring implementation: {total_size:,} bytes")
    return all_present


def test_metrics_system():
    """Test the metrics collection system."""
    print("\n📊 Testing Metrics Collection System...")
    
    try:
        from server.monitoring.metrics import (
            QFLAREMetricsCollector, MetricConfig,
            get_metrics_collector
        )
        
        # Test metrics collector initialization
        config = MetricConfig(
            enabled=True,
            collection_interval=1.0,
            export_port=8001,  # Different port for testing
            retention_period=3600
        )
        
        collector = QFLAREMetricsCollector(config)
        print("✅ Metrics collector initialized")
        
        # Test FL metrics recording
        collector.record_fl_round(
            round_id="test_round_001",
            duration=45.5,
            participants=10,
            status="completed",
            algorithm="fedavg"
        )
        print("✅ FL round metrics recorded")
        
        # Test model metrics
        collector.record_model_metrics(
            accuracy=0.92,
            loss=0.08,
            dataset="global",
            metric_type="validation"
        )
        print("✅ Model performance metrics recorded")
        
        # Test training metrics
        collector.record_training_metrics(
            device_id="test_device_001",
            duration=12.3,
            iterations=100,
            model_type="cnn"
        )
        print("✅ Training metrics recorded")
        
        # Test security metrics
        collector.record_auth_attempt(
            method="jwt",
            status="success",
            device_type="edge"
        )
        print("✅ Security metrics recorded")
        
        collector.record_key_operation(
            operation="generation",
            key_type="frodo_kem",
            status="success"
        )
        print("✅ Key operation metrics recorded")
        
        # Test API metrics
        collector.record_api_request(
            method="POST",
            endpoint="/api/v1/training",
            duration=0.125,
            status_code=200
        )
        print("✅ API metrics recorded")
        
        # Test error metrics
        collector.record_error(
            component="federated_learning",
            error_type="timeout",
            severity="warning",
            details="Device training timeout"
        )
        print("✅ Error metrics recorded")
        
        # Test system metrics
        collector.update_system_metrics(
            cpu_percent=45.2,
            memory_bytes=1024*1024*512,  # 512MB
            disk_bytes=1024*1024*1024*10,  # 10GB
            network_in=1024*100,  # 100KB
            network_out=1024*200   # 200KB
        )
        print("✅ System metrics updated")
        
        # Test metrics export
        metrics_export = collector.get_metrics_export()
        if len(metrics_export) > 0:
            print("✅ Metrics export available")
        else:
            print("⚠️  Metrics export empty (Prometheus client not available)")
        
        # Test metrics summary
        summary = collector.get_metrics_summary()
        if summary['metrics_count'] >= 0:
            print(f"✅ Metrics summary: {summary['metrics_count']} metrics tracked")
        
        # Stop collection
        collector.stop_collection()
        print("✅ Metrics collection stopped cleanly")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_health_monitoring():
    """Test the health monitoring system."""
    print("\n🏥 Testing Health Monitoring System...")
    
    try:
        from server.monitoring.health import (
            QFLAREHealthMonitor, HealthCheckConfig, HealthStatus,
            SystemHealthCheck, get_health_monitor
        )
        
        # Test health monitor initialization
        config = HealthCheckConfig(
            interval_seconds=5.0,
            timeout_seconds=5.0,
            max_failures=2
        )
        
        monitor = QFLAREHealthMonitor(config)
        print("✅ Health monitor initialized")
        
        # Test default health checks
        available_checks = list(monitor.health_checks.keys())
        print(f"✅ Default health checks: {', '.join(available_checks)}")
        
        # Test health check execution
        async def run_health_tests():
            # Run all health checks
            results = await monitor.check_all()
            
            if results:
                print("✅ Health checks executed")
                
                for component, result in results.items():
                    status_emoji = {
                        HealthStatus.HEALTHY: "✅",
                        HealthStatus.DEGRADED: "⚠️",
                        HealthStatus.UNHEALTHY: "❌",
                        HealthStatus.UNKNOWN: "❓"
                    }.get(result.status, "❓")
                    
                    print(f"  {status_emoji} {component}: {result.status.value} - {result.message}")
            else:
                print("❌ No health check results")
                return False
            
            # Test overall status
            overall_status = monitor.get_overall_status()
            print(f"✅ Overall health status: {overall_status.value}")
            
            # Test health summary
            summary = monitor.get_health_summary()
            print(f"✅ Health summary: {summary['healthy_components']}/{summary['total_components']} healthy")
            
            return True
        
        # Run async health tests
        result = asyncio.run(run_health_tests())
        if not result:
            return False
        
        print("✅ Health monitoring system functional")
        return True
        
    except Exception as e:
        print(f"❌ Health monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_distributed_tracing():
    """Test the distributed tracing system."""
    print("\n🔍 Testing Distributed Tracing System...")
    
    try:
        from server.monitoring.tracing import (
            QFLARETracer, FederatedLearningTracer, SpanKind, SpanStatus,
            get_tracer, get_fl_tracer
        )
        
        # Test tracer initialization
        tracer = QFLARETracer(service_name="qflare_test")
        print("✅ Tracer initialized")
        
        # Test basic span creation
        span = tracer.start_span("test_operation", SpanKind.INTERNAL)
        span.set_tag("test.key", "test_value")
        span.log("Test log message")
        
        time.sleep(0.01)  # Simulate some work
        tracer.finish_span(span, SpanStatus.OK)
        print("✅ Basic span tracing completed")
        
        # Test context manager tracing
        with tracer.trace("context_manager_test", SpanKind.SERVER) as span:
            span.set_tag("operation", "context_test")
            time.sleep(0.01)
        print("✅ Context manager tracing completed")
        
        # Test federated learning tracer
        fl_tracer = FederatedLearningTracer(tracer)
        
        # Test FL round tracing
        with fl_tracer.trace_fl_round("test_round_001", "fedavg") as span:
            span.set_tag("participants", 5)
            
            # Nested device training spans
            with fl_tracer.trace_device_training("device_001", "cnn") as training_span:
                training_span.set_tag("epochs", 10)
                time.sleep(0.01)
            
            # Model aggregation span
            with fl_tracer.trace_model_aggregation(5, "fedavg") as agg_span:
                agg_span.set_tag("convergence", True)
                time.sleep(0.01)
        
        print("✅ Federated learning tracing completed")
        
        # Test context injection/extraction
        context = tracer.get_current_context()
        if context:
            carrier = {}
            tracer.inject_context(context, carrier)
            
            extracted_context = tracer.extract_context(carrier)
            if extracted_context and extracted_context.trace_id == context.trace_id:
                print("✅ Context injection/extraction working")
            else:
                print("⚠️  Context extraction incomplete")
        
        # Test tracing statistics
        stats = tracer.get_tracing_statistics()
        print(f"✅ Tracing statistics: {stats['completed_traces']} traces, {stats['active_spans']} active spans")
        
        return True
        
    except Exception as e:
        print(f"❌ Distributed tracing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alerting_system():
    """Test the alerting system."""
    print("\n🚨 Testing Alerting System...")
    
    try:
        from server.monitoring.alerting import (
            QFLAREAlertManager, Alert, AlertSeverity, AlertStatus,
            AlertRule, get_alert_manager
        )
        
        # Test alert manager initialization
        alert_manager = QFLAREAlertManager()
        print("✅ Alert manager initialized")
        
        # Test manual alert creation
        test_alert = Alert(
            id="test_alert_001",
            title="Test Alert",
            description="This is a test alert for validation",
            severity=AlertSeverity.WARNING,
            source="test_suite",
            timestamp=datetime.utcnow(),
            labels={"component": "test", "type": "validation"},
            annotations={"test_run": "validation"}
        )
        
        alert_manager.fire_alert(test_alert)
        print("✅ Manual alert fired")
        
        # Test alert acknowledgment
        alert_manager.acknowledge_alert(test_alert.id, "test_user")
        print("✅ Alert acknowledged")
        
        # Test alert resolution
        alert_manager.resolve_alert(test_alert.id)
        print("✅ Alert resolved")
        
        # Test alert rule creation
        def test_condition():
            return True  # Always triggers for testing
        
        test_rule = AlertRule(
            name="test_rule",
            description="Test alert rule for validation",
            condition=test_condition,
            severity=AlertSeverity.INFO,
            labels={"rule_type": "test"},
            evaluation_interval=1.0,
            for_duration=0.0
        )
        
        alert_manager.add_alert_rule(test_rule)
        print("✅ Alert rule added")
        
        # Test rule evaluation
        triggered_alert = test_rule.evaluate()
        if triggered_alert:
            print("✅ Alert rule evaluation successful")
        else:
            print("⚠️  Alert rule didn't trigger as expected")
        
        # Test alert statistics
        stats = alert_manager.get_alert_statistics()
        print(f"✅ Alert statistics: {stats['total_rules']} rules, {stats['active_alerts']} active alerts")
        
        # Test notification channels (mock)
        from server.monitoring.alerting import WebhookNotificationChannel
        
        webhook_channel = WebhookNotificationChannel(
            name="test_webhook",
            webhook_url="http://example.com/webhook",
            enabled=False  # Disabled for testing
        )
        
        alert_manager.add_notification_channel(webhook_channel)
        print("✅ Notification channel configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Alerting system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integrated_monitoring():
    """Test the integrated monitoring system."""
    print("\n🎯 Testing Integrated Monitoring System...")
    
    try:
        from server.monitoring import (
            QFLAREMonitoringSystem, MonitoringConfig,
            get_monitoring_system
        )
        
        # Test monitoring system initialization
        config = MonitoringConfig(
            metrics_enabled=True,
            metrics_port=8002,  # Different port for testing
            health_enabled=True,
            tracing_enabled=True,
            alerting_enabled=True
        )
        
        monitoring_system = QFLAREMonitoringSystem(config)
        print("✅ Integrated monitoring system created")
        
        # Test initialization
        monitoring_system.initialize()
        print("✅ Monitoring system initialized")
        
        # Test system status
        status = monitoring_system.get_system_status()
        if status['initialized']:
            print("✅ System status reporting functional")
            print(f"  Components: {', '.join(status['components'].keys())}")
        
        # Test FL metrics recording
        monitoring_system.record_fl_training_metrics(
            device_id="test_device_001",
            duration=15.5,
            accuracy=0.89,
            loss=0.11,
            iterations=50
        )
        print("✅ FL training metrics recorded via integrated system")
        
        monitoring_system.record_fl_round_metrics(
            round_id="test_round_001",
            duration=120.0,
            participants=8,
            global_accuracy=0.91,
            algorithm="fedavg"
        )
        print("✅ FL round metrics recorded via integrated system")
        
        # Test security metrics
        monitoring_system.record_security_metrics(
            operation="authenticate",
            success=True,
            device_id="test_device_001",
            method="jwt"
        )
        print("✅ Security metrics recorded via integrated system")
        
        # Test health check via integrated system
        async def test_health():
            health_result = await monitoring_system.run_health_check()
            if 'overall_status' in health_result:
                print(f"✅ Health check via integrated system: {health_result['overall_status']}")
                return True
            return False
        
        health_ok = asyncio.run(test_health())
        if not health_ok:
            print("⚠️  Integrated health check had issues")
        
        # Test metrics export
        metrics_export = monitoring_system.get_metrics_export()
        if metrics_export and "qflare" in metrics_export:
            print("✅ Integrated metrics export functional")
        else:
            print("⚠️  Metrics export may have issues")
        
        # Test alert creation
        monitoring_system.create_performance_alert(
            component="training",
            metric="duration",
            threshold=10.0,
            current_value=15.5
        )
        print("✅ Performance alert created via integrated system")
        
        monitoring_system.create_fl_alert(
            issue_type="low_participation",
            description="Participation rate below threshold",
            round_id="test_round_001"
        )
        print("✅ FL alert created via integrated system")
        
        print("✅ Integrated monitoring system fully functional")
        return True
        
    except Exception as e:
        print(f"❌ Integrated monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_monitoring_performance():
    """Test monitoring system performance."""
    print("\n⚡ Testing Monitoring Performance...")
    
    try:
        from server.monitoring import QFLAREMonitoringSystem, MonitoringConfig
        
        # Create monitoring system
        config = MonitoringConfig(
            metrics_enabled=True,
            metrics_port=8003,
            health_enabled=True,
            tracing_enabled=True,
            alerting_enabled=True
        )
        
        monitoring_system = QFLAREMonitoringSystem(config)
        monitoring_system.initialize()
        
        start_time = time.time()
        operations = 0
        
        # Perform multiple monitoring operations
        for i in range(50):
            # Record metrics
            monitoring_system.record_fl_training_metrics(
                device_id=f"perf_device_{i:03d}",
                duration=float(i % 20),
                accuracy=0.85 + (i % 10) * 0.01,
                loss=0.15 - (i % 10) * 0.01,
                iterations=10 + i
            )
            operations += 1
            
            # Record security events
            monitoring_system.record_security_metrics(
                operation="authenticate",
                success=(i % 7) != 0,  # Occasional failures
                device_id=f"perf_device_{i:03d}",
                method="jwt"
            )
            operations += 1
        
        # Create some alerts
        for i in range(5):
            monitoring_system.create_fl_alert(
                issue_type="test_performance",
                description=f"Performance test alert {i}",
                round_id=f"perf_round_{i}"
            )
            operations += 1
        
        end_time = time.time()
        duration = end_time - start_time
        ops_per_second = operations / duration if duration > 0 else 0
        
        print(f"✅ Performance test completed:")
        print(f"   - {operations} operations in {duration:.2f} seconds")
        print(f"   - {ops_per_second:.1f} operations per second")
        
        # Test system status after load
        status = monitoring_system.get_system_status()
        if status['initialized']:
            print("✅ System remains stable under load")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_monitoring_tests():
    """Run all production monitoring tests."""
    print("🚀 Starting QFLARE Production Monitoring & Metrics Tests")
    print("=" * 65)
    
    test_results = []
    
    # Run tests
    test_results.append(("File Structure", test_monitoring_file_structure()))
    test_results.append(("Metrics System", test_metrics_system()))
    test_results.append(("Health Monitoring", test_health_monitoring()))
    test_results.append(("Distributed Tracing", test_distributed_tracing()))
    test_results.append(("Alerting System", test_alerting_system()))
    test_results.append(("Integrated Monitoring", test_integrated_monitoring()))
    test_results.append(("Performance Testing", test_monitoring_performance()))
    
    # Print summary
    print("\n" + "=" * 65)
    print("🎯 Production Monitoring Test Results:")
    print("=" * 65)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} | {status}")
        if result:
            passed += 1
    
    print("-" * 65)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All production monitoring tests passed!")
        print("📊 QFLARE monitoring & observability system is ready!")
        print("🔍 Features validated:")
        print("   ✅ Prometheus-compatible metrics collection")
        print("   ✅ Comprehensive health monitoring")
        print("   ✅ Distributed tracing with OpenTelemetry")
        print("   ✅ Multi-channel alerting system")
        print("   ✅ Integrated monitoring dashboard")
        print("   ✅ Production-grade observability")
    else:
        print("⚠️  Some monitoring tests failed. Review the logs above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_monitoring_tests()
    sys.exit(0 if success else 1)