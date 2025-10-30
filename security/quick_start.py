#!/usr/bin/env python3
"""
QFLARE Security Quick Start Guide and Test Runner

This script provides a quick start for the QFLARE security scanning framework
and runs basic security tests to validate the implementation.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from security_scanner import QFLARESecurityScanner, SecurityFinding, ScanResult
from config.security_config import SecurityConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_quick_security_scan():
    """Run a quick security scan to validate the setup"""
    print("🛡️ QFLARE Security Scanner - Quick Test")
    print("=" * 50)
    
    project_path = Path.cwd()
    print(f"📁 Project Path: {project_path}")
    
    # Initialize scanner
    scanner = QFLARESecurityScanner(project_path)
    
    try:
        # Load configuration
        config_manager = SecurityConfigManager()
        config = config_manager.load_config()
        print(f"⚙️ Configuration loaded: {config.project_name} v{config.version}")
        
        # Run quick dependency scan only
        print("\n🔍 Running quick dependency scan...")
        start_time = time.time()
        
        # Quick scan with limited scope
        report = await scanner.run_comprehensive_scan(
            include_dependencies=True,
            include_containers=False,  # Skip containers for quick test
            include_dast=False         # Skip DAST for quick test
        )
        
        duration = time.time() - start_time
        
        # Display results
        print(f"\n📊 Scan Results (Duration: {duration:.2f}s)")
        print("-" * 40)
        
        total_findings = report.summary['total_findings']
        severity_breakdown = report.summary['severity_breakdown']
        
        print(f"Total Findings: {total_findings}")
        
        if total_findings > 0:
            for severity, count in severity_breakdown.items():
                if count > 0:
                    emoji = {
                        'CRITICAL': '🔴',
                        'HIGH': '🟠', 
                        'MEDIUM': '🟡',
                        'LOW': '🟢',
                        'INFO': '🔵'
                    }.get(severity, '⚪')
                    print(f"  {emoji} {severity}: {count}")
        else:
            print("✅ No security issues found!")
        
        # Show scanner results
        print(f"\n🔧 Scanner Results:")
        for scan_result in report.scan_results:
            status_emoji = "✅" if scan_result.status == "SUCCESS" else "❌"
            print(f"  {status_emoji} {scan_result.scanner_name}: {len(scan_result.findings)} findings")
            
            if scan_result.error_message:
                print(f"    ⚠️ Error: {scan_result.error_message}")
        
        # Show recommendations
        if report.recommendations:
            print(f"\n💡 Top Recommendations:")
            for i, rec in enumerate(report.recommendations[:3], 1):
                print(f"  {i}. {rec}")
        
        # Validate against security policy
        policy_validation = config_manager.validate_findings(severity_breakdown)
        
        print(f"\n📋 Security Policy Validation:")
        if policy_validation['passed']:
            print("✅ All security policies passed")
        else:
            print("❌ Security policy violations:")
            for violation in policy_validation['violations']:
                print(f"  - {violation}")
        
        # Show report locations
        print(f"\n📁 Reports Generated:")
        reports_dir = scanner.output_dir
        if reports_dir.exists():
            for report_file in reports_dir.iterdir():
                if report_file.is_file():
                    size_kb = report_file.stat().st_size / 1024
                    print(f"  📄 {report_file.name} ({size_kb:.1f} KB)")
        
        return report
        
    except Exception as e:
        logger.error(f"Security scan failed: {e}")
        print(f"❌ Security scan failed: {e}")
        return None

def check_security_tools():
    """Check if required security tools are available"""
    print("\n🔧 Checking Security Tools...")
    print("-" * 30)
    
    tools = {
        'safety': 'pip install safety',
        'bandit': 'pip install bandit', 
        'semgrep': 'pip install semgrep',
        'pip-audit': 'pip install pip-audit'
    }
    
    available_tools = []
    missing_tools = []
    
    for tool, install_cmd in tools.items():
        try:
            import subprocess
            result = subprocess.run([tool, '--version'], 
                                 capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✅ {tool}: Available")
                available_tools.append(tool)
            else:
                print(f"❌ {tool}: Not working properly")
                missing_tools.append((tool, install_cmd))
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            print(f"❌ {tool}: Not installed")
            missing_tools.append((tool, install_cmd))
    
    if missing_tools:
        print(f"\n📦 To install missing tools:")
        for tool, install_cmd in missing_tools:
            print(f"  {install_cmd}")
    
    return len(available_tools), len(missing_tools)

def test_security_config():
    """Test security configuration loading"""
    print("\n⚙️ Testing Security Configuration...")
    print("-" * 35)
    
    try:
        config_manager = SecurityConfigManager()
        config = config_manager.load_config()
        
        print(f"✅ Configuration loaded successfully")
        print(f"  Project: {config.project_name}")
        print(f"  Version: {config.version}")
        
        # Test scanner configurations
        scanners = ['dependency', 'static', 'container', 'dynamic']
        for scanner_name in scanners:
            scanner_config = config_manager.get_scanner_config(scanner_name)
            enabled = "✅ Enabled" if scanner_config.enabled else "❌ Disabled"
            print(f"  {scanner_name.title()} Scanner: {enabled}")
        
        # Test security policy
        policy = config_manager.get_security_policy()
        print(f"  Security Policy: Max Critical={policy.max_critical_findings}, High={policy.max_high_findings}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def test_security_scanner():
    """Test the security scanner with mock data"""
    print("\n🧪 Testing Security Scanner Components...")
    print("-" * 40)
    
    try:
        # Test scanner initialization
        project_path = Path.cwd()
        scanner = QFLARESecurityScanner(project_path)
        print("✅ Scanner initialization successful")
        
        # Test dependency scanner
        try:
            from security_scanner import DependencyScanner
            dep_scanner = DependencyScanner(project_path)
            print("✅ Dependency scanner created")
        except Exception as e:
            print(f"❌ Dependency scanner failed: {e}")
        
        # Test container scanner  
        try:
            from security_scanner import ContainerScanner
            container_scanner = ContainerScanner(project_path)
            print("✅ Container scanner created")
        except Exception as e:
            print(f"❌ Container scanner failed: {e}")
        
        # Test report generator
        try:
            from security_scanner import SecurityReportGenerator
            report_gen = SecurityReportGenerator(Path("security/reports"))
            print("✅ Report generator created")
        except Exception as e:
            print(f"❌ Report generator failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Scanner component test failed: {e}")
        return False

def display_security_summary():
    """Display security framework summary"""
    print("\n" + "=" * 60)
    print("🛡️ QFLARE SECURITY FRAMEWORK SUMMARY")
    print("=" * 60)
    
    features = [
        ("🔍 Dependency Scanning", "Safety, Bandit, Semgrep, pip-audit"),
        ("🐳 Container Security", "Trivy, Docker Scout"),
        ("🌐 Dynamic Testing", "DAST, API security checks"),
        ("📊 Multiple Reports", "HTML, JSON, SARIF, CSV"),
        ("🔄 CI/CD Integration", "GitHub Actions, automated workflows"),
        ("📋 Security Policies", "Configurable thresholds and rules"),
        ("🚨 Real-time Alerts", "Notifications and monitoring"),
        ("🔐 Post-Quantum Ready", "CRYSTALS-Kyber, Dilithium support")
    ]
    
    for feature, description in features:
        print(f"{feature:<25} {description}")
    
    print("\n📁 Key Files Created:")
    files = [
        "security/security_scanner.py - Main scanning framework",
        "security/config/security_config.py - Configuration management", 
        "security/config/security_config.yml - Default configuration",
        "security/ci_integration.py - CI/CD integration tools",
        ".github/workflows/security.yml - GitHub Actions workflow",
        ".github/actions/qflare-security/ - Custom security action",
        "SECURITY.md - Security policy and procedures"
    ]
    
    for file_desc in files:
        print(f"  📄 {file_desc}")

async def main():
    """Main entry point for security quick start"""
    print("🚀 QFLARE Security Framework - Quick Start\n")
    
    # Step 1: Check environment
    available_tools, missing_tools = check_security_tools()
    
    # Step 2: Test configuration
    config_ok = test_security_config()
    
    # Step 3: Test scanner components
    scanner_ok = await test_security_scanner()
    
    # Step 4: Run quick scan if everything is ready
    if available_tools > 0 and config_ok and scanner_ok:
        print(f"\n✅ Prerequisites satisfied - running quick scan...")
        scan_report = await run_quick_security_scan()
        
        if scan_report:
            print("\n🎉 Quick security scan completed successfully!")
        else:
            print("\n⚠️ Quick security scan encountered issues")
    else:
        print(f"\n⚠️ Some prerequisites not met:")
        if missing_tools > 0:
            print(f"  - {missing_tools} security tools missing")
        if not config_ok:
            print(f"  - Configuration test failed")
        if not scanner_ok:
            print(f"  - Scanner component test failed")
        
        print("\nPlease install missing tools and re-run the test.")
    
    # Always show summary
    display_security_summary()
    
    print("\n" + "=" * 60)
    print("🔗 Next Steps:")
    print("  1. Install missing security tools if needed")
    print("  2. Customize security/config/security_config.yml")
    print("  3. Run full scan: python security/security_scanner.py")
    print("  4. Set up GitHub Actions for automated scanning")
    print("  5. Configure notifications and alerts")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())