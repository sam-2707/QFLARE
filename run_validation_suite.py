#!/usr/bin/env python3
"""
QFLARE Complete Validation Suite - Run all three validation tools
Orchestrates evaluation documentation, benchmark harness, and crypto performance testing
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and capture output"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0, result.stdout, result.stderr
    
    except subprocess.TimeoutExpired:
        print("❌ Command timed out after 5 minutes")
        return False, "", "Timeout"
    except Exception as e:
        print(f"❌ Command failed: {e}")
        return False, "", str(e)

def main():
    """Run the complete QFLARE validation suite"""
    
    print("QFLARE COMPLETE VALIDATION SUITE")
    print("=" * 80)
    print("Running all three validation components:")
    print("1. Evaluation Documentation Review")
    print("2. Benchmark Harness Performance Testing")
    print("3. Crypto Performance Testing")
    print("=" * 80)
    
    # Check if we're in the right directory
    project_root = Path("d:/QFLARE_Project_Structure")
    if not project_root.exists():
        project_root = Path.cwd()
    
    os.chdir(project_root)
    print(f"Working directory: {project_root}")
    
    results = {}
    
    # 1. Show evaluation documentation status
    print(f"\n{'='*60}")
    print("1. EVALUATION DOCUMENTATION")
    print(f"{'='*60}")
    
    eval_doc = project_root / "docs" / "qflare-evaluation.md"
    if eval_doc.exists():
        print(f"✅ Evaluation framework available: {eval_doc}")
        with open(eval_doc, 'r') as f:
            lines = f.readlines()
        
        print(f"📊 Document stats: {len(lines)} lines")
        
        # Count test categories
        security_tests = len([l for l in lines if "Security Test" in l or "Crypto Test" in l])
        privacy_tests = len([l for l in lines if "Privacy Test" in l or "DP Test" in l])
        performance_tests = len([l for l in lines if "Performance Test" in l or "Benchmark" in l])
        
        print(f"🔒 Security tests defined: {security_tests}")
        print(f"🔐 Privacy tests defined: {privacy_tests}")
        print(f"⚡ Performance tests defined: {performance_tests}")
        
        results['evaluation_doc'] = {
            'available': True,
            'lines': len(lines),
            'security_tests': security_tests,
            'privacy_tests': privacy_tests,
            'performance_tests': performance_tests
        }
    else:
        print(f"❌ Evaluation documentation not found at {eval_doc}")
        results['evaluation_doc'] = {'available': False}
    
    # 2. Run benchmark harness
    print(f"\n{'='*60}")
    print("2. BENCHMARK HARNESS TESTING")
    print(f"{'='*60}")
    
    benchmark_script = project_root / "tests" / "benchmark_harness.py"
    if benchmark_script.exists():
        success, stdout, stderr = run_command([
            sys.executable, str(benchmark_script), "--quick-test", "--output", "benchmark_results.json"
        ], "Benchmark Harness Performance Testing")
        
        results['benchmark_harness'] = {
            'success': success,
            'output_file': 'benchmark_results.json'
        }
        
        # Try to load results
        if success and os.path.exists("benchmark_results.json"):
            try:
                with open("benchmark_results.json", 'r') as f:
                    benchmark_data = json.load(f)
                results['benchmark_harness']['data'] = benchmark_data
                print(f"\n✅ Benchmark results saved to benchmark_results.json")
            except Exception as e:
                print(f"⚠️ Could not parse benchmark results: {e}")
    else:
        print(f"❌ Benchmark harness not found at {benchmark_script}")
        results['benchmark_harness'] = {'success': False, 'error': 'Script not found'}
    
    # 3. Run crypto performance testing
    print(f"\n{'='*60}")
    print("3. CRYPTO PERFORMANCE TESTING")
    print(f"{'='*60}")
    
    crypto_script = project_root / "tests" / "crypto_performance_tester.py"
    if crypto_script.exists():
        success, stdout, stderr = run_command([
            sys.executable, str(crypto_script), "--quick", "--system-info", "--output", "crypto_results.json"
        ], "Crypto Performance Testing")
        
        results['crypto_performance'] = {
            'success': success,
            'output_file': 'crypto_results.json'
        }
        
        # Try to load results
        if success and os.path.exists("crypto_results.json"):
            try:
                with open("crypto_results.json", 'r') as f:
                    crypto_data = json.load(f)
                results['crypto_performance']['data'] = crypto_data
                print(f"\n✅ Crypto performance results saved to crypto_results.json")
            except Exception as e:
                print(f"⚠️ Could not parse crypto results: {e}")
    else:
        print(f"❌ Crypto performance tester not found at {crypto_script}")
        results['crypto_performance'] = {'success': False, 'error': 'Script not found'}
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("VALIDATION SUITE SUMMARY")
    print(f"{'='*80}")
    
    total_components = 3
    successful_components = sum([
        results.get('evaluation_doc', {}).get('available', False),
        results.get('benchmark_harness', {}).get('success', False),
        results.get('crypto_performance', {}).get('success', False)
    ])
    
    print(f"\n📋 Components Status: {successful_components}/{total_components}")
    
    if results.get('evaluation_doc', {}).get('available'):
        print("✅ Evaluation Framework: Ready")
    else:
        print("❌ Evaluation Framework: Missing")
    
    if results.get('benchmark_harness', {}).get('success'):
        print("✅ Benchmark Harness: Completed")
    else:
        print("❌ Benchmark Harness: Failed")
    
    if results.get('crypto_performance', {}).get('success'):
        print("✅ Crypto Performance: Completed")
    else:
        print("❌ Crypto Performance: Failed")
    
    # Overall assessment
    print(f"\n{'='*60}")
    print("QFLARE EFFECTIVENESS ASSESSMENT")
    print(f"{'='*60}")
    
    if successful_components == total_components:
        assessment = "🎯 COMPLETE: Full validation suite executed successfully"
        print(assessment)
        print("📊 All three validation components are operational")
        print("🔒 Security, privacy, and performance can now be validated")
        print("⚡ QFLARE is ready for comprehensive effectiveness testing")
    elif successful_components >= 2:
        assessment = "⚠️ PARTIAL: Most validation components available"
        print(assessment)
        print("📊 Majority of validation tools are working")
        print("🔧 Minor issues need resolution for complete testing")
    else:
        assessment = "❌ INCOMPLETE: Major validation issues detected"
        print(assessment)
        print("🔧 Significant setup required before validation can proceed")
    
    # Save complete results
    final_report = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'validation_suite_version': '1.0',
        'components': results,
        'summary': {
            'total_components': total_components,
            'successful_components': successful_components,
            'assessment': assessment
        }
    }
    
    with open('qflare_validation_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\n📄 Complete validation report saved to: qflare_validation_report.json")
    
    # Quick start instructions
    print(f"\n{'='*60}")
    print("QUICK START INSTRUCTIONS")
    print(f"{'='*60}")
    print("\nTo run individual components:")
    print(f"1. Evaluation: Open docs/qflare-evaluation.md")
    print(f"2. Benchmark: python tests/benchmark_harness.py --quick-test")
    print(f"3. Crypto Test: python tests/crypto_performance_tester.py --quick")
    print(f"\nTo run this suite again: python run_validation_suite.py")
    
    return successful_components == total_components

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)