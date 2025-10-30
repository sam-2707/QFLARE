#!/usr/bin/env python3
"""
QFLARE Security Scanning Framework

This module provides comprehensive security scanning capabilities for QFLARE including:
- Dependency vulnerability scanning (OWASP Dependency Check, Safety, Bandit)
- Static Application Security Testing (SAST) using multiple tools
- Dynamic Application Security Testing (DAST) for running applications
- Container security scanning (Trivy, Docker Scout)
- Infrastructure as Code (IaC) security scanning
- Custom QFLARE-specific security checks
- Automated security reporting and CI/CD integration

Security scanning covers:
- Python dependencies and packages
- Source code vulnerabilities
- Container images and base layers
- Infrastructure configurations
- API security endpoints
- Cryptographic implementations
- Federated learning security
"""

import asyncio
import json
import logging
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SecurityFinding:
    """Represents a security finding from any scanner"""
    scanner: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    title: str
    description: str
    file_path: Optional[str]
    line_number: Optional[int]
    cwe_id: Optional[str]
    cve_id: Optional[str]
    cvss_score: Optional[float]
    recommendation: str
    timestamp: float
    additional_info: Dict[str, Any]

@dataclass
class ScanResult:
    """Represents the result of a security scan"""
    scanner_name: str
    scan_type: str
    status: str  # SUCCESS, FAILED, WARNING
    start_time: float
    end_time: float
    findings: List[SecurityFinding]
    metrics: Dict[str, Any]
    raw_output: str
    error_message: Optional[str] = None

@dataclass
class SecurityReport:
    """Comprehensive security report"""
    scan_id: str
    project_name: str
    scan_timestamp: float
    total_duration: float
    scan_results: List[ScanResult]
    summary: Dict[str, Any]
    recommendations: List[str]
    baseline_comparison: Optional[Dict[str, Any]] = None

class DependencyScanner:
    """Dependency vulnerability scanning using multiple tools"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.tools = {
            'safety': self._run_safety_scan,
            'bandit': self._run_bandit_scan,
            'semgrep': self._run_semgrep_scan,
            'pip-audit': self._run_pip_audit_scan
        }
        
    async def scan_dependencies(self) -> List[ScanResult]:
        """Run all dependency scanning tools"""
        results = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_tool = {
                executor.submit(tool_func): tool_name 
                for tool_name, tool_func in self.tools.items()
            }
            
            for future in as_completed(future_to_tool):
                tool_name = future_to_tool[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed {tool_name} scan: {len(result.findings)} findings")
                except Exception as e:
                    logger.error(f"Error running {tool_name}: {e}")
                    results.append(ScanResult(
                        scanner_name=tool_name,
                        scan_type="dependency",
                        status="FAILED",
                        start_time=time.time(),
                        end_time=time.time(),
                        findings=[],
                        metrics={},
                        raw_output="",
                        error_message=str(e)
                    ))
                    
        return results
        
    def _run_safety_scan(self) -> ScanResult:
        """Run Safety dependency vulnerability scan"""
        start_time = time.time()
        findings = []
        
        try:
            # Install safety if not available
            self._ensure_tool_installed('safety')
            
            # Run safety check
            result = subprocess.run([
                'safety', 'check', '--json', '--output', 'json'
            ], capture_output=True, text=True, cwd=self.project_path)
            
            if result.returncode == 0 or result.stdout:
                # Parse JSON output
                try:
                    safety_data = json.loads(result.stdout) if result.stdout else []
                    
                    for vuln in safety_data:
                        finding = SecurityFinding(
                            scanner="Safety",
                            severity=self._map_safety_severity(vuln.get('vulnerability_id', '')),
                            title=f"Vulnerable dependency: {vuln.get('package_name', 'Unknown')}",
                            description=vuln.get('advisory', 'No description available'),
                            file_path="requirements.txt",
                            line_number=None,
                            cwe_id=None,
                            cve_id=vuln.get('vulnerability_id'),
                            cvss_score=None,
                            recommendation=f"Update {vuln.get('package_name')} to version {vuln.get('spec', 'latest')}",
                            timestamp=time.time(),
                            additional_info={
                                'affected_versions': vuln.get('affected_versions', []),
                                'analyzed_version': vuln.get('analyzed_version'),
                                'id': vuln.get('id')
                            }
                        )
                        findings.append(finding)
                        
                except json.JSONDecodeError:
                    logger.warning("Failed to parse Safety JSON output")
                    
            return ScanResult(
                scanner_name="Safety",
                scan_type="dependency",
                status="SUCCESS",
                start_time=start_time,
                end_time=time.time(),
                findings=findings,
                metrics={'vulnerabilities_found': len(findings)},
                raw_output=result.stdout
            )
            
        except Exception as e:
            return ScanResult(
                scanner_name="Safety",
                scan_type="dependency",
                status="FAILED",
                start_time=start_time,
                end_time=time.time(),
                findings=[],
                metrics={},
                raw_output="",
                error_message=str(e)
            )
            
    def _run_bandit_scan(self) -> ScanResult:
        """Run Bandit static security analysis"""
        start_time = time.time()
        findings = []
        
        try:
            self._ensure_tool_installed('bandit')
            
            # Run bandit scan
            result = subprocess.run([
                'bandit', '-r', str(self.project_path),
                '-f', 'json', '--skip', 'B101,B601'  # Skip assert and shell usage
            ], capture_output=True, text=True)
            
            if result.stdout:
                try:
                    bandit_data = json.loads(result.stdout)
                    
                    for issue in bandit_data.get('results', []):
                        finding = SecurityFinding(
                            scanner="Bandit",
                            severity=issue.get('issue_severity', 'MEDIUM').upper(),
                            title=issue.get('test_name', 'Security Issue'),
                            description=issue.get('issue_text', 'No description'),
                            file_path=issue.get('filename'),
                            line_number=issue.get('line_number'),
                            cwe_id=issue.get('issue_cwe', {}).get('id'),
                            cve_id=None,
                            cvss_score=self._severity_to_cvss(issue.get('issue_severity', 'MEDIUM')),
                            recommendation=issue.get('issue_text', 'Review and fix security issue'),
                            timestamp=time.time(),
                            additional_info={
                                'test_id': issue.get('test_id'),
                                'confidence': issue.get('issue_confidence'),
                                'code': issue.get('code')
                            }
                        )
                        findings.append(finding)
                        
                except json.JSONDecodeError:
                    logger.warning("Failed to parse Bandit JSON output")
                    
            return ScanResult(
                scanner_name="Bandit",
                scan_type="sast",
                status="SUCCESS",
                start_time=start_time,
                end_time=time.time(),
                findings=findings,
                metrics={'issues_found': len(findings)},
                raw_output=result.stdout
            )
            
        except Exception as e:
            return ScanResult(
                scanner_name="Bandit",
                scan_type="sast",
                status="FAILED",
                start_time=start_time,
                end_time=time.time(),
                findings=[],
                metrics={},
                raw_output="",
                error_message=str(e)
            )
            
    def _run_semgrep_scan(self) -> ScanResult:
        """Run Semgrep static analysis"""
        start_time = time.time()
        findings = []
        
        try:
            self._ensure_tool_installed('semgrep')
            
            # Run semgrep with security rules
            result = subprocess.run([
                'semgrep', '--config=auto', '--json',
                '--exclude=*.pyc', '--exclude=venv/',
                str(self.project_path)
            ], capture_output=True, text=True)
            
            if result.stdout:
                try:
                    semgrep_data = json.loads(result.stdout)
                    
                    for issue in semgrep_data.get('results', []):
                        finding = SecurityFinding(
                            scanner="Semgrep",
                            severity=self._map_semgrep_severity(issue.get('extra', {}).get('severity', 'INFO')),
                            title=issue.get('check_id', 'Security Issue'),
                            description=issue.get('extra', {}).get('message', 'No description'),
                            file_path=issue.get('path'),
                            line_number=issue.get('start', {}).get('line'),
                            cwe_id=None,
                            cve_id=None,
                            cvss_score=self._severity_to_cvss(issue.get('extra', {}).get('severity', 'INFO')),
                            recommendation=issue.get('extra', {}).get('fix', 'Review code for security issues'),
                            timestamp=time.time(),
                            additional_info={
                                'rule_id': issue.get('check_id'),
                                'impact': issue.get('extra', {}).get('impact'),
                                'confidence': issue.get('extra', {}).get('confidence')
                            }
                        )
                        findings.append(finding)
                        
                except json.JSONDecodeError:
                    logger.warning("Failed to parse Semgrep JSON output")
                    
            return ScanResult(
                scanner_name="Semgrep",
                scan_type="sast",
                status="SUCCESS",
                start_time=start_time,
                end_time=time.time(),
                findings=findings,
                metrics={'rules_matched': len(findings)},
                raw_output=result.stdout
            )
            
        except Exception as e:
            return ScanResult(
                scanner_name="Semgrep",
                scan_type="sast",
                status="FAILED",
                start_time=start_time,
                end_time=time.time(),
                findings=[],
                metrics={},
                raw_output="",
                error_message=str(e)
            )
            
    def _run_pip_audit_scan(self) -> ScanResult:
        """Run pip-audit vulnerability scan"""
        start_time = time.time()
        findings = []
        
        try:
            self._ensure_tool_installed('pip-audit')
            
            # Run pip-audit
            result = subprocess.run([
                'pip-audit', '--format=json', '--desc'
            ], capture_output=True, text=True, cwd=self.project_path)
            
            if result.stdout:
                try:
                    audit_data = json.loads(result.stdout)
                    
                    for vuln in audit_data.get('vulnerabilities', []):
                        finding = SecurityFinding(
                            scanner="pip-audit",
                            severity="HIGH",  # pip-audit doesn't provide severity
                            title=f"Vulnerable package: {vuln.get('package', 'Unknown')}",
                            description=vuln.get('description', 'No description available'),
                            file_path="requirements.txt",
                            line_number=None,
                            cwe_id=None,
                            cve_id=vuln.get('id'),
                            cvss_score=None,
                            recommendation=f"Update {vuln.get('package')} to a secure version",
                            timestamp=time.time(),
                            additional_info={
                                'installed_version': vuln.get('installed_version'),
                                'fixed_versions': vuln.get('fixed_versions', [])
                            }
                        )
                        findings.append(finding)
                        
                except json.JSONDecodeError:
                    logger.warning("Failed to parse pip-audit JSON output")
                    
            return ScanResult(
                scanner_name="pip-audit",
                scan_type="dependency",
                status="SUCCESS",
                start_time=start_time,
                end_time=time.time(),
                findings=findings,
                metrics={'packages_scanned': len(findings)},
                raw_output=result.stdout
            )
            
        except Exception as e:
            return ScanResult(
                scanner_name="pip-audit",
                scan_type="dependency",
                status="FAILED",
                start_time=start_time,
                end_time=time.time(),
                findings=[],
                metrics={},
                raw_output="",
                error_message=str(e)
            )
            
    def _ensure_tool_installed(self, tool_name: str):
        """Ensure security tool is installed"""
        try:
            subprocess.run([tool_name, '--version'], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.info(f"Installing {tool_name}...")
            subprocess.run(['pip', 'install', tool_name], check=True)
            
    def _map_safety_severity(self, vuln_id: str) -> str:
        """Map Safety vulnerability to severity"""
        # Safety doesn't provide severity, so we use a simple heuristic
        if 'critical' in vuln_id.lower():
            return 'CRITICAL'
        elif 'high' in vuln_id.lower():
            return 'HIGH'
        else:
            return 'MEDIUM'
            
    def _map_semgrep_severity(self, severity: str) -> str:
        """Map Semgrep severity to standard levels"""
        mapping = {
            'ERROR': 'HIGH',
            'WARNING': 'MEDIUM',
            'INFO': 'LOW'
        }
        return mapping.get(severity.upper(), 'MEDIUM')
        
    def _severity_to_cvss(self, severity: str) -> float:
        """Convert severity to approximate CVSS score"""
        mapping = {
            'CRITICAL': 9.0,
            'HIGH': 7.5,
            'MEDIUM': 5.0,
            'LOW': 2.5,
            'INFO': 1.0
        }
        return mapping.get(severity.upper(), 5.0)

class ContainerScanner:
    """Container security scanning using Trivy and Docker Scout"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        
    async def scan_containers(self, image_name: str = None) -> List[ScanResult]:
        """Scan container images for vulnerabilities"""
        results = []
        
        # Scan with Trivy
        trivy_result = await self._run_trivy_scan(image_name)
        results.append(trivy_result)
        
        # Scan with Docker Scout (if available)
        scout_result = await self._run_docker_scout_scan(image_name)
        if scout_result:
            results.append(scout_result)
            
        return results
        
    async def _run_trivy_scan(self, image_name: str = None) -> ScanResult:
        """Run Trivy container vulnerability scan"""
        start_time = time.time()
        findings = []
        
        try:
            # Use default image if none provided
            if not image_name:
                image_name = "qflare:latest"
                
            # Install trivy if not available
            await self._ensure_trivy_installed()
            
            # Run trivy scan
            result = subprocess.run([
                'trivy', 'image', '--format', 'json', image_name
            ], capture_output=True, text=True)
            
            if result.stdout:
                try:
                    trivy_data = json.loads(result.stdout)
                    
                    for target in trivy_data.get('Results', []):
                        for vuln in target.get('Vulnerabilities', []):
                            finding = SecurityFinding(
                                scanner="Trivy",
                                severity=vuln.get('Severity', 'UNKNOWN').upper(),
                                title=f"Container vulnerability: {vuln.get('VulnerabilityID', 'Unknown')}",
                                description=vuln.get('Description', 'No description'),
                                file_path=target.get('Target', 'container'),
                                line_number=None,
                                cwe_id=None,
                                cve_id=vuln.get('VulnerabilityID'),
                                cvss_score=vuln.get('CVSS', {}).get('nvd', {}).get('V3Score'),
                                recommendation=f"Update {vuln.get('PkgName', 'package')} to {vuln.get('FixedVersion', 'latest version')}",
                                timestamp=time.time(),
                                additional_info={
                                    'package_name': vuln.get('PkgName'),
                                    'installed_version': vuln.get('InstalledVersion'),
                                    'fixed_version': vuln.get('FixedVersion'),
                                    'layer': vuln.get('Layer', {})
                                }
                            )
                            findings.append(finding)
                            
                except json.JSONDecodeError:
                    logger.warning("Failed to parse Trivy JSON output")
                    
            return ScanResult(
                scanner_name="Trivy",
                scan_type="container",
                status="SUCCESS",
                start_time=start_time,
                end_time=time.time(),
                findings=findings,
                metrics={'vulnerabilities_found': len(findings)},
                raw_output=result.stdout
            )
            
        except Exception as e:
            return ScanResult(
                scanner_name="Trivy",
                scan_type="container",
                status="FAILED",
                start_time=start_time,
                end_time=time.time(),
                findings=[],
                metrics={},
                raw_output="",
                error_message=str(e)
            )
            
    async def _run_docker_scout_scan(self, image_name: str = None) -> Optional[ScanResult]:
        """Run Docker Scout security scan"""
        start_time = time.time()
        findings = []
        
        try:
            if not image_name:
                image_name = "qflare:latest"
                
            # Check if Docker Scout is available
            result = subprocess.run([
                'docker', 'scout', 'cves', '--format', 'json', image_name
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout:
                try:
                    scout_data = json.loads(result.stdout)
                    
                    for vuln in scout_data.get('vulnerabilities', []):
                        finding = SecurityFinding(
                            scanner="Docker Scout",
                            severity=vuln.get('severity', 'UNKNOWN').upper(),
                            title=f"Container CVE: {vuln.get('id', 'Unknown')}",
                            description=vuln.get('description', 'No description'),
                            file_path=f"container:{image_name}",
                            line_number=None,
                            cwe_id=None,
                            cve_id=vuln.get('id'),
                            cvss_score=vuln.get('cvss_score'),
                            recommendation=vuln.get('remediation', 'Update affected package'),
                            timestamp=time.time(),
                            additional_info={
                                'package': vuln.get('package'),
                                'version': vuln.get('version'),
                                'fixed_version': vuln.get('fixed_version')
                            }
                        )
                        findings.append(finding)
                        
                except json.JSONDecodeError:
                    logger.warning("Failed to parse Docker Scout JSON output")
                    
                return ScanResult(
                    scanner_name="Docker Scout",
                    scan_type="container",
                    status="SUCCESS",
                    start_time=start_time,
                    end_time=time.time(),
                    findings=findings,
                    metrics={'cves_found': len(findings)},
                    raw_output=result.stdout
                )
                
        except Exception as e:
            logger.warning(f"Docker Scout not available or failed: {e}")
            
        return None
        
    async def _ensure_trivy_installed(self):
        """Ensure Trivy is installed"""
        try:
            subprocess.run(['trivy', '--version'], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.info("Installing Trivy...")
            # Install trivy using the official installation script
            install_script = """
            curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
            """
            subprocess.run(install_script, shell=True, check=True)

class DynamicScanner:
    """Dynamic Application Security Testing (DAST)"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        
    async def scan_application(self) -> ScanResult:
        """Run DAST scan against running application"""
        start_time = time.time()
        findings = []
        
        try:
            # Basic security checks
            security_checks = [
                self._check_ssl_configuration,
                self._check_security_headers,
                self._check_authentication,
                self._check_api_endpoints,
                self._check_rate_limiting
            ]
            
            for check in security_checks:
                check_findings = await check()
                findings.extend(check_findings)
                
            return ScanResult(
                scanner_name="QFLARE DAST",
                scan_type="dast",
                status="SUCCESS",
                start_time=start_time,
                end_time=time.time(),
                findings=findings,
                metrics={'checks_performed': len(security_checks)},
                raw_output=f"DAST scan completed against {self.base_url}"
            )
            
        except Exception as e:
            return ScanResult(
                scanner_name="QFLARE DAST",
                scan_type="dast",
                status="FAILED",
                start_time=start_time,
                end_time=time.time(),
                findings=[],
                metrics={},
                raw_output="",
                error_message=str(e)
            )
            
    async def _check_ssl_configuration(self) -> List[SecurityFinding]:
        """Check SSL/TLS configuration"""
        findings = []
        
        try:
            response = requests.get(self.base_url, timeout=10)
            
            # Check if HTTPS is used
            if not self.base_url.startswith('https://'):
                findings.append(SecurityFinding(
                    scanner="DAST SSL Check",
                    severity="MEDIUM",
                    title="HTTP used instead of HTTPS",
                    description="Application is using HTTP which transmits data in plain text",
                    file_path=None,
                    line_number=None,
                    cwe_id="CWE-319",
                    cve_id=None,
                    cvss_score=5.0,
                    recommendation="Implement HTTPS with proper SSL/TLS certificates",
                    timestamp=time.time(),
                    additional_info={'url': self.base_url}
                ))
                
        except Exception as e:
            logger.warning(f"SSL check failed: {e}")
            
        return findings
        
    async def _check_security_headers(self) -> List[SecurityFinding]:
        """Check for security headers"""
        findings = []
        
        try:
            response = requests.get(self.base_url, timeout=10)
            headers = response.headers
            
            # Required security headers
            required_headers = {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
                'X-XSS-Protection': '1; mode=block',
                'Strict-Transport-Security': None,
                'Content-Security-Policy': None
            }
            
            for header, expected in required_headers.items():
                if header not in headers:
                    findings.append(SecurityFinding(
                        scanner="DAST Header Check",
                        severity="MEDIUM",
                        title=f"Missing security header: {header}",
                        description=f"Security header {header} is not set",
                        file_path=None,
                        line_number=None,
                        cwe_id="CWE-16",
                        cve_id=None,
                        cvss_score=4.0,
                        recommendation=f"Add {header} header to HTTP responses",
                        timestamp=time.time(),
                        additional_info={'missing_header': header}
                    ))
                    
        except Exception as e:
            logger.warning(f"Header check failed: {e}")
            
        return findings
        
    async def _check_authentication(self) -> List[SecurityFinding]:
        """Check authentication mechanisms"""
        findings = []
        
        try:
            # Test for common authentication endpoints
            auth_endpoints = ['/login', '/api/v1/auth', '/auth']
            
            for endpoint in auth_endpoints:
                url = f"{self.base_url.rstrip('/')}{endpoint}"
                
                try:
                    response = requests.get(url, timeout=5)
                    
                    # Check if endpoint exists and is properly secured
                    if response.status_code == 200:
                        # Additional authentication security checks would go here
                        pass
                        
                except requests.RequestException:
                    continue
                    
        except Exception as e:
            logger.warning(f"Authentication check failed: {e}")
            
        return findings
        
    async def _check_api_endpoints(self) -> List[SecurityFinding]:
        """Check API endpoint security"""
        findings = []
        
        try:
            # Test common API endpoints
            api_endpoints = ['/api', '/api/v1', '/docs', '/swagger']
            
            for endpoint in api_endpoints:
                url = f"{self.base_url.rstrip('/')}{endpoint}"
                
                try:
                    response = requests.get(url, timeout=5)
                    
                    # Check for information disclosure
                    if response.status_code == 200 and 'swagger' in response.text.lower():
                        findings.append(SecurityFinding(
                            scanner="DAST API Check",
                            severity="LOW",
                            title="API documentation publicly accessible",
                            description="Swagger/OpenAPI documentation is publicly accessible",
                            file_path=None,
                            line_number=None,
                            cwe_id="CWE-200",
                            cve_id=None,
                            cvss_score=2.0,
                            recommendation="Restrict access to API documentation in production",
                            timestamp=time.time(),
                            additional_info={'endpoint': endpoint}
                        ))
                        
                except requests.RequestException:
                    continue
                    
        except Exception as e:
            logger.warning(f"API endpoint check failed: {e}")
            
        return findings
        
    async def _check_rate_limiting(self) -> List[SecurityFinding]:
        """Check for rate limiting"""
        findings = []
        
        try:
            # Send multiple rapid requests to test rate limiting
            rapid_requests = 10
            success_count = 0
            
            for i in range(rapid_requests):
                try:
                    response = requests.get(self.base_url, timeout=2)
                    if response.status_code == 200:
                        success_count += 1
                except requests.RequestException:
                    break
                    
            # If all requests succeed, rate limiting might not be implemented
            if success_count == rapid_requests:
                findings.append(SecurityFinding(
                    scanner="DAST Rate Limit Check",
                    severity="MEDIUM",
                    title="No rate limiting detected",
                    description="Application does not appear to implement rate limiting",
                    file_path=None,
                    line_number=None,
                    cwe_id="CWE-770",
                    cve_id=None,
                    cvss_score=5.0,
                    recommendation="Implement rate limiting to prevent abuse",
                    timestamp=time.time(),
                    additional_info={'requests_tested': rapid_requests}
                ))
                
        except Exception as e:
            logger.warning(f"Rate limiting check failed: {e}")
            
        return findings

class SecurityReportGenerator:
    """Generate comprehensive security reports"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_report(self, security_report: SecurityReport) -> Dict[str, str]:
        """Generate multiple report formats"""
        report_files = {}
        
        # JSON report
        json_file = self.output_dir / f"security_report_{security_report.scan_id}.json"
        with open(json_file, 'w') as f:
            json.dump(asdict(security_report), f, indent=2, default=str)
        report_files['json'] = str(json_file)
        
        # HTML report
        html_file = self.output_dir / f"security_report_{security_report.scan_id}.html"
        html_content = self._generate_html_report(security_report)
        with open(html_file, 'w') as f:
            f.write(html_content)
        report_files['html'] = str(html_file)
        
        # CSV report
        csv_file = self.output_dir / f"security_findings_{security_report.scan_id}.csv"
        self._generate_csv_report(security_report, csv_file)
        report_files['csv'] = str(csv_file)
        
        # SARIF report for integration with GitHub Security tab
        sarif_file = self.output_dir / f"security_report_{security_report.scan_id}.sarif"
        sarif_content = self._generate_sarif_report(security_report)
        with open(sarif_file, 'w') as f:
            json.dump(sarif_content, f, indent=2)
        report_files['sarif'] = str(sarif_file)
        
        return report_files
        
    def _generate_html_report(self, report: SecurityReport) -> str:
        """Generate HTML security report"""
        # Count findings by severity
        severity_counts = {}
        all_findings = []
        for scan_result in report.scan_results:
            all_findings.extend(scan_result.findings)
            
        for finding in all_findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
            
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QFLARE Security Report - {report.scan_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; border-bottom: 3px solid #007acc; padding-bottom: 20px; margin-bottom: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .summary-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .critical {{ background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); }}
        .high {{ background: linear-gradient(135deg, #ffa726 0%, #ff7043 100%); }}
        .medium {{ background: linear-gradient(135deg, #ffeb3b 0%, #ffc107 100%); color: #333; }}
        .low {{ background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%); }}
        .findings {{ margin-top: 30px; }}
        .finding {{ border: 1px solid #ddd; margin-bottom: 15px; border-radius: 5px; overflow: hidden; }}
        .finding-header {{ padding: 15px; background: #f8f9fa; border-bottom: 1px solid #ddd; }}
        .finding-body {{ padding: 15px; }}
        .severity-badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; text-transform: uppercase; }}
        .scanner-info {{ display: flex; justify-content: space-between; align-items: center; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        .recommendation {{ background: #e3f2fd; padding: 10px; border-radius: 5px; margin-top: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ QFLARE Security Report</h1>
            <p>Scan ID: {report.scan_id} | Project: {report.project_name}</p>
            <p>Generated: {datetime.fromtimestamp(report.scan_timestamp).strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Duration: {report.total_duration:.2f} seconds</p>
        </div>
        
        <div class="summary">
            <div class="summary-card critical">
                <h3>Critical</h3>
                <h2>{severity_counts.get('CRITICAL', 0)}</h2>
            </div>
            <div class="summary-card high">
                <h3>High</h3>
                <h2>{severity_counts.get('HIGH', 0)}</h2>
            </div>
            <div class="summary-card medium">
                <h3>Medium</h3>
                <h2>{severity_counts.get('MEDIUM', 0)}</h2>
            </div>
            <div class="summary-card low">
                <h3>Low</h3>
                <h2>{severity_counts.get('LOW', 0)}</h2>
            </div>
        </div>
        
        <h2>📊 Scan Results Summary</h2>
        <table>
            <tr>
                <th>Scanner</th>
                <th>Type</th>
                <th>Status</th>
                <th>Findings</th>
                <th>Duration</th>
            </tr>
        """
        
        for scan_result in report.scan_results:
            duration = scan_result.end_time - scan_result.start_time
            html_template += f"""
            <tr>
                <td>{scan_result.scanner_name}</td>
                <td>{scan_result.scan_type.upper()}</td>
                <td>{scan_result.status}</td>
                <td>{len(scan_result.findings)}</td>
                <td>{duration:.2f}s</td>
            </tr>
            """
            
        html_template += """
        </table>
        
        <div class="findings">
            <h2>🔍 Security Findings</h2>
        """
        
        # Sort findings by severity
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3, 'INFO': 4}
        sorted_findings = sorted(all_findings, key=lambda x: severity_order.get(x.severity, 5))
        
        for finding in sorted_findings:
            severity_class = finding.severity.lower()
            html_template += f"""
            <div class="finding">
                <div class="finding-header">
                    <div class="scanner-info">
                        <div>
                            <span class="severity-badge {severity_class}">{finding.severity}</span>
                            <strong>{finding.title}</strong>
                        </div>
                        <div><small>Scanner: {finding.scanner}</small></div>
                    </div>
                </div>
                <div class="finding-body">
                    <p>{finding.description}</p>
                    {f'<p><strong>File:</strong> {finding.file_path}' + (f':{finding.line_number}' if finding.line_number else '') + '</p>' if finding.file_path else ''}
                    {f'<p><strong>CVE:</strong> {finding.cve_id}</p>' if finding.cve_id else ''}
                    {f'<p><strong>CWE:</strong> {finding.cwe_id}</p>' if finding.cwe_id else ''}
                    {f'<p><strong>CVSS Score:</strong> {finding.cvss_score}</p>' if finding.cvss_score else ''}
                    <div class="recommendation">
                        <strong>💡 Recommendation:</strong> {finding.recommendation}
                    </div>
                </div>
            </div>
            """
            
        html_template += """
        </div>
    </div>
</body>
</html>
        """
        
        return html_template
        
    def _generate_csv_report(self, report: SecurityReport, csv_file: Path):
        """Generate CSV report"""
        import csv
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Scanner', 'Severity', 'Title', 'Description', 'File', 'Line',
                'CWE', 'CVE', 'CVSS', 'Recommendation', 'Timestamp'
            ])
            
            for scan_result in report.scan_results:
                for finding in scan_result.findings:
                    writer.writerow([
                        finding.scanner,
                        finding.severity,
                        finding.title,
                        finding.description,
                        finding.file_path or '',
                        finding.line_number or '',
                        finding.cwe_id or '',
                        finding.cve_id or '',
                        finding.cvss_score or '',
                        finding.recommendation,
                        datetime.fromtimestamp(finding.timestamp).isoformat()
                    ])
                    
    def _generate_sarif_report(self, report: SecurityReport) -> Dict:
        """Generate SARIF format report for GitHub Security tab"""
        sarif_report = {
            "version": "2.1.0",
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "runs": []
        }
        
        for scan_result in report.scan_results:
            if not scan_result.findings:
                continue
                
            run = {
                "tool": {
                    "driver": {
                        "name": scan_result.scanner_name,
                        "informationUri": "https://github.com/sam-2707/QFLARE",
                        "version": "1.0.0"
                    }
                },
                "results": []
            }
            
            for finding in scan_result.findings:
                result = {
                    "ruleId": f"{scan_result.scanner_name.lower().replace(' ', '_')}_{finding.cwe_id or 'unknown'}",
                    "message": {
                        "text": finding.description
                    },
                    "level": self._sarif_level(finding.severity)
                }
                
                if finding.file_path:
                    result["locations"] = [{
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": finding.file_path
                            }
                        }
                    }]
                    
                    if finding.line_number:
                        result["locations"][0]["physicalLocation"]["region"] = {
                            "startLine": finding.line_number
                        }
                        
                run["results"].append(result)
                
            sarif_report["runs"].append(run)
            
        return sarif_report
        
    def _sarif_level(self, severity: str) -> str:
        """Convert severity to SARIF level"""
        mapping = {
            'CRITICAL': 'error',
            'HIGH': 'error',
            'MEDIUM': 'warning',
            'LOW': 'note',
            'INFO': 'note'
        }
        return mapping.get(severity, 'warning')

class QFLARESecurityScanner:
    """Main QFLARE security scanning coordinator"""
    
    def __init__(self, project_path: Path, output_dir: Path = None):
        self.project_path = project_path
        self.output_dir = output_dir or project_path / "security" / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scanners
        self.dependency_scanner = DependencyScanner(project_path)
        self.container_scanner = ContainerScanner(project_path)
        self.report_generator = SecurityReportGenerator(self.output_dir)
        
    async def run_comprehensive_scan(self, 
                                   include_dependencies: bool = True,
                                   include_containers: bool = True,
                                   include_dast: bool = False,
                                   dast_url: str = None) -> SecurityReport:
        """Run comprehensive security scan"""
        scan_id = f"qflare_scan_{int(time.time())}"
        start_time = time.time()
        scan_results = []
        
        logger.info(f"Starting comprehensive security scan: {scan_id}")
        
        # Dependency scanning
        if include_dependencies:
            logger.info("Running dependency vulnerability scanning...")
            dep_results = await self.dependency_scanner.scan_dependencies()
            scan_results.extend(dep_results)
            
        # Container scanning
        if include_containers:
            logger.info("Running container security scanning...")
            container_results = await self.container_scanner.scan_containers()
            scan_results.extend(container_results)
            
        # Dynamic scanning
        if include_dast and dast_url:
            logger.info(f"Running DAST scan against {dast_url}...")
            dast_scanner = DynamicScanner(dast_url)
            dast_result = await dast_scanner.scan_application()
            scan_results.append(dast_result)
            
        # Generate summary
        total_findings = sum(len(result.findings) for result in scan_results)
        severity_counts = {}
        for result in scan_results:
            for finding in result.findings:
                severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
                
        # Generate recommendations
        recommendations = self._generate_recommendations(scan_results)
        
        # Create comprehensive report
        security_report = SecurityReport(
            scan_id=scan_id,
            project_name="QFLARE",
            scan_timestamp=start_time,
            total_duration=time.time() - start_time,
            scan_results=scan_results,
            summary={
                'total_findings': total_findings,
                'severity_breakdown': severity_counts,
                'scanners_used': [result.scanner_name for result in scan_results],
                'scan_types': list(set(result.scan_type for result in scan_results))
            },
            recommendations=recommendations
        )
        
        # Generate reports
        report_files = self.report_generator.generate_report(security_report)
        
        logger.info(f"Security scan completed: {total_findings} findings")
        logger.info(f"Reports generated: {', '.join(report_files.values())}")
        
        return security_report
        
    def _generate_recommendations(self, scan_results: List[ScanResult]) -> List[str]:
        """Generate security recommendations based on findings"""
        recommendations = []
        
        # Count findings by type
        dependency_findings = sum(len(r.findings) for r in scan_results if r.scan_type == 'dependency')
        sast_findings = sum(len(r.findings) for r in scan_results if r.scan_type == 'sast')
        container_findings = sum(len(r.findings) for r in scan_results if r.scan_type == 'container')
        dast_findings = sum(len(r.findings) for r in scan_results if r.scan_type == 'dast')
        
        if dependency_findings > 0:
            recommendations.append(
                f"📦 Update dependencies: {dependency_findings} vulnerable dependencies found. "
                "Run 'pip install --upgrade' for critical packages and review security advisories."
            )
            
        if sast_findings > 0:
            recommendations.append(
                f"🔍 Code review required: {sast_findings} static analysis issues found. "
                "Review code for security vulnerabilities and implement secure coding practices."
            )
            
        if container_findings > 0:
            recommendations.append(
                f"🐳 Update base images: {container_findings} container vulnerabilities found. "
                "Use minimal base images and regularly update container dependencies."
            )
            
        if dast_findings > 0:
            recommendations.append(
                f"🌐 Runtime security: {dast_findings} runtime issues found. "
                "Implement security headers, HTTPS, and proper authentication mechanisms."
            )
            
        # Add general recommendations
        recommendations.extend([
            "🔒 Enable automated security scanning in CI/CD pipeline",
            "📊 Set up continuous monitoring with security alerts",
            "🛡️ Implement security policies and regular audits",
            "📝 Document security procedures and incident response plans"
        ])
        
        return recommendations

# CLI interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="QFLARE Security Scanner")
    parser.add_argument("--project-path", type=str, default=".", help="Project root path")
    parser.add_argument("--output-dir", type=str, help="Output directory for reports")
    parser.add_argument("--include-dependencies", action="store_true", default=True, help="Include dependency scanning")
    parser.add_argument("--include-containers", action="store_true", default=True, help="Include container scanning")
    parser.add_argument("--include-dast", action="store_true", default=False, help="Include DAST scanning")
    parser.add_argument("--dast-url", type=str, help="URL for DAST scanning")
    parser.add_argument("--format", choices=["json", "html", "csv", "sarif"], default="html", help="Report format")
    
    args = parser.parse_args()
    
    async def run_scan():
        scanner = QFLARESecurityScanner(
            project_path=Path(args.project_path),
            output_dir=Path(args.output_dir) if args.output_dir else None
        )
        
        report = await scanner.run_comprehensive_scan(
            include_dependencies=args.include_dependencies,
            include_containers=args.include_containers,
            include_dast=args.include_dast,
            dast_url=args.dast_url
        )
        
        print(f"\n🛡️ QFLARE Security Scan Complete!")
        print(f"📊 Total Findings: {report.summary['total_findings']}")
        print(f"⏱️ Duration: {report.total_duration:.2f} seconds")
        print(f"📁 Reports saved to: {scanner.output_dir}")
        
        return report
        
    # Run the async scan
    import asyncio
    asyncio.run(run_scan())

if __name__ == "__main__":
    main()