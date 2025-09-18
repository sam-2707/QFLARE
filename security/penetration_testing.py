"""
Penetration Testing and Security Assessment Tools for QFLARE
"""

import asyncio
import aiohttp
import subprocess
import json
import time
from datetime import datetime
from typing import List, Dict, Any
from pydantic import BaseModel
import socket
import ssl
import requests
from urllib.parse import urljoin
import logging

# Configure logging
pentest_logger = logging.getLogger("qflare.pentest")

class PenetrationTestResult(BaseModel):
    test_id: str
    test_name: str
    test_type: str
    status: str
    started_at: datetime
    completed_at: datetime
    vulnerabilities: List[Dict[str, Any]]
    risk_score: float
    recommendations: List[str]

class NetworkScanResult(BaseModel):
    target: str
    open_ports: List[int]
    services: Dict[int, str]
    vulnerabilities: List[Dict[str, Any]]

class WebAppSecurityTest(BaseModel):
    url: str
    test_results: Dict[str, Any]
    security_headers: Dict[str, str]
    ssl_info: Dict[str, Any]
    vulnerabilities: List[Dict[str, Any]]

class SecurityAssessment:
    def __init__(self, target_host: str = "localhost", target_port: int = 8000):
        self.target_host = target_host
        self.target_port = target_port
        self.base_url = f"http://{target_host}:{target_port}"
        
    async def run_comprehensive_assessment(self) -> PenetrationTestResult:
        """Run comprehensive security assessment"""
        test_id = f"PENTEST-{int(time.time())}"
        started_at = datetime.utcnow()
        
        pentest_logger.info(f"Starting comprehensive security assessment: {test_id}")
        
        vulnerabilities = []
        recommendations = []
        
        # Network scan
        network_results = await self.network_scan()
        vulnerabilities.extend(network_results.vulnerabilities)
        
        # Web application security tests
        webapp_results = await self.web_app_security_test()
        vulnerabilities.extend(webapp_results.vulnerabilities)
        
        # API security tests
        api_vulnerabilities = await self.api_security_tests()
        vulnerabilities.extend(api_vulnerabilities)
        
        # Authentication tests
        auth_vulnerabilities = await self.authentication_tests()
        vulnerabilities.extend(auth_vulnerabilities)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(vulnerabilities)
        
        # Calculate risk score
        risk_score = self.calculate_risk_score(vulnerabilities)
        
        completed_at = datetime.utcnow()
        
        return PenetrationTestResult(
            test_id=test_id,
            test_name="Comprehensive Security Assessment",
            test_type="penetration_test",
            status="completed",
            started_at=started_at,
            completed_at=completed_at,
            vulnerabilities=vulnerabilities,
            risk_score=risk_score,
            recommendations=recommendations
        )
    
    async def network_scan(self) -> NetworkScanResult:
        """Perform network port scan"""
        pentest_logger.info(f"Starting network scan on {self.target_host}")
        
        open_ports = []
        services = {}
        vulnerabilities = []
        
        # Common ports to scan
        ports_to_scan = [22, 80, 443, 3000, 5432, 6379, 8000, 8080, 9090, 9200]
        
        for port in ports_to_scan:
            if await self.is_port_open(self.target_host, port):
                open_ports.append(port)
                service = self.identify_service(port)
                services[port] = service
                
                # Check for common vulnerabilities
                vuln = await self.check_port_vulnerabilities(port, service)
                if vuln:
                    vulnerabilities.append(vuln)
        
        return NetworkScanResult(
            target=self.target_host,
            open_ports=open_ports,
            services=services,
            vulnerabilities=vulnerabilities
        )
    
    async def is_port_open(self, host: str, port: int, timeout: float = 3.0) -> bool:
        """Check if a port is open"""
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=timeout
            )
            writer.close()
            await writer.wait_closed()
            return True
        except:
            return False
    
    def identify_service(self, port: int) -> str:
        """Identify service running on port"""
        service_map = {
            22: "SSH",
            80: "HTTP",
            443: "HTTPS",
            3000: "React Dev Server",
            5432: "PostgreSQL",
            6379: "Redis",
            8000: "QFLARE API",
            8080: "HTTP Alt",
            9090: "Prometheus",
            9200: "Elasticsearch"
        }
        return service_map.get(port, "Unknown")
    
    async def check_port_vulnerabilities(self, port: int, service: str) -> Dict[str, Any]:
        """Check for vulnerabilities on specific port/service"""
        vulnerability = None
        
        if port == 22 and service == "SSH":
            # Check for weak SSH configuration
            vulnerability = {
                "id": f"SSH-{port}",
                "severity": "MEDIUM",
                "title": "SSH service exposed",
                "description": "SSH service is accessible. Ensure strong authentication is configured.",
                "port": port,
                "service": service,
                "remediation": "Use key-based authentication and disable password authentication"
            }
        
        elif port in [5432, 6379] and await self.is_port_open(self.target_host, port):
            # Database ports exposed
            vulnerability = {
                "id": f"DB-{port}",
                "severity": "HIGH",
                "title": f"{service} database exposed",
                "description": f"{service} database is accessible from external networks",
                "port": port,
                "service": service,
                "remediation": "Restrict database access to internal networks only"
            }
        
        return vulnerability
    
    async def web_app_security_test(self) -> WebAppSecurityTest:
        """Test web application security"""
        pentest_logger.info(f"Testing web application security: {self.base_url}")
        
        vulnerabilities = []
        security_headers = {}
        ssl_info = {}
        test_results = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test security headers
                async with session.get(self.base_url) as response:
                    security_headers = dict(response.headers)
                    
                    # Check for missing security headers
                    required_headers = [
                        'X-Content-Type-Options',
                        'X-Frame-Options',
                        'X-XSS-Protection',
                        'Strict-Transport-Security',
                        'Content-Security-Policy'
                    ]
                    
                    for header in required_headers:
                        if header not in security_headers:
                            vulnerabilities.append({
                                "id": f"HEADER-{header}",
                                "severity": "MEDIUM",
                                "title": f"Missing security header: {header}",
                                "description": f"The {header} security header is not set",
                                "remediation": f"Add {header} header to HTTP responses"
                            })
                
                # Test for common web vulnerabilities
                test_results = await self.test_common_web_vulns(session)
                
        except Exception as e:
            pentest_logger.error(f"Web app security test failed: {e}")
        
        # Test SSL/TLS configuration if HTTPS
        if self.target_port == 443:
            ssl_info = await self.test_ssl_configuration()
        
        return WebAppSecurityTest(
            url=self.base_url,
            test_results=test_results,
            security_headers=security_headers,
            ssl_info=ssl_info,
            vulnerabilities=vulnerabilities
        )
    
    async def test_common_web_vulns(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test for common web vulnerabilities"""
        results = {}
        
        # Test for SQL injection
        sql_payloads = ["'", "1' OR '1'='1", "'; DROP TABLE users; --"]
        for payload in sql_payloads:
            try:
                url = f"{self.base_url}/api/v1/devices?id={payload}"
                async with session.get(url) as response:
                    if "error" in (await response.text()).lower():
                        results["sql_injection"] = "Potential SQL injection vulnerability detected"
                        break
            except:
                pass
        
        # Test for XSS
        xss_payload = "<script>alert('XSS')</script>"
        try:
            url = f"{self.base_url}/api/v1/devices"
            data = {"name": xss_payload}
            async with session.post(url, json=data) as response:
                if xss_payload in await response.text():
                    results["xss"] = "Potential XSS vulnerability detected"
        except:
            pass
        
        return results
    
    async def test_ssl_configuration(self) -> Dict[str, Any]:
        """Test SSL/TLS configuration"""
        ssl_info = {}
        
        try:
            context = ssl.create_default_context()
            with socket.create_connection((self.target_host, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=self.target_host) as ssock:
                    ssl_info = {
                        "version": ssock.version(),
                        "cipher": ssock.cipher(),
                        "certificate": {
                            "subject": dict(x[0] for x in ssock.getpeercert()['subject']),
                            "issuer": dict(x[0] for x in ssock.getpeercert()['issuer']),
                            "notAfter": ssock.getpeercert()['notAfter']
                        }
                    }
        except Exception as e:
            ssl_info["error"] = str(e)
        
        return ssl_info
    
    async def api_security_tests(self) -> List[Dict[str, Any]]:
        """Test API security"""
        pentest_logger.info("Testing API security")
        
        vulnerabilities = []
        
        # Test rate limiting
        rate_limit_vuln = await self.test_rate_limiting()
        if rate_limit_vuln:
            vulnerabilities.append(rate_limit_vuln)
        
        # Test authentication bypass
        auth_bypass_vuln = await self.test_authentication_bypass()
        if auth_bypass_vuln:
            vulnerabilities.append(auth_bypass_vuln)
        
        # Test authorization
        authz_vuln = await self.test_authorization()
        if authz_vuln:
            vulnerabilities.append(authz_vuln)
        
        return vulnerabilities
    
    async def test_rate_limiting(self) -> Dict[str, Any]:
        """Test API rate limiting"""
        try:
            async with aiohttp.ClientSession() as session:
                # Send rapid requests
                tasks = []
                for _ in range(50):
                    task = session.get(f"{self.base_url}/api/v1/health")
                    tasks.append(task)
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check if any requests were rate limited
                rate_limited = any(
                    isinstance(r, aiohttp.ClientResponse) and r.status == 429
                    for r in responses
                )
                
                if not rate_limited:
                    return {
                        "id": "RATE-LIMIT-001",
                        "severity": "MEDIUM",
                        "title": "Missing or weak rate limiting",
                        "description": "API endpoints do not implement proper rate limiting",
                        "remediation": "Implement rate limiting on all API endpoints"
                    }
        except:
            pass
        
        return None
    
    async def test_authentication_bypass(self) -> Dict[str, Any]:
        """Test for authentication bypass vulnerabilities"""
        try:
            async with aiohttp.ClientSession() as session:
                # Try to access protected endpoint without authentication
                async with session.get(f"{self.base_url}/api/v1/devices") as response:
                    if response.status == 200:
                        return {
                            "id": "AUTH-BYPASS-001",
                            "severity": "HIGH",
                            "title": "Authentication bypass",
                            "description": "Protected endpoints accessible without authentication",
                            "remediation": "Ensure all protected endpoints require valid authentication"
                        }
        except:
            pass
        
        return None
    
    async def test_authorization(self) -> Dict[str, Any]:
        """Test authorization controls"""
        # This would require creating test users with different roles
        # For now, return a generic recommendation
        return {
            "id": "AUTHZ-001",
            "severity": "LOW",
            "title": "Authorization testing required",
            "description": "Manual testing of role-based access controls is recommended",
            "remediation": "Implement comprehensive RBAC testing with different user roles"
        }
    
    async def authentication_tests(self) -> List[Dict[str, Any]]:
        """Test authentication mechanisms"""
        pentest_logger.info("Testing authentication mechanisms")
        
        vulnerabilities = []
        
        # Test for default credentials
        default_creds_vuln = await self.test_default_credentials()
        if default_creds_vuln:
            vulnerabilities.append(default_creds_vuln)
        
        # Test password policy
        pwd_policy_vuln = await self.test_password_policy()
        if pwd_policy_vuln:
            vulnerabilities.append(pwd_policy_vuln)
        
        return vulnerabilities
    
    async def test_default_credentials(self) -> Dict[str, Any]:
        """Test for default credentials"""
        default_creds = [
            ("admin", "admin"),
            ("admin", "password"),
            ("admin", "123456"),
            ("root", "root"),
            ("admin", "admin123")  # This is actually used in our system
        ]
        
        for username, password in default_creds:
            try:
                data = {"username": username, "password": password}
                response = requests.post(f"{self.base_url}/api/v1/auth/token", data=data)
                if response.status_code == 200:
                    return {
                        "id": "DEFAULT-CREDS-001",
                        "severity": "CRITICAL",
                        "title": "Default credentials detected",
                        "description": f"Default credentials {username}/{password} are active",
                        "remediation": "Change all default passwords immediately"
                    }
            except:
                pass
        
        return None
    
    async def test_password_policy(self) -> Dict[str, Any]:
        """Test password policy enforcement"""
        # This would require testing user registration/password change endpoints
        return {
            "id": "PWD-POLICY-001",
            "severity": "MEDIUM",
            "title": "Password policy testing required",
            "description": "Manual testing of password policy enforcement is recommended",
            "remediation": "Implement and test strong password policy enforcement"
        }
    
    def generate_recommendations(self, vulnerabilities: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations based on findings"""
        recommendations = []
        
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for vuln in vulnerabilities:
            severity_counts[vuln.get("severity", "LOW")] += 1
        
        if severity_counts["CRITICAL"] > 0:
            recommendations.append("Address critical vulnerabilities immediately")
        
        if severity_counts["HIGH"] > 0:
            recommendations.append("Prioritize high-severity vulnerabilities")
        
        recommendations.extend([
            "Implement comprehensive security monitoring",
            "Regular security assessments and penetration testing",
            "Security awareness training for all personnel",
            "Implement defense in depth security architecture",
            "Regular security patch management",
            "Implement network segmentation",
            "Use principle of least privilege for all access controls"
        ])
        
        return recommendations
    
    def calculate_risk_score(self, vulnerabilities: List[Dict[str, Any]]) -> float:
        """Calculate overall risk score"""
        if not vulnerabilities:
            return 0.0
        
        severity_weights = {"CRITICAL": 10, "HIGH": 7, "MEDIUM": 4, "LOW": 1}
        total_score = sum(severity_weights.get(vuln.get("severity", "LOW"), 1) for vuln in vulnerabilities)
        max_possible_score = len(vulnerabilities) * 10
        
        return round((total_score / max_possible_score) * 100, 2) if max_possible_score > 0 else 0.0

# CLI interface for running penetration tests
async def main():
    """Main function for running penetration tests"""
    assessor = SecurityAssessment()
    
    print("🔍 Starting QFLARE Security Assessment...")
    print("=" * 50)
    
    result = await assessor.run_comprehensive_assessment()
    
    print(f"\n📊 Assessment Complete: {result.test_id}")
    print(f"⏱️  Duration: {(result.completed_at - result.started_at).total_seconds()} seconds")
    print(f"🎯 Risk Score: {result.risk_score}/100")
    print(f"🚨 Vulnerabilities Found: {len(result.vulnerabilities)}")
    
    if result.vulnerabilities:
        print("\n🔍 Vulnerabilities:")
        for vuln in result.vulnerabilities:
            print(f"  [{vuln['severity']}] {vuln['title']}")
            print(f"    {vuln['description']}")
            print(f"    Remediation: {vuln['remediation']}")
            print()
    
    print("\n💡 Recommendations:")
    for rec in result.recommendations:
        print(f"  • {rec}")
    
    # Save results
    with open(f"pentest_results_{result.test_id}.json", "w") as f:
        json.dump(result.dict(), f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: pentest_results_{result.test_id}.json")

if __name__ == "__main__":
    asyncio.run(main())