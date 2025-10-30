#!/usr/bin/env python3
"""
QFLARE Security Scanning Configuration

This module provides configuration management for the QFLARE security scanning framework.
It defines security policies, scanner configurations, and integration settings.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class ScannerConfig:
    """Configuration for a specific security scanner"""
    enabled: bool = True
    timeout: int = 300  # 5 minutes
    severity_threshold: str = "LOW"  # Minimum severity to report
    exclude_patterns: List[str] = None
    custom_rules: List[str] = None
    arguments: Dict[str, Any] = None

@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    max_critical_findings: int = 0
    max_high_findings: int = 5
    max_medium_findings: int = 20
    fail_on_error: bool = True
    require_approval: bool = True
    notification_channels: List[str] = None

@dataclass
class CIIntegrationConfig:
    """CI/CD integration configuration"""
    github_actions: bool = True
    azure_pipelines: bool = False
    jenkins: bool = False
    slack_webhook: Optional[str] = None
    teams_webhook: Optional[str] = None
    email_notifications: List[str] = None
    sarif_upload: bool = True
    badge_generation: bool = True

@dataclass
class QFLARESecurityConfig:
    """Main QFLARE security configuration"""
    project_name: str = "QFLARE"
    version: str = "1.0.0"
    
    # Scanner configurations
    dependency_scanning: ScannerConfig = None
    static_analysis: ScannerConfig = None
    container_scanning: ScannerConfig = None
    dynamic_scanning: ScannerConfig = None
    
    # Security policies
    security_policy: SecurityPolicy = None
    
    # Integration settings
    ci_integration: CIIntegrationConfig = None
    
    # Report settings
    report_formats: List[str] = None
    report_retention_days: int = 30
    baseline_comparison: bool = True
    
    def __post_init__(self):
        """Initialize default configurations"""
        if self.dependency_scanning is None:
            self.dependency_scanning = ScannerConfig(
                enabled=True,
                timeout=180,
                severity_threshold="LOW",
                exclude_patterns=[
                    "*/venv/*",
                    "*/node_modules/*",
                    "*/.git/*",
                    "*/build/*",
                    "*/__pycache__/*"
                ],
                arguments={
                    "safety": ["--json", "--output", "json"],
                    "bandit": ["-f", "json", "--skip", "B101,B601"],
                    "semgrep": ["--config=auto", "--json"],
                    "pip-audit": ["--format=json", "--desc"]
                }
            )
            
        if self.static_analysis is None:
            self.static_analysis = ScannerConfig(
                enabled=True,
                timeout=300,
                severity_threshold="MEDIUM",
                exclude_patterns=[
                    "*/tests/*",
                    "*/test_*",
                    "*_test.py",
                    "*/migrations/*",
                    "*/alembic/versions/*"
                ],
                custom_rules=[
                    "security/rules/qflare_crypto_rules.yml",
                    "security/rules/federated_learning_rules.yml"
                ]
            )
            
        if self.container_scanning is None:
            self.container_scanning = ScannerConfig(
                enabled=True,
                timeout=600,
                severity_threshold="HIGH",
                arguments={
                    "trivy": ["--format", "json", "--timeout", "10m"],
                    "docker_scout": ["--format", "json"]
                }
            )
            
        if self.dynamic_scanning is None:
            self.dynamic_scanning = ScannerConfig(
                enabled=False,  # Only enable in staging/test environments
                timeout=1800,   # 30 minutes
                severity_threshold="MEDIUM",
                arguments={
                    "base_urls": ["http://localhost:8000", "https://qflare-staging.example.com"],
                    "auth_endpoints": ["/api/v1/auth/login"],
                    "api_endpoints": ["/api/v1", "/docs", "/swagger"]
                }
            )
            
        if self.security_policy is None:
            self.security_policy = SecurityPolicy(
                max_critical_findings=0,
                max_high_findings=3,
                max_medium_findings=15,
                fail_on_error=True,
                require_approval=True,
                notification_channels=["security-team", "dev-team"]
            )
            
        if self.ci_integration is None:
            self.ci_integration = CIIntegrationConfig(
                github_actions=True,
                sarif_upload=True,
                badge_generation=True,
                email_notifications=["security@qflare.dev"]
            )
            
        if self.report_formats is None:
            self.report_formats = ["html", "json", "sarif", "csv"]

class SecurityConfigManager:
    """Manages QFLARE security configuration"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("security/config/security_config.yml")
        self.config: Optional[QFLARESecurityConfig] = None
        
    def load_config(self) -> QFLARESecurityConfig:
        """Load security configuration from file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    
                # Convert dict to dataclass
                self.config = self._dict_to_config(config_data)
                return self.config
                
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_path}: {e}")
                print("Using default configuration")
                
        # Return default configuration
        self.config = QFLARESecurityConfig()
        return self.config
        
    def save_config(self, config: QFLARESecurityConfig):
        """Save security configuration to file"""
        self.config = config
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(asdict(config), f, default_flow_style=False, indent=2)
            
    def _dict_to_config(self, config_data: Dict) -> QFLARESecurityConfig:
        """Convert dictionary to QFLARESecurityConfig"""
        # Helper function to convert dict to dataclass
        def dict_to_dataclass(cls, data):
            if data is None:
                return cls()
            if isinstance(data, dict):
                field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
                kwargs = {}
                for key, value in data.items():
                    if key in field_types:
                        field_type = field_types[key]
                        # Handle nested dataclasses
                        if hasattr(field_type, '__dataclass_fields__'):
                            kwargs[key] = dict_to_dataclass(field_type, value)
                        else:
                            kwargs[key] = value
                return cls(**kwargs)
            return data
            
        return dict_to_dataclass(QFLARESecurityConfig, config_data)
        
    def get_scanner_config(self, scanner_name: str) -> ScannerConfig:
        """Get configuration for a specific scanner"""
        if not self.config:
            self.load_config()
            
        scanner_configs = {
            'dependency': self.config.dependency_scanning,
            'static': self.config.static_analysis,
            'container': self.config.container_scanning,
            'dynamic': self.config.dynamic_scanning
        }
        
        return scanner_configs.get(scanner_name, ScannerConfig())
        
    def is_scanner_enabled(self, scanner_name: str) -> bool:
        """Check if a scanner is enabled"""
        config = self.get_scanner_config(scanner_name)
        return config.enabled
        
    def get_security_policy(self) -> SecurityPolicy:
        """Get security policy configuration"""
        if not self.config:
            self.load_config()
        return self.config.security_policy
        
    def validate_findings(self, findings_by_severity: Dict[str, int]) -> Dict[str, Any]:
        """Validate findings against security policy"""
        policy = self.get_security_policy()
        
        validation_result = {
            'passed': True,
            'violations': [],
            'metrics': findings_by_severity
        }
        
        # Check against policy limits
        if findings_by_severity.get('CRITICAL', 0) > policy.max_critical_findings:
            validation_result['passed'] = False
            validation_result['violations'].append(
                f"Critical findings exceed limit: {findings_by_severity['CRITICAL']} > {policy.max_critical_findings}"
            )
            
        if findings_by_severity.get('HIGH', 0) > policy.max_high_findings:
            validation_result['passed'] = False
            validation_result['violations'].append(
                f"High severity findings exceed limit: {findings_by_severity['HIGH']} > {policy.max_high_findings}"
            )
            
        if findings_by_severity.get('MEDIUM', 0) > policy.max_medium_findings:
            validation_result['passed'] = False
            validation_result['violations'].append(
                f"Medium severity findings exceed limit: {findings_by_severity['MEDIUM']} > {policy.max_medium_findings}"
            )
            
        return validation_result

# Default security configuration template
DEFAULT_SECURITY_CONFIG = """
# QFLARE Security Scanning Configuration
project_name: "QFLARE"
version: "1.0.0"

# Dependency Scanning Configuration
dependency_scanning:
  enabled: true
  timeout: 180
  severity_threshold: "LOW"
  exclude_patterns:
    - "*/venv/*"
    - "*/node_modules/*"
    - "*/.git/*"
    - "*/build/*"
    - "*/__pycache__/*"
  arguments:
    safety:
      - "--json"
      - "--output"
      - "json"
    bandit:
      - "-f"
      - "json"
      - "--skip"
      - "B101,B601"
    semgrep:
      - "--config=auto"
      - "--json"
    pip-audit:
      - "--format=json"
      - "--desc"

# Static Analysis Configuration
static_analysis:
  enabled: true
  timeout: 300
  severity_threshold: "MEDIUM"
  exclude_patterns:
    - "*/tests/*"
    - "*/test_*"
    - "*_test.py"
    - "*/migrations/*"
    - "*/alembic/versions/*"
  custom_rules:
    - "security/rules/qflare_crypto_rules.yml"
    - "security/rules/federated_learning_rules.yml"

# Container Scanning Configuration
container_scanning:
  enabled: true
  timeout: 600
  severity_threshold: "HIGH"
  arguments:
    trivy:
      - "--format"
      - "json"
      - "--timeout"
      - "10m"
    docker_scout:
      - "--format"
      - "json"

# Dynamic Application Security Testing
dynamic_scanning:
  enabled: false  # Only enable in staging/test environments
  timeout: 1800   # 30 minutes
  severity_threshold: "MEDIUM"
  arguments:
    base_urls:
      - "http://localhost:8000"
      - "https://qflare-staging.example.com"
    auth_endpoints:
      - "/api/v1/auth/login"
    api_endpoints:
      - "/api/v1"
      - "/docs"
      - "/swagger"

# Security Policy
security_policy:
  max_critical_findings: 0
  max_high_findings: 3
  max_medium_findings: 15
  fail_on_error: true
  require_approval: true
  notification_channels:
    - "security-team"
    - "dev-team"

# CI/CD Integration
ci_integration:
  github_actions: true
  azure_pipelines: false
  jenkins: false
  slack_webhook: null
  teams_webhook: null
  email_notifications:
    - "security@qflare.dev"
  sarif_upload: true
  badge_generation: true

# Report Settings
report_formats:
  - "html"
  - "json"
  - "sarif"
  - "csv"
report_retention_days: 30
baseline_comparison: true
"""

def create_default_config(config_path: Path):
    """Create default security configuration file"""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(DEFAULT_SECURITY_CONFIG)
        
    print(f"Default security configuration created at: {config_path}")

def main():
    """CLI for security configuration management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QFLARE Security Configuration Manager")
    parser.add_argument("--config-path", type=str, help="Path to security configuration file")
    parser.add_argument("--create-default", action="store_true", help="Create default configuration")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    parser.add_argument("--show", action="store_true", help="Show current configuration")
    
    args = parser.parse_args()
    
    config_path = Path(args.config_path) if args.config_path else Path("security/config/security_config.yml")
    
    if args.create_default:
        create_default_config(config_path)
        return
        
    config_manager = SecurityConfigManager(config_path)
    
    if args.validate:
        try:
            config = config_manager.load_config()
            print("✅ Configuration is valid")
            print(f"Project: {config.project_name} v{config.version}")
            
            # Show scanner status
            scanners = {
                'Dependency Scanning': config.dependency_scanning.enabled,
                'Static Analysis': config.static_analysis.enabled,
                'Container Scanning': config.container_scanning.enabled,
                'Dynamic Scanning': config.dynamic_scanning.enabled
            }
            
            print("\n📊 Scanner Status:")
            for scanner, enabled in scanners.items():
                status = "✅ Enabled" if enabled else "❌ Disabled"
                print(f"  {scanner}: {status}")
                
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            
    if args.show:
        config = config_manager.load_config()
        print(yaml.dump(asdict(config), default_flow_style=False, indent=2))

if __name__ == "__main__":
    main()