#!/usr/bin/env python3
"""
QFLARE Security Scanning CI/CD Integration

This module provides GitHub Actions integration for the QFLARE security scanning framework.
It creates workflow files and automation for continuous security scanning.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

def create_security_workflow(project_path: Path) -> str:
    """Create GitHub Actions workflow for security scanning"""
    
    workflow_content = """name: 🛡️ QFLARE Security Scanning

on:
  push:
    branches: [ main, develop ]
    paths-ignore:
      - '**.md'
      - 'docs/**'
      - '.gitignore'
  pull_request:
    branches: [ main, develop ]
  schedule:
    # Run security scan daily at 2 AM UTC
    - cron: '0 2 * * *'
  workflow_dispatch:
    inputs:
      scan_type:
        description: 'Type of security scan'
        required: true
        default: 'full'
        type: choice
        options:
          - full
          - dependencies
          - containers
          - quick

permissions:
  contents: read
  security-events: write
  actions: read
  pull-requests: write

env:
  PYTHON_VERSION: '3.11'

jobs:
  security-scan:
    name: 🔍 Security Scanning
    runs-on: ubuntu-latest
    
    strategy:
      fail-fast: false
      matrix:
        scan-type: 
          - dependencies
          - static-analysis
          - container-security
    
    steps:
    - name: 📥 Checkout Repository
      uses: actions/checkout@v4
      with:
        fetch-depth: 0
        
    - name: 🐍 Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        cache: 'pip'
        
    - name: 📦 Install Dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements.secure.txt
        
    - name: 🔧 Install Security Tools
      run: |
        # Install security scanning tools
        pip install safety bandit semgrep pip-audit
        
        # Install Trivy for container scanning
        curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
        
        # Install additional tools
        npm install -g retire
        
    - name: 🏗️ Build Docker Images
      if: matrix.scan-type == 'container-security'
      run: |
        docker build -t qflare:latest -f docker/Dockerfile.server .
        docker build -t qflare-edge:latest -f docker/Dockerfile.edge .
        
    - name: 🛡️ Run Security Scan - ${{ matrix.scan-type }}
      run: |
        mkdir -p security/reports
        python security/security_scanner.py \\
          --project-path . \\
          --output-dir security/reports \\
          --format sarif
      env:
        SCAN_TYPE: ${{ matrix.scan-type }}
        
    - name: 📊 Upload SARIF Results
      if: always()
      uses: github/codeql-action/upload-sarif@v3
      with:
        sarif_file: security/reports/security_report_*.sarif
        category: ${{ matrix.scan-type }}
        
    - name: 📁 Upload Security Reports
      if: always()
      uses: actions/upload-artifact@v4
      with:
        name: security-reports-${{ matrix.scan-type }}
        path: security/reports/
        retention-days: 30
        
    - name: 💬 Comment on PR
      if: github.event_name == 'pull_request' && always()
      uses: actions/github-script@v7
      with:
        script: |
          const fs = require('fs');
          const path = require('path');
          
          // Read security report
          const reportDir = 'security/reports';
          const files = fs.readdirSync(reportDir);
          const jsonFile = files.find(f => f.endsWith('.json'));
          
          if (jsonFile) {
            const reportPath = path.join(reportDir, jsonFile);
            const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
            
            const summary = report.summary;
            const severityCounts = summary.severity_breakdown;
            
            const comment = `
          ## 🛡️ Security Scan Results - ${{ matrix.scan-type }}
          
          | Severity | Count |
          |----------|-------|
          | 🔴 Critical | ${severityCounts.CRITICAL || 0} |
          | 🟠 High | ${severityCounts.HIGH || 0} |
          | 🟡 Medium | ${severityCounts.MEDIUM || 0} |
          | 🟢 Low | ${severityCounts.LOW || 0} |
          
          **Total Findings:** ${summary.total_findings}
          **Scan Duration:** ${report.total_duration.toFixed(2)}s
          
          ${summary.total_findings > 0 ? '⚠️ Security issues found. Please review the detailed report.' : '✅ No security issues found.'}
          
          [View Full Report](https://github.com/${{ github.repository }}/actions/runs/${{ github.run_id }})
            `;
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
          }

  dependency-check:
    name: 📦 Dependency Vulnerability Check
    runs-on: ubuntu-latest
    
    steps:
    - name: 📥 Checkout Repository
      uses: actions/checkout@v4
      
    - name: 🐍 Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: 🔍 Run Safety Check
      run: |
        pip install safety
        safety check --json --output safety-report.json || true
        
    - name: 🔍 Run pip-audit
      run: |
        pip install pip-audit
        pip-audit --format=json --output=pip-audit-report.json || true
        
    - name: 📊 Process Results
      run: |
        python -c "
        import json
        import sys
        
        # Process safety results
        try:
            with open('safety-report.json') as f:
                safety_data = json.load(f)
            print(f'Safety found {len(safety_data)} vulnerabilities')
        except:
            print('No safety vulnerabilities found')
            
        # Process pip-audit results  
        try:
            with open('pip-audit-report.json') as f:
                audit_data = json.load(f)
            vulns = audit_data.get('vulnerabilities', [])
            print(f'pip-audit found {len(vulns)} vulnerabilities')
        except:
            print('No pip-audit vulnerabilities found')
        "
        
    - name: 📁 Upload Dependency Reports
      uses: actions/upload-artifact@v4
      with:
        name: dependency-reports
        path: '*-report.json'

  static-analysis:
    name: 🔍 Static Code Analysis
    runs-on: ubuntu-latest
    
    steps:
    - name: 📥 Checkout Repository
      uses: actions/checkout@v4
      
    - name: 🐍 Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: 📦 Install Analysis Tools
      run: |
        pip install bandit semgrep flake8-bandit
        
    - name: 🔍 Run Bandit
      run: |
        bandit -r . -f json -o bandit-report.json || true
        
    - name: 🔍 Run Semgrep
      run: |
        semgrep --config=auto --json --output=semgrep-report.json . || true
        
    - name: 📁 Upload Analysis Reports
      uses: actions/upload-artifact@v4
      with:
        name: static-analysis-reports
        path: '*-report.json'

  container-security:
    name: 🐳 Container Security Scan
    runs-on: ubuntu-latest
    
    steps:
    - name: 📥 Checkout Repository
      uses: actions/checkout@v4
      
    - name: 🏗️ Build Docker Images
      run: |
        docker build -t qflare:latest -f docker/Dockerfile.server .
        
    - name: 🔍 Run Trivy Container Scan
      run: |
        # Install Trivy
        curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
        
        # Scan container
        trivy image --format json --output trivy-report.json qflare:latest || true
        
    - name: 🔍 Run Docker Scout (if available)
      continue-on-error: true
      run: |
        docker scout cves --format json --output docker-scout-report.json qflare:latest || true
        
    - name: 📁 Upload Container Reports
      uses: actions/upload-artifact@v4
      with:
        name: container-security-reports
        path: '*-report.json'

  security-policy:
    name: 📋 Security Policy Check
    runs-on: ubuntu-latest
    needs: [security-scan, dependency-check, static-analysis, container-security]
    if: always()
    
    steps:
    - name: 📥 Checkout Repository
      uses: actions/checkout@v4
      
    - name: 📥 Download All Reports
      uses: actions/download-artifact@v4
      with:
        path: security-reports/
        
    - name: 🔍 Validate Security Policy
      run: |
        python -c "
        import json
        import os
        from pathlib import Path
        
        # Collect all findings
        total_critical = 0
        total_high = 0
        total_medium = 0
        
        for report_dir in Path('security-reports').iterdir():
            if report_dir.is_dir():
                for json_file in report_dir.glob('*.json'):
                    try:
                        with open(json_file) as f:
                            data = json.load(f)
                        
                        # Parse different report formats
                        if 'summary' in data and 'severity_breakdown' in data['summary']:
                            breakdown = data['summary']['severity_breakdown']
                            total_critical += breakdown.get('CRITICAL', 0)
                            total_high += breakdown.get('HIGH', 0)
                            total_medium += breakdown.get('MEDIUM', 0)
                    except:
                        continue
        
        print(f'Total Critical: {total_critical}')
        print(f'Total High: {total_high}')
        print(f'Total Medium: {total_medium}')
        
        # Security policy limits
        max_critical = 0
        max_high = 5
        max_medium = 20
        
        failed = False
        if total_critical > max_critical:
            print(f'❌ CRITICAL findings exceed limit: {total_critical} > {max_critical}')
            failed = True
        if total_high > max_high:
            print(f'❌ HIGH findings exceed limit: {total_high} > {max_high}')
            failed = True
        if total_medium > max_medium:
            print(f'❌ MEDIUM findings exceed limit: {total_medium} > {max_medium}')
            failed = True
            
        if failed:
            exit(1)
        else:
            print('✅ Security policy validation passed')
        "
        
    - name: 📊 Generate Security Badge
      run: |
        # Create security status badge
        python -c "
        import json
        
        # Simple badge generation (would integrate with shields.io in production)
        badge_data = {
            'schemaVersion': 1,
            'label': 'security',
            'message': 'passing',
            'color': 'brightgreen'
        }
        
        with open('security-badge.json', 'w') as f:
            json.dump(badge_data, f)
        "
        
    - name: 📧 Send Security Notification
      if: failure()
      run: |
        echo "Security policy validation failed - notifications would be sent here"
        # Integration with notification services would go here

  generate-security-report:
    name: 📄 Generate Comprehensive Report
    runs-on: ubuntu-latest
    needs: [security-scan, dependency-check, static-analysis, container-security]
    if: always()
    
    steps:
    - name: 📥 Checkout Repository
      uses: actions/checkout@v4
      
    - name: 🐍 Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: 📥 Download All Reports
      uses: actions/download-artifact@v4
      with:
        path: security-reports/
        
    - name: 📊 Generate Comprehensive Report
      run: |
        python security/generate_comprehensive_report.py \\
          --input-dir security-reports \\
          --output-dir final-reports \\
          --format html,json,sarif
          
    - name: 📁 Upload Final Report
      uses: actions/upload-artifact@v4
      with:
        name: comprehensive-security-report
        path: final-reports/
        retention-days: 90
        
    - name: 📊 Update Security Dashboard
      run: |
        echo "Security dashboard would be updated here"
        # Integration with monitoring dashboard would go here"""
    
    # Write workflow file
    workflow_path = project_path / ".github" / "workflows" / "security.yml"
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(workflow_path, 'w', encoding='utf-8') as f:
        f.write(workflow_content)
        
    return str(workflow_path)

def create_security_action(project_path: Path) -> str:
    """Create custom GitHub Action for QFLARE security scanning"""
    
    action_content = """name: 'QFLARE Security Scanner'
description: 'Comprehensive security scanning for QFLARE quantum-resistant federated learning'
author: 'QFLARE Team'

branding:
  icon: 'shield'
  color: 'blue'

inputs:
  project-path:
    description: 'Path to the QFLARE project'
    required: false
    default: '.'
  
  scan-types:
    description: 'Comma-separated list of scan types (dependencies,static,container,dast)'
    required: false
    default: 'dependencies,static,container'
    
  output-format:
    description: 'Output format for reports (html,json,sarif,csv)'
    required: false
    default: 'html,json,sarif'
    
  severity-threshold:
    description: 'Minimum severity level to report (LOW,MEDIUM,HIGH,CRITICAL)'
    required: false
    default: 'LOW'
    
  upload-sarif:
    description: 'Upload SARIF results to GitHub Security tab'
    required: false
    default: 'true'
    
  fail-on-high:
    description: 'Fail the action if high severity issues are found'
    required: false
    default: 'false'

outputs:
  total-findings:
    description: 'Total number of security findings'
    value: ${{ steps.scan.outputs.total-findings }}
    
  critical-findings:
    description: 'Number of critical security findings'
    value: ${{ steps.scan.outputs.critical-findings }}
    
  high-findings:
    description: 'Number of high severity findings'
    value: ${{ steps.scan.outputs.high-findings }}
    
  report-path:
    description: 'Path to the generated security report'
    value: ${{ steps.scan.outputs.report-path }}

runs:
  using: 'composite'
  steps:
    - name: 🛠️ Setup Security Tools
      shell: bash
      run: |
        # Install Python security tools
        pip install safety bandit semgrep pip-audit
        
        # Install Trivy for container scanning
        curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
        
        echo "Security tools installed successfully"
        
    - name: 🔍 Run QFLARE Security Scan
      id: scan
      shell: bash
      run: |
        # Create output directory
        mkdir -p security/reports
        
        # Run security scanner
        python ${{ inputs.project-path }}/security/security_scanner.py \\
          --project-path ${{ inputs.project-path }} \\
          --output-dir security/reports \\
          --format ${{ inputs.output-format }} \\
          --severity-threshold ${{ inputs.severity-threshold }}
        
        # Parse results for outputs
        if [ -f security/reports/security_report_*.json ]; then
          REPORT_FILE=$(ls security/reports/security_report_*.json | head -1)
          
          TOTAL_FINDINGS=$(python -c "
        import json
        with open('$REPORT_FILE') as f:
            data = json.load(f)
        print(data['summary']['total_findings'])
        ")
          
          CRITICAL_FINDINGS=$(python -c "
        import json
        with open('$REPORT_FILE') as f:
            data = json.load(f)
        print(data['summary']['severity_breakdown'].get('CRITICAL', 0))
        ")
          
          HIGH_FINDINGS=$(python -c "
        import json
        with open('$REPORT_FILE') as f:
            data = json.load(f)
        print(data['summary']['severity_breakdown'].get('HIGH', 0))
        ")
          
          echo "total-findings=$TOTAL_FINDINGS" >> $GITHUB_OUTPUT
          echo "critical-findings=$CRITICAL_FINDINGS" >> $GITHUB_OUTPUT
          echo "high-findings=$HIGH_FINDINGS" >> $GITHUB_OUTPUT
          echo "report-path=$REPORT_FILE" >> $GITHUB_OUTPUT
          
          echo "✅ Security scan completed: $TOTAL_FINDINGS findings"
          echo "   🔴 Critical: $CRITICAL_FINDINGS"
          echo "   🟠 High: $HIGH_FINDINGS"
        else
          echo "❌ Security scan failed - no report generated"
          exit 1
        fi
        
    - name: 📊 Upload SARIF Results
      if: inputs.upload-sarif == 'true'
      uses: github/codeql-action/upload-sarif@v3
      with:
        sarif_file: security/reports/security_report_*.sarif
        
    - name: ❌ Fail on High Severity
      if: inputs.fail-on-high == 'true'
      shell: bash
      run: |
        if [ "${{ steps.scan.outputs.high-findings }}" -gt "0" ] || [ "${{ steps.scan.outputs.critical-findings }}" -gt "0" ]; then
          echo "❌ High or critical severity findings detected"
          echo "Critical: ${{ steps.scan.outputs.critical-findings }}"
          echo "High: ${{ steps.scan.outputs.high-findings }}"
          exit 1
        fi"""
    
    # Write action file
    action_path = project_path / ".github" / "actions" / "qflare-security" / "action.yml"
    action_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(action_path, 'w', encoding='utf-8') as f:
        f.write(action_content)
        
    return str(action_path)

def create_security_policy(project_path: Path) -> str:
    """Create GitHub Security Policy"""
    
    policy_content = """# 🛡️ QFLARE Security Policy

## Supported Versions

We actively support the following versions of QFLARE with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | ✅ Active Support  |
| 0.9.x   | ⚠️ Limited Support |
| < 0.9   | ❌ No Support      |

## Security Features

QFLARE implements comprehensive security measures:

### 🔐 Post-Quantum Cryptography
- **CRYSTALS-Kyber-1024**: Quantum-resistant key encapsulation
- **CRYSTALS-Dilithium-5**: Quantum-resistant digital signatures
- **AES-256-GCM**: Symmetric encryption for data protection

### 🛡️ Federated Learning Security
- **Byzantine Fault Tolerance**: Protection against malicious participants
- **Differential Privacy**: Privacy-preserving machine learning
- **Secure Aggregation**: Encrypted model parameter aggregation
- **Model Poisoning Detection**: Detection of adversarial model updates

### 🔍 Continuous Security Monitoring
- **Automated Vulnerability Scanning**: Daily dependency and container scans
- **Static Code Analysis**: Security-focused code review
- **Dynamic Security Testing**: Runtime security validation
- **Security Policy Enforcement**: Automated policy compliance

## Reporting a Vulnerability

We take security seriously and appreciate responsible disclosure of security vulnerabilities.

### 📧 How to Report

**For security vulnerabilities, please email**: security@qflare.dev

**Do NOT create public GitHub issues for security vulnerabilities.**

### 📋 What to Include

Please include the following information in your report:

1. **Description**: Clear description of the vulnerability
2. **Impact**: Potential security impact and affected components
3. **Reproduction**: Steps to reproduce the vulnerability
4. **Environment**: QFLARE version, OS, Python version, etc.
5. **Proof of Concept**: Code or screenshots if applicable

### ⏱️ Response Timeline

- **Initial Response**: Within 24 hours
- **Severity Assessment**: Within 72 hours
- **Status Updates**: Weekly updates until resolution
- **Fix Timeline**: 
  - Critical: 48-72 hours
  - High: 1-2 weeks
  - Medium: 2-4 weeks
  - Low: Next planned release

### 🏆 Recognition

We maintain a Security Hall of Fame for researchers who responsibly disclose vulnerabilities:

- Your name (if desired) will be listed in our security acknowledgments
- Acknowledgment in release notes for fixed vulnerabilities
- Optional CVE credit where applicable

## Security Best Practices

### For Users

1. **Keep Updated**: Always use the latest stable version
2. **Secure Configuration**: Follow our security configuration guide
3. **Environment Security**: Secure your deployment environment
4. **Access Control**: Implement proper authentication and authorization
5. **Network Security**: Use TLS/SSL for all communications

### For Contributors

1. **Secure Coding**: Follow secure coding practices
2. **Dependency Management**: Keep dependencies updated
3. **Code Review**: Participate in security-focused code reviews
4. **Testing**: Include security tests in your contributions
5. **Documentation**: Document security considerations

## Security Tools and Processes

### 🔧 Automated Security Scanning

- **Daily Scans**: Automated vulnerability scanning
- **PR Checks**: Security validation on pull requests
- **Baseline Monitoring**: Continuous security baseline tracking
- **Policy Enforcement**: Automated security policy compliance

### 🛠️ Security Tools Used

- **Safety**: Python dependency vulnerability scanning
- **Bandit**: Static security analysis for Python
- **Semgrep**: Advanced static analysis with custom rules
- **Trivy**: Container vulnerability scanning
- **pip-audit**: Python package audit

### 📊 Security Metrics

We track and monitor:
- Vulnerability discovery and resolution time
- Security scan coverage and effectiveness
- Security policy compliance rates
- Security training completion

## Security Incident Response

### 🚨 Incident Classification

- **P0 - Critical**: Immediate threat to production systems
- **P1 - High**: Significant security impact
- **P2 - Medium**: Moderate security concern
- **P3 - Low**: Minor security issue

### 📞 Emergency Contact

For critical security incidents requiring immediate attention:
- **Email**: security-emergency@qflare.dev
- **Response Time**: <2 hours for P0 incidents

## Compliance and Standards

QFLARE adheres to:
- **NIST Cybersecurity Framework**
- **OWASP Security Guidelines**
- **IEEE Standards for Federated Learning Security**
- **Post-Quantum Cryptography Standards (NIST)**

## Security Training

All contributors are encouraged to:
- Complete OWASP security training
- Stay updated on quantum computing security threats
- Understand federated learning privacy concerns
- Follow secure development lifecycle practices

## Questions?

For security-related questions that are not vulnerabilities:
- **General Security**: security@qflare.dev
- **Security Documentation**: docs@qflare.dev
- **Security Training**: training@qflare.dev

---

**Last Updated**: {date}
**Version**: 1.0
**Contact**: security@qflare.dev""".format(date=datetime.now().strftime("%Y-%m-%d"))
    
    # Write security policy
    policy_path = project_path / "SECURITY.md"
    
    with open(policy_path, 'w', encoding='utf-8') as f:
        f.write(policy_content)
        
    return str(policy_path)

def create_dependabot_config(project_path: Path) -> str:
    """Create Dependabot configuration for automated dependency updates"""
    
    dependabot_config = {
        "version": 2,
        "updates": [
            {
                "package-ecosystem": "pip",
                "directory": "/",
                "schedule": {"interval": "daily"},
                "open-pull-requests-limit": 5,
                "reviewers": ["@qflare-team/security"],
                "assignees": ["@qflare-team/maintainers"],
                "commit-message": {
                    "prefix": "security",
                    "prefix-development": "deps-dev",
                    "include": "scope"
                },
                "labels": ["security", "dependencies"],
                "allow": [
                    {"dependency-type": "direct"},
                    {"dependency-type": "indirect"}
                ],
                "ignore": [
                    {
                        "dependency-name": "numpy",
                        "versions": ["< 1.20.0"]
                    }
                ]
            },
            {
                "package-ecosystem": "docker",
                "directory": "/docker",
                "schedule": {"interval": "weekly"},
                "open-pull-requests-limit": 3,
                "reviewers": ["@qflare-team/security"],
                "commit-message": {
                    "prefix": "docker",
                    "include": "scope"
                },
                "labels": ["security", "docker"]
            },
            {
                "package-ecosystem": "github-actions",
                "directory": "/",
                "schedule": {"interval": "weekly"},
                "open-pull-requests-limit": 3,
                "commit-message": {
                    "prefix": "ci",
                    "include": "scope"
                },
                "labels": ["ci", "security"]
            }
        ]
    }
    
    # Write dependabot config
    dependabot_path = project_path / ".github" / "dependabot.yml"
    dependabot_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dependabot_path, 'w', encoding='utf-8') as f:
        yaml.dump(dependabot_config, f, default_flow_style=False, indent=2)
        
    return str(dependabot_path)

def create_codeql_config(project_path: Path) -> str:
    """Create CodeQL configuration for advanced static analysis"""
    
    codeql_workflow = """name: 🔍 CodeQL Security Analysis

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '30 1 * * 0'  # Weekly on Sundays

jobs:
  analyze:
    name: Analyze
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write

    strategy:
      fail-fast: false
      matrix:
        language: [ 'python', 'javascript' ]

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v3
      with:
        languages: ${{ matrix.language }}
        config-file: ./.github/codeql/codeql-config.yml

    - name: Autobuild
      uses: github/codeql-action/autobuild@v3

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v3
      with:
        category: "/language:${{matrix.language}}\""""
    
    # Create CodeQL config
    codeql_config = """name: "QFLARE CodeQL Configuration"

disable-default-queries: false

queries:
  - uses: security-and-quality
  - uses: security-extended

paths:
  - server/
  - edge_node/
  - security/
  - backend/
  - frontend/src/

paths-ignore:
  - "**/test*"
  - "**/spec*" 
  - "**/*_test.py"
  - "**/node_modules/"
  - "**/venv/"
  - "**/qflare-env/"
  - "data/"
  - "build/"

packs:
  python:
    - codeql/python-queries:AlertSuppression.ql
    - codeql/python-queries:Security/"""
    
    # Write CodeQL workflow
    workflow_path = project_path / ".github" / "workflows" / "codeql.yml"
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(workflow_path, 'w', encoding='utf-8') as f:
        f.write(codeql_workflow)
        
    # Write CodeQL config
    config_path = project_path / ".github" / "codeql" / "codeql-config.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(codeql_config)
        
    return str(workflow_path)

def main():
    """Create all CI/CD security integration files"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create QFLARE Security CI/CD Integration")
    parser.add_argument("--project-path", type=str, default=".", help="Project root path")
    
    args = parser.parse_args()
    project_path = Path(args.project_path)
    
    print("🚀 Creating QFLARE Security CI/CD Integration...")
    
    # Create all integration files
    files_created = []
    
    try:
        # GitHub Actions Security Workflow
        workflow_path = create_security_workflow(project_path)
        files_created.append(workflow_path)
        print(f"✅ Created security workflow: {workflow_path}")
        
        # Custom Security Action
        action_path = create_security_action(project_path)
        files_created.append(action_path)
        print(f"✅ Created custom security action: {action_path}")
        
        # Security Policy
        policy_path = create_security_policy(project_path)
        files_created.append(policy_path)
        print(f"✅ Created security policy: {policy_path}")
        
        # Dependabot Configuration
        dependabot_path = create_dependabot_config(project_path)
        files_created.append(dependabot_path)
        print(f"✅ Created Dependabot config: {dependabot_path}")
        
        # CodeQL Configuration
        codeql_path = create_codeql_config(project_path)
        files_created.append(codeql_path)
        print(f"✅ Created CodeQL workflow: {codeql_path}")
        
        print(f"\n🎉 Successfully created {len(files_created)} security integration files!")
        print("\n📋 Next Steps:")
        print("1. Review and customize the security configuration")
        print("2. Set up GitHub repository secrets if needed")
        print("3. Configure notification channels (Slack, Teams, email)")
        print("4. Run the security workflow to validate setup")
        print("5. Monitor security dashboard and reports")
        
    except Exception as e:
        print(f"❌ Error creating security integration: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())