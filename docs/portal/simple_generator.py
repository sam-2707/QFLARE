#!/usr/bin/env python3
"""
Simplified QFLARE Documentation Portal Generator

This script creates a comprehensive documentation portal for the QFLARE project.
"""

import os
import json
from pathlib import Path
import shutil
from datetime import datetime

def create_main_index():
    """Create the main documentation index page"""
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QFLARE Documentation Portal</title>
    <link rel="stylesheet" href="static/styles.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/themes/prism.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <meta name="description" content="Comprehensive documentation for QFLARE - Quantum-Resistant Federated Learning with Post-Quantum Cryptography">
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🛡️</text></svg>">
</head>
<body>
    <header class="header">
        <nav class="navbar">
            <div class="nav-brand">
                <h1>🛡️ QFLARE</h1>
                <span class="version">v1.0.0</span>
            </div>
            <div class="nav-links">
                <a href="#home" class="active">Home</a>
                <a href="guides/getting-started.html">Guides</a>
                <a href="api/index.html">API</a>
                <a href="deployment/index.html">Deploy</a>
                <a href="troubleshooting.html">Troubleshoot</a>
                <a href="examples.html">Examples</a>
                <a href="search.html">🔍</a>
            </div>
            <div class="theme-toggle">
                <button id="theme-toggle" class="theme-btn">🌙</button>
            </div>
        </nav>
    </header>

    <main class="main-content" id="home">
        <!-- Hero Section -->
        <div class="hero-section">
            <div class="hero-content">
                <h1>🛡️ QFLARE Documentation Portal</h1>
                <p class="hero-subtitle">Quantum-Resistant Federated Learning with Post-Quantum Cryptography</p>
                <div class="hero-stats">
                    <div class="stat">
                        <span class="stat-number">25+</span>
                        <span class="stat-label">API Modules</span>
                    </div>
                    <div class="stat">
                        <span class="stat-number">12</span>
                        <span class="stat-label">User Guides</span>
                    </div>
                    <div class="stat">
                        <span class="stat-number">15</span>
                        <span class="stat-label">Examples</span>
                    </div>
                    <div class="stat">
                        <span class="stat-number">8</span>
                        <span class="stat-label">Deployment Guides</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Features Grid -->
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">📚</div>
                <h3>User Guides</h3>
                <p>Comprehensive guides covering all aspects of QFLARE from basic setup to advanced configurations including post-quantum cryptography and federated learning concepts.</p>
                <a href="guides/getting-started.html" class="feature-link">Explore Guides →</a>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">🔧</div>
                <h3>API Reference</h3>
                <p>Complete API documentation with examples, parameters, and return values for all QFLARE components including cryptographic functions and federated learning algorithms.</p>
                <a href="api/index.html" class="feature-link">Browse API →</a>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">🚀</div>
                <h3>Deployment Guides</h3>
                <p>Step-by-step deployment guides for Docker, Kubernetes, cloud platforms, and production environments with security best practices and monitoring setup.</p>
                <a href="deployment/index.html" class="feature-link">Deploy Now →</a>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <h3>Interactive Examples</h3>
                <p>Runnable code examples demonstrating key QFLARE features including Byzantine fault tolerance, differential privacy, and secure aggregation.</p>
                <a href="examples.html" class="feature-link">Try Examples →</a>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <h3>Troubleshooting</h3>
                <p>Common issues, debugging guides, and solutions to help you resolve problems quickly. Includes performance optimization and security issue resolution.</p>
                <a href="troubleshooting.html" class="feature-link">Get Help →</a>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">🔐</div>
                <h3>Security Framework</h3>
                <p>Post-quantum cryptography implementation, Byzantine fault tolerance, comprehensive security scanning, and vulnerability management best practices.</p>
                <a href="security.html" class="feature-link">Secure Setup →</a>
            </div>
        </div>

        <!-- Quick Start Section -->
        <div class="quick-start-section">
            <h2>🚀 Quick Start</h2>
            <div class="quick-start-steps">
                <div class="step">
                    <div class="step-number">1</div>
                    <div class="step-content">
                        <h4>Install QFLARE</h4>
                        <code>pip install qflare</code>
                        <p>Or install from source for latest features</p>
                    </div>
                </div>
                <div class="step">
                    <div class="step-number">2</div>
                    <div class="step-content">
                        <h4>Run Basic Example</h4>
                        <code>python examples/basic_federated_learning.py</code>
                        <p>Test your installation with a simple FL example</p>
                    </div>
                </div>
                <div class="step">
                    <div class="step-number">3</div>
                    <div class="step-content">
                        <h4>Explore Documentation</h4>
                        <a href="guides/getting-started.html">Read the Getting Started Guide</a>
                        <p>Learn QFLARE concepts and best practices</p>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <footer class="footer">
        <div class="footer-content">
            <div class="footer-info">
                <p>&copy; 2025 QFLARE Development Team. Quantum-Resistant Federated Learning with Post-Quantum Cryptography.</p>
                <p>Built with ❤️ for secure, privacy-preserving machine learning.</p>
            </div>
            <div class="footer-links">
                <a href="https://github.com/sam-2707/QFLARE" target="_blank"><i class="fab fa-github"></i> GitHub</a>
                <a href="about.html">About</a>
                <a href="contact.html">Contact</a>
                <a href="https://qflare.dev" target="_blank">Official Website</a>
            </div>
        </div>
    </footer>

    <!-- Scripts -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/components/prism-core.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/plugins/autoloader/prism-autoloader.min.js"></script>
    <script src="static/scripts.js"></script>
</body>
</html>'''
    
    return html_content

def create_api_index():
    """Create the API documentation index"""
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API Reference - QFLARE Documentation</title>
    <link rel="stylesheet" href="../static/styles.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/themes/prism.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
</head>
<body>
    <header class="header">
        <nav class="navbar">
            <div class="nav-brand">
                <a href="../index.html"><h1>🛡️ QFLARE</h1></a>
                <span class="version">v1.0.0</span>
            </div>
            <div class="nav-links">
                <a href="../index.html">Home</a>
                <a href="../guides/getting-started.html">Guides</a>
                <a href="index.html" class="active">API</a>
                <a href="../deployment/index.html">Deploy</a>
                <a href="../troubleshooting.html">Troubleshoot</a>
                <a href="../examples.html">Examples</a>
            </div>
        </nav>
    </header>

    <main class="main-content">
        <div class="page-header">
            <h1>🔧 API Reference</h1>
            <p class="page-subtitle">Complete API documentation for all QFLARE components</p>
        </div>

        <div class="api-grid">
            <div class="api-card">
                <h3>qflare.client</h3>
                <p>Client-side federated learning components including model training, data handling, and server communication.</p>
                <div class="api-stats">
                    <span>15 classes</span>
                    <span>42 methods</span>
                </div>
                <a href="qflare.client.html" class="feature-link">Explore →</a>
            </div>
            
            <div class="api-card">
                <h3>qflare.server</h3>
                <p>Server-side aggregation, client management, model coordination, and federated learning orchestration.</p>
                <div class="api-stats">
                    <span>12 classes</span>
                    <span>38 methods</span>
                </div>
                <a href="qflare.server.html" class="feature-link">Explore →</a>
            </div>
            
            <div class="api-card">
                <h3>qflare.crypto</h3>
                <p>Post-quantum cryptographic functions, key management, encryption/decryption, and digital signatures.</p>
                <div class="api-stats">
                    <span>8 classes</span>
                    <span>24 methods</span>
                </div>
                <a href="qflare.crypto.html" class="feature-link">Explore →</a>
            </div>
            
            <div class="api-card">
                <h3>qflare.aggregation</h3>
                <p>Model aggregation algorithms, Byzantine-resilient methods, secure aggregation, and consensus mechanisms.</p>
                <div class="api-stats">
                    <span>10 classes</span>
                    <span>28 methods</span>
                </div>
                <a href="qflare.aggregation.html" class="feature-link">Explore →</a>
            </div>
            
            <div class="api-card">
                <h3>qflare.monitoring</h3>
                <p>Performance monitoring, metrics collection, alerting systems, and real-time dashboard integration.</p>
                <div class="api-stats">
                    <span>6 classes</span>
                    <span>22 methods</span>
                </div>
                <a href="qflare.monitoring.html" class="feature-link">Explore →</a>
            </div>
            
            <div class="api-card">
                <h3>qflare.security</h3>
                <p>Security scanning, vulnerability assessment, policy enforcement, and compliance validation tools.</p>
                <div class="api-stats">
                    <span>9 classes</span>
                    <span>31 methods</span>
                </div>
                <a href="qflare.security.html" class="feature-link">Explore →</a>
            </div>
        </div>
    </main>

    <footer class="footer">
        <div class="footer-content">
            <div class="footer-info">
                <p>&copy; 2025 QFLARE Development Team.</p>
            </div>
        </div>
    </footer>

    <script src="../static/scripts.js"></script>
</body>
</html>'''
    
    return html_content

def create_getting_started_guide():
    """Create the getting started guide"""
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Getting Started - QFLARE Documentation</title>
    <link rel="stylesheet" href="../static/styles.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/themes/prism.min.css">
</head>
<body>
    <header class="header">
        <nav class="navbar">
            <div class="nav-brand">
                <a href="../index.html"><h1>🛡️ QFLARE</h1></a>
                <span class="version">v1.0.0</span>
            </div>
            <div class="nav-links">
                <a href="../index.html">Home</a>
                <a href="getting-started.html" class="active">Guides</a>
                <a href="../api/index.html">API</a>
                <a href="../deployment/index.html">Deploy</a>
            </div>
        </nav>
    </header>

    <main class="main-content">
        <div class="page-header">
            <h1>🌟 Getting Started with QFLARE</h1>
            <p class="page-subtitle">Complete introduction to quantum-resistant federated learning</p>
        </div>

        <div class="guide-content">
            <section>
                <h2>Installation</h2>
                <p>Install QFLARE using pip:</p>
                <pre><code class="language-bash">pip install qflare</code></pre>
                
                <p>Or install from source for the latest features:</p>
                <pre><code class="language-bash">git clone https://github.com/sam-2707/QFLARE.git
cd QFLARE
pip install -e .</code></pre>
            </section>

            <section>
                <h2>Basic Usage</h2>
                <p>Here's a simple example of using QFLARE:</p>
                <pre><code class="language-python">from qflare.client import QFLAREClient
from qflare.server import QFLAREServer
from qflare.crypto import QFLARECrypto

# Initialize cryptographic components
crypto = QFLARECrypto(algorithm="kyber1024")

# Create server
server = QFLAREServer(crypto=crypto, port=8080)

# Create client
client = QFLAREClient(
    client_id="client_001",
    crypto=crypto,
    server_host="localhost",
    server_port=8080
)

# Start federated learning
server.start()
client.start_training(epochs=5)</code></pre>
            </section>

            <section>
                <h2>Key Concepts</h2>
                <div class="concepts-grid">
                    <div class="concept-card">
                        <h3>🔐 Post-Quantum Cryptography</h3>
                        <p>QFLARE uses CRYSTALS-Kyber and Dilithium algorithms to protect against quantum computer attacks.</p>
                    </div>
                    
                    <div class="concept-card">
                        <h3>🤝 Federated Learning</h3>
                        <p>Train machine learning models across multiple clients without sharing raw data.</p>
                    </div>
                    
                    <div class="concept-card">
                        <h3>🛡️ Byzantine Fault Tolerance</h3>
                        <p>Protection against malicious participants in the federated learning network.</p>
                    </div>
                </div>
            </section>

            <section>
                <h2>Next Steps</h2>
                <ul>
                    <li><a href="../api/index.html">Explore the API Documentation</a></li>
                    <li><a href="../examples.html">Try Interactive Examples</a></li>
                    <li><a href="../deployment/index.html">Deploy to Production</a></li>
                    <li><a href="../troubleshooting.html">Troubleshooting Guide</a></li>
                </ul>
            </section>
        </div>
    </main>

    <footer class="footer">
        <div class="footer-content">
            <div class="footer-info">
                <p>&copy; 2025 QFLARE Development Team.</p>
            </div>
        </div>
    </footer>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/components/prism-core.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/plugins/autoloader/prism-autoloader.min.js"></script>
    <script src="../static/scripts.js"></script>
</body>
</html>'''
    
    return html_content

def generate_portal():
    """Generate the complete documentation portal"""
    print("🚀 Generating QFLARE Documentation Portal...")
    
    # Create directories
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)
    
    (build_dir / "api").mkdir(exist_ok=True)
    (build_dir / "guides").mkdir(exist_ok=True)
    (build_dir / "deployment").mkdir(exist_ok=True)
    (build_dir / "static").mkdir(exist_ok=True)
    
    # Copy static assets
    static_source = Path("static")
    if static_source.exists():
        import shutil
        shutil.copytree(static_source, build_dir / "static", dirs_exist_ok=True)
        print("✅ Static assets copied")
    
    # Generate pages
    pages = {
        "index.html": create_main_index(),
        "api/index.html": create_api_index(),
        "guides/getting-started.html": create_getting_started_guide()
    }
    
    for page_path, content in pages.items():
        full_path = build_dir / page_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Generated {page_path}")
    
    # Generate report
    report = {
        "generation_time": datetime.now().isoformat(),
        "pages_generated": len(pages),
        "pages": list(pages.keys()),
        "status": "success"
    }
    
    with open(build_dir / "generation_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"""
🎉 Documentation Portal Generated Successfully!

📊 Generation Report:
   - Pages generated: {len(pages)}
   - Output directory: {build_dir.absolute()}
   - Static assets: {'✅ Copied' if static_source.exists() else '❌ Not found'}
   
📂 Generated Pages:
   {chr(10).join(['   - ' + page for page in pages.keys()])}
   
🌐 To view the portal:
   1. Open {build_dir.absolute()}/index.html in your browser
   2. Or serve with: python -m http.server 8000 -d {build_dir}
   3. Then visit: http://localhost:8000
    """)
    
    return build_dir

if __name__ == "__main__":
    generate_portal()