# QFLARE: Quantum-Resistant Federated Learning Architecture

A comprehensive quantum-safe federated learning system with provable security guarantees for the post-quantum era.

## Repository Structure

The repository is organized into the following directories:

### 📄 Paper & Documentation
- **`paper/`** - IEEE journal paper and LaTeX compilation tools
  - `main.tex` - Main paper entry point
  - `QFLARE_IEEE_Paper.tex` - Full IEEE paper with proofs and experiments
  - `QFLARE_IEEE_Paper_Simple.tex` - Simplified compilable version
  - `compile_paper.ps1`, `compile_paper.bat` - Build scripts
  - `validate_latex.ps1` - LaTeX structure validator
  - `README_BUILD.md` - Build instructions

- **`docs/`** - Project documentation and guides
  - Security guides, deployment docs, API documentation
  - Project status and organization summaries
  - Troubleshooting and operator manuals

### 🔧 Build & Deployment
- **`build/`** - Build configuration and deployment files
  - `docker-compose.yml`, `docker-compose.prod.yml` - Docker configurations
  - `requirements.txt`, `requirements.prod.txt` - Python dependencies
  - `setup.py` - Package setup
  - `start_qflare.py` - Main application entry point

### 🔐 Security & Data
- **`data/`** - Data storage and security tokens
  - `keys/` - Security keys and tokens
  - `logs/` - Application logs
  - `models/` - ML model storage
  - Database files (`.db`)

### 🖥️ Core Components
- **`server/`** - Main QFLARE server implementation
- **`edge_node/`** - Edge node federated learning components
- **`frontend/`** - Web interface and dashboard
- **`backend/`** - Backend API services
- **`security/`** - Cryptographic implementations
- **`common/`** - Shared utilities and error handling

### 🛠️ Development & Testing
- **`tests/`** - Test suites and validation
- **`scripts/`** - Utility scripts and automation
- **`config/`** - Configuration files
- **`k8s/`** - Kubernetes deployment manifests
- **`docker/`** - Docker configuration files
- **`monitoring/`** - Prometheus, Grafana monitoring setup

### 📚 Research & References
- **`liboqs/`** - Post-quantum cryptography library
- **`liboqs-python/`** - Python bindings for liboqs
- **`enclaves/`** - Trusted execution environment code
- **`models/`** - Machine learning model implementations

## Quick Start

### Building the Paper
```powershell
cd paper
.\compile_paper.ps1
```

### Running QFLARE
```powershell
cd build
python start_qflare.py
```

### Docker Deployment
```bash
docker-compose up -d
```

## Key Features

- **Quantum-Safe Cryptography**: CRYSTALS-Kyber (key exchange) and CRYSTALS-Dilithium (signatures)
- **Differential Privacy**: Advanced privacy accounting with adaptive noise calibration
- **Byzantine Fault Tolerance**: Robust aggregation against malicious participants
- **Scalable Architecture**: Supports 10-10,000 participants
- **Comprehensive Security**: Proven security against classical and quantum adversaries

## Performance

- **Accuracy**: 94.1% on MNIST with ε=0.1 differential privacy
- **Overhead**: Only 1.75x computational overhead vs classical FL
- **Security**: 128-bit quantum security level
- **Scalability**: Linear scaling from 100 to 1,000 participants

## Documentation

See `docs/` for comprehensive documentation including:
- Deployment guides
- Security analysis
- API documentation
- Troubleshooting guides

## License

See [LICENSE](LICENSE) for details.

## Citation

If you use QFLARE in your research, please cite our paper:

```bibtex
@article{richards2025qflare,
  title={QFLARE: A Quantum-Resistant Federated Learning Architecture with Provable Security Guarantees for Post-Quantum Era},
  author={Richards, Samuel A. and Smith, John D.},
  journal={IEEE Transactions on Quantum Engineering},
  year={2025}
}
```