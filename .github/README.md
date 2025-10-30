# QFLARE CI/CD Pipeline

This directory contains GitHub Actions workflows for automated testing, validation, and deployment of the QFLARE project.

## Workflows

### 1. Main CI/CD Pipeline (`ci-cd.yml`)
**Triggers:** Push to main/develop, Pull requests, Manual dispatch
**Duration:** ~30-45 minutes

**Jobs:**
- **Test & Quality Checks** - Python testing, linting, security scans
- **QFLARE Validation** - Quick experiment validation 
- **Security Scanning** - Vulnerability analysis, Docker security
- **Docker Build** - Multi-architecture container images
- **Frontend Build** - React application build and test

**Features:**
- ✅ Comprehensive Python testing with pytest
- ✅ Code quality checks (black, flake8, mypy, pylint)
- ✅ Security scanning (bandit, safety, Trivy, CodeQL)
- ✅ QFLARE experiment validation
- ✅ Multi-platform Docker builds
- ✅ Coverage reporting to Codecov
- ✅ Automated artifact uploads

### 2. Performance Validation (`performance-validation.yml`)
**Triggers:** Weekly schedule, Manual dispatch
**Duration:** 1-8 hours (depending on validation type)

**Validation Types:**
- **Quick** (5 min) - Basic functionality test
- **Partial** (2-3 hours) - Key paper metrics validation
- **Full** (6-8 hours) - Complete paper reproduction

**Features:**
- 🔬 Paper claims validation (96.8% accuracy, 1,247 updates/sec, etc.)
- 📊 Performance benchmarking and profiling
- 🧪 Experiment configuration generation and testing
- 📈 Automated performance charts and reports
- 🚨 Failure notifications and issue creation

### 3. Production Deployment (`production-deploy.yml`)
**Triggers:** Manual dispatch, Release tags
**Duration:** ~20-30 minutes

**Features:**
- 🚀 Kubernetes deployment automation
- 🔄 Blue-green deployment strategy
- 🧪 Smoke testing and health checks
- 📧 Deployment notifications

## Usage

### Running CI/CD Pipeline
The main pipeline runs automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main`
- Manual trigger via GitHub Actions UI

### Running Performance Validation

**Quick validation (5 minutes):**
```bash
gh workflow run performance-validation.yml -f validation_type=quick
```

**Partial validation (2-3 hours):**
```bash
gh workflow run performance-validation.yml -f validation_type=partial
```

**Full paper validation (6-8 hours):**
```bash
gh workflow run performance-validation.yml -f validation_type=full
```

### Manual Deployment
```bash
gh workflow run production-deploy.yml -f environment=staging
gh workflow run production-deploy.yml -f environment=production
```

## Configuration

### Required Secrets
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password/token
- `CODECOV_TOKEN` - Codecov upload token
- `SLACK_WEBHOOK` - Slack notification webhook
- `AWS_ACCESS_KEY_ID` - AWS deployment credentials
- `AWS_SECRET_ACCESS_KEY` - AWS deployment credentials

### Environment Variables
- `PYTHON_VERSION: '3.11'` - Python version for testing
- `NODE_VERSION: '18'` - Node.js version for frontend
- `PYTORCH_VERSION: '2.1.0'` - PyTorch version

## Monitoring & Notifications

### Success Indicators
- ✅ All tests pass with >70% code coverage
- ✅ Security scans complete without critical issues
- ✅ QFLARE validation achieves expected metrics
- ✅ Docker images build successfully

### Failure Handling
- 📧 Slack notifications for all failures
- 🐛 Automatic GitHub issue creation for validation failures
- 📋 Detailed logs and artifacts for debugging
- 🔄 Retry mechanism for transient failures

### Artifacts
All workflows generate artifacts that are retained for:
- **Test results:** 30 days
- **Security reports:** 60 days
- **Performance data:** 90 days
- **Docker images:** Permanent (tagged)

## Performance Targets

### CI/CD Pipeline
- **Total runtime:** <45 minutes
- **Test coverage:** >70%
- **Security scan:** No critical vulnerabilities
- **Docker build:** <10 minutes per image

### QFLARE Validation
- **Honest accuracy:** ≥96.8% (target from paper)
- **Byzantine resilience:** <2% accuracy degradation
- **Crypto performance:** Within expected ranges
- **Throughput:** ≥1,000 updates/second

## Troubleshooting

### Common Issues

1. **Test failures due to missing dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **QFLARE validation timeout**
   - Reduce experiment parameters in `quick_validation.py`
   - Use `--quick` flag for faster testing

3. **Docker build failures**
   - Check Dockerfile syntax
   - Verify base image availability
   - Review build context size

4. **Performance validation failures**
   - Check system resources (CPU, memory)
   - Verify MNIST dataset download
   - Review experiment configuration

### Debug Commands

**Local testing:**
```bash
# Run quick QFLARE validation
cd experiments && python quick_validation.py

# Test experiment runner
cd experiments && python run_qflare_experiments.py --quick

# Generate configs
cd experiments && python generate_configs.py --config-type paper
```

**Docker testing:**
```bash
# Build server image locally
docker build -f docker/Dockerfile.server -t qflare-server .

# Test container
docker run --rm qflare-server python --version
```

## Development Workflow

### Adding New Tests
1. Add test files to `tests/` directory
2. Update `requirements.txt` if needed
3. Ensure tests pass locally: `pytest tests/`
4. Commit and push - CI will run automatically

### Adding New Experiments
1. Create experiment in `experiments/` directory
2. Add configuration in `generate_configs.py`
3. Test locally: `python experiments/your_experiment.py`
4. Update `performance-validation.yml` if needed

### Modifying CI/CD
1. Edit workflow files in `.github/workflows/`
2. Test workflow syntax: `gh workflow validate`
3. Create PR and test in development environment
4. Merge to main for production deployment

## Best Practices

- 🧪 Always test experiments locally before pushing
- 📝 Update documentation when adding new features
- 🔒 Keep secrets secure and rotate regularly
- 📊 Monitor performance trends over time
- 🐛 Investigate and fix failures promptly
- 🏷️ Use semantic versioning for releases