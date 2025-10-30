# QFLARE Data Validation Scripts

This directory contains comprehensive data validation and testing scripts for the QFLARE project. These scripts ensure data integrity, validate model artifacts, and test the complete data pipeline.

## Scripts Overview

### 1. Data Artifacts Validation (`validate_data_artifacts.py`)
**Purpose:** Validates integrity of datasets, models, databases, and cryptographic keys

**Features:**
- ✅ MNIST dataset integrity checking
- ✅ Model artifact validation (.pth/.pt files)
- ✅ SQLite database validation
- ✅ Cryptographic key validation
- ✅ Custom dataset discovery and validation
- ✅ Comprehensive reporting with JSON output

**Usage:**
```bash
# Basic validation
python scripts/validate_data_artifacts.py

# Specify data directory
python scripts/validate_data_artifacts.py --data-path ./data

# Save report to custom location
python scripts/validate_data_artifacts.py --output my_validation_report.json

# Verbose output
python scripts/validate_data_artifacts.py --verbose
```

**Example Output:**
```
============================================================
QFLARE DATA ARTIFACTS VALIDATION SUMMARY
============================================================
Total Checks: 10
Passed: 8
Failed: 0
Warnings: 2
Success Rate: 80.0%
Overall Status: PASS
```

### 2. Checksum Management (`generate_checksums.py`)
**Purpose:** Generates and verifies SHA256/MD5 checksums for critical data files

**Features:**
- 🔐 SHA256 and MD5 checksum generation
- ✅ File integrity verification
- 📊 Comprehensive difference reporting
- 🔄 Automatic checksum updates
- 📁 Support for custom file patterns

**Usage:**
```bash
# Generate checksums for all data files
python scripts/generate_checksums.py generate

# Verify against stored checksums
python scripts/generate_checksums.py verify

# Update checksums (generate and compare)
python scripts/generate_checksums.py update

# Custom data path
python scripts/generate_checksums.py generate --data-path ./custom_data

# Custom file patterns
python scripts/generate_checksums.py generate --patterns "**/*.pth" "**/*.db"

# Save verification results
python scripts/generate_checksums.py verify --output verification_results.json
```

**Checksum File Format:**
```json
{
  "metadata": {
    "generated_at": 1698123456.789,
    "total_files": 15,
    "data_path": "./data"
  },
  "checksums": {
    "MNIST/raw/train-images-idx3-ubyte": {
      "sha256": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
      "md5": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
      "size_bytes": 47040016,
      "timestamp": 1698123456.789
    }
  }
}
```

### 3. Data Pipeline Testing (`test_data_pipeline.py`)
**Purpose:** End-to-end testing of the data pipeline from ingestion to model training

**Features:**
- 📥 Data ingestion testing (MNIST download/loading)
- 🔄 Data preprocessing pipeline validation
- 🧠 Model training pipeline testing
- 🔮 Model inference validation
- 💾 Model serialization/deserialization testing
- ⚡ Performance benchmarking

**Usage:**
```bash
# Run complete data pipeline tests
python scripts/test_data_pipeline.py

# Custom data directory
python scripts/test_data_pipeline.py --data-path ./data

# Save detailed report
python scripts/test_data_pipeline.py --output pipeline_test_results.json

# Verbose testing
python scripts/test_data_pipeline.py --verbose
```

**Test Categories:**
1. **Data Ingestion Tests**
   - MNIST download and loading
   - Data preprocessing validation
   - Data augmentation testing

2. **Model Pipeline Tests**
   - Model training validation
   - Inference accuracy testing
   - Model serialization/loading

**Example Output:**
```
============================================================
QFLARE DATA PIPELINE TEST SUMMARY
============================================================
Total Tests: 5
Passed: 5
Failed: 0
Warnings: 0
Success Rate: 100.0%
Total Duration: 45.32s
Overall Status: PASS

Detailed Test Results:
--------------------------------------------------
✓ MNIST Download & Loading    PASS     (12.45s)
  → Dataset size: 60000
✓ Data Preprocessing          PASS     (2.31s)
✓ Model Training             PASS     (25.67s)
  → Loss decreased: True
✓ Model Inference            PASS     (3.12s)
  → Accuracy: 0.8234
✓ Model Serialization        PASS     (1.77s)
```

## Integration with CI/CD

These scripts are integrated into the GitHub Actions workflows:

### Main CI/CD Pipeline
```yaml
- name: Validate data artifacts
  run: |
    python scripts/validate_data_artifacts.py --data-path data
    python scripts/generate_checksums.py verify
```

### Performance Validation Pipeline
```yaml
- name: Test data pipeline
  run: |
    python scripts/test_data_pipeline.py --verbose
  timeout-minutes: 60
```

## Configuration

### File Patterns
Default patterns for checksum generation:
- `**/*.pth` - PyTorch model files
- `**/*.pt` - PyTorch tensor files
- `**/*.db` - SQLite databases
- `**/*.sqlite` - SQLite databases
- `**/*.json` - Configuration files
- `MNIST/raw/*` - MNIST raw data files
- `keys/**/*` - Cryptographic keys
- `models/**/*` - Model artifacts

### Validation Thresholds
- **Dataset size:** Exact match required
- **Model accuracy:** Warning if < 70%
- **File integrity:** Fail on checksum mismatch
- **Database integrity:** Fail on corruption

## Troubleshooting

### Common Issues

1. **MNIST Dataset Not Found**
   ```bash
   # Download MNIST manually
   python -c "from torchvision.datasets import MNIST; MNIST('./data', download=True)"
   ```

2. **Missing Dependencies**
   ```bash
   pip install torch torchvision numpy
   ```

3. **Permission Errors (Keys)**
   ```bash
   # Check file permissions
   ls -la data/keys/
   chmod 600 data/keys/*.pem
   ```

4. **Database Locked Errors**
   ```bash
   # Stop any running QFLARE processes
   ps aux | grep qflare
   kill <process_id>
   ```

### Debug Mode

Enable detailed debugging:
```bash
python scripts/validate_data_artifacts.py --verbose
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Best Practices

### Regular Validation
- Run data validation before each release
- Generate checksums after data updates
- Test data pipeline on multiple environments

### Checksum Management
- Generate checksums for critical files
- Verify checksums before deployment
- Update checksums when files are intentionally modified

### Performance Monitoring
- Track data pipeline performance over time
- Monitor model accuracy trends
- Alert on significant performance degradation

## File Structure

```
scripts/
├── validate_data_artifacts.py    # Main validation script
├── generate_checksums.py         # Checksum management
├── test_data_pipeline.py         # Pipeline testing
└── README.md                     # This file

data/
├── checksums.json                # Generated checksums
├── MNIST/                        # MNIST dataset
├── models/                       # Model artifacts
├── keys/                         # Cryptographic keys
└── *.db                          # SQLite databases

outputs/
├── data_validation_report.json   # Validation results
├── pipeline_test_results.json    # Pipeline test results
└── verification_results.json     # Checksum verification
```

## API Reference

### ValidationResult Class
```python
@dataclass
class ValidationResult:
    name: str              # Test name
    status: str            # PASS, FAIL, WARNING, SKIP
    message: str           # Human-readable message
    details: Dict          # Additional test details
    timestamp: float       # Unix timestamp
```

### ChecksumRecord Class
```python
@dataclass
class ChecksumRecord:
    file_path: str         # Relative file path
    sha256: str            # SHA256 hash
    md5: str               # MD5 hash
    size_bytes: int        # File size
    timestamp: float       # Generation timestamp
```

## Contributing

### Adding New Validators
1. Create validator class inheriting from base patterns
2. Implement validation methods returning `ValidationResult`
3. Add to main validation pipeline
4. Update tests and documentation

### Adding New Tests
1. Add test methods to appropriate tester class
2. Return `PipelineTestResult` with detailed metrics
3. Update comprehensive test runner
4. Add to CI/CD pipeline if needed

### Error Handling
- Always catch and report exceptions gracefully
- Provide detailed error messages with context
- Use appropriate status codes (PASS/FAIL/WARNING)
- Include suggestions for fixing common issues