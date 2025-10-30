#!/usr/bin/env python3
"""
QFLARE Data Artifacts Validation Suite
Validates integrity of datasets, models, and data pipeline components
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import sqlite3

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Represents the result of a validation check"""
    name: str
    status: str  # PASS, FAIL, WARNING, SKIP
    message: str
    details: Dict = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class DatasetValidator:
    """Validates ML datasets used in QFLARE"""
    
    def __init__(self, data_path: str = "./data"):
        self.data_path = Path(data_path)
        self.validation_results = []
    
    def validate_mnist_dataset(self) -> ValidationResult:
        """Validate MNIST dataset integrity"""
        logger.info("Validating MNIST dataset...")
        
        try:
            mnist_path = self.data_path / "MNIST"
            
            # Check if MNIST directory exists
            if not mnist_path.exists():
                return ValidationResult(
                    name="MNIST Dataset",
                    status="FAIL",
                    message="MNIST directory not found",
                    details={"expected_path": str(mnist_path)}
                )
            
            # Validate training set
            try:
                train_data = MNIST(root=str(self.data_path), train=True, download=False)
            except:
                # Try downloading if not found
                train_data = MNIST(root=str(self.data_path), train=True, download=True)
                
            train_size = len(train_data)
            expected_train_size = 60000
            
            if train_size != expected_train_size:
                return ValidationResult(
                    name="MNIST Dataset",
                    status="FAIL",
                    message=f"Training set size mismatch: {train_size} != {expected_train_size}"
                )
            
            # Validate test set
            try:
                test_data = MNIST(root=str(self.data_path), train=False, download=False)
            except:
                test_data = MNIST(root=str(self.data_path), train=False, download=True)
                
            test_size = len(test_data)
            expected_test_size = 10000
            
            if test_size != expected_test_size:
                return ValidationResult(
                    name="MNIST Dataset",
                    status="FAIL",
                    message=f"Test set size mismatch: {test_size} != {expected_test_size}"
                )
            
            # Validate data format and ranges
            sample_data, sample_label = train_data[0]
            
            # Convert PIL Image to tensor if needed
            if hasattr(sample_data, 'mode'):  # PIL Image
                transform = transforms.ToTensor()
                sample_data = transform(sample_data)
            
            if not isinstance(sample_data, torch.Tensor):
                return ValidationResult(
                    name="MNIST Dataset",
                    status="FAIL",
                    message=f"Sample data is not a torch.Tensor, got {type(sample_data)}"
                )
            
            if sample_data.shape != (1, 28, 28):
                return ValidationResult(
                    name="MNIST Dataset",
                    status="FAIL",
                    message=f"Sample data shape incorrect: {sample_data.shape} != (1, 28, 28)"
                )
            
            if sample_data.min() < 0 or sample_data.max() > 1:
                return ValidationResult(
                    name="MNIST Dataset",
                    status="WARNING",
                    message=f"Sample data range unusual: [{sample_data.min():.3f}, {sample_data.max():.3f}]"
                )
            
            if not (0 <= sample_label <= 9):
                return ValidationResult(
                    name="MNIST Dataset",
                    status="FAIL",
                    message=f"Sample label out of range: {sample_label}"
                )
            
            # Calculate checksums for key files
            checksums = self._calculate_mnist_checksums()
            
            return ValidationResult(
                name="MNIST Dataset",
                status="PASS",
                message="MNIST dataset validation successful",
                details={
                    "train_size": train_size,
                    "test_size": test_size,
                    "data_shape": list(sample_data.shape),
                    "data_range": [float(sample_data.min()), float(sample_data.max())],
                    "label_range": [0, 9],
                    "checksums": checksums
                }
            )
            
        except Exception as e:
            return ValidationResult(
                name="MNIST Dataset",
                status="FAIL",
                message=f"MNIST validation error: {str(e)}",
                details={"error_type": type(e).__name__}
            )
    
    def _calculate_mnist_checksums(self) -> Dict[str, str]:
        """Calculate checksums for MNIST files"""
        checksums = {}
        mnist_files = [
            "MNIST/raw/train-images-idx3-ubyte",
            "MNIST/raw/train-labels-idx1-ubyte", 
            "MNIST/raw/t10k-images-idx3-ubyte",
            "MNIST/raw/t10k-labels-idx1-ubyte"
        ]
        
        for file_path in mnist_files:
            full_path = self.data_path / file_path
            if full_path.exists():
                checksums[file_path] = self._calculate_file_checksum(full_path)
        
        return checksums
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def validate_custom_datasets(self) -> List[ValidationResult]:
        """Validate any custom datasets in the data directory"""
        results = []
        
        # Check for additional dataset directories
        dataset_dirs = [d for d in self.data_path.iterdir() if d.is_dir() and d.name != "MNIST"]
        
        for dataset_dir in dataset_dirs:
            try:
                # Basic directory structure validation
                file_count = len(list(dataset_dir.rglob("*")))
                size_mb = sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file()) / (1024*1024)
                
                result = ValidationResult(
                    name=f"Custom Dataset: {dataset_dir.name}",
                    status="PASS",
                    message=f"Dataset found with {file_count} files ({size_mb:.1f} MB)",
                    details={
                        "path": str(dataset_dir),
                        "file_count": file_count,
                        "size_mb": round(size_mb, 1)
                    }
                )
                results.append(result)
                
            except Exception as e:
                result = ValidationResult(
                    name=f"Custom Dataset: {dataset_dir.name}",
                    status="FAIL",
                    message=f"Error validating dataset: {str(e)}"
                )
                results.append(result)
        
        return results

class ModelArtifactValidator:
    """Validates ML model artifacts and checkpoints"""
    
    def __init__(self, models_path: str = "./data/models"):
        self.models_path = Path(models_path)
        self.validation_results = []
    
    def validate_model_artifacts(self) -> List[ValidationResult]:
        """Validate all model artifacts"""
        results = []
        
        if not self.models_path.exists():
            results.append(ValidationResult(
                name="Model Directory",
                status="WARNING",
                message="Models directory not found",
                details={"expected_path": str(self.models_path)}
            ))
            return results
        
        # Find all model files
        model_files = list(self.models_path.rglob("*.pth")) + list(self.models_path.rglob("*.pt"))
        
        if not model_files:
            results.append(ValidationResult(
                name="Model Artifacts",
                status="WARNING", 
                message="No PyTorch model files found",
                details={"search_path": str(self.models_path)}
            ))
        
        for model_file in model_files:
            result = self._validate_single_model(model_file)
            results.append(result)
        
        return results
    
    def _validate_single_model(self, model_path: Path) -> ValidationResult:
        """Validate a single model file"""
        try:
            # Check if file is readable
            if not model_path.exists():
                return ValidationResult(
                    name=f"Model: {model_path.name}",
                    status="FAIL",
                    message="Model file not found"
                )
            
            file_size = model_path.stat().st_size
            
            # Try to load the model
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Analyze checkpoint structure
                details = {
                    "file_size_mb": round(file_size / (1024*1024), 2),
                    "file_path": str(model_path)
                }
                
                if isinstance(checkpoint, dict):
                    details["checkpoint_keys"] = list(checkpoint.keys())
                    
                    # Check for common checkpoint components
                    if "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                        details["model_parameters"] = len(state_dict)
                        details["parameter_shapes"] = {k: list(v.shape) for k, v in list(state_dict.items())[:5]}
                    
                    if "optimizer_state_dict" in checkpoint:
                        details["has_optimizer_state"] = True
                    
                    if "epoch" in checkpoint:
                        details["epoch"] = checkpoint["epoch"]
                    
                    if "loss" in checkpoint:
                        details["loss"] = checkpoint["loss"]
                
                return ValidationResult(
                    name=f"Model: {model_path.name}",
                    status="PASS",
                    message="Model loaded successfully",
                    details=details
                )
                
            except Exception as load_error:
                return ValidationResult(
                    name=f"Model: {model_path.name}",
                    status="FAIL",
                    message=f"Failed to load model: {str(load_error)}",
                    details={"file_size_mb": round(file_size / (1024*1024), 2)}
                )
                
        except Exception as e:
            return ValidationResult(
                name=f"Model: {model_path.name}",
                status="FAIL",
                message=f"Model validation error: {str(e)}"
            )

class DatabaseValidator:
    """Validates SQLite databases used by QFLARE"""
    
    def __init__(self, data_path: str = "./data"):
        self.data_path = Path(data_path)
    
    def validate_databases(self) -> List[ValidationResult]:
        """Validate all SQLite databases"""
        results = []
        
        # Find all SQLite database files
        db_files = list(self.data_path.rglob("*.db")) + list(self.data_path.rglob("*.sqlite"))
        
        if not db_files:
            results.append(ValidationResult(
                name="Database Files",
                status="WARNING",
                message="No database files found",
                details={"search_path": str(self.data_path)}
            ))
            return results
        
        for db_file in db_files:
            result = self._validate_single_database(db_file)
            results.append(result)
        
        return results
    
    def _validate_single_database(self, db_path: Path) -> ValidationResult:
        """Validate a single SQLite database"""
        try:
            if not db_path.exists():
                return ValidationResult(
                    name=f"Database: {db_path.name}",
                    status="FAIL",
                    message="Database file not found"
                )
            
            file_size = db_path.stat().st_size
            
            # Connect to database and validate
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()
                
                # Get database info
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                table_info = {}
                total_rows = 0
                
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    row_count = cursor.fetchone()[0]
                    table_info[table] = row_count
                    total_rows += row_count
                
                # Check database integrity
                cursor.execute("PRAGMA integrity_check;")
                integrity_result = cursor.fetchone()[0]
                
                status = "PASS" if integrity_result == "ok" else "FAIL"
                message = "Database validation successful" if status == "PASS" else f"Integrity check failed: {integrity_result}"
                
                return ValidationResult(
                    name=f"Database: {db_path.name}",
                    status=status,
                    message=message,
                    details={
                        "file_size_mb": round(file_size / (1024*1024), 2),
                        "tables": tables,
                        "table_row_counts": table_info,
                        "total_rows": total_rows,
                        "integrity_check": integrity_result
                    }
                )
                
        except Exception as e:
            return ValidationResult(
                name=f"Database: {db_path.name}",
                status="FAIL",
                message=f"Database validation error: {str(e)}"
            )

class KeysValidator:
    """Validates cryptographic keys and certificates"""
    
    def __init__(self, keys_path: str = "./data/keys"):
        self.keys_path = Path(keys_path)
    
    def validate_keys(self) -> List[ValidationResult]:
        """Validate cryptographic keys and certificates"""
        results = []
        
        if not self.keys_path.exists():
            results.append(ValidationResult(
                name="Keys Directory",
                status="WARNING",
                message="Keys directory not found",
                details={"expected_path": str(self.keys_path)}
            ))
            return results
        
        # Find key files
        key_patterns = ["*.pem", "*.key", "*.crt", "*.cert", "*.pub", "*.priv"]
        key_files = []
        for pattern in key_patterns:
            key_files.extend(self.keys_path.rglob(pattern))
        
        if not key_files:
            results.append(ValidationResult(
                name="Cryptographic Keys",
                status="WARNING",
                message="No key files found",
                details={"search_path": str(self.keys_path)}
            ))
            return results
        
        for key_file in key_files:
            result = self._validate_single_key(key_file)
            results.append(result)
        
        return results
    
    def _validate_single_key(self, key_path: Path) -> ValidationResult:
        """Validate a single key file"""
        try:
            if not key_path.exists():
                return ValidationResult(
                    name=f"Key: {key_path.name}",
                    status="FAIL", 
                    message="Key file not found"
                )
            
            file_size = key_path.stat().st_size
            
            # Read file content
            with open(key_path, 'r') as f:
                content = f.read()
            
            # Basic validation
            details = {
                "file_size_bytes": file_size,
                "file_path": str(key_path),
                "content_length": len(content)
            }
            
            # Check for common key formats
            if "-----BEGIN" in content and "-----END" in content:
                # PEM format
                if "PRIVATE KEY" in content:
                    details["key_type"] = "Private Key (PEM)"
                elif "PUBLIC KEY" in content:
                    details["key_type"] = "Public Key (PEM)"
                elif "CERTIFICATE" in content:
                    details["key_type"] = "Certificate (PEM)"
                else:
                    details["key_type"] = "Unknown PEM format"
                
                return ValidationResult(
                    name=f"Key: {key_path.name}",
                    status="PASS",
                    message="Key file validation successful",
                    details=details
                )
            else:
                # Binary or other format
                details["key_type"] = "Binary or unknown format"
                
                return ValidationResult(
                    name=f"Key: {key_path.name}",
                    status="WARNING",
                    message="Key file format not recognized as PEM",
                    details=details
                )
                
        except Exception as e:
            return ValidationResult(
                name=f"Key: {key_path.name}",
                status="FAIL",
                message=f"Key validation error: {str(e)}"
            )

class DataPipelineValidator:
    """Validates the complete data pipeline"""
    
    def __init__(self, data_path: str = "./data"):
        self.data_path = Path(data_path)
        self.validators = {
            "dataset": DatasetValidator(data_path),
            "models": ModelArtifactValidator(str(Path(data_path) / "models")),
            "database": DatabaseValidator(data_path),
            "keys": KeysValidator(str(Path(data_path) / "keys"))
        }
    
    def run_complete_validation(self) -> Dict:
        """Run complete data artifacts validation"""
        logger.info("Starting complete data artifacts validation...")
        
        all_results = []
        
        # Dataset validation
        mnist_result = self.validators["dataset"].validate_mnist_dataset()
        all_results.append(mnist_result)
        
        custom_results = self.validators["dataset"].validate_custom_datasets()
        all_results.extend(custom_results)
        
        # Model artifacts validation
        model_results = self.validators["models"].validate_model_artifacts()
        all_results.extend(model_results)
        
        # Database validation
        db_results = self.validators["database"].validate_databases()
        all_results.extend(db_results)
        
        # Keys validation
        key_results = self.validators["keys"].validate_keys()
        all_results.extend(key_results)
        
        # Compile summary
        total_checks = len(all_results)
        passed_checks = len([r for r in all_results if r.status == "PASS"])
        failed_checks = len([r for r in all_results if r.status == "FAIL"])
        warning_checks = len([r for r in all_results if r.status == "WARNING"])
        skipped_checks = len([r for r in all_results if r.status == "SKIP"])
        
        summary = {
            "total_checks": total_checks,
            "passed": passed_checks,
            "failed": failed_checks,
            "warnings": warning_checks,
            "skipped": skipped_checks,
            "success_rate": passed_checks / total_checks if total_checks > 0 else 0,
            "overall_status": "PASS" if failed_checks == 0 else "FAIL"
        }
        
        return {
            "summary": summary,
            "results": [asdict(r) for r in all_results],
            "validation_timestamp": time.time()
        }
    
    def save_validation_report(self, results: Dict, output_file: str = "data_validation_report.json"):
        """Save validation results to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {output_path}")
    
    def print_validation_summary(self, results: Dict):
        """Print validation summary to console"""
        summary = results["summary"]
        
        print("\n" + "="*60)
        print("QFLARE DATA ARTIFACTS VALIDATION SUMMARY")
        print("="*60)
        print(f"Total Checks: {summary['total_checks']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Warnings: {summary['warnings']}")
        print(f"Skipped: {summary['skipped']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Overall Status: {summary['overall_status']}")
        
        print("\nDetailed Results:")
        print("-" * 50)
        
        for result in results["results"]:
            status_symbol = {
                "PASS": "✓",
                "FAIL": "✗", 
                "WARNING": "⚠",
                "SKIP": "⏭"
            }.get(result["status"], "?")
            
            print(f"{status_symbol} {result['name']:<30} {result['status']}")
            if result["status"] != "PASS":
                print(f"  → {result['message']}")
        
        print("="*60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="QFLARE Data Artifacts Validation")
    parser.add_argument("--data-path", type=str, default="./data",
                       help="Path to data directory")
    parser.add_argument("--output", type=str, default="data_validation_report.json",
                       help="Output file for validation report")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run validation
    validator = DataPipelineValidator(args.data_path)
    results = validator.run_complete_validation()
    
    # Save and display results
    validator.save_validation_report(results, args.output)
    validator.print_validation_summary(results)
    
    # Exit with appropriate code
    if results["summary"]["overall_status"] == "PASS":
        logger.info("All critical validations passed!")
        sys.exit(0)
    else:
        logger.error(f"Validation failed! {results['summary']['failed']} checks failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()