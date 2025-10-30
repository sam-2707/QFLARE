#!/usr/bin/env python3
"""
QFLARE Data Pipeline Testing
End-to-end testing of the data pipeline from ingestion to model training
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PipelineTestResult:
    """Result of a data pipeline test"""
    name: str
    status: str  # PASS, FAIL, WARNING
    duration_seconds: float
    details: Dict
    error_message: str = None

class DataIngestionTester:
    """Tests data ingestion pipeline"""
    
    def __init__(self, data_path: str = "./data"):
        self.data_path = Path(data_path)
    
    def test_mnist_download(self) -> PipelineTestResult:
        """Test MNIST dataset download and loading"""
        start_time = time.time()
        
        try:
            # Create temporary directory for test
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Download MNIST to temporary location
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                
                train_dataset = MNIST(root=str(temp_path), train=True, download=True, transform=transform)
                test_dataset = MNIST(root=str(temp_path), train=False, download=True, transform=transform)
                
                # Verify dataset properties
                train_size = len(train_dataset)
                test_size = len(test_dataset)
                
                # Test data loading
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                batch_data, batch_labels = next(iter(train_loader))
                
                duration = time.time() - start_time
                
                return PipelineTestResult(
                    name="MNIST Download & Loading",
                    status="PASS",
                    duration_seconds=duration,
                    details={
                        "train_size": train_size,
                        "test_size": test_size,
                        "batch_shape": list(batch_data.shape),
                        "label_shape": list(batch_labels.shape),
                        "data_range": [float(batch_data.min()), float(batch_data.max())],
                        "download_location": str(temp_path)
                    }
                )
                
        except Exception as e:
            duration = time.time() - start_time
            return PipelineTestResult(
                name="MNIST Download & Loading",
                status="FAIL",
                duration_seconds=duration,
                details={},
                error_message=str(e)
            )
    
    def test_data_preprocessing(self) -> PipelineTestResult:
        """Test data preprocessing pipeline"""
        start_time = time.time()
        
        try:
            # Load existing MNIST data
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            dataset = MNIST(root=str(self.data_path), train=True, download=False, transform=transform)
            
            # Test various preprocessing steps
            sample_data, sample_label = dataset[0]
            
            # Test normalization
            mean = float(sample_data.mean())
            std = float(sample_data.std())
            
            # Test data augmentation
            augmented_transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            # Create a PIL augmentation dataset
            pil_dataset = MNIST(root=str(self.data_path), train=True, download=False, transform=augmented_transform)
            augmented_data, augmented_label = pil_dataset[0]
            
            duration = time.time() - start_time
            
            return PipelineTestResult(
                name="Data Preprocessing",
                status="PASS",
                duration_seconds=duration,
                details={
                    "original_shape": list(sample_data.shape),
                    "normalized_mean": mean,
                    "normalized_std": std,
                    "augmented_shape": list(augmented_data.shape),
                    "preprocessing_steps": ["normalization", "augmentation"]
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return PipelineTestResult(
                name="Data Preprocessing",
                status="FAIL",
                duration_seconds=duration,
                details={},
                error_message=str(e)
            )

class ModelPipelineTester:
    """Tests model training and inference pipeline"""
    
    def __init__(self, data_path: str = "./data"):
        self.data_path = Path(data_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def create_simple_model(self) -> nn.Module:
        """Create a simple CNN for testing"""
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.fc1 = nn.Linear(320, 50)
                self.fc2 = nn.Linear(50, 10)
                
            def forward(self, x):
                x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
                x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
                x = x.view(-1, 320)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        return SimpleCNN()
    
    def test_model_training(self) -> PipelineTestResult:
        """Test model training pipeline"""
        start_time = time.time()
        
        try:
            # Load data
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            train_dataset = MNIST(root=str(self.data_path), train=True, download=False, transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            
            # Create model
            model = self.create_simple_model().to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # Train for a few batches
            model.train()
            initial_loss = None
            final_loss = None
            
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 5:  # Only train for 5 batches
                    break
                
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                
                if initial_loss is None:
                    initial_loss = float(loss.item())
                final_loss = float(loss.item())
                
                loss.backward()
                optimizer.step()
            
            duration = time.time() - start_time
            
            # Verify training occurred
            loss_decreased = final_loss < initial_loss
            
            return PipelineTestResult(
                name="Model Training",
                status="PASS" if loss_decreased else "WARNING",
                duration_seconds=duration,
                details={
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                    "loss_decreased": loss_decreased,
                    "model_parameters": sum(p.numel() for p in model.parameters()),
                    "device": str(self.device),
                    "batches_trained": 5
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return PipelineTestResult(
                name="Model Training",
                status="FAIL",
                duration_seconds=duration,
                details={},
                error_message=str(e)
            )
    
    def test_model_inference(self) -> PipelineTestResult:
        """Test model inference pipeline"""
        start_time = time.time()
        
        try:
            # Load test data
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            test_dataset = MNIST(root=str(self.data_path), train=False, download=False, transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
            
            # Create and test model
            model = self.create_simple_model().to(self.device)
            model.eval()
            
            correct = 0
            total = 0
            inference_times = []
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_loader):
                    if batch_idx >= 10:  # Only test 10 batches
                        break
                    
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Measure inference time
                    inference_start = time.time()
                    output = model(data)
                    inference_time = time.time() - inference_start
                    inference_times.append(inference_time)
                    
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            accuracy = correct / total if total > 0 else 0
            avg_inference_time = np.mean(inference_times)
            
            duration = time.time() - start_time
            
            return PipelineTestResult(
                name="Model Inference",
                status="PASS",
                duration_seconds=duration,
                details={
                    "accuracy": accuracy,
                    "correct_predictions": correct,
                    "total_predictions": total,
                    "avg_inference_time_ms": avg_inference_time * 1000,
                    "device": str(self.device),
                    "batches_tested": min(10, len(test_loader))
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return PipelineTestResult(
                name="Model Inference",
                status="FAIL",
                duration_seconds=duration,
                details={},
                error_message=str(e)
            )
    
    def test_model_serialization(self) -> PipelineTestResult:
        """Test model save/load pipeline"""
        start_time = time.time()
        
        try:
            # Create and train a simple model
            model = self.create_simple_model()
            
            # Create some dummy data for testing
            dummy_input = torch.randn(1, 1, 28, 28)
            original_output = model(dummy_input)
            
            # Save model to temporary file
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Test different save formats
                torch.save(model.state_dict(), temp_path)
                
                # Load model
                loaded_model = self.create_simple_model()
                loaded_model.load_state_dict(torch.load(temp_path, map_location='cpu'))
                loaded_model.eval()
                
                # Test that loaded model produces same output
                with torch.no_grad():
                    loaded_output = loaded_model(dummy_input)
                
                # Check if outputs are approximately equal
                output_diff = torch.abs(original_output - loaded_output).max().item()
                serialization_success = output_diff < 1e-6
                
                duration = time.time() - start_time
                
                return PipelineTestResult(
                    name="Model Serialization",
                    status="PASS" if serialization_success else "FAIL",
                    duration_seconds=duration,
                    details={
                        "output_difference": output_diff,
                        "serialization_success": serialization_success,
                        "model_file_size": os.path.getsize(temp_path),
                        "save_format": "state_dict"
                    }
                )
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            duration = time.time() - start_time
            return PipelineTestResult(
                name="Model Serialization",
                status="FAIL",
                duration_seconds=duration,
                details={},
                error_message=str(e)
            )

class DataPipelineTester:
    """Comprehensive data pipeline testing"""
    
    def __init__(self, data_path: str = "./data"):
        self.data_path = Path(data_path)
        self.ingestion_tester = DataIngestionTester(data_path)
        self.model_tester = ModelPipelineTester(data_path)
    
    def run_comprehensive_tests(self) -> Dict:
        """Run all data pipeline tests"""
        logger.info("Starting comprehensive data pipeline tests...")
        
        test_results = []
        
        # Data ingestion tests
        logger.info("Running data ingestion tests...")
        test_results.append(self.ingestion_tester.test_mnist_download())
        test_results.append(self.ingestion_tester.test_data_preprocessing())
        
        # Model pipeline tests
        logger.info("Running model pipeline tests...")
        test_results.append(self.model_tester.test_model_training())
        test_results.append(self.model_tester.test_model_inference())
        test_results.append(self.model_tester.test_model_serialization())
        
        # Compile summary
        total_tests = len(test_results)
        passed_tests = len([r for r in test_results if r.status == "PASS"])
        failed_tests = len([r for r in test_results if r.status == "FAIL"])
        warning_tests = len([r for r in test_results if r.status == "WARNING"])
        
        total_duration = sum(r.duration_seconds for r in test_results)
        
        summary = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "warnings": warning_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_duration_seconds": total_duration,
            "overall_status": "PASS" if failed_tests == 0 else "FAIL"
        }
        
        return {
            "summary": summary,
            "test_results": [
                {
                    "name": r.name,
                    "status": r.status,
                    "duration_seconds": r.duration_seconds,
                    "details": r.details,
                    "error_message": r.error_message
                }
                for r in test_results
            ],
            "test_timestamp": time.time()
        }
    
    def save_test_report(self, results: Dict, output_file: str = "data_pipeline_test_report.json"):
        """Save test results to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Test report saved to {output_path}")
    
    def print_test_summary(self, results: Dict):
        """Print test summary to console"""
        summary = results["summary"]
        
        print("\n" + "="*60)
        print("QFLARE DATA PIPELINE TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Warnings: {summary['warnings']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Duration: {summary['total_duration_seconds']:.2f}s")
        print(f"Overall Status: {summary['overall_status']}")
        
        print("\nDetailed Test Results:")
        print("-" * 50)
        
        for test in results["test_results"]:
            status_symbol = {
                "PASS": "✓",
                "FAIL": "✗",
                "WARNING": "⚠"
            }.get(test["status"], "?")
            
            print(f"{status_symbol} {test['name']:<25} {test['status']:<8} ({test['duration_seconds']:.2f}s)")
            
            if test["status"] == "FAIL" and test["error_message"]:
                print(f"  → Error: {test['error_message']}")
            elif test["status"] == "PASS" and test["details"]:
                # Show key metrics for passed tests
                details = test["details"]
                if "accuracy" in details:
                    print(f"  → Accuracy: {details['accuracy']:.4f}")
                if "train_size" in details:
                    print(f"  → Dataset size: {details['train_size']}")
                if "loss_decreased" in details:
                    print(f"  → Loss decreased: {details['loss_decreased']}")
        
        print("="*60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="QFLARE Data Pipeline Testing")
    parser.add_argument("--data-path", type=str, default="./data",
                       help="Path to data directory")
    parser.add_argument("--output", type=str, default="data_pipeline_test_report.json",
                       help="Output file for test report")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run tests
    tester = DataPipelineTester(args.data_path)
    results = tester.run_comprehensive_tests()
    
    # Save and display results
    tester.save_test_report(results, args.output)
    tester.print_test_summary(results)
    
    # Exit with appropriate code
    if results["summary"]["overall_status"] == "PASS":
        logger.info("All data pipeline tests passed!")
        sys.exit(0)
    else:
        logger.error(f"Data pipeline tests failed! {results['summary']['failed']} tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()