#!/usr/bin/env python3
"""
QFLARE Checksum Management
Generates and verifies checksums for critical data files
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional
import time
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ChecksumRecord:
    """Represents a file checksum record"""
    file_path: str
    sha256: str
    md5: str
    size_bytes: int
    timestamp: float
    
class ChecksumManager:
    """Manages checksums for QFLARE data files"""
    
    def __init__(self, data_path: str = "./data"):
        self.data_path = Path(data_path)
        self.checksums_file = self.data_path / "checksums.json"
        
    def calculate_file_checksums(self, file_path: Path) -> ChecksumRecord:
        """Calculate SHA256 and MD5 checksums for a file"""
        sha256_hash = hashlib.sha256()
        md5_hash = hashlib.md5()
        
        file_size = file_path.stat().st_size
        
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256_hash.update(chunk)
                md5_hash.update(chunk)
        
        return ChecksumRecord(
            file_path=str(file_path.relative_to(self.data_path)),
            sha256=sha256_hash.hexdigest(),
            md5=md5_hash.hexdigest(),
            size_bytes=file_size,
            timestamp=time.time()
        )
    
    def generate_checksums(self, file_patterns: List[str] = None) -> Dict:
        """Generate checksums for all relevant files"""
        if file_patterns is None:
            file_patterns = [
                "**/*.pth",     # PyTorch models
                "**/*.pt",      # PyTorch tensors
                "**/*.db",      # SQLite databases
                "**/*.sqlite",  # SQLite databases
                "**/*.json",    # Configuration files
                "**/*.yaml",    # Configuration files
                "**/*.yml",     # Configuration files
                "MNIST/raw/*",  # MNIST raw data
                "keys/**/*",    # Cryptographic keys
                "models/**/*"   # Model artifacts
            ]
        
        logger.info("Generating checksums for data files...")
        
        checksums = {}
        file_count = 0
        
        for pattern in file_patterns:
            for file_path in self.data_path.glob(pattern):
                if file_path.is_file():
                    try:
                        record = self.calculate_file_checksums(file_path)
                        checksums[record.file_path] = {
                            "sha256": record.sha256,
                            "md5": record.md5,
                            "size_bytes": record.size_bytes,
                            "timestamp": record.timestamp
                        }
                        file_count += 1
                        logger.debug(f"Generated checksum for {record.file_path}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to checksum {file_path}: {e}")
        
        # Add metadata
        result = {
            "metadata": {
                "generated_at": time.time(),
                "total_files": file_count,
                "data_path": str(self.data_path),
                "patterns": file_patterns
            },
            "checksums": checksums
        }
        
        logger.info(f"Generated checksums for {file_count} files")
        return result
    
    def save_checksums(self, checksums: Dict):
        """Save checksums to file"""
        self.checksums_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.checksums_file, 'w') as f:
            json.dump(checksums, f, indent=2)
        
        logger.info(f"Checksums saved to {self.checksums_file}")
    
    def load_checksums(self) -> Optional[Dict]:
        """Load checksums from file"""
        if not self.checksums_file.exists():
            logger.warning(f"Checksums file not found: {self.checksums_file}")
            return None
        
        with open(self.checksums_file, 'r') as f:
            return json.load(f)
    
    def verify_checksums(self, checksums: Dict = None) -> Dict:
        """Verify current files against stored checksums"""
        if checksums is None:
            checksums = self.load_checksums()
            if checksums is None:
                return {"error": "No checksums found to verify against"}
        
        logger.info("Verifying file checksums...")
        
        verification_results = {
            "verified": [],
            "modified": [],
            "missing": [],
            "new_files": [],
            "errors": []
        }
        
        stored_checksums = checksums.get("checksums", {})
        
        # Check stored files
        for relative_path, stored_data in stored_checksums.items():
            file_path = self.data_path / relative_path
            
            if not file_path.exists():
                verification_results["missing"].append({
                    "file": relative_path,
                    "reason": "File not found"
                })
                continue
            
            try:
                current_record = self.calculate_file_checksums(file_path)
                
                if current_record.sha256 == stored_data["sha256"]:
                    verification_results["verified"].append({
                        "file": relative_path,
                        "sha256": current_record.sha256
                    })
                else:
                    verification_results["modified"].append({
                        "file": relative_path,
                        "stored_sha256": stored_data["sha256"],
                        "current_sha256": current_record.sha256,
                        "stored_size": stored_data["size_bytes"],
                        "current_size": current_record.size_bytes
                    })
                    
            except Exception as e:
                verification_results["errors"].append({
                    "file": relative_path,
                    "error": str(e)
                })
        
        # Check for new files
        current_files = set()
        for pattern in ["**/*.pth", "**/*.pt", "**/*.db", "**/*.sqlite", "MNIST/raw/*"]:
            for file_path in self.data_path.glob(pattern):
                if file_path.is_file():
                    relative_path = str(file_path.relative_to(self.data_path))
                    current_files.add(relative_path)
        
        stored_files = set(stored_checksums.keys())
        new_files = current_files - stored_files
        
        for new_file in new_files:
            verification_results["new_files"].append(new_file)
        
        # Summary
        verification_results["summary"] = {
            "total_verified": len(verification_results["verified"]),
            "total_modified": len(verification_results["modified"]),
            "total_missing": len(verification_results["missing"]),
            "total_new": len(verification_results["new_files"]),
            "total_errors": len(verification_results["errors"]),
            "verification_timestamp": time.time()
        }
        
        logger.info(f"Verification complete: {verification_results['summary']}")
        return verification_results
    
    def print_verification_report(self, results: Dict):
        """Print verification results to console"""
        summary = results["summary"]
        
        print("\n" + "="*60)
        print("QFLARE CHECKSUM VERIFICATION REPORT")
        print("="*60)
        print(f"Verified: {summary['total_verified']}")
        print(f"Modified: {summary['total_modified']}")
        print(f"Missing: {summary['total_missing']}")
        print(f"New Files: {summary['total_new']}")
        print(f"Errors: {summary['total_errors']}")
        
        if results["modified"]:
            print(f"\n⚠️  Modified Files ({len(results['modified'])}):")
            for mod in results["modified"]:
                print(f"  - {mod['file']}")
                print(f"    Stored: {mod['stored_sha256'][:16]}... ({mod['stored_size']} bytes)")
                print(f"    Current: {mod['current_sha256'][:16]}... ({mod['current_size']} bytes)")
        
        if results["missing"]:
            print(f"\n❌ Missing Files ({len(results['missing'])}):")
            for missing in results["missing"]:
                print(f"  - {missing['file']}")
        
        if results["new_files"]:
            print(f"\n📁 New Files ({len(results['new_files'])}):")
            for new_file in results["new_files"]:
                print(f"  - {new_file}")
        
        if results["errors"]:
            print(f"\n💥 Errors ({len(results['errors'])}):")
            for error in results["errors"]:
                print(f"  - {error['file']}: {error['error']}")
        
        print("="*60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="QFLARE Checksum Management")
    parser.add_argument("action", choices=["generate", "verify", "update"],
                       help="Action to perform")
    parser.add_argument("--data-path", type=str, default="./data",
                       help="Path to data directory")
    parser.add_argument("--output", type=str, help="Output file for verification results")
    parser.add_argument("--patterns", nargs="*", help="File patterns to include")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    manager = ChecksumManager(args.data_path)
    
    if args.action == "generate":
        # Generate new checksums
        checksums = manager.generate_checksums(args.patterns)
        manager.save_checksums(checksums)
        print(f"Generated checksums for {checksums['metadata']['total_files']} files")
        
    elif args.action == "verify":
        # Verify existing checksums
        results = manager.verify_checksums()
        manager.print_verification_report(results)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Verification results saved to {args.output}")
        
        # Exit with error if verification failed
        if results["summary"]["total_modified"] > 0 or results["summary"]["total_missing"] > 0:
            sys.exit(1)
            
    elif args.action == "update":
        # Update checksums (generate and save)
        checksums = manager.generate_checksums(args.patterns)
        manager.save_checksums(checksums)
        
        # Also verify to show differences
        old_checksums = manager.load_checksums()
        if old_checksums:
            results = manager.verify_checksums(old_checksums)
            manager.print_verification_report(results)
        
        print(f"Updated checksums for {checksums['metadata']['total_files']} files")

if __name__ == "__main__":
    main()