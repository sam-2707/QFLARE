#!/usr/bin/env python3
"""
QFLARE Crypto Performance Tester - Measure PQC handshake and crypto operation performance
Tests CRYSTALS-Kyber, CRYSTALS-Dilithium, and hybrid TLS performance
"""

import argparse
import json
import platform
import subprocess
import time
import statistics
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import hashlib
import os
import secrets

try:
    # Try to import liboqs for real PQC operations
    import oqs
    HAS_LIBOQS = True
except ImportError:
    HAS_LIBOQS = False
    print("Warning: liboqs not available, using simulated crypto operations")

try:
    import cryptography
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False
    print("Warning: cryptography library not available")

@dataclass
class CryptoPerformanceResult:
    """Results from crypto performance testing"""
    operation: str
    algorithm: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    stddev_ms: float
    ops_per_second: float
    key_size_bytes: Optional[int] = None
    ciphertext_size_bytes: Optional[int] = None

class PQCTester:
    """Post-Quantum Cryptography performance tester"""
    
    def __init__(self):
        self.results: List[CryptoPerformanceResult] = []
        
    def test_kyber_kem(self, iterations: int = 1000) -> CryptoPerformanceResult:
        """Test CRYSTALS-Kyber Key Encapsulation Mechanism"""
        if not HAS_LIBOQS:
            return self._simulate_kyber(iterations)
            
        # Test Kyber-768 (NIST Level 3)
        kem = oqs.KeyEncapsulation("Kyber768")
        
        # Key generation timing
        keygen_times = []
        encaps_times = []
        decaps_times = []
        
        for i in range(iterations):
            # Key generation
            start = time.perf_counter()
            public_key = kem.generate_keypair()
            keygen_time = (time.perf_counter() - start) * 1000
            keygen_times.append(keygen_time)
            
            # Encapsulation
            start = time.perf_counter()
            ciphertext, shared_secret1 = kem.encap(public_key)
            encaps_time = (time.perf_counter() - start) * 1000
            encaps_times.append(encaps_time)
            
            # Decapsulation
            start = time.perf_counter()
            shared_secret2 = kem.decap(ciphertext)
            decaps_time = (time.perf_counter() - start) * 1000
            decaps_times.append(decaps_time)
            
            # Verify correctness
            assert shared_secret1 == shared_secret2
        
        # Calculate combined handshake time (keygen + encaps + decaps)
        handshake_times = [kg + enc + dec for kg, enc, dec in zip(keygen_times, encaps_times, decaps_times)]
        
        result = CryptoPerformanceResult(
            operation="KEM Handshake",
            algorithm="CRYSTALS-Kyber768",
            iterations=iterations,
            total_time_ms=sum(handshake_times),
            avg_time_ms=statistics.mean(handshake_times),
            min_time_ms=min(handshake_times),
            max_time_ms=max(handshake_times),
            stddev_ms=statistics.stdev(handshake_times) if len(handshake_times) > 1 else 0.0,
            ops_per_second=1000.0 / statistics.mean(handshake_times),
            key_size_bytes=len(public_key),
            ciphertext_size_bytes=len(ciphertext)
        )
        
        self.results.append(result)
        return result
    
    def test_dilithium_signatures(self, iterations: int = 1000) -> CryptoPerformanceResult:
        """Test CRYSTALS-Dilithium Digital Signatures"""
        if not HAS_LIBOQS:
            return self._simulate_dilithium(iterations)
            
        # Test Dilithium3 (NIST Level 3)
        sig = oqs.Signature("Dilithium3")
        
        # Generate test message
        message = b"QFLARE federated learning update batch #12345"
        
        sign_times = []
        verify_times = []
        
        # Generate keypair once
        public_key = sig.generate_keypair()
        
        for i in range(iterations):
            # Signing
            start = time.perf_counter()
            signature = sig.sign(message)
            sign_time = (time.perf_counter() - start) * 1000
            sign_times.append(sign_time)
            
            # Verification
            start = time.perf_counter()
            is_valid = sig.verify(message, signature, public_key)
            verify_time = (time.perf_counter() - start) * 1000
            verify_times.append(verify_time)
            
            # Verify correctness
            assert is_valid
        
        # Calculate combined sign+verify time
        combined_times = [s + v for s, v in zip(sign_times, verify_times)]
        
        result = CryptoPerformanceResult(
            operation="Digital Signature",
            algorithm="CRYSTALS-Dilithium3",
            iterations=iterations,
            total_time_ms=sum(combined_times),
            avg_time_ms=statistics.mean(combined_times),
            min_time_ms=min(combined_times),
            max_time_ms=max(combined_times),
            stddev_ms=statistics.stdev(combined_times) if len(combined_times) > 1 else 0.0,
            ops_per_second=1000.0 / statistics.mean(combined_times),
            key_size_bytes=len(public_key),
            ciphertext_size_bytes=len(signature)
        )
        
        self.results.append(result)
        return result
    
    def test_hybrid_handshake(self, iterations: int = 100) -> CryptoPerformanceResult:
        """Test hybrid classical + PQC handshake"""
        if not HAS_CRYPTOGRAPHY:
            return self._simulate_hybrid(iterations)
            
        handshake_times = []
        
        for i in range(iterations):
            start = time.perf_counter()
            
            # Classical ECDH (simulated with RSA for simplicity)
            if HAS_CRYPTOGRAPHY:
                # Generate RSA keypair (simulating ECDH)
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048
                )
                public_key = private_key.public_key()
                
                # Simulate key exchange
                message = secrets.token_bytes(32)
                ciphertext = public_key.encrypt(
                    message,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                plaintext = private_key.decrypt(
                    ciphertext,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                assert message == plaintext
            
            # PQC KEM (if available)
            if HAS_LIBOQS:
                kem = oqs.KeyEncapsulation("Kyber768")
                public_key_pqc = kem.generate_keypair()
                ciphertext_pqc, shared_secret = kem.encap(public_key_pqc)
                shared_secret2 = kem.decap(ciphertext_pqc)
                assert shared_secret == shared_secret2
            
            handshake_time = (time.perf_counter() - start) * 1000
            handshake_times.append(handshake_time)
        
        result = CryptoPerformanceResult(
            operation="Hybrid Handshake",
            algorithm="RSA-2048 + Kyber768",
            iterations=iterations,
            total_time_ms=sum(handshake_times),
            avg_time_ms=statistics.mean(handshake_times),
            min_time_ms=min(handshake_times),
            max_time_ms=max(handshake_times),
            stddev_ms=statistics.stdev(handshake_times) if len(handshake_times) > 1 else 0.0,
            ops_per_second=1000.0 / statistics.mean(handshake_times),
        )
        
        self.results.append(result)
        return result
    
    def _simulate_kyber(self, iterations: int) -> CryptoPerformanceResult:
        """Simulate Kyber performance when liboqs is not available"""
        # Based on known Kyber performance characteristics
        base_time = 0.5  # ~0.5ms for Kyber768 on modern hardware
        handshake_times = []
        
        for i in range(iterations):
            # Add realistic variance
            time_ms = base_time + secrets.randbelow(100) / 1000.0  # 0-0.1ms variance
            handshake_times.append(time_ms)
        
        return CryptoPerformanceResult(
            operation="KEM Handshake (Simulated)",
            algorithm="CRYSTALS-Kyber768",
            iterations=iterations,
            total_time_ms=sum(handshake_times),
            avg_time_ms=statistics.mean(handshake_times),
            min_time_ms=min(handshake_times),
            max_time_ms=max(handshake_times),
            stddev_ms=statistics.stdev(handshake_times) if len(handshake_times) > 1 else 0.0,
            ops_per_second=1000.0 / statistics.mean(handshake_times),
            key_size_bytes=1184,  # Kyber768 public key size
            ciphertext_size_bytes=1088  # Kyber768 ciphertext size
        )
    
    def _simulate_dilithium(self, iterations: int) -> CryptoPerformanceResult:
        """Simulate Dilithium performance when liboqs is not available"""
        base_time = 0.8  # ~0.8ms for Dilithium3 sign+verify
        times = []
        
        for i in range(iterations):
            time_ms = base_time + secrets.randbelow(200) / 1000.0  # 0-0.2ms variance
            times.append(time_ms)
        
        return CryptoPerformanceResult(
            operation="Digital Signature (Simulated)",
            algorithm="CRYSTALS-Dilithium3",
            iterations=iterations,
            total_time_ms=sum(times),
            avg_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            stddev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
            ops_per_second=1000.0 / statistics.mean(times),
            key_size_bytes=1952,  # Dilithium3 public key size
            ciphertext_size_bytes=3293  # Dilithium3 signature size
        )
    
    def _simulate_hybrid(self, iterations: int) -> CryptoPerformanceResult:
        """Simulate hybrid handshake when crypto libraries unavailable"""
        base_time = 50.0  # ~50ms for full hybrid handshake
        times = []
        
        for i in range(iterations):
            time_ms = base_time + secrets.randbelow(20000) / 1000.0  # 0-20ms variance
            times.append(time_ms)
        
        return CryptoPerformanceResult(
            operation="Hybrid Handshake (Simulated)",
            algorithm="RSA-2048 + Kyber768",
            iterations=iterations,
            total_time_ms=sum(times),
            avg_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            stddev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
            ops_per_second=1000.0 / statistics.mean(times),
        )

class SystemBenchmark:
    """System-level crypto performance testing"""
    
    @staticmethod
    def get_system_info() -> Dict:
        """Collect system information for benchmarking context"""
        try:
            import psutil
            
            return {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "cpu_count": psutil.cpu_count(),
                "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown",
                "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "python_version": platform.python_version()
            }
        except ImportError:
            return {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "python_version": platform.python_version()
            }
    
    @staticmethod
    def test_openssl_performance() -> Optional[Dict]:
        """Test OpenSSL performance using system commands"""
        if platform.system() == "Windows":
            return SystemBenchmark._test_openssl_windows()
        else:
            return SystemBenchmark._test_openssl_unix()
    
    @staticmethod
    def _test_openssl_windows() -> Optional[Dict]:
        """Test OpenSSL on Windows (if available)"""
        try:
            # Try to find OpenSSL
            result = subprocess.run(
                ["openssl", "version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return None
            
            openssl_version = result.stdout.strip()
            
            # Benchmark RSA operations
            rsa_result = subprocess.run(
                ["openssl", "speed", "-seconds", "3", "rsa2048"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "openssl_version": openssl_version,
                "rsa_benchmark_available": rsa_result.returncode == 0,
                "rsa_output": rsa_result.stdout if rsa_result.returncode == 0 else "Failed"
            }
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
    
    @staticmethod
    def _test_openssl_unix() -> Optional[Dict]:
        """Test OpenSSL on Unix-like systems"""
        try:
            # Check OpenSSL availability
            result = subprocess.run(
                ["openssl", "version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return None
            
            openssl_version = result.stdout.strip()
            
            # Quick RSA benchmark
            rsa_result = subprocess.run(
                ["openssl", "speed", "-seconds", "3", "rsa2048"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "openssl_version": openssl_version,
                "rsa_benchmark_available": rsa_result.returncode == 0,
                "rsa_output": rsa_result.stdout if rsa_result.returncode == 0 else "Failed"
            }
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

def create_powershell_test_script() -> str:
    """Create a PowerShell script for Windows crypto testing"""
    script_content = '''
# QFLARE Crypto Performance Test - PowerShell Component
param(
    [int]$Iterations = 100,
    [switch]$Verbose
)

Write-Host "QFLARE Crypto Performance Test (PowerShell Component)"
Write-Host "Platform: Windows PowerShell"
Write-Host "Iterations: $Iterations"
Write-Host ""

# Test .NET crypto performance as baseline
function Test-DotNetCrypto {
    param([int]$iterations)
    
    $results = @()
    
    # RSA key generation and operations
    for ($i = 0; $i -lt $iterations; $i++) {
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        
        # Generate RSA keypair
        $rsa = [System.Security.Cryptography.RSA]::Create(2048)
        $publicKey = $rsa.ExportRSAPublicKey()
        $privateKey = $rsa.ExportRSAPrivateKey()
        
        # Encrypt and decrypt a small message
        $message = [System.Text.Encoding]::UTF8.GetBytes("QFLARE test message")
        $encrypted = $rsa.Encrypt($message, [System.Security.Cryptography.RSAEncryptionPadding]::OaepSHA256)
        $decrypted = $rsa.Decrypt($encrypted, [System.Security.Cryptography.RSAEncryptionPadding]::OaepSHA256)
        
        $stopwatch.Stop()
        $results += $stopwatch.ElapsedMilliseconds
        
        $rsa.Dispose()
        
        if ($Verbose -and ($i % 10 -eq 0)) {
            Write-Host "Completed $i/$iterations iterations..."
        }
    }
    
    $avgTime = ($results | Measure-Object -Average).Average
    $minTime = ($results | Measure-Object -Minimum).Minimum
    $maxTime = ($results | Measure-Object -Maximum).Maximum
    
    Write-Host "RSA-2048 Performance (.NET Baseline):"
    Write-Host "  Average: $([math]::Round($avgTime, 2)) ms"
    Write-Host "  Minimum: $minTime ms"
    Write-Host "  Maximum: $maxTime ms"
    Write-Host "  Ops/sec: $([math]::Round(1000 / $avgTime, 2))"
    Write-Host ""
    
    return @{
        "algorithm" = "RSA-2048 (.NET)"
        "avg_time_ms" = $avgTime
        "min_time_ms" = $minTime
        "max_time_ms" = $maxTime
        "ops_per_second" = 1000 / $avgTime
    }
}

# Test AES encryption performance
function Test-AESPerformance {
    param([int]$iterations)
    
    $results = @()
    $testData = [System.Text.Encoding]::UTF8.GetBytes("QFLARE federated learning model update data " * 100)  # ~4KB
    
    for ($i = 0; $i -lt $iterations; $i++) {
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        
        # AES-256 encryption/decryption
        $aes = [System.Security.Cryptography.Aes]::Create()
        $aes.KeySize = 256
        $aes.GenerateKey()
        $aes.GenerateIV()
        
        $encryptor = $aes.CreateEncryptor()
        $decryptor = $aes.CreateDecryptor()
        
        # Encrypt
        $encrypted = $encryptor.TransformFinalBlock($testData, 0, $testData.Length)
        # Decrypt
        $decrypted = $decryptor.TransformFinalBlock($encrypted, 0, $encrypted.Length)
        
        $stopwatch.Stop()
        $results += $stopwatch.ElapsedMilliseconds
        
        $aes.Dispose()
    }
    
    $avgTime = ($results | Measure-Object -Average).Average
    $throughputMBps = ($testData.Length / 1024 / 1024) * (1000 / $avgTime)
    
    Write-Host "AES-256 Performance:"
    Write-Host "  Average: $([math]::Round($avgTime, 2)) ms (for $($testData.Length) bytes)"
    Write-Host "  Throughput: $([math]::Round($throughputMBps, 2)) MB/s"
    Write-Host ""
    
    return @{
        "algorithm" = "AES-256"
        "avg_time_ms" = $avgTime
        "throughput_mbps" = $throughputMBps
        "data_size_bytes" = $testData.Length
    }
}

# Execute tests
$dotnetResult = Test-DotNetCrypto -iterations $Iterations
$aesResult = Test-AESPerformance -iterations ($Iterations / 10)  # AES is much faster

# Output results as JSON for Python integration
$results = @{
    "platform" = "Windows PowerShell"
    "iterations" = $Iterations
    "timestamp" = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    "results" = @($dotnetResult, $aesResult)
}

$json = $results | ConvertTo-Json -Depth 3
Write-Host "JSON Results:"
$json

# Save to file
$json | Out-File -FilePath "powershell_crypto_results.json" -Encoding UTF8
Write-Host "Results saved to: powershell_crypto_results.json"
'''
    return script_content

def main():
    """Main crypto performance testing function"""
    parser = argparse.ArgumentParser(description='QFLARE Crypto Performance Tester')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations for crypto tests')
    parser.add_argument('--quick', action='store_true', help='Run quick tests (fewer iterations)')
    parser.add_argument('--kyber-only', action='store_true', help='Test only Kyber KEM')
    parser.add_argument('--dilithium-only', action='store_true', help='Test only Dilithium signatures')
    parser.add_argument('--hybrid-only', action='store_true', help='Test only hybrid handshake')
    parser.add_argument('--system-info', action='store_true', help='Include system information')
    parser.add_argument('--powershell', action='store_true', help='Generate and run PowerShell tests')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    if args.quick:
        args.iterations = min(100, args.iterations)
    
    print("QFLARE Crypto Performance Tester")
    print("=" * 50)
    
    # System information
    if args.system_info:
        system_info = SystemBenchmark.get_system_info()
        print("\nSystem Information:")
        for key, value in system_info.items():
            print(f"  {key}: {value}")
        
        openssl_info = SystemBenchmark.test_openssl_performance()
        if openssl_info:
            print(f"  OpenSSL: {openssl_info['openssl_version']}")
    
    # Initialize tester
    tester = PQCTester()
    print(f"\nRunning crypto performance tests ({args.iterations} iterations)...")
    print(f"PQC Libraries Available: liboqs={HAS_LIBOQS}, cryptography={HAS_CRYPTOGRAPHY}")
    print()
    
    # Run tests based on arguments
    if not any([args.kyber_only, args.dilithium_only, args.hybrid_only]):
        # Run all tests
        print("Testing CRYSTALS-Kyber KEM...")
        kyber_result = tester.test_kyber_kem(args.iterations)
        
        print("Testing CRYSTALS-Dilithium signatures...")
        dilithium_result = tester.test_dilithium_signatures(args.iterations)
        
        print("Testing hybrid handshake...")
        hybrid_result = tester.test_hybrid_handshake(min(100, args.iterations // 10))
    else:
        if args.kyber_only:
            print("Testing CRYSTALS-Kyber KEM...")
            tester.test_kyber_kem(args.iterations)
        if args.dilithium_only:
            print("Testing CRYSTALS-Dilithium signatures...")
            tester.test_dilithium_signatures(args.iterations)
        if args.hybrid_only:
            print("Testing hybrid handshake...")
            tester.test_hybrid_handshake(args.iterations)
    
    # PowerShell integration for Windows
    if args.powershell and platform.system() == "Windows":
        print("\nGenerating PowerShell crypto tests...")
        ps_script = create_powershell_test_script()
        
        with open("qflare_crypto_test.ps1", "w") as f:
            f.write(ps_script)
        
        print("PowerShell script created: qflare_crypto_test.ps1")
        print("Run with: powershell -ExecutionPolicy Bypass -File qflare_crypto_test.ps1")
        
        # Try to run it
        try:
            result = subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File", "qflare_crypto_test.ps1", "-Iterations", str(min(100, args.iterations))],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                print("PowerShell tests completed successfully!")
            else:
                print(f"PowerShell tests failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Could not run PowerShell tests: {e}")
    
    # Display results
    print("\n" + "=" * 60)
    print("CRYPTO PERFORMANCE RESULTS")
    print("=" * 60)
    
    for result in tester.results:
        status = "✅" if result.avg_time_ms < 500 else "⚠️" if result.avg_time_ms < 1000 else "❌"
        
        print(f"\n{result.operation} ({result.algorithm}) {status}")
        print(f"  Average Time: {result.avg_time_ms:.2f} ms")
        print(f"  Min/Max: {result.min_time_ms:.2f} / {result.max_time_ms:.2f} ms")
        print(f"  Std Dev: {result.stddev_ms:.2f} ms")
        print(f"  Operations/sec: {result.ops_per_second:.2f}")
        if result.key_size_bytes:
            print(f"  Key Size: {result.key_size_bytes} bytes")
        if result.ciphertext_size_bytes:
            print(f"  Ciphertext Size: {result.ciphertext_size_bytes} bytes")
    
    # Generate performance assessment
    print("\n" + "=" * 60)
    print("PERFORMANCE ASSESSMENT")
    print("=" * 60)
    
    handshake_results = [r for r in tester.results if "Handshake" in r.operation]
    if handshake_results:
        avg_handshake = statistics.mean([r.avg_time_ms for r in handshake_results])
        print(f"\nAverage Handshake Time: {avg_handshake:.2f} ms")
        
        if avg_handshake <= 100:
            assessment = "✅ EXCELLENT - Sub-100ms handshakes suitable for real-time applications"
        elif avg_handshake <= 500:
            assessment = "✅ GOOD - Handshakes within acceptable range for most applications"
        elif avg_handshake <= 1000:
            assessment = "⚠️ ACCEPTABLE - May impact user experience in interactive applications"
        else:
            assessment = "❌ POOR - Handshakes too slow for production use"
        
        print(f"Assessment: {assessment}")
    
    # Save results
    if args.output:
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": SystemBenchmark.get_system_info() if args.system_info else {},
            "test_config": {
                "iterations": args.iterations,
                "libraries_available": {
                    "liboqs": HAS_LIBOQS,
                    "cryptography": HAS_CRYPTOGRAPHY
                }
            },
            "results": [asdict(r) for r in tester.results]
        }
        
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")

if __name__ == "__main__":
    main()