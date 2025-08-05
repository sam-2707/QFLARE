"""
SSL Certificate Management for QFLARE.

This module handles SSL/TLS certificate generation, validation, and management
for secure communication in the QFLARE system.
"""

import os
import ssl
import socket
import tempfile
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
import subprocess
import shutil

logger = logging.getLogger(__name__)

# SSL Configuration
SSL_CERT_DIR = Path("ssl")
SSL_CERT_FILE = SSL_CERT_DIR / "server.crt"
SSL_KEY_FILE = SSL_CERT_DIR / "server.key"
SSL_CA_FILE = SSL_CERT_DIR / "ca.crt"
SSL_CA_KEY_FILE = SSL_CERT_DIR / "ca.key"


class SSLManager:
    """Manages SSL certificates for secure communication."""
    
    def __init__(self, cert_dir: str = "ssl"):
        """Initialize SSL manager.
        
        Args:
            cert_dir: Directory to store SSL certificates
        """
        self.cert_dir = Path(cert_dir)
        self.cert_dir.mkdir(exist_ok=True)
        
        # Check if OpenSSL is available
        self.openssl_available = self._check_openssl()
        
        if not self.openssl_available:
            logger.warning("OpenSSL not available - using self-signed certificates")
    
    def _check_openssl(self) -> bool:
        """Check if OpenSSL is available on the system."""
        try:
            result = subprocess.run(
                ["openssl", "version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def generate_self_signed_certificate(
        self,
        common_name: str = "localhost",
        country: str = "US",
        state: str = "CA",
        locality: str = "San Francisco",
        organization: str = "QFLARE",
        organizational_unit: str = "Development",
        days_valid: int = 365
    ) -> Tuple[bool, str]:
        """
        Generate a self-signed SSL certificate.
        
        Args:
            common_name: Common name for the certificate
            country: Country code
            state: State or province
            locality: City or locality
            organization: Organization name
            organizational_unit: Organizational unit
            days_valid: Number of days the certificate is valid
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if not self.openssl_available:
                return False, "OpenSSL not available"
            
            # Create certificate configuration
            config_content = f"""
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = {country}
ST = {state}
L = {locality}
O = {organization}
OU = {organizational_unit}
CN = {common_name}

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = {common_name}
DNS.2 = localhost
IP.1 = 127.0.0.1
"""
            
            config_file = self.cert_dir / "openssl.conf"
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            # Generate private key
            subprocess.run([
                "openssl", "genrsa", "-out", str(self.cert_dir / "server.key"), "2048"
            ], check=True, capture_output=True)
            
            # Generate certificate signing request
            subprocess.run([
                "openssl", "req", "-new", "-key", str(self.cert_dir / "server.key"),
                "-out", str(self.cert_dir / "server.csr"), "-config", str(config_file)
            ], check=True, capture_output=True)
            
            # Generate self-signed certificate
            subprocess.run([
                "openssl", "x509", "-req", "-in", str(self.cert_dir / "server.csr"),
                "-signkey", str(self.cert_dir / "server.key"), "-out", str(self.cert_dir / "server.crt"),
                "-days", str(days_valid), "-extensions", "v3_req", "-extfile", str(config_file)
            ], check=True, capture_output=True)
            
            # Clean up temporary files
            (self.cert_dir / "server.csr").unlink(missing_ok=True)
            config_file.unlink(missing_ok=True)
            
            logger.info(f"Generated self-signed certificate for {common_name}")
            return True, "Certificate generated successfully"
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to generate certificate: {e.stderr.decode()}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error generating certificate: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def generate_ca_certificate(
        self,
        common_name: str = "QFLARE CA",
        country: str = "US",
        state: str = "CA",
        locality: str = "San Francisco",
        organization: str = "QFLARE",
        organizational_unit: str = "Certificate Authority",
        days_valid: int = 3650
    ) -> Tuple[bool, str]:
        """
        Generate a Certificate Authority (CA) certificate.
        
        Args:
            common_name: Common name for the CA
            country: Country code
            state: State or province
            locality: City or locality
            organization: Organization name
            organizational_unit: Organizational unit
            days_valid: Number of days the certificate is valid
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if not self.openssl_available:
                return False, "OpenSSL not available"
            
            # Create CA configuration
            config_content = f"""
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = {country}
ST = {state}
L = {locality}
O = {organization}
OU = {organizational_unit}
CN = {common_name}

[v3_req]
basicConstraints = CA:TRUE
keyUsage = keyCertSign, cRLSign
"""
            
            config_file = self.cert_dir / "ca.conf"
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            # Generate CA private key
            subprocess.run([
                "openssl", "genrsa", "-out", str(self.cert_dir / "ca.key"), "4096"
            ], check=True, capture_output=True)
            
            # Generate CA certificate
            subprocess.run([
                "openssl", "req", "-new", "-x509", "-key", str(self.cert_dir / "ca.key"),
                "-out", str(self.cert_dir / "ca.crt"), "-days", str(days_valid),
                "-config", str(config_file)
            ], check=True, capture_output=True)
            
            # Clean up
            config_file.unlink(missing_ok=True)
            
            logger.info(f"Generated CA certificate: {common_name}")
            return True, "CA certificate generated successfully"
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to generate CA certificate: {e.stderr.decode()}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error generating CA certificate: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def generate_server_certificate(
        self,
        common_name: str,
        ca_cert_file: Optional[str] = None,
        ca_key_file: Optional[str] = None,
        days_valid: int = 365
    ) -> Tuple[bool, str]:
        """
        Generate a server certificate signed by a CA.
        
        Args:
            common_name: Common name for the server
            ca_cert_file: Path to CA certificate file
            ca_key_file: Path to CA private key file
            days_valid: Number of days the certificate is valid
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if not self.openssl_available:
                return False, "OpenSSL not available"
            
            # Use default CA files if not specified
            if ca_cert_file is None:
                ca_cert_file = str(self.cert_dir / "ca.crt")
            if ca_key_file is None:
                ca_key_file = str(self.cert_dir / "ca.key")
            
            # Check if CA files exist
            if not Path(ca_cert_file).exists() or not Path(ca_key_file).exists():
                return False, "CA certificate or key file not found"
            
            # Create server configuration
            config_content = f"""
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = CA
L = San Francisco
O = QFLARE
OU = Development
CN = {common_name}

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = {common_name}
DNS.2 = localhost
IP.1 = 127.0.0.1
"""
            
            config_file = self.cert_dir / "server.conf"
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            # Generate server private key
            subprocess.run([
                "openssl", "genrsa", "-out", str(self.cert_dir / "server.key"), "2048"
            ], check=True, capture_output=True)
            
            # Generate certificate signing request
            subprocess.run([
                "openssl", "req", "-new", "-key", str(self.cert_dir / "server.key"),
                "-out", str(self.cert_dir / "server.csr"), "-config", str(config_file)
            ], check=True, capture_output=True)
            
            # Sign with CA
            subprocess.run([
                "openssl", "x509", "-req", "-in", str(self.cert_dir / "server.csr"),
                "-CA", ca_cert_file, "-CAkey", ca_key_file, "-CAcreateserial",
                "-out", str(self.cert_dir / "server.crt"), "-days", str(days_valid),
                "-extensions", "v3_req", "-extfile", str(config_file)
            ], check=True, capture_output=True)
            
            # Clean up
            (self.cert_dir / "server.csr").unlink(missing_ok=True)
            config_file.unlink(missing_ok=True)
            
            logger.info(f"Generated server certificate for {common_name}")
            return True, "Server certificate generated successfully"
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to generate server certificate: {e.stderr.decode()}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error generating server certificate: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def validate_certificate(self, cert_file: str) -> Dict[str, Any]:
        """
        Validate an SSL certificate.
        
        Args:
            cert_file: Path to certificate file
            
        Returns:
            Dictionary with validation results
        """
        try:
            if not Path(cert_file).exists():
                return {
                    "valid": False,
                    "error": "Certificate file not found"
                }
            
            # Get certificate information
            result = subprocess.run([
                "openssl", "x509", "-in", cert_file, "-text", "-noout"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return {
                    "valid": False,
                    "error": f"Invalid certificate: {result.stderr}"
                }
            
            # Parse certificate dates
            dates_result = subprocess.run([
                "openssl", "x509", "-in", cert_file, "-noout", "-dates"
            ], capture_output=True, text=True, timeout=10)
            
            if dates_result.returncode == 0:
                dates = dates_result.stdout.strip().split('\n')
                not_before = dates[0].split('=')[1]
                not_after = dates[1].split('=')[1]
            else:
                not_before = "Unknown"
                not_after = "Unknown"
            
            # Check if certificate is expired
            now = datetime.now()
            try:
                not_after_date = datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z")
                is_expired = now > not_after_date
            except ValueError:
                is_expired = False
            
            return {
                "valid": True,
                "not_before": not_before,
                "not_after": not_after,
                "expired": is_expired,
                "days_remaining": (not_after_date - now).days if not is_expired else 0
            }
            
        except subprocess.TimeoutExpired:
            return {
                "valid": False,
                "error": "Certificate validation timed out"
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Certificate validation error: {e}"
            }
    
    def create_ssl_context(
        self,
        cert_file: Optional[str] = None,
        key_file: Optional[str] = None,
        ca_file: Optional[str] = None
    ) -> Optional[ssl.SSLContext]:
        """
        Create an SSL context for secure communication.
        
        Args:
            cert_file: Path to certificate file
            key_file: Path to private key file
            ca_file: Path to CA certificate file
            
        Returns:
            SSL context or None if creation fails
        """
        try:
            # Use default files if not specified
            if cert_file is None:
                cert_file = str(self.cert_dir / "server.crt")
            if key_file is None:
                key_file = str(self.cert_dir / "server.key")
            
            # Check if files exist
            if not Path(cert_file).exists():
                logger.error(f"Certificate file not found: {cert_file}")
                return None
            if not Path(key_file).exists():
                logger.error(f"Private key file not found: {key_file}")
                return None
            
            # Create SSL context
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            context.load_cert_chain(cert_file, key_file)
            
            # Set verification mode
            context.verify_mode = ssl.CERT_REQUIRED
            context.check_hostname = True
            
            # Load CA certificate if provided
            if ca_file and Path(ca_file).exists():
                context.load_verify_locations(ca_file)
            
            logger.info("SSL context created successfully")
            return context
            
        except Exception as e:
            logger.error(f"Failed to create SSL context: {e}")
            return None
    
    def check_certificate_expiry(self, cert_file: str, days_warning: int = 30) -> Dict[str, Any]:
        """
        Check certificate expiry and warn if close to expiration.
        
        Args:
            cert_file: Path to certificate file
            days_warning: Days before expiry to start warning
            
        Returns:
            Dictionary with expiry information
        """
        validation = self.validate_certificate(cert_file)
        
        if not validation["valid"]:
            return validation
        
        days_remaining = validation.get("days_remaining", 0)
        
        if days_remaining <= 0:
            return {
                "expired": True,
                "warning": "Certificate has expired",
                "days_remaining": days_remaining
            }
        elif days_remaining <= days_warning:
            return {
                "expired": False,
                "warning": f"Certificate expires in {days_remaining} days",
                "days_remaining": days_remaining
            }
        else:
            return {
                "expired": False,
                "warning": None,
                "days_remaining": days_remaining
            }


# Global SSL manager instance
ssl_manager = SSLManager()


def setup_ssl_certificates(force_regenerate: bool = False) -> bool:
    """
    Set up SSL certificates for the server.
    
    Args:
        force_regenerate: Force regeneration of certificates
        
    Returns:
        True if setup successful, False otherwise
    """
    try:
        # Check if certificates already exist
        if not force_regenerate and SSL_CERT_FILE.exists() and SSL_KEY_FILE.exists():
            logger.info("SSL certificates already exist")
            return True
        
        # Generate self-signed certificate
        success, message = ssl_manager.generate_self_signed_certificate()
        
        if success:
            logger.info("SSL certificates generated successfully")
            return True
        else:
            logger.error(f"Failed to generate SSL certificates: {message}")
            return False
            
    except Exception as e:
        logger.error(f"Error setting up SSL certificates: {e}")
        return False


def get_ssl_context() -> Optional[ssl.SSLContext]:
    """Get SSL context for the server."""
    return ssl_manager.create_ssl_context()


def check_certificate_status() -> Dict[str, Any]:
    """Check the status of SSL certificates."""
    if not SSL_CERT_FILE.exists():
        return {
            "status": "missing",
            "message": "SSL certificate not found"
        }
    
    expiry_info = ssl_manager.check_certificate_expiry(str(SSL_CERT_FILE))
    
    if expiry_info.get("expired", False):
        return {
            "status": "expired",
            "message": expiry_info.get("warning", "Certificate expired"),
            "days_remaining": expiry_info.get("days_remaining", 0)
        }
    elif expiry_info.get("warning"):
        return {
            "status": "warning",
            "message": expiry_info.get("warning"),
            "days_remaining": expiry_info.get("days_remaining", 0)
        }
    else:
        return {
            "status": "valid",
            "message": "Certificate is valid",
            "days_remaining": expiry_info.get("days_remaining", 0)
        } 