#!/usr/bin/env python3
"""
QFLARE Secure Key Exchange for Server
Implements secure key delivery methods to prevent MITM attacks
"""

import os
import json
import time
import hashlib
import secrets
import base64
import hmac
from datetime import datetime, timedelta


class SecureKeyExchange:
    """
    Implements multiple secure key exchange methods to prevent MITM attacks
    """
    
    def __init__(self, admin_private_key_path="admin_master_key.pem"):
        self.admin_private_key_path = admin_private_key_path
        self.pending_registrations = {}
        self.verified_devices = {}
        
    def generate_admin_master_key(self):
        """Generate admin master key for signing operations (simplified)"""
        if not os.path.exists(self.admin_private_key_path):
            # Generate simplified admin key
            admin_key = secrets.token_hex(32)
            with open(self.admin_private_key_path, 'w') as f:
                f.write(admin_key)
            return admin_key
        else:
            # Load existing key
            with open(self.admin_private_key_path, 'r') as f:
                return f.read().strip()
    
    def method_1_qr_code_with_otp(self, user_request):
        """METHOD 1: QR Code + One-Time Password"""
        device_id = user_request['device_id']
        
        # Generate quantum keys (simplified)
        quantum_keys = {
            'kyber_public': secrets.token_hex(1568),
            'kyber_private': secrets.token_hex(3168),
            'dilithium_public': secrets.token_hex(1952),
            'dilithium_private': secrets.token_hex(4000)
        }
        
        # Generate OTP
        otp = str(secrets.randbelow(1000000)).zfill(6)
        expires_at = time.time() + 1800  # 30 minutes
        
        # Encrypt keys with OTP (simplified)
        key_data = json.dumps(quantum_keys)
        encrypted_data = self._simple_encrypt(key_data, otp)
        
        # Create QR code data
        qr_data = {
            'device_id': device_id,
            'encrypted_keys': encrypted_data,
            'expires_at': expires_at,
            'method': 'qr_otp'
        }
        
        # Save QR code to file (simplified)
        qr_file = f"qr_code_{device_id}.json"
        with open(qr_file, 'w') as f:
            json.dump(qr_data, f)
        
        # Store registration
        self.pending_registrations[device_id] = {
            'method': 'qr_otp',
            'otp': otp,
            'expires_at': expires_at,
            'quantum_keys': quantum_keys,
            'user_request': user_request
        }
        
        return {
            'qr_code_file': qr_file,
            'otp': otp,
            'expires_in': 1800,
            'device_id': device_id
        }
    
    def method_2_secure_email_with_pgp(self, user_request):
        """METHOD 2: Secure Email with PGP Encryption"""
        device_id = user_request['device_id']
        
        # Generate quantum keys
        quantum_keys = {
            'kyber_public': secrets.token_hex(1568),
            'kyber_private': secrets.token_hex(3168),
            'dilithium_public': secrets.token_hex(1952),
            'dilithium_private': secrets.token_hex(4000)
        }
        
        # Simulate PGP encryption (simplified)
        key_data = json.dumps(quantum_keys)
        pgp_encrypted = self._simple_encrypt(key_data, user_request['email'])
        
        # Store registration
        self.pending_registrations[device_id] = {
            'method': 'pgp_email',
            'encrypted_keys': pgp_encrypted,
            'quantum_keys': quantum_keys,
            'user_request': user_request
        }
        
        return {
            'encrypted_keys': pgp_encrypted,
            'device_id': device_id,
            'delivery_method': 'pgp_email'
        }
    
    def method_3_totp_based_exchange(self, user_request):
        """METHOD 3: TOTP (Time-based One-Time Password)"""
        device_id = user_request['device_id']
        
        # Generate quantum keys
        quantum_keys = {
            'kyber_public': secrets.token_hex(1568),
            'kyber_private': secrets.token_hex(3168),
            'dilithium_public': secrets.token_hex(1952),
            'dilithium_private': secrets.token_hex(4000)
        }
        
        # Generate TOTP secret
        totp_secret = base64.b32encode(secrets.token_bytes(20)).decode()
        
        # Encrypt keys with TOTP (simplified)
        key_data = json.dumps(quantum_keys)
        encrypted_keys = self._simple_encrypt(key_data, totp_secret)
        
        # Store registration
        self.pending_registrations[device_id] = {
            'method': 'totp',
            'totp_secret': totp_secret,
            'encrypted_keys': encrypted_keys,
            'quantum_keys': quantum_keys,
            'user_request': user_request
        }
        
        return {
            'totp_secret': totp_secret,
            'device_id': device_id,
            'encrypted_keys': encrypted_keys
        }
    
    def method_4_physical_token_exchange(self, user_request):
        """METHOD 4: Physical Hardware Token"""
        device_id = user_request['device_id']
        
        # Generate quantum keys
        quantum_keys = {
            'kyber_public': secrets.token_hex(1568),
            'kyber_private': secrets.token_hex(3168),
            'dilithium_public': secrets.token_hex(1952),
            'dilithium_private': secrets.token_hex(4000)
        }
        
        # Generate token PIN
        token_pin = str(secrets.randbelow(100000000)).zfill(8)
        
        # Encrypt keys with PIN
        key_data = json.dumps(quantum_keys)
        encrypted_keys = self._simple_encrypt(key_data, token_pin)
        
        # Create token file
        token_file = f"token_{device_id}.json"
        token_data = {
            'device_id': device_id,
            'encrypted_keys': encrypted_keys,
            'created_at': time.time(),
            'method': 'physical_token'
        }
        
        with open(token_file, 'w') as f:
            json.dump(token_data, f)
        
        # Store registration
        self.pending_registrations[device_id] = {
            'method': 'physical_token',
            'token_pin': token_pin,
            'token_file': token_file,
            'quantum_keys': quantum_keys,
            'user_request': user_request
        }
        
        return {
            'token_file': token_file,
            'token_pin': token_pin,
            'device_id': device_id
        }
    
    def method_5_blockchain_verification(self, user_request):
        """METHOD 5: Blockchain-based Key Verification"""
        device_id = user_request['device_id']
        
        # Generate quantum keys
        quantum_keys = {
            'kyber_public': secrets.token_hex(1568),
            'kyber_private': secrets.token_hex(3168),
            'dilithium_public': secrets.token_hex(1952),
            'dilithium_private': secrets.token_hex(4000)
        }
        
        # Create key fingerprint
        key_data = json.dumps(quantum_keys, sort_keys=True)
        key_fingerprint = hashlib.sha256(key_data.encode()).hexdigest()
        
        # Simulate blockchain transaction
        tx_hash = hashlib.sha256(f"{device_id}{time.time()}".encode()).hexdigest()
        block_height = 1000000 + secrets.randbelow(10000)
        
        # Store registration
        self.pending_registrations[device_id] = {
            'method': 'blockchain',
            'tx_hash': tx_hash,
            'key_fingerprint': key_fingerprint,
            'block_height': block_height,
            'quantum_keys': quantum_keys,
            'user_request': user_request
        }
        
        return {
            'tx_hash': tx_hash,
            'key_fingerprint': key_fingerprint,
            'block_height': block_height,
            'device_id': device_id
        }
    
    def verify_and_deliver_keys(self, device_id, verification_data):
        """Verify user credentials and deliver quantum keys"""
        if device_id not in self.pending_registrations:
            raise ValueError("No pending registration found for this device")
        
        registration = self.pending_registrations[device_id]
        method = registration['method']
        
        # Verify based on method
        if method == 'qr_otp':
            if registration.get('otp') != verification_data.get('otp'):
                raise ValueError("Invalid OTP")
            if time.time() > registration.get('expires_at', 0):
                raise ValueError("OTP expired")
                
        elif method == 'totp':
            # Simplified TOTP verification
            current_totp = self._generate_totp(registration['totp_secret'])
            if current_totp != verification_data.get('totp_code'):
                raise ValueError("Invalid TOTP code")
                
        elif method == 'physical_token':
            if registration.get('token_pin') != verification_data.get('token_pin'):
                raise ValueError("Invalid token PIN")
        
        # Deliver quantum keys
        quantum_keys = registration['quantum_keys']
        
        # Move to verified devices
        self.verified_devices[device_id] = registration
        del self.pending_registrations[device_id]
        
        return quantum_keys
    
    def _simple_encrypt(self, data, key):
        """Simplified encryption for demonstration"""
        key_hash = hashlib.sha256(key.encode()).digest()
        return base64.b64encode(data.encode() + key_hash[:16]).decode()
    
    def _simple_decrypt(self, encrypted_data, key):
        """Simplified decryption for demonstration"""
        try:
            decoded = base64.b64decode(encrypted_data.encode())
            data = decoded[:-16]
            return data.decode()
        except:
            raise ValueError("Decryption failed")
    
    def _generate_totp(self, secret, interval=30):
        """Generate TOTP code (simplified)"""
        current_time = int(time.time() // interval)
        return str((current_time + hash(secret)) % 1000000).zfill(6)


# Global instance for server use
secure_key_exchange_instance = None

def get_secure_key_exchange():
    """Get or create secure key exchange instance"""
    global secure_key_exchange_instance
    if secure_key_exchange_instance is None:
        secure_key_exchange_instance = SecureKeyExchange()
        secure_key_exchange_instance.generate_admin_master_key()
    return secure_key_exchange_instance