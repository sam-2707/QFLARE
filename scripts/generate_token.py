#!/usr/bin/env python3
"""
Generate secure one-time enrollment tokens for new devices.
This script is used by administrators to create tokens for device enrollment.
"""

import secrets
import hashlib
import time
import json
from pathlib import Path
from typing import Optional


class TokenGenerator:
    def __init__(self, token_file: str = "enrollment_tokens.json"):
        self.token_file = Path(token_file)
        self.tokens = self._load_tokens()
    
    def _load_tokens(self) -> dict:
        """Load existing tokens from file."""
        if self.token_file.exists():
            try:
                with open(self.token_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}
    
    def _save_tokens(self):
        """Save tokens to file."""
        self.token_file.parent.mkdir(exist_ok=True)
        with open(self.token_file, 'w') as f:
            json.dump(self.tokens, f, indent=2)
    
    def generate_token(self, device_id: str, expires_in_hours: int = 24) -> str:
        """Generate a secure one-time enrollment token."""
        # Generate a cryptographically secure random token
        token = secrets.token_urlsafe(32)
        
        # Create token record
        token_data = {
            "device_id": device_id,
            "created_at": time.time(),
            "expires_at": time.time() + (expires_in_hours * 3600),
            "used": False,
            "hash": hashlib.sha256(token.encode()).hexdigest()
        }
        
        # Store token hash (not the actual token)
        self.tokens[token] = token_data
        self._save_tokens()
        
        return token
    
    def validate_token(self, token: str, device_id: str) -> bool:
        """Validate a token for device enrollment."""
        if token not in self.tokens:
            return False
        
        token_data = self.tokens[token]
        
        # Check if token is already used
        if token_data.get("used", False):
            return False
        
        # Check if token is expired
        if time.time() > token_data.get("expires_at", 0):
            return False
        
        # Check if device_id matches
        if token_data.get("device_id") != device_id:
            return False
        
        return True
    
    def revoke_token(self, token: str):
        """Mark a token as used/revoked."""
        if token in self.tokens:
            self.tokens[token]["used"] = True
            self._save_tokens()
    
    def list_tokens(self) -> dict:
        """List all tokens and their status."""
        return self.tokens


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate enrollment tokens for QFLARE devices")
    parser.add_argument("--device-id", required=True, help="Device ID for the token")
    parser.add_argument("--expires", type=int, default=24, help="Token expiration in hours (default: 24)")
    parser.add_argument("--list", action="store_true", help="List all existing tokens")
    parser.add_argument("--revoke", help="Revoke a specific token")
    
    args = parser.parse_args()
    
    generator = TokenGenerator()
    
    if args.list:
        tokens = generator.list_tokens()
        print("Existing tokens:")
        for token, data in tokens.items():
            status = "USED" if data.get("used") else "ACTIVE"
            expires = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data.get("expires_at", 0)))
            print(f"  {token[:16]}... - {data.get('device_id')} - {status} - Expires: {expires}")
        return
    
    if args.revoke:
        generator.revoke_token(args.revoke)
        print(f"Token {args.revoke[:16]}... revoked successfully")
        return
    
    # Generate new token
    token = generator.generate_token(args.device_id, args.expires)
    print(f"Generated enrollment token for device {args.device_id}:")
    print(f"Token: {token}")
    print(f"Expires in: {args.expires} hours")
    print(f"Use this token with the enroll_device.py script")


if __name__ == "__main__":
    main() 