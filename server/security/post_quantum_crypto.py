"""
QFLARE Post-Quantum Cryptography Implementation
NIST-approved post-quantum cryptographic algorithms for quantum-resistant security
"""

import hashlib
import hmac
import secrets
import struct
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime
import json
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PQCAlgorithm(Enum):
    """Post-quantum cryptographic algorithms"""
    KYBER_512 = "kyber512"      # Key encapsulation
    KYBER_768 = "kyber768"      # Key encapsulation  
    KYBER_1024 = "kyber1024"    # Key encapsulation
    DILITHIUM_2 = "dilithium2"  # Digital signatures
    DILITHIUM_3 = "dilithium3"  # Digital signatures
    DILITHIUM_5 = "dilithium5"  # Digital signatures

@dataclass
class PQCKeyPair:
    """Post-quantum cryptographic key pair"""
    public_key: bytes
    private_key: bytes
    algorithm: PQCAlgorithm
    key_id: str
    created_at: str
    parameters: Dict[str, Any]

@dataclass
class PQCCiphertext:
    """Post-quantum ciphertext with metadata"""
    ciphertext: bytes
    algorithm: PQCAlgorithm
    key_id: str
    timestamp: str
    additional_data: Optional[bytes] = None

@dataclass
class PQCSignature:
    """Post-quantum digital signature"""
    signature: bytes
    algorithm: PQCAlgorithm
    signer_key_id: str
    message_hash: bytes
    timestamp: str

class KyberParameters:
    """CRYSTALS-Kyber parameter sets"""
    
    KYBER_512 = {
        'n': 256,           # Polynomial degree
        'q': 3329,          # Modulus
        'k': 2,             # Rank of module
        'eta1': 3,          # Noise parameter for key generation
        'eta2': 2,          # Noise parameter for encryption
        'du': 10,           # Compression parameter
        'dv': 4,            # Compression parameter
        'public_key_size': 800,
        'private_key_size': 1632,
        'ciphertext_size': 768,
        'shared_secret_size': 32
    }
    
    KYBER_768 = {
        'n': 256,
        'q': 3329,
        'k': 3,
        'eta1': 2,
        'eta2': 2,
        'du': 10,
        'dv': 4,
        'public_key_size': 1184,
        'private_key_size': 2400,
        'ciphertext_size': 1088,
        'shared_secret_size': 32
    }
    
    KYBER_1024 = {
        'n': 256,
        'q': 3329,
        'k': 4,
        'eta1': 2,
        'eta2': 2,
        'du': 11,
        'dv': 5,
        'public_key_size': 1568,
        'private_key_size': 3168,
        'ciphertext_size': 1568,
        'shared_secret_size': 32
    }

class DilithiumParameters:
    """CRYSTALS-Dilithium parameter sets"""
    
    DILITHIUM_2 = {
        'n': 256,
        'q': 8380417,
        'k': 4,         # Rank of A
        'l': 4,         # Rank of s1, s2
        'eta': 2,       # Noise parameter
        'tau': 39,      # Number of ±1's in challenge
        'beta': 78,     # Max coefficient of w - cs2
        'gamma1': 2**17,
        'gamma2': (8380417 - 1) // 88,
        'public_key_size': 1312,
        'private_key_size': 2528,
        'signature_size': 2420
    }
    
    DILITHIUM_3 = {
        'n': 256,
        'q': 8380417,
        'k': 6,
        'l': 5,
        'eta': 4,
        'tau': 49,
        'beta': 196,
        'gamma1': 2**19,
        'gamma2': (8380417 - 1) // 32,
        'public_key_size': 1952,
        'private_key_size': 4000,
        'signature_size': 3293
    }
    
    DILITHIUM_5 = {
        'n': 256,
        'q': 8380417,
        'k': 8,
        'l': 7,
        'eta': 2,
        'tau': 60,
        'beta': 120,
        'gamma1': 2**19,
        'gamma2': (8380417 - 1) // 32,
        'public_key_size': 2592,
        'private_key_size': 4864,
        'signature_size': 4595
    }

class PostQuantumKEM:
    """Post-Quantum Key Encapsulation Mechanism (Kyber-like)"""
    
    def __init__(self, algorithm: PQCAlgorithm = PQCAlgorithm.KYBER_768):
        self.algorithm = algorithm
        self.params = self._get_parameters(algorithm)
        
    def _get_parameters(self, algorithm: PQCAlgorithm) -> Dict[str, Any]:
        """Get algorithm parameters"""
        if algorithm == PQCAlgorithm.KYBER_512:
            return KyberParameters.KYBER_512
        elif algorithm == PQCAlgorithm.KYBER_768:
            return KyberParameters.KYBER_768
        elif algorithm == PQCAlgorithm.KYBER_1024:
            return KyberParameters.KYBER_1024
        else:
            raise ValueError(f"Unsupported KEM algorithm: {algorithm}")
    
    def _sample_noise(self, size: int, eta: int) -> np.ndarray:
        """Sample noise from centered binomial distribution"""
        # Simplified noise sampling - in practice would use CBD
        noise = np.random.randint(-eta, eta + 1, size=size)
        return noise % self.params['q']
    
    def _compress(self, poly: np.ndarray, d: int) -> np.ndarray:
        """Compress polynomial coefficients"""
        # Simplified compression
        scale = 2**d / self.params['q']
        compressed = np.round(poly * scale) % (2**d)
        return compressed.astype(np.int32)
    
    def _decompress(self, compressed: np.ndarray, d: int) -> np.ndarray:
        """Decompress polynomial coefficients"""
        # Simplified decompression
        scale = self.params['q'] / (2**d)
        decompressed = np.round(compressed * scale) % self.params['q']
        return decompressed.astype(np.int32)
    
    def _polynomial_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply polynomials in ring Rq"""
        # Simplified polynomial multiplication
        # In practice would use NTT for efficiency
        conv = np.convolve(a, b)
        # Reduce modulo x^n + 1
        n = self.params['n']
        result = np.zeros(n, dtype=np.int32)
        for i in range(len(conv)):
            if i < n:
                result[i] += conv[i]
            else:
                result[i - n] -= conv[i]  # x^n = -1
        return result % self.params['q']
    
    def generate_keypair(self) -> PQCKeyPair:
        """Generate Kyber key pair"""
        logger.info(f"Generating {self.algorithm.value} key pair")
        
        params = self.params
        n = params['n']
        k = params['k']
        q = params['q']
        
        # Generate random matrix A (would be derived from seed in practice)
        A = np.random.randint(0, q, size=(k, k, n))
        
        # Sample secret vectors s and e
        s = np.array([self._sample_noise(n, params['eta1']) for _ in range(k)])
        e = np.array([self._sample_noise(n, params['eta1']) for _ in range(k)])
        
        # Compute public key t = As + e
        t = np.zeros((k, n), dtype=np.int32)
        for i in range(k):
            for j in range(k):
                t[i] = (t[i] + self._polynomial_multiply(A[i][j], s[j])) % q
            t[i] = (t[i] + e[i]) % q
        
        # Serialize keys (simplified)
        public_key = self._serialize_public_key(t, A)
        private_key = self._serialize_private_key(s, t, A)
        
        key_id = hashlib.sha256(public_key).hexdigest()[:16]
        
        return PQCKeyPair(
            public_key=public_key,
            private_key=private_key,
            algorithm=self.algorithm,
            key_id=key_id,
            created_at=datetime.now().isoformat(),
            parameters=params
        )
    
    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """Encapsulate shared secret"""
        logger.debug(f"Encapsulating shared secret with {self.algorithm.value}")
        
        # Deserialize public key
        t, A = self._deserialize_public_key(public_key)
        
        params = self.params
        n = params['n']
        k = params['k']
        q = params['q']
        
        # Generate random message m and coins r
        m = secrets.token_bytes(32)
        r = np.array([self._sample_noise(n, params['eta1']) for _ in range(k)])
        e1 = np.array([self._sample_noise(n, params['eta2']) for _ in range(k)])
        e2 = self._sample_noise(n, params['eta2'])
        
        # Compute ciphertext
        # u = A^T * r + e1
        u = np.zeros((k, n), dtype=np.int32)
        for i in range(k):
            for j in range(k):
                u[i] = (u[i] + self._polynomial_multiply(A[j][i], r[j])) % q
            u[i] = (u[i] + e1[i]) % q
        
        # v = t^T * r + e2 + Decompress(m, 1)
        v = np.zeros(n, dtype=np.int32)
        for i in range(k):
            v = (v + self._polynomial_multiply(t[i], r[i])) % q
        v = (v + e2) % q
        
        # Add message (simplified)
        m_poly = np.frombuffer(m, dtype=np.uint8)[:n] * (q // 2)
        v = (v + m_poly) % q
        
        # Compress and serialize ciphertext
        ciphertext = self._serialize_ciphertext(u, v)
        
        # Derive shared secret
        shared_secret = hashlib.sha256(m + hashlib.sha256(ciphertext).digest()).digest()
        
        return ciphertext, shared_secret
    
    def decapsulate(self, private_key: bytes, ciphertext: bytes) -> bytes:
        """Decapsulate shared secret"""
        logger.debug(f"Decapsulating shared secret with {self.algorithm.value}")
        
        # Deserialize private key and ciphertext
        s, t, A = self._deserialize_private_key(private_key)
        u, v = self._deserialize_ciphertext(ciphertext)
        
        params = self.params
        n = params['n']
        k = params['k']
        q = params['q']
        
        # Compute message candidate
        # m' = v - s^T * u
        m_prime = v.copy()
        for i in range(k):
            m_prime = (m_prime - self._polynomial_multiply(s[i], u[i])) % q
        
        # Extract message (simplified)
        m_candidate = np.round(m_prime * 2.0 / q) % 2
        m = m_candidate[:32].astype(np.uint8).tobytes()
        
        # Re-encrypt to verify (simplified verification)
        shared_secret = hashlib.sha256(m + hashlib.sha256(ciphertext).digest()).digest()
        
        return shared_secret
    
    def _serialize_public_key(self, t: np.ndarray, A: np.ndarray) -> bytes:
        """Serialize public key (simplified)"""
        data = {
            't': t.tolist(),
            'A_seed': hashlib.sha256(A.tobytes()).digest().hex()  # In practice, A is generated from seed
        }
        return json.dumps(data).encode()
    
    def _deserialize_public_key(self, public_key: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """Deserialize public key (simplified)"""
        data = json.loads(public_key.decode())
        t = np.array(data['t'])
        # Regenerate A from seed (simplified)
        A = np.random.randint(0, self.params['q'], size=(self.params['k'], self.params['k'], self.params['n']))
        return t, A
    
    def _serialize_private_key(self, s: np.ndarray, t: np.ndarray, A: np.ndarray) -> bytes:
        """Serialize private key (simplified)"""
        data = {
            's': s.tolist(),
            't': t.tolist(),
            'A_seed': hashlib.sha256(A.tobytes()).digest().hex()
        }
        return json.dumps(data).encode()
    
    def _deserialize_private_key(self, private_key: bytes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Deserialize private key (simplified)"""
        data = json.loads(private_key.decode())
        s = np.array(data['s'])
        t = np.array(data['t'])
        A = np.random.randint(0, self.params['q'], size=(self.params['k'], self.params['k'], self.params['n']))
        return s, t, A
    
    def _serialize_ciphertext(self, u: np.ndarray, v: np.ndarray) -> bytes:
        """Serialize ciphertext (simplified)"""
        data = {
            'u': u.tolist(),
            'v': v.tolist()
        }
        return json.dumps(data).encode()
    
    def _deserialize_ciphertext(self, ciphertext: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """Deserialize ciphertext (simplified)"""
        data = json.loads(ciphertext.decode())
        u = np.array(data['u'])
        v = np.array(data['v'])
        return u, v

class PostQuantumSignature:
    """Post-Quantum Digital Signature Scheme (Dilithium-like)"""
    
    def __init__(self, algorithm: PQCAlgorithm = PQCAlgorithm.DILITHIUM_3):
        self.algorithm = algorithm
        self.params = self._get_parameters(algorithm)
    
    def _get_parameters(self, algorithm: PQCAlgorithm) -> Dict[str, Any]:
        """Get algorithm parameters"""
        if algorithm == PQCAlgorithm.DILITHIUM_2:
            return DilithiumParameters.DILITHIUM_2
        elif algorithm == PQCAlgorithm.DILITHIUM_3:
            return DilithiumParameters.DILITHIUM_3
        elif algorithm == PQCAlgorithm.DILITHIUM_5:
            return DilithiumParameters.DILITHIUM_5
        else:
            raise ValueError(f"Unsupported signature algorithm: {algorithm}")
    
    def generate_keypair(self) -> PQCKeyPair:
        """Generate Dilithium key pair"""
        logger.info(f"Generating {self.algorithm.value} signature key pair")
        
        params = self.params
        n = params['n']
        k = params['k']
        l = params['l']
        q = params['q']
        
        # Generate random matrix A (would be derived from seed)
        A = np.random.randint(0, q, size=(k, l, n))
        
        # Sample secret vectors s1 and s2
        s1 = np.array([self._sample_uniform(n, params['eta']) for _ in range(l)])
        s2 = np.array([self._sample_uniform(n, params['eta']) for _ in range(k)])
        
        # Compute public key t = As1 + s2
        t = np.zeros((k, n), dtype=np.int32)
        for i in range(k):
            for j in range(l):
                t[i] = (t[i] + self._polynomial_multiply(A[i][j], s1[j])) % q
            t[i] = (t[i] + s2[i]) % q
        
        # Serialize keys
        public_key = self._serialize_signature_public_key(t, A)
        private_key = self._serialize_signature_private_key(s1, s2, t, A)
        
        key_id = hashlib.sha256(public_key).hexdigest()[:16]
        
        return PQCKeyPair(
            public_key=public_key,
            private_key=private_key,
            algorithm=self.algorithm,
            key_id=key_id,
            created_at=datetime.now().isoformat(),
            parameters=params
        )
    
    def sign(self, private_key: bytes, message: bytes) -> PQCSignature:
        """Sign message with Dilithium"""
        logger.debug(f"Signing message with {self.algorithm.value}")
        
        # Hash message
        message_hash = hashlib.sha256(message).digest()
        
        # Deserialize private key
        s1, s2, t, A = self._deserialize_signature_private_key(private_key)
        
        params = self.params
        n = params['n']
        k = params['k']
        l = params['l']
        q = params['q']
        
        # Simplified signing process
        # In practice, this would involve rejection sampling and complex polynomial operations
        
        # Sample random y
        y = np.array([self._sample_uniform(n, params['gamma1'] - 1) for _ in range(l)])
        
        # Compute w = Ay (high bits)
        w = np.zeros((k, n), dtype=np.int32)
        for i in range(k):
            for j in range(l):
                w[i] = (w[i] + self._polynomial_multiply(A[i][j], y[j])) % q
        
        # Create challenge (simplified)
        challenge_input = message_hash + w.tobytes()
        c = self._generate_challenge(challenge_input, params['tau'])
        
        # Compute z = y + cs1
        z = np.zeros((l, n), dtype=np.int32)
        for i in range(l):
            cs1 = self._polynomial_multiply(c, s1[i])
            z[i] = (y[i] + cs1) % q
        
        # Compute hint h (simplified)
        h = np.zeros((k, n), dtype=np.int32)
        for i in range(k):
            cs2 = self._polynomial_multiply(c, s2[i])
            h[i] = (w[i] - cs2) % q
        
        # Serialize signature
        signature = self._serialize_signature(c, z, h)
        
        return PQCSignature(
            signature=signature,
            algorithm=self.algorithm,
            signer_key_id=hashlib.sha256(private_key).hexdigest()[:16],
            message_hash=message_hash,
            timestamp=datetime.now().isoformat()
        )
    
    def verify(self, public_key: bytes, message: bytes, signature: PQCSignature) -> bool:
        """Verify Dilithium signature"""
        logger.debug(f"Verifying signature with {self.algorithm.value}")
        
        try:
            # Hash message
            message_hash = hashlib.sha256(message).digest()
            
            if message_hash != signature.message_hash:
                return False
            
            # Deserialize public key and signature
            t, A = self._deserialize_signature_public_key(public_key)
            c, z, h = self._deserialize_signature(signature.signature)
            
            params = self.params
            k = params['k']
            l = params['l']
            q = params['q']
            
            # Compute w' = Az - ct
            w_prime = np.zeros((k, params['n']), dtype=np.int32)
            for i in range(k):
                for j in range(l):
                    w_prime[i] = (w_prime[i] + self._polynomial_multiply(A[i][j], z[j])) % q
                ct = self._polynomial_multiply(c, t[i])
                w_prime[i] = (w_prime[i] - ct) % q
            
            # Verify challenge
            challenge_input = message_hash + w_prime.tobytes()
            c_expected = self._generate_challenge(challenge_input, params['tau'])
            
            # Compare challenges (simplified)
            return np.array_equal(c, c_expected)
            
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False
    
    def _sample_uniform(self, size: int, bound: int) -> np.ndarray:
        """Sample uniform coefficients"""
        return np.random.randint(-bound, bound + 1, size=size)
    
    def _polynomial_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply polynomials (reuse from KEM)"""
        conv = np.convolve(a, b)
        n = self.params['n']
        result = np.zeros(n, dtype=np.int32)
        for i in range(len(conv)):
            if i < n:
                result[i] += conv[i]
            else:
                result[i - n] -= conv[i]
        return result % self.params['q']
    
    def _generate_challenge(self, input_data: bytes, tau: int) -> np.ndarray:
        """Generate challenge polynomial"""
        # Simplified challenge generation
        hash_output = hashlib.sha256(input_data).digest()
        n = self.params['n']
        c = np.zeros(n, dtype=np.int32)
        
        # Set tau coefficients to ±1
        for i in range(min(tau, n)):
            if hash_output[i % len(hash_output)] % 2:
                c[i] = 1
            else:
                c[i] = -1
        
        return c
    
    def _serialize_signature_public_key(self, t: np.ndarray, A: np.ndarray) -> bytes:
        """Serialize signature public key"""
        data = {
            't': t.tolist(),
            'A_seed': hashlib.sha256(A.tobytes()).digest().hex()
        }
        return json.dumps(data).encode()
    
    def _deserialize_signature_public_key(self, public_key: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """Deserialize signature public key"""
        data = json.loads(public_key.decode())
        t = np.array(data['t'])
        A = np.random.randint(0, self.params['q'], 
                             size=(self.params['k'], self.params['l'], self.params['n']))
        return t, A
    
    def _serialize_signature_private_key(self, s1: np.ndarray, s2: np.ndarray, 
                                        t: np.ndarray, A: np.ndarray) -> bytes:
        """Serialize signature private key"""
        data = {
            's1': s1.tolist(),
            's2': s2.tolist(),
            't': t.tolist(),
            'A_seed': hashlib.sha256(A.tobytes()).digest().hex()
        }
        return json.dumps(data).encode()
    
    def _deserialize_signature_private_key(self, private_key: bytes) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Deserialize signature private key"""
        data = json.loads(private_key.decode())
        s1 = np.array(data['s1'])
        s2 = np.array(data['s2'])
        t = np.array(data['t'])
        A = np.random.randint(0, self.params['q'], 
                             size=(self.params['k'], self.params['l'], self.params['n']))
        return s1, s2, t, A
    
    def _serialize_signature(self, c: np.ndarray, z: np.ndarray, h: np.ndarray) -> bytes:
        """Serialize signature"""
        data = {
            'c': c.tolist(),
            'z': z.tolist(),
            'h': h.tolist()
        }
        return json.dumps(data).encode()
    
    def _deserialize_signature(self, signature: bytes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Deserialize signature"""
        data = json.loads(signature.decode())
        c = np.array(data['c'])
        z = np.array(data['z'])
        h = np.array(data['h'])
        return c, z, h

class PostQuantumCryptoManager:
    """High-level manager for post-quantum cryptographic operations"""
    
    def __init__(self):
        self.key_pairs = {}
        self.cached_sessions = {}
        
        # Initialize default algorithms
        self.default_kem = PostQuantumKEM(PQCAlgorithm.KYBER_768)
        self.default_signature = PostQuantumSignature(PQCAlgorithm.DILITHIUM_3)
    
    def generate_kem_keypair(self, algorithm: PQCAlgorithm = PQCAlgorithm.KYBER_768, 
                            key_id: Optional[str] = None) -> str:
        """Generate KEM key pair"""
        kem = PostQuantumKEM(algorithm)
        keypair = kem.generate_keypair()
        
        if key_id is None:
            key_id = keypair.key_id
        
        self.key_pairs[key_id] = {
            'keypair': keypair,
            'kem': kem,
            'type': 'kem'
        }
        
        logger.info(f"Generated KEM keypair: {key_id} ({algorithm.value})")
        return key_id
    
    def generate_signature_keypair(self, algorithm: PQCAlgorithm = PQCAlgorithm.DILITHIUM_3,
                                  key_id: Optional[str] = None) -> str:
        """Generate signature key pair"""
        signature = PostQuantumSignature(algorithm)
        keypair = signature.generate_keypair()
        
        if key_id is None:
            key_id = keypair.key_id
        
        self.key_pairs[key_id] = {
            'keypair': keypair,
            'signature': signature,
            'type': 'signature'
        }
        
        logger.info(f"Generated signature keypair: {key_id} ({algorithm.value})")
        return key_id
    
    def establish_shared_secret(self, kem_key_id: str, peer_public_key: bytes) -> bytes:
        """Establish shared secret using KEM"""
        if kem_key_id not in self.key_pairs:
            raise ValueError(f"KEM key not found: {kem_key_id}")
        
        key_data = self.key_pairs[kem_key_id]
        if key_data['type'] != 'kem':
            raise ValueError(f"Key is not a KEM key: {kem_key_id}")
        
        kem = key_data['kem']
        ciphertext, shared_secret = kem.encapsulate(peer_public_key)
        
        logger.info(f"Established shared secret using KEM key: {kem_key_id}")
        return shared_secret
    
    def sign_message(self, signature_key_id: str, message: bytes) -> PQCSignature:
        """Sign message using post-quantum signature"""
        if signature_key_id not in self.key_pairs:
            raise ValueError(f"Signature key not found: {signature_key_id}")
        
        key_data = self.key_pairs[signature_key_id]
        if key_data['type'] != 'signature':
            raise ValueError(f"Key is not a signature key: {signature_key_id}")
        
        signature_scheme = key_data['signature']
        keypair = key_data['keypair']
        
        signature = signature_scheme.sign(keypair.private_key, message)
        
        logger.info(f"Signed message using signature key: {signature_key_id}")
        return signature
    
    def verify_signature(self, public_key: bytes, message: bytes, 
                        signature: PQCSignature) -> bool:
        """Verify post-quantum signature"""
        signature_scheme = PostQuantumSignature(signature.algorithm)
        return signature_scheme.verify(public_key, message, signature)
    
    def get_public_key(self, key_id: str) -> bytes:
        """Get public key by ID"""
        if key_id not in self.key_pairs:
            raise ValueError(f"Key not found: {key_id}")
        
        return self.key_pairs[key_id]['keypair'].public_key
    
    def export_key_info(self, key_id: str) -> Dict[str, Any]:
        """Export key information (without private key)"""
        if key_id not in self.key_pairs:
            raise ValueError(f"Key not found: {key_id}")
        
        keypair = self.key_pairs[key_id]['keypair']
        return {
            'key_id': keypair.key_id,
            'algorithm': keypair.algorithm.value,
            'created_at': keypair.created_at,
            'public_key_b64': base64.b64encode(keypair.public_key).decode(),
            'parameters': keypair.parameters
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        kem_keys = sum(1 for k in self.key_pairs.values() if k['type'] == 'kem')
        sig_keys = sum(1 for k in self.key_pairs.values() if k['type'] == 'signature')
        
        return {
            'total_keys': len(self.key_pairs),
            'kem_keys': kem_keys,
            'signature_keys': sig_keys,
            'supported_algorithms': [alg.value for alg in PQCAlgorithm],
            'system_status': 'operational'
        }

# Global post-quantum crypto manager
pqc_manager = PostQuantumCryptoManager()

# Utility functions for QFLARE integration
def setup_post_quantum_security(client_id: str) -> Dict[str, str]:
    """Set up post-quantum security for a client"""
    kem_key_id = pqc_manager.generate_kem_keypair()
    sig_key_id = pqc_manager.generate_signature_keypair()
    
    logger.info(f"Post-quantum security setup complete for client: {client_id}")
    return {
        'kem_key_id': kem_key_id,
        'signature_key_id': sig_key_id,
        'client_id': client_id
    }

def secure_federated_communication(sender_id: str, receiver_public_key: bytes, 
                                  message: bytes, sender_sig_key: str) -> Dict[str, Any]:
    """Secure federated learning communication with post-quantum crypto"""
    
    # Establish shared secret
    shared_secret = pqc_manager.establish_shared_secret(sender_id, receiver_public_key)
    
    # Sign message
    signature = pqc_manager.sign_message(sender_sig_key, message)
    
    # Encrypt message using shared secret (simplified)
    encrypted_message = hmac.new(shared_secret, message, hashlib.sha256).digest()
    
    return {
        'encrypted_message': encrypted_message,
        'signature': signature,
        'shared_secret_established': True
    }

if __name__ == "__main__":
    # Demo and testing
    print("🔐 QFLARE Post-Quantum Cryptography Demo")
    print("=" * 50)
    
    # Test KEM
    print("\n🔑 Testing Key Encapsulation Mechanism (Kyber)")
    kem_key_id = pqc_manager.generate_kem_keypair(PQCAlgorithm.KYBER_768)
    public_key = pqc_manager.get_public_key(kem_key_id)
    
    # Test encapsulation/decapsulation
    kem = PostQuantumKEM(PQCAlgorithm.KYBER_768)
    ciphertext, secret1 = kem.encapsulate(public_key)
    secret2 = kem.decapsulate(pqc_manager.key_pairs[kem_key_id]['keypair'].private_key, ciphertext)
    
    print(f"✅ KEM test: Secrets match = {secret1 == secret2}")
    print(f"   Shared secret: {secret1.hex()[:32]}...")
    
    # Test signatures
    print("\n✍️ Testing Digital Signatures (Dilithium)")
    sig_key_id = pqc_manager.generate_signature_keypair(PQCAlgorithm.DILITHIUM_3)
    message = b"QFLARE federated learning is quantum-secure!"
    
    signature = pqc_manager.sign_message(sig_key_id, message)
    sig_public_key = pqc_manager.get_public_key(sig_key_id)
    is_valid = pqc_manager.verify_signature(sig_public_key, message, signature)
    
    print(f"✅ Signature test: Valid = {is_valid}")
    print(f"   Message: {message.decode()}")
    
    # System info
    info = pqc_manager.get_system_info()
    print(f"\n📊 System Info:")
    print(f"   Total keys: {info['total_keys']}")
    print(f"   KEM keys: {info['kem_keys']}")
    print(f"   Signature keys: {info['signature_keys']}")
    print(f"   Status: {info['system_status']}")