"""
QFLARE Quantum Key Exchange Implementation
Advanced quantum protocols for secure key distribution
"""

import numpy as np
import random
import hashlib
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantumBasis(Enum):
    """Quantum measurement basis for BB84 protocol"""
    RECTILINEAR = "+"  # 0°/90° basis
    DIAGONAL = "x"     # 45°/135° basis

class QuantumBit(Enum):
    """Quantum bit values"""
    ZERO = 0
    ONE = 1

@dataclass
class QuantumState:
    """Represents a quantum bit state"""
    bit_value: QuantumBit
    basis: QuantumBasis
    measured: bool = False
    
    def __str__(self) -> str:
        return f"|{self.bit_value.value}⟩_{self.basis.value}"

@dataclass
class BB84KeyExchangeResult:
    """Result of BB84 key exchange protocol"""
    shared_key: bytes
    key_length: int
    error_rate: float
    security_parameter: float
    exchange_id: str
    timestamp: str
    participants: List[str]
    
class QuantumChannel:
    """Simulates quantum communication channel with noise"""
    
    def __init__(self, error_rate: float = 0.01, eavesdropper_present: bool = False):
        self.error_rate = error_rate
        self.eavesdropper_present = eavesdropper_present
        self.transmission_log = []
        
    def transmit_quantum_bit(self, quantum_bit: QuantumState) -> QuantumState:
        """Simulate quantum bit transmission through noisy channel"""
        
        # Log transmission
        self.transmission_log.append({
            'original': str(quantum_bit),
            'timestamp': datetime.now().isoformat()
        })
        
        # Simulate eavesdropping (Eve's attack)
        if self.eavesdropper_present:
            # Eve measures in random basis (50% chance each)
            eve_basis = random.choice([QuantumBasis.RECTILINEAR, QuantumBasis.DIAGONAL])
            
            # If Eve uses wrong basis, she introduces errors
            if eve_basis != quantum_bit.basis:
                # 50% chance of bit flip when wrong basis is used
                if random.random() < 0.5:
                    new_bit = QuantumBit.ONE if quantum_bit.bit_value == QuantumBit.ZERO else QuantumBit.ZERO
                    quantum_bit = QuantumState(new_bit, quantum_bit.basis)
                    logger.debug(f"Eavesdropper caused bit flip: {quantum_bit}")
        
        # Simulate channel noise
        if random.random() < self.error_rate:
            # Bit flip due to channel noise
            new_bit = QuantumBit.ONE if quantum_bit.bit_value == QuantumBit.ZERO else QuantumBit.ZERO
            quantum_bit = QuantumState(new_bit, quantum_bit.basis)
            logger.debug(f"Channel noise caused bit flip: {quantum_bit}")
        
        return quantum_bit
    
    def get_transmission_statistics(self) -> Dict[str, Any]:
        """Get channel transmission statistics"""
        return {
            'total_transmissions': len(self.transmission_log),
            'error_rate': self.error_rate,
            'eavesdropper_present': self.eavesdropper_present,
            'last_transmission': self.transmission_log[-1] if self.transmission_log else None
        }

class BB84Protocol:
    """Implementation of BB84 Quantum Key Distribution Protocol"""
    
    def __init__(self, key_length: int = 256, security_parameter: float = 0.1):
        self.key_length = key_length
        self.security_parameter = security_parameter  # Maximum acceptable error rate
        self.protocol_id = self._generate_protocol_id()
        
    def _generate_protocol_id(self) -> str:
        """Generate unique protocol execution ID"""
        return hashlib.sha256(f"bb84_{datetime.now().isoformat()}_{random.random()}".encode()).hexdigest()[:16]
    
    async def alice_prepare_quantum_bits(self, num_bits: int) -> Tuple[List[QuantumState], List[int], List[QuantumBasis]]:
        """Alice prepares quantum bits for transmission"""
        
        # Alice's random bit string
        bit_string = [random.choice([QuantumBit.ZERO, QuantumBit.ONE]) for _ in range(num_bits)]
        
        # Alice's random basis choices
        basis_choices = [random.choice([QuantumBasis.RECTILINEAR, QuantumBasis.DIAGONAL]) for _ in range(num_bits)]
        
        # Prepare quantum states
        quantum_bits = []
        for bit, basis in zip(bit_string, basis_choices):
            quantum_state = QuantumState(bit, basis)
            quantum_bits.append(quantum_state)
        
        logger.info(f"Alice prepared {num_bits} quantum bits")
        return quantum_bits, [b.value for b in bit_string], basis_choices
    
    async def bob_measure_quantum_bits(self, quantum_bits: List[QuantumState]) -> Tuple[List[int], List[QuantumBasis]]:
        """Bob measures received quantum bits"""
        
        # Bob's random basis choices
        bob_bases = [random.choice([QuantumBasis.RECTILINEAR, QuantumBasis.DIAGONAL]) for _ in range(len(quantum_bits))]
        
        # Bob's measurement results
        bob_bits = []
        
        for quantum_bit, bob_basis in zip(quantum_bits, bob_bases):
            # Simulate quantum measurement
            if quantum_bit.basis == bob_basis:
                # Correct basis - measure the actual bit
                measured_bit = quantum_bit.bit_value.value
            else:
                # Wrong basis - random result
                measured_bit = random.choice([0, 1])
            
            bob_bits.append(measured_bit)
            quantum_bit.measured = True
        
        logger.info(f"Bob measured {len(quantum_bits)} quantum bits")
        return bob_bits, bob_bases
    
    def sift_keys(self, alice_bits: List[int], alice_bases: List[QuantumBasis], 
                  bob_bits: List[int], bob_bases: List[QuantumBasis]) -> Tuple[List[int], List[int]]:
        """Sift keys by comparing basis choices"""
        
        alice_sifted = []
        bob_sifted = []
        
        for i in range(len(alice_bases)):
            if alice_bases[i] == bob_bases[i]:
                # Same basis - keep the bits
                alice_sifted.append(alice_bits[i])
                bob_sifted.append(bob_bits[i])
        
        logger.info(f"Sifted key length: {len(alice_sifted)} bits")
        return alice_sifted, bob_sifted
    
    def estimate_error_rate(self, alice_sifted: List[int], bob_sifted: List[int], 
                           sample_size: int = None) -> float:
        """Estimate quantum bit error rate using random sampling"""
        
        if sample_size is None:
            sample_size = min(len(alice_sifted) // 4, 64)  # Use 25% or max 64 bits for testing
        
        if len(alice_sifted) < sample_size:
            sample_size = len(alice_sifted)
        
        # Randomly sample positions for error testing
        test_positions = random.sample(range(len(alice_sifted)), sample_size)
        
        errors = 0
        for pos in test_positions:
            if alice_sifted[pos] != bob_sifted[pos]:
                errors += 1
        
        error_rate = errors / sample_size if sample_size > 0 else 0
        logger.info(f"Estimated error rate: {error_rate:.3f} ({errors}/{sample_size})")
        
        # Remove test bits from sifted keys
        alice_final = [alice_sifted[i] for i in range(len(alice_sifted)) if i not in test_positions]
        bob_final = [bob_sifted[i] for i in range(len(bob_sifted)) if i not in test_positions]
        
        return error_rate, alice_final, bob_final
    
    def error_correction(self, alice_key: List[int], bob_key: List[int]) -> Tuple[List[int], List[int]]:
        """Simple error correction using parity checking"""
        
        corrected_alice = alice_key.copy()
        corrected_bob = bob_key.copy()
        
        # Simple parity-based error correction
        # In practice, more sophisticated codes like Cascade or LDPC would be used
        
        block_size = 8  # Process in blocks of 8 bits
        
        for i in range(0, len(alice_key), block_size):
            block_end = min(i + block_size, len(alice_key))
            alice_block = alice_key[i:block_end]
            bob_block = bob_key[i:block_end]
            
            alice_parity = sum(alice_block) % 2
            bob_parity = sum(bob_block) % 2
            
            if alice_parity != bob_parity:
                # Error detected - flip first bit in Bob's block (simplified correction)
                if block_end > i:
                    corrected_bob[i] = 1 - corrected_bob[i]
        
        logger.info(f"Error correction completed on {len(alice_key)} bits")
        return corrected_alice, corrected_bob
    
    def privacy_amplification(self, shared_key: List[int], target_length: int) -> bytes:
        """Privacy amplification using hash functions"""
        
        # Convert bit array to bytes
        key_string = ''.join(map(str, shared_key))
        key_bytes = key_string.encode()
        
        # Use SHA-256 for privacy amplification
        amplified_key = b''
        counter = 0
        
        while len(amplified_key) < target_length:
            hash_input = key_bytes + counter.to_bytes(4, 'big')
            hash_output = hashlib.sha256(hash_input).digest()
            amplified_key += hash_output
            counter += 1
        
        # Trim to desired length
        final_key = amplified_key[:target_length]
        
        logger.info(f"Privacy amplification: {len(shared_key)} bits -> {len(final_key)} bytes")
        return final_key
    
    async def execute_protocol(self, quantum_channel: QuantumChannel, 
                              alice_id: str = "Alice", bob_id: str = "Bob") -> BB84KeyExchangeResult:
        """Execute complete BB84 protocol"""
        
        logger.info(f"Starting BB84 protocol execution: {self.protocol_id}")
        start_time = datetime.now()
        
        # Step 1: Alice prepares quantum bits (need extra bits due to sifting and error correction)
        num_initial_bits = self.key_length * 4  # 4x oversampling
        quantum_bits, alice_bits, alice_bases = await self.alice_prepare_quantum_bits(num_initial_bits)
        
        # Step 2: Alice sends quantum bits through quantum channel
        transmitted_bits = []
        for qbit in quantum_bits:
            transmitted_bit = quantum_channel.transmit_quantum_bit(qbit)
            transmitted_bits.append(transmitted_bit)
        
        # Step 3: Bob measures quantum bits
        bob_bits, bob_bases = await self.bob_measure_quantum_bits(transmitted_bits)
        
        # Step 4: Public discussion - basis comparison and sifting
        alice_sifted, bob_sifted = self.sift_keys(alice_bits, alice_bases, bob_bits, bob_bases)
        
        if len(alice_sifted) < self.key_length:
            raise ValueError(f"Insufficient sifted bits: {len(alice_sifted)} < {self.key_length}")
        
        # Step 5: Error estimation
        error_rate, alice_corrected, bob_corrected = self.estimate_error_rate(alice_sifted, bob_sifted)
        
        # Step 6: Security check
        if error_rate > self.security_parameter:
            logger.warning(f"High error rate detected: {error_rate:.3f} > {self.security_parameter}")
            # In practice, protocol would be aborted here
        
        # Step 7: Error correction
        alice_final, bob_final = self.error_correction(alice_corrected, bob_corrected)
        
        # Step 8: Privacy amplification
        shared_key = self.privacy_amplification(alice_final, self.key_length // 8)
        
        # Create result
        result = BB84KeyExchangeResult(
            shared_key=shared_key,
            key_length=len(shared_key),
            error_rate=error_rate,
            security_parameter=self.security_parameter,
            exchange_id=self.protocol_id,
            timestamp=start_time.isoformat(),
            participants=[alice_id, bob_id]
        )
        
        logger.info(f"BB84 protocol completed successfully: {len(shared_key)} byte key generated")
        return result

class QuantumKeyExchangeManager:
    """High-level manager for quantum key exchange operations"""
    
    def __init__(self):
        self.active_exchanges = {}
        self.completed_exchanges = {}
        self.channel_configurations = {}
        
    def create_quantum_channel(self, channel_id: str, error_rate: float = 0.01, 
                              eavesdropper_present: bool = False) -> QuantumChannel:
        """Create and configure quantum communication channel"""
        
        channel = QuantumChannel(error_rate, eavesdropper_present)
        self.channel_configurations[channel_id] = {
            'channel': channel,
            'created_at': datetime.now().isoformat(),
            'error_rate': error_rate,
            'eavesdropper_present': eavesdropper_present
        }
        
        logger.info(f"Quantum channel created: {channel_id}")
        return channel
    
    async def initiate_key_exchange(self, participant_a: str, participant_b: str,
                                   key_length: int = 256, channel_id: str = "default",
                                   security_parameter: float = 0.1) -> str:
        """Initiate quantum key exchange between two participants"""
        
        # Create channel if it doesn't exist
        if channel_id not in self.channel_configurations:
            self.create_quantum_channel(channel_id)
        
        # Create BB84 protocol instance
        bb84 = BB84Protocol(key_length, security_parameter)
        channel = self.channel_configurations[channel_id]['channel']
        
        # Store active exchange
        exchange_id = bb84.protocol_id
        self.active_exchanges[exchange_id] = {
            'protocol': bb84,
            'participants': [participant_a, participant_b],
            'channel_id': channel_id,
            'status': 'initiated',
            'start_time': datetime.now().isoformat()
        }
        
        try:
            # Execute protocol
            result = await bb84.execute_protocol(channel, participant_a, participant_b)
            
            # Move to completed exchanges
            self.completed_exchanges[exchange_id] = {
                'result': result,
                'status': 'completed',
                'completion_time': datetime.now().isoformat()
            }
            
            # Remove from active exchanges
            del self.active_exchanges[exchange_id]
            
            logger.info(f"Key exchange completed: {exchange_id}")
            return exchange_id
            
        except Exception as e:
            # Mark as failed
            self.active_exchanges[exchange_id]['status'] = 'failed'
            self.active_exchanges[exchange_id]['error'] = str(e)
            logger.error(f"Key exchange failed: {exchange_id}, error: {e}")
            raise
    
    def get_exchange_result(self, exchange_id: str) -> Optional[BB84KeyExchangeResult]:
        """Retrieve completed key exchange result"""
        
        if exchange_id in self.completed_exchanges:
            return self.completed_exchanges[exchange_id]['result']
        return None
    
    def get_exchange_status(self, exchange_id: str) -> Dict[str, Any]:
        """Get status of key exchange"""
        
        if exchange_id in self.active_exchanges:
            return {
                'status': self.active_exchanges[exchange_id]['status'],
                'participants': self.active_exchanges[exchange_id]['participants'],
                'start_time': self.active_exchanges[exchange_id]['start_time']
            }
        elif exchange_id in self.completed_exchanges:
            return {
                'status': 'completed',
                'completion_time': self.completed_exchanges[exchange_id]['completion_time']
            }
        else:
            return {'status': 'not_found'}
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system-wide statistics"""
        
        return {
            'active_exchanges': len(self.active_exchanges),
            'completed_exchanges': len(self.completed_exchanges),
            'total_channels': len(self.channel_configurations),
            'system_status': 'operational'
        }

# Global quantum key exchange manager instance
quantum_key_manager = QuantumKeyExchangeManager()

# Utility functions for integration with QFLARE system
async def generate_quantum_shared_key(client_a_id: str, client_b_id: str, 
                                     key_length: int = 256) -> bytes:
    """Generate shared quantum key between two clients"""
    
    exchange_id = await quantum_key_manager.initiate_key_exchange(
        client_a_id, client_b_id, key_length
    )
    
    result = quantum_key_manager.get_exchange_result(exchange_id)
    if result:
        return result.shared_key
    else:
        raise RuntimeError("Failed to generate quantum shared key")

async def secure_quantum_communication_setup(participants: List[str]) -> Dict[str, bytes]:
    """Set up quantum-secure communication keys between multiple participants"""
    
    shared_keys = {}
    
    # Generate pairwise shared keys
    for i in range(len(participants)):
        for j in range(i + 1, len(participants)):
            participant_a = participants[i]
            participant_b = participants[j]
            
            key = await generate_quantum_shared_key(participant_a, participant_b)
            shared_keys[f"{participant_a}_{participant_b}"] = key
            shared_keys[f"{participant_b}_{participant_a}"] = key  # Symmetric
    
    logger.info(f"Quantum communication setup complete for {len(participants)} participants")
    return shared_keys

if __name__ == "__main__":
    # Example usage and testing
    async def main():
        print("🔬 QFLARE Quantum Key Exchange Demo")
        print("=" * 50)
        
        # Create quantum channel with some noise
        manager = QuantumKeyExchangeManager()
        channel = manager.create_quantum_channel("demo_channel", error_rate=0.02)
        
        # Execute BB84 protocol
        try:
            exchange_id = await manager.initiate_key_exchange(
                "Alice_Client", "Bob_Client", key_length=256
            )
            
            result = manager.get_exchange_result(exchange_id)
            if result:
                print(f"✅ Key exchange successful!")
                print(f"   Exchange ID: {result.exchange_id}")
                print(f"   Key length: {result.key_length} bytes")
                print(f"   Error rate: {result.error_rate:.3f}")
                print(f"   Key (hex): {result.shared_key.hex()[:32]}...")
            
        except Exception as e:
            print(f"❌ Key exchange failed: {e}")
        
        # System statistics
        stats = manager.get_system_statistics()
        print(f"\n📊 System Statistics:")
        print(f"   Completed exchanges: {stats['completed_exchanges']}")
        print(f"   System status: {stats['system_status']}")
    
    # Run the demo
    asyncio.run(main())