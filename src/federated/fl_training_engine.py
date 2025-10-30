"""
QFLARE Federated Learning Training Engine
Implements secure, privacy-preserving federated learning with PQC and Byzantine resilience
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import logging
import time
import json
import math
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import copy
from collections import defaultdict, OrderedDict
import pickle
import hashlib
import asyncio

# Import QFLARE components
try:
    from ..crypto.post_quantum_crypto import QFLARECrypto, KyberKeyPair, DilithiumKeyPair
    from ..privacy.differential_privacy_engine import DifferentialPrivacyEngine, PrivacyParameters
    from ..security.byzantine_resilience import ByzantineResilienceSystem, AggregationMethod
except ImportError:
    # Handle relative imports when running as main module
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from crypto.post_quantum_crypto import QFLARECrypto, KyberKeyPair, DilithiumKeyPair
    from privacy.differential_privacy_engine import DifferentialPrivacyEngine, PrivacyParameters
    from security.byzantine_resilience import ByzantineResilienceSystem, AggregationMethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederationStrategy(Enum):
    """Federated learning strategies"""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"
    FEDOPT = "fedopt"
    FEDBN = "fedbn"

class ClientSelectionStrategy(Enum):
    """Client selection strategies"""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    REPUTATION_BASED = "reputation_based"
    GRADIENT_DIVERSITY = "gradient_diversity"

@dataclass
class TrainingConfig:
    """Configuration for federated learning training"""
    # Basic training parameters
    local_epochs: int = 5
    local_batch_size: int = 32
    local_learning_rate: float = 0.01
    
    # Federated parameters
    num_rounds: int = 100
    clients_per_round: int = 10
    min_clients_per_round: int = 5
    
    # Strategy parameters
    federation_strategy: FederationStrategy = FederationStrategy.FEDAVG
    client_selection: ClientSelectionStrategy = ClientSelectionStrategy.RANDOM
    
    # Security parameters
    enable_differential_privacy: bool = True
    enable_byzantine_resilience: bool = True
    enable_post_quantum_crypto: bool = True
    
    # Privacy parameters
    dp_epsilon: float = 0.1
    dp_delta: float = 1e-6
    gradient_clipping_norm: float = 1.0
    
    # Byzantine resilience parameters
    aggregation_method: AggregationMethod = AggregationMethod.KRUM
    expected_byzantine_ratio: float = 0.33
    
    # Communication parameters
    compression_enabled: bool = False
    secure_aggregation: bool = True
    
    def to_dict(self) -> Dict:
        """Export configuration to dictionary"""
        return {
            'local_epochs': self.local_epochs,
            'local_batch_size': self.local_batch_size,
            'local_learning_rate': self.local_learning_rate,
            'num_rounds': self.num_rounds,
            'clients_per_round': self.clients_per_round,
            'min_clients_per_round': self.min_clients_per_round,
            'federation_strategy': self.federation_strategy.value,
            'client_selection': self.client_selection.value,
            'enable_differential_privacy': self.enable_differential_privacy,
            'enable_byzantine_resilience': self.enable_byzantine_resilience,
            'enable_post_quantum_crypto': self.enable_post_quantum_crypto,
            'dp_epsilon': self.dp_epsilon,
            'dp_delta': self.dp_delta,
            'gradient_clipping_norm': self.gradient_clipping_norm,
            'aggregation_method': self.aggregation_method.value,
            'expected_byzantine_ratio': self.expected_byzantine_ratio,
            'compression_enabled': self.compression_enabled,
            'secure_aggregation': self.secure_aggregation
        }

@dataclass
class ClientProfile:
    """Profile information for a federated learning client"""
    client_id: str
    data_size: int
    model_accuracy: float = 0.0
    computation_capacity: float = 1.0
    communication_bandwidth: float = 1.0
    availability: float = 1.0
    last_participation_round: int = -1
    total_rounds_participated: int = 0
    
    # Cryptographic keys
    kyber_keypair: Optional[KyberKeyPair] = None
    dilithium_keypair: Optional[DilithiumKeyPair] = None
    
    # Performance metrics
    training_time_history: List[float] = field(default_factory=list)
    accuracy_history: List[float] = field(default_factory=list)
    
    def update_performance(self, accuracy: float, training_time: float):
        """Update client performance metrics"""
        self.model_accuracy = accuracy
        self.accuracy_history.append(accuracy)
        self.training_time_history.append(training_time)
        
        # Keep only recent history
        if len(self.accuracy_history) > 10:
            self.accuracy_history = self.accuracy_history[-10:]
        if len(self.training_time_history) > 10:
            self.training_time_history = self.training_time_history[-10:]
    
    def get_average_accuracy(self) -> float:
        """Get average accuracy over recent rounds"""
        return np.mean(self.accuracy_history) if self.accuracy_history else 0.0
    
    def get_average_training_time(self) -> float:
        """Get average training time over recent rounds"""
        return np.mean(self.training_time_history) if self.training_time_history else 0.0

class SecureModelUpdate:
    """Encrypted and signed model update"""
    
    def __init__(self, client_id: str, round_num: int, 
                 encrypted_gradients: Dict[str, bytes],
                 signature: bytes,
                 metadata: Dict):
        self.client_id = client_id
        self.round_num = round_num
        self.encrypted_gradients = encrypted_gradients
        self.signature = signature
        self.metadata = metadata
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict:
        """Export update to dictionary"""
        return {
            'client_id': self.client_id,
            'round_num': self.round_num,
            'encrypted_gradients': {k: v.hex() for k, v in self.encrypted_gradients.items()},
            'signature': self.signature.hex(),
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }

class FederatedClient:
    """Federated learning client with security features"""
    
    def __init__(self, client_id: str, model: nn.Module, train_data: Dataset, 
                 config: TrainingConfig):
        """
        Initialize federated client
        
        Args:
            client_id: Unique client identifier
            model: Neural network model
            train_data: Client's training dataset
            config: Training configuration
        """
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.train_data = train_data
        self.config = config
        
        # Create data loader
        self.train_loader = DataLoader(
            train_data, 
            batch_size=config.local_batch_size, 
            shuffle=True
        )
        
        # Initialize optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=config.local_learning_rate
        )
        
        # Security components
        if config.enable_post_quantum_crypto:
            self.crypto = QFLARECrypto()
            self.kyber_keypair = self.crypto.generate_kyber_keypair(1024)
            self.dilithium_keypair = self.crypto.generate_dilithium_keypair(2)
        
        if config.enable_differential_privacy:
            privacy_params = PrivacyParameters(
                epsilon=config.dp_epsilon,
                delta=config.dp_delta,
                sensitivity=config.gradient_clipping_norm
            )
            self.dp_engine = DifferentialPrivacyEngine(privacy_params)
        
        logger.info(f"Federated client {client_id} initialized")
    
    def local_train(self, global_model_state: OrderedDict, 
                   num_epochs: Optional[int] = None) -> Tuple[OrderedDict, Dict]:
        """
        Perform local training on client data
        
        Args:
            global_model_state: Global model parameters
            num_epochs: Number of local epochs (optional)
            
        Returns:
            Tuple of (model_update, training_metrics)
        """
        start_time = time.time()
        
        if num_epochs is None:
            num_epochs = self.config.local_epochs
        
        # Update model with global parameters
        self.model.load_state_dict(global_model_state)
        
        # Store initial model for gradient calculation
        initial_params = copy.deepcopy(global_model_state)
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                loss = nn.functional.cross_entropy(output, target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for differential privacy
                if self.config.enable_differential_privacy:
                    gradients = [param.grad for param in self.model.parameters() if param.grad is not None]
                    clipped_gradients = self.dp_engine.clip_gradients(gradients, self.config.gradient_clipping_norm)
                    
                    # Update parameter gradients with clipped values
                    for param, clipped_grad in zip(self.model.parameters(), clipped_gradients):
                        if param.grad is not None:
                            param.grad.data = clipped_grad.data
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                total_loss += loss.item()
                num_batches += 1
        
        # Calculate gradients (model update)
        model_update = OrderedDict()
        for name, param in self.model.named_parameters():
            if name in initial_params:
                gradient = initial_params[name] - param.data
                model_update[name] = gradient
        
        # Add differential privacy noise to gradients
        if self.config.enable_differential_privacy:
            for name, grad in model_update.items():
                model_update[name] = self.dp_engine.add_gaussian_noise(grad)
        
        training_time = time.time() - start_time
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Calculate training accuracy
        accuracy = self._evaluate_model()
        
        metrics = {
            'training_time': training_time,
            'average_loss': avg_loss,
            'accuracy': accuracy,
            'num_samples': len(self.train_data),
            'local_epochs': num_epochs,
            'differential_privacy_used': self.config.enable_differential_privacy
        }
        
        logger.debug(f"Client {self.client_id} local training: "
                    f"loss={avg_loss:.4f}, accuracy={accuracy:.3f}, time={training_time:.2f}s")
        
        return model_update, metrics
    
    def _evaluate_model(self) -> float:
        """Evaluate model accuracy on local data"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.train_loader:
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def create_secure_update(self, model_update: OrderedDict, round_num: int,
                           server_public_key: bytes) -> SecureModelUpdate:
        """
        Create encrypted and signed model update
        
        Args:
            model_update: Model gradients
            round_num: Current round number
            server_public_key: Server's public key for encryption
            
        Returns:
            Secure model update
        """
        if not self.config.enable_post_quantum_crypto:
            raise ValueError("Post-quantum crypto not enabled")
        
        # Serialize model update
        serialized_update = pickle.dumps(model_update)
        
        # Encrypt with hybrid encryption
        encrypted_data = self.crypto.hybrid_encrypt(serialized_update, server_public_key, 1024)
        
        # Create metadata
        metadata = {
            'client_id': self.client_id,
            'round_num': round_num,
            'data_size': len(self.train_data),
            'model_params': len(model_update),
            'encryption_algorithm': 'Kyber-1024 + AES-256-GCM',
            'signature_algorithm': 'Dilithium-2'
        }
        
        # Sign the encrypted data and metadata
        message_to_sign = json.dumps(metadata, sort_keys=True).encode() + encrypted_data['kyber_ciphertext']
        signature = self.crypto.dilithium_sign(message_to_sign, self.dilithium_keypair.private_key)
        
        return SecureModelUpdate(
            client_id=self.client_id,
            round_num=round_num,
            encrypted_gradients=encrypted_data,
            signature=signature,
            metadata=metadata
        )

class FederatedServer:
    """Federated learning server with security and privacy features"""
    
    def __init__(self, global_model: nn.Module, config: TrainingConfig):
        """
        Initialize federated server
        
        Args:
            global_model: Global neural network model
            config: Training configuration
        """
        self.global_model = global_model
        self.config = config
        self.current_round = 0
        self.client_profiles: Dict[str, ClientProfile] = {}
        self.training_history: List[Dict] = []
        
        # Security components
        if config.enable_post_quantum_crypto:
            self.crypto = QFLARECrypto()
            self.kyber_keypair = self.crypto.generate_kyber_keypair(1024)
            self.dilithium_keypair = self.crypto.generate_dilithium_keypair(2)
        
        if config.enable_differential_privacy:
            privacy_params = PrivacyParameters(
                epsilon=config.dp_epsilon,
                delta=config.dp_delta,
                sensitivity=config.gradient_clipping_norm
            )
            self.dp_engine = DifferentialPrivacyEngine(privacy_params)
        
        if config.enable_byzantine_resilience:
            self.byzantine_system = ByzantineResilienceSystem(
                aggregation_method=config.aggregation_method,
                byzantine_ratio=config.expected_byzantine_ratio
            )
        
        logger.info(f"Federated server initialized with {config.federation_strategy.value} strategy")
    
    def register_client(self, client: FederatedClient) -> ClientProfile:
        """
        Register a new client with the federation
        
        Args:
            client: Federated client to register
            
        Returns:
            Client profile
        """
        profile = ClientProfile(
            client_id=client.client_id,
            data_size=len(client.train_data)
        )
        
        # Store cryptographic keys if available
        if hasattr(client, 'kyber_keypair'):
            profile.kyber_keypair = client.kyber_keypair
        if hasattr(client, 'dilithium_keypair'):
            profile.dilithium_keypair = client.dilithium_keypair
        
        self.client_profiles[client.client_id] = profile
        
        logger.info(f"Client {client.client_id} registered with {profile.data_size} samples")
        
        return profile
    
    def select_clients(self, available_clients: List[str]) -> List[str]:
        """
        Select clients for the current round
        
        Args:
            available_clients: List of available client IDs
            
        Returns:
            List of selected client IDs
        """
        num_to_select = min(self.config.clients_per_round, len(available_clients))
        
        if self.config.client_selection == ClientSelectionStrategy.RANDOM:
            selected = np.random.choice(available_clients, size=num_to_select, replace=False)
        
        elif self.config.client_selection == ClientSelectionStrategy.ROUND_ROBIN:
            # Sort by last participation round
            clients_sorted = sorted(available_clients, 
                                  key=lambda cid: self.client_profiles[cid].last_participation_round)
            selected = clients_sorted[:num_to_select]
        
        elif self.config.client_selection == ClientSelectionStrategy.REPUTATION_BASED:
            if self.config.enable_byzantine_resilience:
                # Select based on reputation scores
                client_scores = []
                for cid in available_clients:
                    if cid in self.byzantine_system.client_reputations:
                        score = self.byzantine_system.client_reputations[cid].reputation_score
                    else:
                        score = 1.0
                    client_scores.append((cid, score))
                
                # Select clients with highest reputation
                client_scores.sort(key=lambda x: x[1], reverse=True)
                selected = [cid for cid, _ in client_scores[:num_to_select]]
            else:
                # Fall back to random selection
                selected = np.random.choice(available_clients, size=num_to_select, replace=False)
        
        else:  # GRADIENT_DIVERSITY - simplified version
            selected = np.random.choice(available_clients, size=num_to_select, replace=False)
        
        # Update participation tracking
        for client_id in selected:
            self.client_profiles[client_id].last_participation_round = self.current_round
            self.client_profiles[client_id].total_rounds_participated += 1
        
        logger.info(f"Round {self.current_round}: Selected {len(selected)} clients using {self.config.client_selection.value}")
        
        return list(selected)
    
    def decrypt_and_verify_updates(self, secure_updates: List[SecureModelUpdate]) -> List[Tuple[str, OrderedDict]]:
        """
        Decrypt and verify client model updates
        
        Args:
            secure_updates: List of encrypted model updates
            
        Returns:
            List of (client_id, model_update) tuples
        """
        if not self.config.enable_post_quantum_crypto:
            raise ValueError("Post-quantum crypto not enabled")
        
        verified_updates = []
        
        for update in secure_updates:
            try:
                # Verify signature
                client_profile = self.client_profiles[update.client_id]
                message_to_verify = (json.dumps(update.metadata, sort_keys=True).encode() + 
                                   update.encrypted_gradients['kyber_ciphertext'])
                
                is_valid = self.crypto.dilithium_verify(
                    message_to_verify,
                    update.signature,
                    client_profile.dilithium_keypair.public_key
                )
                
                if not is_valid:
                    logger.warning(f"Invalid signature from client {update.client_id}")
                    continue
                
                # Decrypt model update
                decrypted_data = self.crypto.hybrid_decrypt(
                    update.encrypted_gradients,
                    self.kyber_keypair.private_key
                )
                
                model_update = pickle.loads(decrypted_data)
                verified_updates.append((update.client_id, model_update))
                
                logger.debug(f"Successfully decrypted and verified update from {update.client_id}")
                
            except Exception as e:
                logger.error(f"Failed to decrypt/verify update from {update.client_id}: {str(e)}")
                continue
        
        logger.info(f"Verified {len(verified_updates)}/{len(secure_updates)} model updates")
        
        return verified_updates
    
    def aggregate_updates(self, client_updates: List[Tuple[str, OrderedDict]]) -> OrderedDict:
        """
        Aggregate client model updates with Byzantine resilience
        
        Args:
            client_updates: List of (client_id, model_update) tuples
            
        Returns:
            Aggregated model update
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        # Extract gradients and client IDs
        client_ids = [client_id for client_id, _ in client_updates]
        
        # Convert to tensor format for aggregation
        parameter_names = list(client_updates[0][1].keys())
        aggregated_params = OrderedDict()
        
        for param_name in parameter_names:
            # Collect parameter gradients from all clients
            param_gradients = [update[param_name] for _, update in client_updates]
            
            if self.config.enable_byzantine_resilience:
                # Use Byzantine-resilient aggregation
                results = self.byzantine_system.process_round(param_gradients, client_ids)
                aggregated_params[param_name] = results['aggregated_gradient']
            else:
                # Simple averaging
                aggregated_params[param_name] = torch.stack(param_gradients).mean(dim=0)
        
        logger.info(f"Aggregated updates from {len(client_updates)} clients")
        
        return aggregated_params
    
    def update_global_model(self, aggregated_update: OrderedDict):
        """
        Update global model with aggregated gradients
        
        Args:
            aggregated_update: Aggregated model update
        """
        # Apply aggregated gradients to global model
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_update:
                    param.data -= aggregated_update[name]  # Gradient descent
        
        logger.debug("Global model updated with aggregated gradients")
    
    def evaluate_global_model(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate global model on test dataset
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.global_model(data)
                loss = nn.functional.cross_entropy(output, target)
                total_loss += loss.item()
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'total_samples': total
        }
        
        logger.info(f"Global model evaluation: loss={avg_loss:.4f}, accuracy={accuracy:.3f}")
        
        return metrics
    
    def federated_round(self, clients: List[FederatedClient], 
                       test_loader: Optional[DataLoader] = None) -> Dict:
        """
        Execute one complete federated learning round
        
        Args:
            clients: List of available clients
            test_loader: Test data loader for evaluation
            
        Returns:
            Round results and metrics
        """
        round_start_time = time.time()
        self.current_round += 1
        
        logger.info(f"Starting federated round {self.current_round}")
        
        # Select clients for this round
        available_client_ids = [client.client_id for client in clients]
        selected_client_ids = self.select_clients(available_client_ids)
        selected_clients = [c for c in clients if c.client_id in selected_client_ids]
        
        if len(selected_clients) < self.config.min_clients_per_round:
            logger.warning(f"Insufficient clients: {len(selected_clients)} < {self.config.min_clients_per_round}")
            return {'error': 'Insufficient clients for round'}
        
        # Get current global model state
        global_model_state = self.global_model.state_dict()
        
        # Collect client updates
        client_updates = []
        client_metrics = {}
        
        for client in selected_clients:
            try:
                # Client performs local training
                model_update, metrics = client.local_train(global_model_state)
                
                if self.config.enable_post_quantum_crypto:
                    # Create secure update
                    secure_update = client.create_secure_update(
                        model_update, self.current_round, self.kyber_keypair.public_key
                    )
                    client_updates.append(secure_update)
                else:
                    # Plain model update
                    client_updates.append((client.client_id, model_update))
                
                client_metrics[client.client_id] = metrics
                
                # Update client profile
                self.client_profiles[client.client_id].update_performance(
                    metrics['accuracy'], metrics['training_time']
                )
                
            except Exception as e:
                logger.error(f"Client {client.client_id} training failed: {str(e)}")
                continue
        
        if not client_updates:
            return {'error': 'No successful client updates'}
        
        # Process updates based on security configuration
        if self.config.enable_post_quantum_crypto:
            verified_updates = self.decrypt_and_verify_updates(client_updates)
        else:
            verified_updates = client_updates
        
        # Aggregate updates
        aggregated_update = self.aggregate_updates(verified_updates)
        
        # Update global model
        self.update_global_model(aggregated_update)
        
        # Evaluate global model
        evaluation_metrics = {}
        if test_loader is not None:
            evaluation_metrics = self.evaluate_global_model(test_loader)
        
        # Compile round results
        round_time = time.time() - round_start_time
        
        round_results = {
            'round': self.current_round,
            'selected_clients': selected_client_ids,
            'successful_updates': len(verified_updates),
            'client_metrics': client_metrics,
            'global_metrics': evaluation_metrics,
            'round_time': round_time,
            'security_features': {
                'post_quantum_crypto': self.config.enable_post_quantum_crypto,
                'differential_privacy': self.config.enable_differential_privacy,
                'byzantine_resilience': self.config.enable_byzantine_resilience
            }
        }
        
        # Add privacy and security metrics
        if self.config.enable_differential_privacy:
            privacy_report = self.dp_engine.get_privacy_report()
            round_results['privacy_metrics'] = privacy_report
        
        if self.config.enable_byzantine_resilience:
            byzantine_stats = self.byzantine_system.get_defense_effectiveness(0.3)
            round_results['byzantine_metrics'] = byzantine_stats
        
        self.training_history.append(round_results)
        
        logger.info(f"Round {self.current_round} completed: "
                   f"{len(verified_updates)} updates, "
                   f"accuracy={evaluation_metrics.get('accuracy', 0.0):.3f}, "
                   f"time={round_time:.2f}s")
        
        return round_results

def demo_federated_training():
    """Demonstration of QFLARE federated learning"""
    print("🤖 QFLARE Federated Learning Training Engine Demo")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create simple neural network model
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(28*28, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = x.view(-1, 28*28)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # Create synthetic dataset
    class SyntheticMNIST(Dataset):
        def __init__(self, num_samples=1000):
            self.data = torch.randn(num_samples, 28, 28)
            self.targets = torch.randint(0, 10, (num_samples,))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    
    # Training configuration with all security features enabled
    config = TrainingConfig(
        local_epochs=3,
        local_batch_size=32,
        local_learning_rate=0.01,
        num_rounds=5,
        clients_per_round=6,
        min_clients_per_round=4,
        federation_strategy=FederationStrategy.FEDAVG,
        client_selection=ClientSelectionStrategy.REPUTATION_BASED,
        enable_differential_privacy=True,
        enable_byzantine_resilience=True,
        enable_post_quantum_crypto=True,
        dp_epsilon=0.1,
        dp_delta=1e-6,
        gradient_clipping_norm=1.0,
        aggregation_method=AggregationMethod.KRUM,
        expected_byzantine_ratio=0.33
    )
    
    print("Configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Initialize global model and server
    global_model = SimpleNet()
    server = FederatedServer(global_model, config)
    
    # Create federated clients with different data distributions
    num_clients = 8
    clients = []
    
    print(f"\nCreating {num_clients} federated clients...")
    
    for i in range(num_clients):
        # Create client-specific dataset (simulating data heterogeneity)
        if i < 6:  # Honest clients
            dataset = SyntheticMNIST(num_samples=500 + i*100)
        else:  # Potentially Byzantine clients
            dataset = SyntheticMNIST(num_samples=300)
        
        client = FederatedClient(f"client_{i:02d}", global_model, dataset, config)
        clients.append(client)
        
        # Register client with server
        profile = server.register_client(client)
        print(f"  Client {client.client_id}: {profile.data_size} samples, "
              f"PQC keys: {profile.kyber_keypair is not None}")
    
    # Create test dataset
    test_dataset = SyntheticMNIST(num_samples=1000)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"\nStarting federated training for {config.num_rounds} rounds...")
    
    # Execute federated training rounds
    for round_num in range(config.num_rounds):
        print(f"\n--- Round {round_num + 1} ---")
        
        # Execute federated round
        results = server.federated_round(clients, test_loader)
        
        if 'error' in results:
            print(f"Round failed: {results['error']}")
            continue
        
        # Display round results
        print(f"Selected clients: {results['selected_clients']}")
        print(f"Successful updates: {results['successful_updates']}")
        print(f"Global accuracy: {results['global_metrics'].get('accuracy', 0.0):.3f}")
        print(f"Round time: {results['round_time']:.2f}s")
        
        # Security metrics
        if 'privacy_metrics' in results:
            privacy = results['privacy_metrics']['privacy_ledger']
            print(f"Privacy budget used: {privacy['total_epsilon']:.4f}/{privacy['budget_limit']:.4f}")
        
        if 'byzantine_metrics' in results:
            byzantine = results['byzantine_metrics']
            print(f"Byzantine defense success: {byzantine['defense_success_rate']:.1%}")
    
    # Training summary
    print(f"\n--- Training Summary ---")
    print(f"Total rounds: {server.current_round}")
    print(f"Security features enabled:")
    print(f"  Post-Quantum Crypto: {config.enable_post_quantum_crypto}")
    print(f"  Differential Privacy: {config.enable_differential_privacy} (ε={config.dp_epsilon})")
    print(f"  Byzantine Resilience: {config.enable_byzantine_resilience} ({config.aggregation_method.value})")
    
    # Client performance summary
    print(f"\nClient Performance:")
    for client_id, profile in server.client_profiles.items():
        print(f"  {client_id}: rounds={profile.total_rounds_participated}, "
              f"avg_accuracy={profile.get_average_accuracy():.3f}")
    
    # Final evaluation
    final_metrics = server.evaluate_global_model(test_loader)
    print(f"\nFinal Global Model:")
    print(f"  Accuracy: {final_metrics['accuracy']:.3f}")
    print(f"  Loss: {final_metrics['loss']:.4f}")
    
    print(f"\n🤖 QFLARE federated training completed with quantum-safe security!")

if __name__ == "__main__":
    demo_federated_training()