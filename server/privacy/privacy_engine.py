"""
QFLARE Privacy Engine
Advanced privacy orchestration and management for federated learning
"""

import logging
import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import json
import hashlib
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

# Import QFLARE privacy components
try:
    from .differential_privacy import DifferentialPrivacyEngine, PrivacyAccountant, NoiseType
except ImportError:
    from server.privacy.differential_privacy import DifferentialPrivacyEngine, PrivacyAccountant, NoiseType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrivacyLevel(Enum):
    """Privacy protection levels"""
    MINIMAL = "minimal"      # ε = 10.0, δ = 1e-3
    STANDARD = "standard"    # ε = 1.0, δ = 1e-5
    HIGH = "high"           # ε = 0.1, δ = 1e-6
    MAXIMUM = "maximum"     # ε = 0.01, δ = 1e-8

class PrivacyMechanism(Enum):
    """Available privacy mechanisms"""
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    LOCAL_DIFFERENTIAL_PRIVACY = "local_differential_privacy"
    SECURE_AGGREGATION = "secure_aggregation"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"

@dataclass
class PrivacyBudget:
    """Privacy budget configuration"""
    epsilon: float
    delta: float
    mechanism: PrivacyMechanism
    allocated_at: str
    expires_at: Optional[str] = None
    consumed: float = 0.0
    
    @property
    def remaining(self) -> float:
        """Get remaining privacy budget"""
        return max(0.0, self.epsilon - self.consumed)
    
    @property
    def is_exhausted(self) -> bool:
        """Check if budget is exhausted"""
        return self.remaining <= 0.0001  # Small threshold for floating point precision

@dataclass
class PrivacyPolicy:
    """Privacy policy configuration"""
    client_id: str
    privacy_level: PrivacyLevel
    mechanisms: List[PrivacyMechanism]
    budget_allocation: Dict[str, PrivacyBudget]
    data_retention_days: int
    allow_model_updates: bool
    require_secure_aggregation: bool
    max_noise_multiplier: float
    created_at: str
    updated_at: str

@dataclass
class PrivacyAuditLog:
    """Privacy operation audit log entry"""
    timestamp: str
    client_id: str
    operation: str
    mechanism: PrivacyMechanism
    epsilon_consumed: float
    delta_consumed: float
    noise_added: float
    data_size: int
    success: bool
    details: Dict[str, Any]

class PrivacyBudgetManager:
    """Advanced privacy budget management with composition"""
    
    def __init__(self):
        self.client_budgets = {}
        self.global_budget = {}
        self.budget_history = defaultdict(list)
        self.composition_accountant = {}
        self.lock = threading.RLock()
    
    def allocate_budget(self, client_id: str, epsilon: float, delta: float, 
                       mechanism: PrivacyMechanism, duration_hours: int = 24) -> str:
        """Allocate privacy budget to client"""
        with self.lock:
            budget_id = hashlib.sha256(f"{client_id}_{datetime.now().isoformat()}_{epsilon}_{delta}".encode()).hexdigest()[:16]
            
            expires_at = (datetime.now() + timedelta(hours=duration_hours)).isoformat()
            
            budget = PrivacyBudget(
                epsilon=epsilon,
                delta=delta,
                mechanism=mechanism,
                allocated_at=datetime.now().isoformat(),
                expires_at=expires_at
            )
            
            if client_id not in self.client_budgets:
                self.client_budgets[client_id] = {}
            
            self.client_budgets[client_id][budget_id] = budget
            self.budget_history[client_id].append({
                'budget_id': budget_id,
                'action': 'allocated',
                'epsilon': epsilon,
                'delta': delta,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Allocated privacy budget {budget_id} to client {client_id}: ε={epsilon}, δ={delta}")
            return budget_id
    
    def consume_budget(self, client_id: str, budget_id: str, epsilon_consumed: float) -> bool:
        """Consume privacy budget"""
        with self.lock:
            if client_id not in self.client_budgets or budget_id not in self.client_budgets[client_id]:
                logger.error(f"Budget not found: {client_id}/{budget_id}")
                return False
            
            budget = self.client_budgets[client_id][budget_id]
            
            # Check if budget is expired
            if budget.expires_at and datetime.fromisoformat(budget.expires_at) < datetime.now():
                logger.error(f"Budget expired: {budget_id}")
                return False
            
            # Check if sufficient budget remains
            if budget.remaining < epsilon_consumed:
                logger.error(f"Insufficient budget: {budget.remaining} < {epsilon_consumed}")
                return False
            
            # Consume budget
            budget.consumed += epsilon_consumed
            
            self.budget_history[client_id].append({
                'budget_id': budget_id,
                'action': 'consumed',
                'epsilon_consumed': epsilon_consumed,
                'remaining': budget.remaining,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.debug(f"Consumed privacy budget: {epsilon_consumed} from {budget_id}")
            return True
    
    def get_available_budget(self, client_id: str) -> Dict[str, float]:
        """Get available privacy budget for client"""
        with self.lock:
            if client_id not in self.client_budgets:
                return {'total_epsilon': 0.0, 'total_delta': 0.0, 'active_budgets': 0}
            
            total_epsilon = 0.0
            total_delta = 0.0
            active_budgets = 0
            
            current_time = datetime.now()
            
            for budget_id, budget in self.client_budgets[client_id].items():
                # Skip expired budgets
                if budget.expires_at and datetime.fromisoformat(budget.expires_at) < current_time:
                    continue
                
                if not budget.is_exhausted:
                    total_epsilon += budget.remaining
                    total_delta += budget.delta
                    active_budgets += 1
            
            return {
                'total_epsilon': total_epsilon,
                'total_delta': total_delta,
                'active_budgets': active_budgets
            }
    
    def compute_composition_bounds(self, client_id: str) -> Dict[str, float]:
        """Compute privacy composition bounds using advanced composition"""
        with self.lock:
            if client_id not in self.client_budgets:
                return {'composed_epsilon': 0.0, 'composed_delta': 0.0}
            
            # Use advanced composition theorem for better bounds
            epsilons = []
            deltas = []
            
            for budget in self.client_budgets[client_id].values():
                if budget.consumed > 0:
                    epsilons.append(budget.consumed)
                    deltas.append(budget.delta)
            
            if not epsilons:
                return {'composed_epsilon': 0.0, 'composed_delta': 0.0}
            
            # Advanced composition (simplified)
            # In practice, would use more sophisticated composition theorems
            total_epsilon = sum(epsilons)
            total_delta = sum(deltas)
            
            # Apply advanced composition improvement
            k = len(epsilons)  # Number of mechanisms
            if k > 1:
                # Improved bound: ε' = √(2k ln(1/δ'))ε + kε(e^ε - 1)
                epsilon_prime = total_epsilon * (1 + np.sqrt(2 * k * np.log(1/max(total_delta, 1e-10))))
                composed_epsilon = min(total_epsilon, epsilon_prime)
            else:
                composed_epsilon = total_epsilon
            
            return {
                'composed_epsilon': composed_epsilon,
                'composed_delta': total_delta,
                'mechanism_count': k
            }

class AdaptivePrivacyManager:
    """Adaptive privacy parameter adjustment based on utility feedback"""
    
    def __init__(self, utility_threshold: float = 0.8):
        self.utility_threshold = utility_threshold
        self.privacy_history = deque(maxlen=100)
        self.utility_history = deque(maxlen=100)
        self.adaptation_params = {
            'learning_rate': 0.1,
            'epsilon_min': 0.01,
            'epsilon_max': 10.0,
            'utility_weight': 0.7,
            'privacy_weight': 0.3
        }
    
    def record_privacy_utility_feedback(self, epsilon: float, delta: float, 
                                       utility_score: float, accuracy_loss: float):
        """Record privacy-utility trade-off feedback"""
        feedback = {
            'epsilon': epsilon,
            'delta': delta,
            'utility_score': utility_score,
            'accuracy_loss': accuracy_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        self.privacy_history.append(feedback)
        self.utility_history.append(utility_score)
        
        logger.debug(f"Recorded privacy-utility feedback: ε={epsilon}, utility={utility_score}")
    
    def suggest_privacy_parameters(self, target_utility: float = None) -> Dict[str, float]:
        """Suggest optimal privacy parameters based on historical data"""
        if not self.privacy_history:
            # Default parameters
            return {'epsilon': 1.0, 'delta': 1e-5, 'confidence': 0.0}
        
        if target_utility is None:
            target_utility = self.utility_threshold
        
        # Analyze historical performance
        recent_data = list(self.privacy_history)[-20:]  # Last 20 entries
        
        # Find parameters that achieved target utility
        good_params = []
        for entry in recent_data:
            if entry['utility_score'] >= target_utility:
                good_params.append(entry)
        
        if not good_params:
            # If no good parameters found, relax utility requirement
            target_utility *= 0.9
            good_params = [entry for entry in recent_data if entry['utility_score'] >= target_utility]
        
        if good_params:
            # Use parameters from best performing entry
            best_entry = max(good_params, key=lambda x: x['utility_score'])
            suggested_epsilon = best_entry['epsilon']
            suggested_delta = best_entry['delta']
            confidence = 0.8
        else:
            # Use average of recent parameters
            avg_epsilon = np.mean([entry['epsilon'] for entry in recent_data])
            avg_delta = np.mean([entry['delta'] for entry in recent_data])
            suggested_epsilon = avg_epsilon
            suggested_delta = avg_delta
            confidence = 0.3
        
        # Apply bounds
        suggested_epsilon = np.clip(suggested_epsilon, 
                                   self.adaptation_params['epsilon_min'],
                                   self.adaptation_params['epsilon_max'])
        
        return {
            'epsilon': suggested_epsilon,
            'delta': suggested_delta,
            'confidence': confidence,
            'target_utility': target_utility
        }
    
    def adaptive_noise_calibration(self, data_sensitivity: float, 
                                  current_utility: float) -> float:
        """Adaptively calibrate noise based on current utility"""
        if current_utility < self.utility_threshold:
            # Reduce noise to improve utility
            noise_multiplier = 0.8
        elif current_utility > self.utility_threshold * 1.2:
            # Increase noise to strengthen privacy
            noise_multiplier = 1.2
        else:
            # Maintain current noise level
            noise_multiplier = 1.0
        
        return data_sensitivity * noise_multiplier

class PrivacyAuditManager:
    """Privacy audit and compliance management"""
    
    def __init__(self, max_logs: int = 10000):
        self.audit_logs = deque(maxlen=max_logs)
        self.compliance_checks = {}
        self.violation_alerts = []
        self.lock = threading.RLock()
    
    def log_privacy_operation(self, client_id: str, operation: str, 
                             mechanism: PrivacyMechanism, epsilon_consumed: float,
                             delta_consumed: float, noise_added: float,
                             data_size: int, success: bool, details: Dict[str, Any]):
        """Log privacy operation for audit"""
        with self.lock:
            log_entry = PrivacyAuditLog(
                timestamp=datetime.now().isoformat(),
                client_id=client_id,
                operation=operation,
                mechanism=mechanism,
                epsilon_consumed=epsilon_consumed,
                delta_consumed=delta_consumed,
                noise_added=noise_added,
                data_size=data_size,
                success=success,
                details=details
            )
            
            self.audit_logs.append(log_entry)
            
            # Check for compliance violations
            self._check_compliance_violations(log_entry)
    
    def _check_compliance_violations(self, log_entry: PrivacyAuditLog):
        """Check for privacy compliance violations"""
        violations = []
        
        # Check for excessive privacy budget consumption
        if log_entry.epsilon_consumed > 5.0:
            violations.append(f"High epsilon consumption: {log_entry.epsilon_consumed}")
        
        # Check for insufficient noise
        if log_entry.noise_added < 0.01 and log_entry.mechanism == PrivacyMechanism.DIFFERENTIAL_PRIVACY:
            violations.append(f"Low noise addition: {log_entry.noise_added}")
        
        # Check for large data exposure
        if log_entry.data_size > 1000000:  # 1MB threshold
            violations.append(f"Large data size: {log_entry.data_size}")
        
        if violations:
            alert = {
                'timestamp': log_entry.timestamp,
                'client_id': log_entry.client_id,
                'violations': violations,
                'severity': 'high' if len(violations) > 1 else 'medium'
            }
            self.violation_alerts.append(alert)
            logger.warning(f"Privacy compliance violations detected for {log_entry.client_id}: {violations}")
    
    def generate_audit_report(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        with self.lock:
            # Filter logs by date range if specified
            filtered_logs = list(self.audit_logs)
            if start_date:
                filtered_logs = [log for log in filtered_logs if log.timestamp >= start_date]
            if end_date:
                filtered_logs = [log for log in filtered_logs if log.timestamp <= end_date]
            
            # Aggregate statistics
            total_operations = len(filtered_logs)
            successful_operations = sum(1 for log in filtered_logs if log.success)
            
            # Privacy budget consumption
            total_epsilon = sum(log.epsilon_consumed for log in filtered_logs)
            total_delta = sum(log.delta_consumed for log in filtered_logs)
            
            # Client statistics
            client_stats = defaultdict(lambda: {'operations': 0, 'epsilon_consumed': 0.0})
            for log in filtered_logs:
                client_stats[log.client_id]['operations'] += 1
                client_stats[log.client_id]['epsilon_consumed'] += log.epsilon_consumed
            
            # Mechanism usage
            mechanism_stats = defaultdict(int)
            for log in filtered_logs:
                mechanism_stats[log.mechanism.value] += 1
            
            return {
                'report_period': {
                    'start_date': start_date or 'all_time',
                    'end_date': end_date or 'all_time'
                },
                'summary': {
                    'total_operations': total_operations,
                    'successful_operations': successful_operations,
                    'success_rate': successful_operations / total_operations if total_operations > 0 else 0,
                    'total_epsilon_consumed': total_epsilon,
                    'total_delta_consumed': total_delta
                },
                'client_statistics': dict(client_stats),
                'mechanism_usage': dict(mechanism_stats),
                'compliance_violations': len(self.violation_alerts),
                'recent_violations': self.violation_alerts[-10:] if self.violation_alerts else []
            }

class PrivacyEngine:
    """Main privacy engine orchestrating all privacy mechanisms"""
    
    def __init__(self):
        self.dp_engine = DifferentialPrivacyEngine()
        self.budget_manager = PrivacyBudgetManager()
        self.adaptive_manager = AdaptivePrivacyManager()
        self.audit_manager = PrivacyAuditManager()
        self.privacy_policies = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Default privacy levels
        self.privacy_levels = {
            PrivacyLevel.MINIMAL: {'epsilon': 10.0, 'delta': 1e-3},
            PrivacyLevel.STANDARD: {'epsilon': 1.0, 'delta': 1e-5},
            PrivacyLevel.HIGH: {'epsilon': 0.1, 'delta': 1e-6},
            PrivacyLevel.MAXIMUM: {'epsilon': 0.01, 'delta': 1e-8}
        }
        
        logger.info("Privacy Engine initialized")
    
    def setup_client_privacy(self, client_id: str, privacy_level: PrivacyLevel,
                           mechanisms: List[PrivacyMechanism] = None,
                           custom_params: Dict[str, Any] = None) -> PrivacyPolicy:
        """Set up privacy configuration for client"""
        if mechanisms is None:
            mechanisms = [PrivacyMechanism.DIFFERENTIAL_PRIVACY]
        
        # Get privacy parameters
        level_params = self.privacy_levels[privacy_level]
        if custom_params:
            level_params.update(custom_params)
        
        # Allocate privacy budgets
        budget_allocation = {}
        for mechanism in mechanisms:
            budget_id = self.budget_manager.allocate_budget(
                client_id, 
                level_params['epsilon'],
                level_params['delta'],
                mechanism
            )
            budget_allocation[mechanism.value] = self.budget_manager.client_budgets[client_id][budget_id]
        
        # Create privacy policy
        policy = PrivacyPolicy(
            client_id=client_id,
            privacy_level=privacy_level,
            mechanisms=mechanisms,
            budget_allocation=budget_allocation,
            data_retention_days=custom_params.get('data_retention_days', 30),
            allow_model_updates=custom_params.get('allow_model_updates', True),
            require_secure_aggregation=custom_params.get('require_secure_aggregation', False),
            max_noise_multiplier=custom_params.get('max_noise_multiplier', 2.0),
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        self.privacy_policies[client_id] = policy
        
        logger.info(f"Privacy policy created for client {client_id}: {privacy_level.value}")
        return policy
    
    async def apply_privacy_protection(self, client_id: str, data: np.ndarray,
                                     operation: str = "model_update",
                                     mechanism: PrivacyMechanism = PrivacyMechanism.DIFFERENTIAL_PRIVACY,
                                     custom_epsilon: float = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply privacy protection to data"""
        
        if client_id not in self.privacy_policies:
            raise ValueError(f"No privacy policy found for client: {client_id}")
        
        policy = self.privacy_policies[client_id]
        
        # Check if mechanism is allowed
        if mechanism not in policy.mechanisms:
            raise ValueError(f"Mechanism {mechanism.value} not allowed for client {client_id}")
        
        # Get privacy parameters
        if custom_epsilon is not None:
            epsilon = custom_epsilon
        else:
            # Use adaptive parameter suggestion
            suggestions = self.adaptive_manager.suggest_privacy_parameters()
            epsilon = suggestions['epsilon']
        
        budget_key = mechanism.value
        if budget_key not in policy.budget_allocation:
            raise ValueError(f"No budget allocated for mechanism: {mechanism.value}")
        
        budget = policy.budget_allocation[budget_key]
        delta = budget.delta
        
        try:
            # Apply privacy mechanism
            if mechanism == PrivacyMechanism.DIFFERENTIAL_PRIVACY:
                protected_data = await self._apply_differential_privacy(
                    data, epsilon, delta, policy.max_noise_multiplier
                )
                noise_added = np.linalg.norm(protected_data - data)
            else:
                # Other mechanisms would be implemented here
                protected_data = data.copy()
                noise_added = 0.0
            
            # Consume privacy budget
            budget_consumed = self.budget_manager.consume_budget(
                client_id, list(policy.budget_allocation.keys())[0], epsilon
            )
            
            if not budget_consumed:
                raise RuntimeError("Failed to consume privacy budget")
            
            # Log operation
            self.audit_manager.log_privacy_operation(
                client_id=client_id,
                operation=operation,
                mechanism=mechanism,
                epsilon_consumed=epsilon,
                delta_consumed=delta,
                noise_added=noise_added,
                data_size=data.nbytes,
                success=True,
                details={
                    'data_shape': data.shape,
                    'noise_type': 'gaussian',
                    'adaptive_params': self.adaptive_manager.suggest_privacy_parameters()
                }
            )
            
            # Prepare result metadata
            metadata = {
                'epsilon_consumed': epsilon,
                'delta_consumed': delta,
                'noise_added': noise_added,
                'mechanism': mechanism.value,
                'privacy_level': policy.privacy_level.value,
                'remaining_budget': self.budget_manager.get_available_budget(client_id),
                'operation_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Privacy protection applied for {client_id}: ε={epsilon:.3f}, noise={noise_added:.3f}")
            return protected_data, metadata
            
        except Exception as e:
            # Log failed operation
            self.audit_manager.log_privacy_operation(
                client_id=client_id,
                operation=operation,
                mechanism=mechanism,
                epsilon_consumed=0.0,
                delta_consumed=0.0,
                noise_added=0.0,
                data_size=data.nbytes,
                success=False,
                details={'error': str(e)}
            )
            raise
    
    async def _apply_differential_privacy(self, data: np.ndarray, epsilon: float,
                                        delta: float, max_noise_multiplier: float) -> np.ndarray:
        """Apply differential privacy protection"""
        
        # Calculate sensitivity (simplified - would be more sophisticated in practice)
        sensitivity = np.std(data) * 2  # Rough estimate
        
        # Apply adaptive noise calibration
        calibrated_sensitivity = self.adaptive_manager.adaptive_noise_calibration(
            sensitivity, 0.8  # Assume current utility of 0.8
        )
        
        # Limit noise multiplier
        noise_multiplier = min(calibrated_sensitivity / sensitivity, max_noise_multiplier)
        final_sensitivity = sensitivity * noise_multiplier
        
        # Add Gaussian noise for (ε, δ)-DP
        noise_scale = np.sqrt(2 * np.log(1.25 / delta)) * final_sensitivity / epsilon
        noise = np.random.normal(0, noise_scale, data.shape)
        
        return data + noise
    
    def update_privacy_utility_feedback(self, client_id: str, accuracy_before: float,
                                       accuracy_after: float, epsilon_used: float,
                                       delta_used: float):
        """Update privacy-utility feedback for adaptive learning"""
        utility_score = accuracy_after / accuracy_before if accuracy_before > 0 else 0.0
        accuracy_loss = accuracy_before - accuracy_after
        
        self.adaptive_manager.record_privacy_utility_feedback(
            epsilon_used, delta_used, utility_score, accuracy_loss
        )
        
        logger.info(f"Updated privacy-utility feedback for {client_id}: utility={utility_score:.3f}")
    
    def get_client_privacy_status(self, client_id: str) -> Dict[str, Any]:
        """Get comprehensive privacy status for client"""
        if client_id not in self.privacy_policies:
            return {'error': 'No privacy policy found'}
        
        policy = self.privacy_policies[client_id]
        budget_status = self.budget_manager.get_available_budget(client_id)
        composition_bounds = self.budget_manager.compute_composition_bounds(client_id)
        
        return {
            'client_id': client_id,
            'privacy_level': policy.privacy_level.value,
            'mechanisms': [m.value for m in policy.mechanisms],
            'budget_status': budget_status,
            'composition_bounds': composition_bounds,
            'policy_created': policy.created_at,
            'policy_updated': policy.updated_at,
            'data_retention_days': policy.data_retention_days,
            'adaptive_suggestions': self.adaptive_manager.suggest_privacy_parameters()
        }
    
    def generate_privacy_report(self) -> Dict[str, Any]:
        """Generate comprehensive privacy system report"""
        audit_report = self.audit_manager.generate_audit_report()
        
        # Add system-wide statistics
        total_clients = len(self.privacy_policies)
        active_budgets = sum(
            self.budget_manager.get_available_budget(client_id)['active_budgets']
            for client_id in self.privacy_policies.keys()
        )
        
        privacy_levels = defaultdict(int)
        for policy in self.privacy_policies.values():
            privacy_levels[policy.privacy_level.value] += 1
        
        return {
            'system_overview': {
                'total_clients': total_clients,
                'active_budgets': active_budgets,
                'privacy_level_distribution': dict(privacy_levels),
                'supported_mechanisms': [m.value for m in PrivacyMechanism]
            },
            'audit_report': audit_report,
            'adaptive_learning': {
                'total_feedback_entries': len(self.adaptive_manager.privacy_history),
                'utility_threshold': self.adaptive_manager.utility_threshold,
                'current_suggestions': self.adaptive_manager.suggest_privacy_parameters()
            },
            'report_generated': datetime.now().isoformat()
        }

# Global privacy engine instance
privacy_engine = PrivacyEngine()

# Utility functions for QFLARE integration
async def setup_federated_privacy(client_ids: List[str], 
                                privacy_level: PrivacyLevel = PrivacyLevel.STANDARD) -> Dict[str, PrivacyPolicy]:
    """Set up privacy for multiple federated learning clients"""
    policies = {}
    
    for client_id in client_ids:
        policy = privacy_engine.setup_client_privacy(client_id, privacy_level)
        policies[client_id] = policy
    
    logger.info(f"Privacy setup complete for {len(client_ids)} clients")
    return policies

async def privacy_preserving_aggregation(client_data: Dict[str, np.ndarray],
                                       privacy_level: PrivacyLevel = PrivacyLevel.STANDARD) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Perform privacy-preserving model aggregation"""
    
    protected_updates = {}
    aggregation_metadata = {}
    
    # Apply privacy protection to each client's data
    for client_id, data in client_data.items():
        protected_data, metadata = await privacy_engine.apply_privacy_protection(
            client_id, data, "aggregation", PrivacyMechanism.DIFFERENTIAL_PRIVACY
        )
        protected_updates[client_id] = protected_data
        aggregation_metadata[client_id] = metadata
    
    # Aggregate protected updates
    if protected_updates:
        aggregated = np.mean(list(protected_updates.values()), axis=0)
    else:
        raise ValueError("No client data provided for aggregation")
    
    # Calculate total privacy cost
    total_epsilon = sum(meta['epsilon_consumed'] for meta in aggregation_metadata.values())
    total_delta = sum(meta['delta_consumed'] for meta in aggregation_metadata.values())
    
    final_metadata = {
        'total_epsilon_consumed': total_epsilon,
        'total_delta_consumed': total_delta,
        'participating_clients': len(protected_updates),
        'aggregation_timestamp': datetime.now().isoformat(),
        'client_metadata': aggregation_metadata
    }
    
    logger.info(f"Privacy-preserving aggregation complete: {len(protected_updates)} clients, ε={total_epsilon:.3f}")
    return aggregated, final_metadata

if __name__ == "__main__":
    # Demo and testing
    async def main():
        print("🛡️ QFLARE Privacy Engine Demo")
        print("=" * 50)
        
        # Setup client privacy
        client_id = "test_client_001"
        policy = privacy_engine.setup_client_privacy(
            client_id, 
            PrivacyLevel.STANDARD,
            [PrivacyMechanism.DIFFERENTIAL_PRIVACY]
        )
        
        print(f"✅ Privacy policy created for {client_id}")
        print(f"   Privacy level: {policy.privacy_level.value}")
        print(f"   Mechanisms: {[m.value for m in policy.mechanisms]}")
        
        # Test privacy protection
        test_data = np.random.randn(100, 10)  # Simulated model parameters
        
        protected_data, metadata = await privacy_engine.apply_privacy_protection(
            client_id, test_data, "model_update"
        )
        
        print(f"\n🔒 Privacy protection applied:")
        print(f"   Epsilon consumed: {metadata['epsilon_consumed']:.3f}")
        print(f"   Noise added: {metadata['noise_added']:.3f}")
        print(f"   Remaining budget: {metadata['remaining_budget']}")
        
        # Update utility feedback
        privacy_engine.update_privacy_utility_feedback(
            client_id, 
            accuracy_before=0.95,
            accuracy_after=0.92,
            epsilon_used=metadata['epsilon_consumed'],
            delta_used=metadata['delta_consumed']
        )
        
        # Get privacy status
        status = privacy_engine.get_client_privacy_status(client_id)
        print(f"\n📊 Privacy Status:")
        print(f"   Active budgets: {status['budget_status']['active_budgets']}")
        print(f"   Composition bounds: ε={status['composition_bounds']['composed_epsilon']:.3f}")
        print(f"   Adaptive suggestions: ε={status['adaptive_suggestions']['epsilon']:.3f}")
        
        # Generate system report
        report = privacy_engine.generate_privacy_report()
        print(f"\n📋 System Report:")
        print(f"   Total clients: {report['system_overview']['total_clients']}")
        print(f"   Total operations: {report['audit_report']['summary']['total_operations']}")
        print(f"   Success rate: {report['audit_report']['summary']['success_rate']:.1%}")
    
    # Run the demo
    asyncio.run(main())