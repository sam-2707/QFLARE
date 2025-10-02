"""
Byzantine Fault Tolerance API Endpoints

FastAPI endpoints for Byzantine protection configuration and monitoring.
Provides REST API for robust federated learning operations.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from ..byzantine.byzantine_fl_controller import ByzantineFLController, create_byzantine_fl_controller
from ..websocket.manager import WebSocketManager

logger = logging.getLogger(__name__)

# Create router
byzantine_router = APIRouter(prefix="/api/byzantine", tags=["byzantine"])

# Global Byzantine controller (will be initialized)
byzantine_controller: Optional[ByzantineFLController] = None


# Pydantic models for request/response
class ByzantineConfigRequest(BaseModel):
    detection_method: str = Field(..., description="Detection method: krum, multi_krum, trimmed_mean, clustering")
    max_malicious_ratio: float = Field(..., description="Maximum ratio of malicious clients (0.0-0.5)")
    aggregation_method: Optional[str] = Field(None, description="Aggregation method override")


class RobustTrainingRoundRequest(BaseModel):
    num_clients: int = Field(default=10, description="Number of clients to use")
    client_fraction: float = Field(default=1.0, description="Fraction of clients to select")
    epochs: int = Field(default=1, description="Number of training epochs")
    batch_size: int = Field(default=32, description="Batch size for training")
    learning_rate: float = Field(default=0.01, description="Learning rate")
    attack_probability: float = Field(default=0.0, description="Probability of attack (simulation only)")


class AttackSimulationRequest(BaseModel):
    num_clients: int = Field(default=10, description="Total number of clients")
    num_malicious: int = Field(default=2, description="Number of malicious clients")
    attack_type: str = Field(default="gaussian_noise", description="Type of attack: gaussian_noise, sign_flipping, large_deviation")


# Dependency to get Byzantine controller
def get_byzantine_controller() -> ByzantineFLController:
    global byzantine_controller
    if byzantine_controller is None:
        # Initialize with default settings
        byzantine_controller = create_byzantine_fl_controller(detection_method="krum")
    return byzantine_controller


def set_websocket_manager(websocket_manager: WebSocketManager):
    """Set the WebSocket manager for the Byzantine controller."""
    global byzantine_controller
    if byzantine_controller is None:
        byzantine_controller = create_byzantine_fl_controller(
            websocket_manager=websocket_manager,
            detection_method="krum"
        )
    else:
        byzantine_controller.websocket_manager = websocket_manager


@byzantine_router.get("/status", response_model=Dict[str, Any])
async def get_byzantine_status(controller: ByzantineFLController = Depends(get_byzantine_controller)):
    """
    Get current Byzantine protection status.
    
    Returns:
        Byzantine protection status and configuration
    """
    try:
        dashboard_data = await controller.get_byzantine_dashboard_data()
        
        return {
            "success": True,
            "data": dashboard_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get Byzantine status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get Byzantine status: {str(e)}")


@byzantine_router.get("/dashboard", response_model=Dict[str, Any])
async def get_byzantine_dashboard(controller: ByzantineFLController = Depends(get_byzantine_controller)):
    """
    Get comprehensive Byzantine protection dashboard data.
    
    Returns:
        Complete Byzantine metrics for dashboard display
    """
    try:
        dashboard_data = await controller.get_byzantine_dashboard_data()
        
        return {
            "success": True,
            "dashboard_data": dashboard_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get Byzantine dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get Byzantine dashboard: {str(e)}")


@byzantine_router.post("/configure", response_model=Dict[str, Any])
async def configure_byzantine_protection(
    config: ByzantineConfigRequest,
    controller: ByzantineFLController = Depends(get_byzantine_controller)
):
    """
    Configure Byzantine protection parameters.
    
    Args:
        config: Byzantine configuration request
        
    Returns:
        Configuration result
    """
    try:
        # Validate detection method
        valid_methods = ["krum", "multi_krum", "trimmed_mean", "clustering"]
        if config.detection_method not in valid_methods:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid detection method. Must be one of: {valid_methods}"
            )
        
        # Validate malicious ratio
        if not 0 < config.max_malicious_ratio < 0.5:
            raise HTTPException(
                status_code=400,
                detail="max_malicious_ratio must be between 0 and 0.5"
            )
        
        # Update configuration
        result = await controller.update_byzantine_config({
            "detection_method": config.detection_method,
            "max_malicious_ratio": config.max_malicious_ratio
        })
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        logger.info(f"Byzantine protection configured: {config.detection_method}")
        
        return {
            "success": True,
            "message": f"Byzantine protection configured with {config.detection_method}",
            "configuration": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to configure Byzantine protection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to configure protection: {str(e)}")


@byzantine_router.post("/robust-training-round", response_model=Dict[str, Any])
async def run_robust_training_round(
    training_request: RobustTrainingRoundRequest,
    background_tasks: BackgroundTasks,
    controller: ByzantineFLController = Depends(get_byzantine_controller)
):
    """
    Start a Byzantine-robust federated learning training round.
    
    Args:
        training_request: Robust training round configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        Training round initiation result
    """
    try:
        # Prepare round configuration
        round_config = {
            "round_id": f"byzantine_round_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "num_clients": training_request.num_clients,
            "client_fraction": training_request.client_fraction,
            "epochs": training_request.epochs,
            "batch_size": training_request.batch_size,
            "learning_rate": training_request.learning_rate,
            "attack_probability": training_request.attack_probability
        }
        
        # Start training round as background task
        async def run_training():
            try:
                result = await controller.run_byzantine_robust_training_round(round_config)
                logger.info(f"Background Byzantine training round completed: {result.get('success', False)}")
            except Exception as e:
                logger.error(f"Background Byzantine training round failed: {str(e)}")
        
        background_tasks.add_task(run_training)
        
        logger.info(f"Started Byzantine-robust training round: {round_config['round_id']}")
        
        return {
            "success": True,
            "message": "Byzantine-robust training round started",
            "round_id": round_config["round_id"],
            "configuration": round_config,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start robust training round: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start training round: {str(e)}")


@byzantine_router.get("/attacks", response_model=Dict[str, Any])
async def get_attack_history(
    limit: int = 10,
    controller: ByzantineFLController = Depends(get_byzantine_controller)
):
    """
    Get history of detected Byzantine attacks.
    
    Args:
        limit: Maximum number of attack records to return
        
    Returns:
        Attack history and statistics
    """
    try:
        # Get attack statistics from aggregator
        attack_stats = controller.robust_aggregator.get_attack_statistics()
        
        # Get recent attacks
        recent_attacks = attack_stats.get("recent_attacks", [])[-limit:]
        
        return {
            "success": True,
            "attack_history": recent_attacks,
            "attack_statistics": {
                "total_attacks": attack_stats.get("total_attacks", 0),
                "attack_rate": attack_stats.get("attack_rate", 0.0),
                "average_severity": attack_stats.get("average_attack_severity", 0.0),
                "max_severity": attack_stats.get("max_attack_severity", 0.0),
                "most_malicious_clients": attack_stats.get("most_malicious_clients", [])
            },
            "system_resilience": {
                "total_rounds": controller.byzantine_rounds_completed,
                "successful_defenses": controller.byzantine_rounds_completed - controller.total_attacks_detected,
                "defense_rate": (controller.byzantine_rounds_completed - controller.total_attacks_detected) / max(controller.byzantine_rounds_completed, 1)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get attack history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get attack history: {str(e)}")


@byzantine_router.get("/robustness-report", response_model=Dict[str, Any])
async def get_robustness_report(controller: ByzantineFLController = Depends(get_byzantine_controller)):
    """
    Get comprehensive robustness report.
    
    Returns:
        Detailed robustness analysis and recommendations
    """
    try:
        robustness_report = controller.robust_aggregator.get_robustness_report()
        
        return {
            "success": True,
            "robustness_report": robustness_report,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get robustness report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get robustness report: {str(e)}")


@byzantine_router.get("/client-trust", response_model=Dict[str, Any])
async def get_client_trust_scores(controller: ByzantineFLController = Depends(get_byzantine_controller)):
    """
    Get trust scores for all clients.
    
    Returns:
        Client trust scores and reputation analysis
    """
    try:
        detection_summary = controller.robust_aggregator.byzantine_detector.get_detection_summary()
        
        # Get trust scores for all clients that have been evaluated
        client_reputation = detection_summary.get("client_reputation", {})
        trust_scores = {}
        
        for client_id in client_reputation.keys():
            trust_score = controller.robust_aggregator.byzantine_detector.get_client_trust_score(int(client_id))
            trust_scores[client_id] = trust_score
        
        # Classify clients by trust level
        trusted_clients = [client_id for client_id, score in trust_scores.items() if score > 0.8]
        suspicious_clients = [client_id for client_id, score in trust_scores.items() if 0.3 <= score <= 0.8]
        malicious_clients = [client_id for client_id, score in trust_scores.items() if score < 0.3]
        
        return {
            "success": True,
            "client_trust_scores": trust_scores,
            "client_classification": {
                "trusted": trusted_clients,
                "suspicious": suspicious_clients,
                "malicious": malicious_clients
            },
            "reputation_summary": {
                "total_evaluated_clients": len(trust_scores),
                "trusted_count": len(trusted_clients),
                "suspicious_count": len(suspicious_clients),
                "malicious_count": len(malicious_clients)
            },
            "client_reputation": client_reputation,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get client trust scores: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get client trust scores: {str(e)}")


@byzantine_router.post("/simulate-attack", response_model=Dict[str, Any])
async def simulate_byzantine_attack(
    attack_request: AttackSimulationRequest,
    controller: ByzantineFLController = Depends(get_byzantine_controller)
):
    """
    Simulate a Byzantine attack for testing purposes.
    
    WARNING: This endpoint is for testing and demonstration only.
    
    Args:
        attack_request: Attack simulation parameters
        
    Returns:
        Attack simulation result
    """
    try:
        # Validate attack parameters
        if attack_request.num_malicious >= attack_request.num_clients:
            raise HTTPException(
                status_code=400,
                detail="Number of malicious clients must be less than total clients"
            )
        
        if attack_request.num_malicious / attack_request.num_clients > controller.max_malicious_ratio:
            raise HTTPException(
                status_code=400,
                detail=f"Attack exceeds maximum malicious ratio ({controller.max_malicious_ratio:.1%})"
            )
        
        # Create round configuration for attack simulation
        round_config = {
            "round_id": f"attack_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "num_clients": attack_request.num_clients,
            "client_fraction": 1.0,
            "epochs": 1,
            "attack_probability": 1.0,  # Force attack
            "num_malicious": attack_request.num_malicious,
            "attack_type": attack_request.attack_type
        }
        
        # Run simulated attack
        result = await controller.run_byzantine_robust_training_round(round_config)
        
        logger.warning(f"Attack simulation completed: {attack_request.attack_type} with {attack_request.num_malicious} malicious clients")
        
        return {
            "success": True,
            "message": "Byzantine attack simulation completed",
            "simulation_result": result,
            "attack_detected": result.get("attack_detected", False),
            "detection_method": result.get("detection_method", "unknown"),
            "malicious_clients_detected": result.get("malicious_clients", []),
            "honest_clients": result.get("honest_clients", []),
            "warning": "This was a simulated attack for testing purposes",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to simulate attack: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to simulate attack: {str(e)}")


@byzantine_router.get("/detection-methods", response_model=Dict[str, Any])
async def get_detection_methods():
    """
    Get information about available Byzantine detection methods.
    
    Returns:
        Detection methods and their descriptions
    """
    try:
        detection_methods = {
            "krum": {
                "name": "Krum",
                "description": "Selects the update closest to its k-nearest neighbors",
                "strengths": ["Robust against coordinated attacks", "Theoretically proven"],
                "weaknesses": ["Only selects one client update", "May lose information"],
                "recommended_for": "High-security environments"
            },
            "multi_krum": {
                "name": "Multi-Krum",
                "description": "Selects multiple honest clients and averages their updates",
                "strengths": ["Preserves more information than Krum", "Good balance of robustness and utility"],
                "weaknesses": ["More complex than single Krum", "Parameter tuning required"],
                "recommended_for": "General federated learning scenarios"
            },
            "trimmed_mean": {
                "name": "Trimmed Mean",
                "description": "Removes extreme values and computes mean of remaining updates",
                "strengths": ["Simple and effective", "Good against outlier attacks"],
                "weaknesses": ["May remove legitimate extreme values", "Fixed trimming ratio"],
                "recommended_for": "Scenarios with moderate attack intensity"
            },
            "clustering": {
                "name": "Clustering-based Detection",
                "description": "Groups similar updates and identifies outlier clusters",
                "strengths": ["Adaptive to attack patterns", "Can detect complex attacks"],
                "weaknesses": ["Parameter sensitive", "May struggle with sophisticated attacks"],
                "recommended_for": "Exploratory analysis and diverse attack patterns"
            }
        }
        
        aggregation_methods = {
            "federated_averaging": "Standard FedAvg on honest clients only",
            "trimmed_mean": "Coordinate-wise trimmed mean aggregation",
            "median": "Coordinate-wise median aggregation",
            "krum": "Single client selection based on Krum score",
            "multi_krum": "Multiple client selection and averaging"
        }
        
        return {
            "success": True,
            "detection_methods": detection_methods,
            "aggregation_methods": aggregation_methods,
            "current_configuration": {
                "detection_method": byzantine_controller.detection_method if byzantine_controller else "krum",
                "aggregation_method": byzantine_controller.aggregation_method if byzantine_controller else "krum",
                "max_malicious_ratio": byzantine_controller.max_malicious_ratio if byzantine_controller else 0.33
            },
            "recommendations": {
                "high_security": "Use Krum or Multi-Krum with low malicious ratio",
                "balanced": "Use Multi-Krum or Trimmed Mean with moderate malicious ratio",
                "high_utility": "Use Clustering with careful parameter tuning"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get detection methods: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get detection methods: {str(e)}")


@byzantine_router.get("/health", response_model=Dict[str, Any])
async def byzantine_health_check(controller: ByzantineFLController = Depends(get_byzantine_controller)):
    """
    Health check for Byzantine protection system.
    
    Returns:
        Byzantine protection system health status
    """
    try:
        dashboard_data = await controller.get_byzantine_dashboard_data()
        
        health_status = {
            "byzantine_protection_active": True,
            "detection_method": controller.detection_method,
            "aggregation_method": controller.aggregation_method,
            "max_malicious_ratio": controller.max_malicious_ratio,
            "rounds_completed": controller.byzantine_rounds_completed,
            "attacks_detected": controller.total_attacks_detected,
            "system_resilience": dashboard_data.get("system_resilience", {}).get("system_status", "unknown"),
            "last_activity": controller.robust_aggregator.robust_aggregation_history[-1]["timestamp"] if controller.robust_aggregator.robust_aggregation_history else None
        }
        
        return {
            "success": True,
            "health": health_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Byzantine health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Byzantine health check failed: {str(e)}")


# Export router
__all__ = ["byzantine_router", "set_websocket_manager"]