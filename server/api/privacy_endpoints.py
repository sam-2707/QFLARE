"""
Privacy API Endpoints

FastAPI endpoints for differential privacy configuration and monitoring.
Provides REST API for privacy-preserving federated learning operations.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from ..privacy.private_fl_controller import PrivateFLController, create_private_fl_controller
from ..websocket.manager import WebSocketManager

logger = logging.getLogger(__name__)

# Create router
privacy_router = APIRouter(prefix="/api/privacy", tags=["privacy"])

# Global privacy controller (will be initialized)
privacy_controller: Optional[PrivateFLController] = None


# Pydantic models for request/response
class PrivacyConfigRequest(BaseModel):
    privacy_level: str = Field(..., description="Privacy level: strong, moderate, or weak")
    epsilon: Optional[float] = Field(None, description="Custom epsilon value")
    delta: Optional[float] = Field(None, description="Custom delta value")


class TrainingRoundRequest(BaseModel):
    num_clients: int = Field(default=5, description="Number of clients to use")
    client_fraction: float = Field(default=1.0, description="Fraction of clients to select")
    epochs: int = Field(default=1, description="Number of training epochs")
    batch_size: int = Field(default=32, description="Batch size for training")
    learning_rate: float = Field(default=0.01, description="Learning rate")
    privacy_level: Optional[str] = Field(None, description="Override privacy level for this round")


class PrivacyParameterValidation(BaseModel):
    epsilon: float = Field(..., description="Epsilon value to validate")
    delta: float = Field(..., description="Delta value to validate")


# Dependency to get privacy controller
def get_privacy_controller() -> PrivateFLController:
    global privacy_controller
    if privacy_controller is None:
        # Initialize with default settings
        privacy_controller = create_private_fl_controller(privacy_level="strong")
    return privacy_controller


def set_websocket_manager(websocket_manager: WebSocketManager):
    """Set the WebSocket manager for the privacy controller."""
    global privacy_controller
    if privacy_controller is None:
        privacy_controller = create_private_fl_controller(
            websocket_manager=websocket_manager,
            privacy_level="strong"
        )
    else:
        privacy_controller.websocket_manager = websocket_manager


@privacy_router.get("/status", response_model=Dict[str, Any])
async def get_privacy_status(controller: PrivateFLController = Depends(get_privacy_controller)):
    """
    Get current privacy status and configuration.
    
    Returns:
        Privacy status including budget, configuration, and metrics
    """
    try:
        dashboard_data = await controller.get_privacy_dashboard_data()
        
        return {
            "success": True,
            "data": dashboard_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get privacy status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get privacy status: {str(e)}")


@privacy_router.get("/dashboard", response_model=Dict[str, Any])
async def get_privacy_dashboard(controller: PrivateFLController = Depends(get_privacy_controller)):
    """
    Get comprehensive privacy dashboard data.
    
    Returns:
        Complete privacy metrics for dashboard display
    """
    try:
        dashboard_data = await controller.get_privacy_dashboard_data()
        
        return {
            "success": True,
            "dashboard_data": dashboard_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get privacy dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get privacy dashboard: {str(e)}")


@privacy_router.post("/configure", response_model=Dict[str, Any])
async def configure_privacy(
    config: PrivacyConfigRequest,
    controller: PrivateFLController = Depends(get_privacy_controller)
):
    """
    Configure differential privacy parameters.
    
    Args:
        config: Privacy configuration request
        
    Returns:
        Configuration result
    """
    try:
        # Validate privacy level
        if config.privacy_level not in ["strong", "moderate", "weak"]:
            raise HTTPException(status_code=400, detail=f"Invalid privacy level: {config.privacy_level}")
        
        # If custom parameters provided, validate them
        if config.epsilon is not None and config.delta is not None:
            validation = await controller.validate_privacy_parameters(config.epsilon, config.delta)
            if not validation["valid"]:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid privacy parameters: {validation.get('warnings', [])}"
                )
        
        # Set privacy level
        result = await controller.set_privacy_level(config.privacy_level)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        logger.info(f"Privacy configured: {config.privacy_level}")
        
        return {
            "success": True,
            "message": f"Privacy level set to {config.privacy_level}",
            "configuration": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to configure privacy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to configure privacy: {str(e)}")


@privacy_router.post("/validate-parameters", response_model=Dict[str, Any])
async def validate_privacy_parameters(
    validation_request: PrivacyParameterValidation,
    controller: PrivateFLController = Depends(get_privacy_controller)
):
    """
    Validate differential privacy parameters.
    
    Args:
        validation_request: Parameters to validate
        
    Returns:
        Validation result with recommendations
    """
    try:
        validation_result = await controller.validate_privacy_parameters(
            validation_request.epsilon,
            validation_request.delta
        )
        
        return {
            "success": True,
            "validation": validation_result,
            "parameters": {
                "epsilon": validation_request.epsilon,
                "delta": validation_request.delta
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to validate privacy parameters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to validate parameters: {str(e)}")


@privacy_router.post("/training-round", response_model=Dict[str, Any])
async def run_private_training_round(
    training_request: TrainingRoundRequest,
    background_tasks: BackgroundTasks,
    controller: PrivateFLController = Depends(get_privacy_controller)
):
    """
    Start a privacy-preserving federated learning training round.
    
    Args:
        training_request: Training round configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        Training round initiation result
    """
    try:
        # Check if privacy budget is available
        status = await controller.get_privacy_dashboard_data()
        if not status.get("privacy_budget", {}).get("budget_valid", False):
            raise HTTPException(
                status_code=400, 
                detail="Privacy budget exhausted - cannot start new training round"
            )
        
        # Prepare round configuration
        round_config = {
            "round_id": f"private_round_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "num_clients": training_request.num_clients,
            "client_fraction": training_request.client_fraction,
            "epochs": training_request.epochs,
            "batch_size": training_request.batch_size,
            "learning_rate": training_request.learning_rate
        }
        
        # If privacy level override provided, apply it
        if training_request.privacy_level:
            if training_request.privacy_level not in ["strong", "moderate", "weak"]:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid privacy level: {training_request.privacy_level}"
                )
            await controller.set_privacy_level(training_request.privacy_level)
        
        # Start training round as background task
        async def run_training():
            try:
                result = await controller.run_private_training_round(round_config)
                logger.info(f"Background training round completed: {result.get('success', False)}")
            except Exception as e:
                logger.error(f"Background training round failed: {str(e)}")
        
        background_tasks.add_task(run_training)
        
        logger.info(f"Started private training round: {round_config['round_id']}")
        
        return {
            "success": True,
            "message": "Private training round started",
            "round_id": round_config["round_id"],
            "configuration": round_config,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start private training round: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start training round: {str(e)}")


@privacy_router.get("/history", response_model=Dict[str, Any])
async def get_privacy_history(
    limit: int = 10,
    controller: PrivateFLController = Depends(get_privacy_controller)
):
    """
    Get privacy training history.
    
    Args:
        limit: Maximum number of history entries to return
        
    Returns:
        Privacy training history
    """
    try:
        # Get recent privacy history
        history = controller.privacy_history[-limit:] if controller.privacy_history else []
        
        # Calculate summary statistics
        total_rounds = len(controller.privacy_history)
        total_epsilon_spent = sum(event.get("epsilon_spent", 0) for event in controller.privacy_history)
        
        return {
            "success": True,
            "history": history,
            "summary": {
                "total_privacy_rounds": total_rounds,
                "total_epsilon_spent": round(total_epsilon_spent, 4),
                "privacy_level": controller.privacy_level,
                "rounds_returned": len(history)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get privacy history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get privacy history: {str(e)}")


@privacy_router.get("/budget", response_model=Dict[str, Any])
async def get_privacy_budget(controller: PrivateFLController = Depends(get_privacy_controller)):
    """
    Get current privacy budget status.
    
    Returns:
        Privacy budget information
    """
    try:
        dashboard_data = await controller.get_privacy_dashboard_data()
        budget_info = dashboard_data.get("privacy_budget", {})
        
        return {
            "success": True,
            "budget": budget_info,
            "budget_valid": budget_info.get("budget_valid", False),
            "can_train": budget_info.get("budget_valid", False),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get privacy budget: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get privacy budget: {str(e)}")


@privacy_router.post("/reset-budget", response_model=Dict[str, Any])
async def reset_privacy_budget(controller: PrivateFLController = Depends(get_privacy_controller)):
    """
    Reset privacy budget (creates new privacy engine).
    
    WARNING: This should only be used in development/testing scenarios.
    
    Returns:
        Reset confirmation
    """
    try:
        # Create new private FL controller (resets budget)
        global privacy_controller
        old_privacy_level = controller.privacy_level
        websocket_manager = controller.websocket_manager
        
        privacy_controller = create_private_fl_controller(
            websocket_manager=websocket_manager,
            privacy_level=old_privacy_level
        )
        
        logger.warning("Privacy budget has been reset - this should only be done in development")
        
        return {
            "success": True,
            "message": "Privacy budget reset",
            "warning": "Budget reset should only be used in development",
            "privacy_level": old_privacy_level,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to reset privacy budget: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reset privacy budget: {str(e)}")


@privacy_router.get("/mechanisms", response_model=Dict[str, Any])
async def get_privacy_mechanisms():
    """
    Get information about available privacy mechanisms.
    
    Returns:
        Privacy mechanisms and their descriptions
    """
    try:
        mechanisms = {
            "gaussian_mechanism": {
                "name": "Gaussian Mechanism",
                "description": "Adds calibrated Gaussian noise to provide (ε, δ)-differential privacy",
                "parameters": ["epsilon", "delta", "noise_multiplier"],
                "guarantees": "Provides formal (ε, δ)-DP guarantees"
            },
            "gradient_clipping": {
                "name": "Gradient Clipping",
                "description": "Bounds the L2 norm of gradients to limit sensitivity",
                "parameters": ["max_grad_norm"],
                "purpose": "Prerequisite for meaningful noise calibration"
            },
            "privacy_accounting": {
                "name": "Privacy Accounting",
                "description": "Tracks privacy budget consumption across training rounds",
                "features": ["composition_tracking", "budget_validation", "remaining_budget_calculation"]
            }
        }
        
        privacy_levels = {
            "strong": {"epsilon": 0.1, "delta": 1e-6, "description": "High privacy protection"},
            "moderate": {"epsilon": 1.0, "delta": 1e-5, "description": "Balanced privacy/utility"},
            "weak": {"epsilon": 5.0, "delta": 1e-4, "description": "Lower privacy protection"}
        }
        
        return {
            "success": True,
            "mechanisms": mechanisms,
            "privacy_levels": privacy_levels,
            "current_implementation": "Gaussian mechanism with gradient clipping",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get privacy mechanisms: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get privacy mechanisms: {str(e)}")


# Health check endpoint
@privacy_router.get("/health", response_model=Dict[str, Any])
async def privacy_health_check(controller: PrivateFLController = Depends(get_privacy_controller)):
    """
    Health check for privacy system.
    
    Returns:
        Privacy system health status
    """
    try:
        status = await controller.get_privacy_dashboard_data()
        
        health_status = {
            "privacy_engine_active": True,
            "privacy_level": controller.privacy_level,
            "budget_valid": status.get("privacy_budget", {}).get("budget_valid", False),
            "training_possible": status.get("privacy_budget", {}).get("budget_valid", False),
            "rounds_completed": controller.privacy_rounds_completed,
            "last_activity": controller.privacy_history[-1]["timestamp"] if controller.privacy_history else None
        }
        
        return {
            "success": True,
            "health": health_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Privacy health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Privacy health check failed: {str(e)}")


# Export router
__all__ = ["privacy_router", "set_websocket_manager"]