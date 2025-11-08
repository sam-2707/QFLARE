"""
FL API Extensions
Additional endpoints for federated learning visualization
Add these routes to your FastAPI backend
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List
import random
from datetime import datetime

router = APIRouter()

# Mock data for FL nodes (replace with real data from your FL system)
MOCK_NODES = [
    {
        "node_id": "node-001",
        "node_name": "Edge Node 1",
        "status": "training",
        "last_seen": datetime.now().isoformat(),
        "total_samples": 6000,
        "local_accuracy": 0.92,
        "local_loss": 0.23,
        "contribution_weight": 0.25,
        "rounds_participated": 8,
        "data_distribution": "Non-IID",
        "computing_power": 85,
        "network_latency": 45,
        "is_byzantine": False,
    },
    {
        "node_id": "node-002",
        "node_name": "Edge Node 2",
        "status": "active",
        "last_seen": datetime.now().isoformat(),
        "total_samples": 4500,
        "local_accuracy": 0.88,
        "local_loss": 0.31,
        "contribution_weight": 0.18,
        "rounds_participated": 7,
        "data_distribution": "Skewed",
        "computing_power": 72,
        "network_latency": 67,
        "is_byzantine": False,
    },
    {
        "node_id": "node-003",
        "node_name": "Edge Node 3",
        "status": "training",
        "last_seen": datetime.now().isoformat(),
        "total_samples": 7200,
        "local_accuracy": 0.94,
        "local_loss": 0.19,
        "contribution_weight": 0.32,
        "rounds_participated": 9,
        "data_distribution": "IID",
        "computing_power": 93,
        "network_latency": 32,
        "is_byzantine": False,
    },
    {
        "node_id": "node-004",
        "node_name": "Edge Node 4",
        "status": "error",
        "last_seen": datetime.now().isoformat(),
        "total_samples": 3800,
        "local_accuracy": 0.45,
        "local_loss": 1.87,
        "contribution_weight": 0.05,
        "rounds_participated": 3,
        "data_distribution": "Non-IID",
        "computing_power": 54,
        "network_latency": 156,
        "is_byzantine": True,
    },
    {
        "node_id": "node-005",
        "node_name": "Edge Node 5",
        "status": "idle",
        "last_seen": datetime.now().isoformat(),
        "total_samples": 5200,
        "local_accuracy": 0.90,
        "local_loss": 0.27,
        "contribution_weight": 0.20,
        "rounds_participated": 6,
        "data_distribution": "Non-IID",
        "computing_power": 78,
        "network_latency": 54,
        "is_byzantine": False,
    },
]


@router.get("/api/training/nodes")
async def get_all_nodes():
    """Get all edge nodes in the system"""
    return {"nodes": MOCK_NODES}


@router.get("/api/training/{training_id}/nodes")
async def get_training_nodes(training_id: str):
    """Get nodes participating in a specific training session"""
    # Filter to only active/training nodes for this session
    active_nodes = [n for n in MOCK_NODES if n["status"] in ["training", "active"]]
    return {"nodes": active_nodes, "training_id": training_id}


@router.get("/api/training/{training_id}/nodes/{node_id}")
async def get_node_details(training_id: str, node_id: str):
    """Get detailed metrics for a specific node"""
    node = next((n for n in MOCK_NODES if n["node_id"] == node_id), None)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    # Add detailed history for this node
    node_details = {
        **node,
        "round_history": [
            {
                "round": i,
                "accuracy": 0.5 + (i * 0.05) + random.uniform(-0.02, 0.02),
                "loss": 2.0 - (i * 0.2) + random.uniform(-0.1, 0.1),
                "samples_processed": node["total_samples"] // 10,
                "training_time": 120 + random.randint(-20, 20),
            }
            for i in range(1, 11)
        ],
    }
    return node_details


@router.get("/api/training/{training_id}/convergence")
async def get_convergence_data(training_id: str):
    """Get model convergence data across all rounds"""
    # Generate mock convergence data
    rounds = 10
    history = []
    
    for round_num in range(1, rounds + 1):
        round_data = {
            "round": round_num,
            "global_accuracy": 0.5 + (round_num * 0.045) + random.uniform(-0.01, 0.01),
            "global_loss": 2.0 - (round_num * 0.18) + random.uniform(-0.05, 0.05),
            "nodes": {},
        }
        
        # Add per-node metrics
        for node in MOCK_NODES[:3]:  # Only include training nodes
            node_id = node["node_id"]
            round_data["nodes"][node_id] = {
                "accuracy": 0.5 + (round_num * 0.04) + random.uniform(-0.03, 0.03),
                "loss": 2.0 - (round_num * 0.17) + random.uniform(-0.1, 0.1),
            }
        
        history.append(round_data)
    
    return {
        "training_id": training_id,
        "history": history,
        "total_rounds": rounds,
    }


@router.get("/api/training/{training_id}/aggregation")
async def get_aggregation_info(training_id: str):
    """Get federated aggregation process information"""
    return {
        "training_id": training_id,
        "aggregation_method": "FedAvg",
        "current_round": 8,
        "nodes_contributing": 3,
        "aggregation_status": "completed",
        "node_contributions": [
            {
                "node_id": "node-001",
                "weight": 0.35,
                "model_size_mb": 12.4,
                "gradient_norm": 0.023,
            },
            {
                "node_id": "node-002",
                "weight": 0.28,
                "model_size_mb": 12.4,
                "gradient_norm": 0.019,
            },
            {
                "node_id": "node-003",
                "weight": 0.37,
                "model_size_mb": 12.4,
                "gradient_norm": 0.021,
            },
        ],
        "privacy_metrics": {
            "epsilon_spent": 0.45,
            "delta": 1e-5,
            "noise_scale": 0.01,
        },
        "security_metrics": {
            "byzantine_detected": 1,
            "nodes_rejected": ["node-004"],
            "aggregation_method_used": "Multi-Krum",
        },
    }


@router.get("/api/training/{training_id}/round/{round_num}")
async def get_round_details(training_id: str, round_num: int):
    """Get detailed information for a specific training round"""
    return {
        "training_id": training_id,
        "round": round_num,
        "status": "completed",
        "start_time": "2025-01-10T10:30:00Z",
        "end_time": "2025-01-10T10:35:30Z",
        "duration_seconds": 330,
        "global_metrics": {
            "accuracy": 0.92,
            "loss": 0.23,
            "validation_accuracy": 0.89,
        },
        "node_updates": [
            {
                "node_id": f"node-{i:03d}",
                "local_accuracy": 0.88 + random.uniform(-0.05, 0.05),
                "local_loss": 0.25 + random.uniform(-0.03, 0.03),
                "samples_trained": 6000,
                "epochs_completed": 5,
                "training_time_seconds": 120,
            }
            for i in range(1, 4)
        ],
        "aggregation_info": {
            "method": "FedAvg",
            "nodes_aggregated": 3,
            "weights": [0.35, 0.28, 0.37],
        },
    }


# Add these routes to your main FastAPI app:
# from fl_api_extensions import router as fl_router
# app.include_router(fl_router)
