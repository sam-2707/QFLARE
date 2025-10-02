"""
QFLARE WebSocket Endpoints

FastAPI WebSocket endpoints for real-time communication.
Provides live updates for FL training, device status, and system events.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from typing import Optional
import logging
import json

from ..websocket.manager import websocket_manager, websocket_endpoint

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws")
async def general_websocket(websocket: WebSocket):
    """General purpose WebSocket endpoint."""
    await websocket_endpoint(websocket, "general")


@router.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket endpoint specifically for dashboard connections."""
    await websocket_endpoint(websocket, "dashboard")


@router.websocket("/ws/clients")
async def clients_websocket(websocket: WebSocket):
    """WebSocket endpoint for edge node clients."""
    await websocket_endpoint(websocket, "clients")


@router.websocket("/ws/admin")  
async def admin_websocket(websocket: WebSocket):
    """WebSocket endpoint for admin connections."""
    await websocket_endpoint(websocket, "admin")


@router.get("/ws/stats")
async def websocket_stats():
    """Get WebSocket connection statistics."""
    stats = websocket_manager.get_connection_stats()
    return {
        "success": True,
        "websocket_stats": stats
    }


@router.post("/ws/broadcast")
async def broadcast_message(
    event_type: str,
    message: dict,
    target_types: Optional[list] = None
):
    """
    Admin endpoint to broadcast messages to WebSocket connections.
    
    Args:
        event_type: Type of event to broadcast
        message: Message data to send
        target_types: Target connection types (default: all)
    """
    try:
        target_set = set(target_types) if target_types else None
        
        await websocket_manager.broadcast_event(
            event_type=event_type,
            data=message,
            target_types=target_set
        )
        
        return {
            "success": True,
            "message": f"Broadcasted {event_type} to {target_types or 'all'} connections"
        }
        
    except Exception as e:
        logger.error(f"Error broadcasting message: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }