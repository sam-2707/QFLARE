"""
QFLARE WebSocket Package

Real-time communication module for federated learning updates.
Provides WebSocket connections for dashboard, clients, and admin interfaces.
"""

from .manager import (
    WebSocketManager,
    websocket_manager,
    broadcast_fl_status_update,
    broadcast_training_progress,
    broadcast_model_aggregation,
    broadcast_device_status,
    broadcast_error_notification,
    websocket_endpoint,
    handle_websocket_message
)

__all__ = [
    "WebSocketManager",
    "websocket_manager",
    "broadcast_fl_status_update",
    "broadcast_training_progress", 
    "broadcast_model_aggregation",
    "broadcast_device_status",
    "broadcast_error_notification",
    "websocket_endpoint",
    "handle_websocket_message"
]