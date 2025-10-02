"""
QFLARE WebSocket Manager

This module implements real-time WebSocket communication for live updates
to the federated learning dashboard and client applications.
"""

import json
import logging
import asyncio
from typing import Dict, Set, Any, Optional
from datetime import datetime
import weakref

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections for real-time updates.
    Supports multiple connection types: dashboard, clients, admin.
    """
    
    def __init__(self):
        # Store connections by type
        self.connections: Dict[str, Set[WebSocket]] = {
            "dashboard": set(),
            "clients": set(), 
            "admin": set(),
            "general": set()
        }
        
        # Store connection metadata
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = weakref.WeakKeyDictionary()
        
        # Event history for new connections
        self.recent_events = []
        self.max_recent_events = 50
        
        logger.info("WebSocket Manager initialized")
    
    async def connect(self, websocket: WebSocket, connection_type: str = "general", 
                     client_info: Optional[Dict[str, Any]] = None):
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: WebSocket instance
            connection_type: Type of connection (dashboard, clients, admin, general)
            client_info: Additional client information
        """
        await websocket.accept()
        
        # Add to appropriate connection set
        if connection_type not in self.connections:
            connection_type = "general"
        
        self.connections[connection_type].add(websocket)
        
        # Store connection metadata
        self.connection_info[websocket] = {
            "type": connection_type,
            "connected_at": datetime.now().isoformat(),
            "client_info": client_info or {}
        }
        
        logger.info(f"WebSocket connected: {connection_type}, total: {self.get_connection_count()}")
        
        # Send recent events to new connection
        await self._send_recent_events(websocket)
        
        # Notify about new connection
        await self.broadcast_event("connection", {
            "action": "connected",
            "connection_type": connection_type,
            "total_connections": self.get_connection_count(),
            "timestamp": datetime.now().isoformat()
        }, exclude_types=[connection_type])
    
    def disconnect(self, websocket: WebSocket):
        """
        Remove a WebSocket connection.
        
        Args:
            websocket: WebSocket instance to remove
        """
        connection_type = "general"
        
        # Find and remove from appropriate set
        for conn_type, conn_set in self.connections.items():
            if websocket in conn_set:
                conn_set.remove(websocket)
                connection_type = conn_type
                break
        
        # Remove from metadata
        if websocket in self.connection_info:
            del self.connection_info[websocket]
        
        logger.info(f"WebSocket disconnected: {connection_type}, total: {self.get_connection_count()}")
    
    async def send_personal_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """
        Send a message to a specific WebSocket connection.
        
        Args:
            websocket: Target WebSocket
            message: Message to send
        """
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending personal message: {str(e)}")
                self.disconnect(websocket)
    
    async def broadcast_to_type(self, connection_type: str, message: Dict[str, Any]):
        """
        Broadcast a message to all connections of a specific type.
        
        Args:
            connection_type: Target connection type
            message: Message to broadcast
        """
        if connection_type not in self.connections:
            return
        
        disconnected = []
        for websocket in self.connections[connection_type].copy():
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps(message))
                else:
                    disconnected.append(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting to {connection_type}: {str(e)}")
                disconnected.append(websocket)
        
        # Clean up disconnected sockets
        for websocket in disconnected:
            self.disconnect(websocket)
    
    async def broadcast_event(self, event_type: str, data: Dict[str, Any], 
                            target_types: Optional[Set[str]] = None,
                            exclude_types: Optional[Set[str]] = None):
        """
        Broadcast an event to specified connection types.
        
        Args:
            event_type: Type of event (fl_status, training_update, etc.)
            data: Event data
            target_types: Only send to these connection types (None = all)
            exclude_types: Don't send to these connection types
        """
        message = {
            "event": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to recent events
        self.recent_events.append(message)
        if len(self.recent_events) > self.max_recent_events:
            self.recent_events.pop(0)
        
        # Determine target types
        if target_types is None:
            target_types = set(self.connections.keys())
        
        if exclude_types:
            target_types = target_types - exclude_types
        
        # Broadcast to target types
        for connection_type in target_types:
            await self.broadcast_to_type(connection_type, message)
        
        logger.debug(f"Broadcasted {event_type} event to {len(target_types)} connection types")
    
    async def _send_recent_events(self, websocket: WebSocket):
        """Send recent events to a newly connected client."""
        if self.recent_events:
            recent_message = {
                "event": "recent_events",
                "data": {
                    "events": self.recent_events[-10:],  # Last 10 events
                    "total_recent": len(self.recent_events)
                },
                "timestamp": datetime.now().isoformat()
            }
            await self.send_personal_message(websocket, recent_message)
    
    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return sum(len(conn_set) for conn_set in self.connections.values())
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get detailed connection statistics."""
        stats = {
            "total_connections": self.get_connection_count(),
            "by_type": {
                conn_type: len(conn_set) 
                for conn_type, conn_set in self.connections.items()
            },
            "recent_events": len(self.recent_events)
        }
        return stats
    
    async def ping_all_connections(self):
        """Send ping to all connections to check health."""
        ping_message = {
            "event": "ping",
            "data": {"timestamp": datetime.now().isoformat()},
            "timestamp": datetime.now().isoformat()
        }
        
        for connection_type in self.connections:
            await self.broadcast_to_type(connection_type, ping_message)


# Global WebSocket manager instance
websocket_manager = WebSocketManager()


# Convenience functions for FL events
async def broadcast_fl_status_update(fl_status: Dict[str, Any]):
    """Broadcast federated learning status update."""
    await websocket_manager.broadcast_event("fl_status_update", fl_status, 
                                          target_types={"dashboard", "admin"})


async def broadcast_training_progress(round_number: int, progress_data: Dict[str, Any]):
    """Broadcast training progress update."""
    await websocket_manager.broadcast_event("training_progress", {
        "round_number": round_number,
        **progress_data
    }, target_types={"dashboard", "admin", "clients"})


async def broadcast_model_aggregation(aggregation_results: Dict[str, Any]):
    """Broadcast model aggregation completion."""
    await websocket_manager.broadcast_event("model_aggregation", aggregation_results,
                                          target_types={"dashboard", "admin"})


async def broadcast_device_status(device_id: str, status_data: Dict[str, Any]):
    """Broadcast device status change."""
    await websocket_manager.broadcast_event("device_status", {
        "device_id": device_id,
        **status_data
    }, target_types={"dashboard", "admin"})


async def broadcast_error_notification(error_type: str, error_data: Dict[str, Any]):
    """Broadcast error notification."""
    await websocket_manager.broadcast_event("error_notification", {
        "error_type": error_type,
        **error_data
    }, target_types={"dashboard", "admin"})


# WebSocket connection handler
async def websocket_endpoint(websocket: WebSocket, connection_type: str = "general"):
    """
    Main WebSocket endpoint handler.
    
    Args:
        websocket: WebSocket connection
        connection_type: Type of connection (dashboard, clients, admin)
    """
    await websocket_manager.connect(websocket, connection_type)
    
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await handle_websocket_message(websocket, message)
            except json.JSONDecodeError:
                await websocket_manager.send_personal_message(websocket, {
                    "event": "error",
                    "data": {"message": "Invalid JSON format"},
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        websocket_manager.disconnect(websocket)


async def handle_websocket_message(websocket: WebSocket, message: Dict[str, Any]):
    """
    Handle incoming WebSocket messages from clients.
    
    Args:
        websocket: Source WebSocket
        message: Parsed message data
    """
    message_type = message.get("type", "unknown")
    
    if message_type == "ping":
        # Respond to ping
        await websocket_manager.send_personal_message(websocket, {
            "event": "pong",
            "data": {"timestamp": datetime.now().isoformat()},
            "timestamp": datetime.now().isoformat()
        })
    
    elif message_type == "subscribe":
        # Handle event subscription
        events = message.get("events", [])
        await websocket_manager.send_personal_message(websocket, {
            "event": "subscription_confirmed",
            "data": {"subscribed_events": events},
            "timestamp": datetime.now().isoformat()
        })
    
    elif message_type == "get_stats":
        # Send connection statistics
        stats = websocket_manager.get_connection_stats()
        await websocket_manager.send_personal_message(websocket, {
            "event": "connection_stats",
            "data": stats,
            "timestamp": datetime.now().isoformat()
        })
    
    else:
        logger.warning(f"Unknown WebSocket message type: {message_type}")


if __name__ == "__main__":
    # Test the WebSocket manager
    print("Testing WebSocket Manager...")
    
    manager = WebSocketManager()
    print(f"Initial connection count: {manager.get_connection_count()}")
    
    # Test connection stats
    stats = manager.get_connection_stats()
    print(f"Connection stats: {stats}")
    
    print("WebSocket Manager test completed!")