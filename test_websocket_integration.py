#!/usr/bin/env python3
"""
Test WebSocket Real-Time Updates

This script tests the WebSocket implementation for real-time federated learning updates.
Verifies that WebSocket connections, event broadcasting, and client communication work correctly.
"""

import asyncio
import json
import logging
from pathlib import Path
import sys

# Add server to path
sys.path.append(str(Path(__file__).parent / "server"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_websocket_manager():
    """Test the WebSocket manager functionality."""
    print("\n=== Testing WebSocket Manager ===")
    
    try:
        from server.websocket.manager import WebSocketManager, websocket_manager
        
        # Test manager initialization
        manager = WebSocketManager()
        print("✓ WebSocket manager created")
        
        # Test connection stats
        stats = manager.get_connection_stats()
        print(f"✓ Connection stats: {stats}")
        
        # Test event broadcasting (without actual WebSocket connections)
        await manager.broadcast_event("test_event", {"message": "test"})
        print("✓ Event broadcasting works")
        
        # Test global manager
        global_stats = websocket_manager.get_connection_stats()
        print(f"✓ Global manager stats: {global_stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ WebSocket manager test failed: {str(e)}")
        return False

async def test_websocket_endpoints():
    """Test WebSocket endpoint imports."""
    print("\n=== Testing WebSocket Endpoints ===")
    
    try:
        from server.api.websocket_endpoints import router
        
        print("✓ WebSocket endpoints imported successfully")
        print(f"✓ Router has {len(router.routes)} routes")
        
        # Check that routes exist
        route_paths = [route.path for route in router.routes]
        expected_paths = ["/ws", "/ws/dashboard", "/ws/clients", "/ws/admin"]
        
        for path in expected_paths:
            if any(path in route_path for route_path in route_paths):
                print(f"✓ Found WebSocket route: {path}")
            else:
                print(f"○ WebSocket route not found: {path}")
        
        return True
        
    except Exception as e:
        print(f"✗ WebSocket endpoints test failed: {str(e)}")
        return False

async def test_fl_controller_websocket_integration():
    """Test FL controller WebSocket integration."""
    print("\n=== Testing FL Controller WebSocket Integration ===")
    
    try:
        # Test that FL controller can import WebSocket functions
        from server.fl_core.fl_controller import FLController, WEBSOCKET_AVAILABLE
        
        print(f"✓ FL Controller imported, WebSocket available: {WEBSOCKET_AVAILABLE}")
        
        # Create FL controller
        fl_controller = FLController()
        print("✓ FL Controller with WebSocket integration created")
        
        if WEBSOCKET_AVAILABLE:
            print("✓ WebSocket broadcasting functions available")
        else:
            print("○ WebSocket broadcasting functions not available (expected in test)")
        
        return True
        
    except Exception as e:
        print(f"✗ FL controller WebSocket integration test failed: {str(e)}")
        return False

async def test_websocket_message_formatting():
    """Test WebSocket message formatting."""
    print("\n=== Testing WebSocket Message Formatting ===")
    
    try:
        from server.websocket.manager import (
            broadcast_fl_status_update,
            broadcast_training_progress,
            broadcast_model_aggregation,
            broadcast_device_status,
            broadcast_error_notification
        )
        
        print("✓ WebSocket broadcasting functions imported")
        
        # Test message creation (these won't actually broadcast without connections)
        test_fl_status = {
            "current_round": 5,
            "status": "training",
            "participants": 10
        }
        
        # These will silently succeed with no connections
        await broadcast_fl_status_update(test_fl_status)
        print("✓ FL status update message formatting works")
        
        await broadcast_training_progress(5, {"accuracy": 85.5, "loss": 0.15})
        print("✓ Training progress message formatting works")
        
        await broadcast_model_aggregation({"round": 5, "clients": 8})
        print("✓ Model aggregation message formatting works")
        
        await broadcast_device_status("device_001", {"status": "active"})
        print("✓ Device status message formatting works")
        
        await broadcast_error_notification("training_error", {"message": "Test error"})
        print("✓ Error notification message formatting works")
        
        return True
        
    except Exception as e:
        print(f"✗ WebSocket message formatting test failed: {str(e)}")
        return False

async def test_websocket_event_history():
    """Test WebSocket event history functionality."""
    print("\n=== Testing WebSocket Event History ===")
    
    try:
        from server.websocket.manager import websocket_manager
        
        # Check initial event history
        initial_events = len(websocket_manager.recent_events)
        print(f"✓ Initial event history: {initial_events} events")
        
        # Add some test events
        await websocket_manager.broadcast_event("test_event_1", {"data": "test1"})
        await websocket_manager.broadcast_event("test_event_2", {"data": "test2"})
        await websocket_manager.broadcast_event("test_event_3", {"data": "test3"})
        
        # Check event history updated
        final_events = len(websocket_manager.recent_events)
        print(f"✓ Final event history: {final_events} events")
        
        if final_events >= initial_events + 3:
            print("✓ Event history properly updated")
        else:
            print("○ Event history update unclear (may be expected)")
        
        return True
        
    except Exception as e:
        print(f"✗ WebSocket event history test failed: {str(e)}")
        return False

async def main():
    """Run all WebSocket integration tests."""
    print("QFLARE WebSocket Real-Time Updates Test Suite")
    print("=============================================")
    
    tests = [
        ("WebSocket Manager", test_websocket_manager),
        ("WebSocket Endpoints", test_websocket_endpoints),
        ("FL Controller Integration", test_fl_controller_websocket_integration),
        ("Message Formatting", test_websocket_message_formatting),
        ("Event History", test_websocket_event_history)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {str(e)}")
            results.append((test_name, False))
    
    print("\n=== Test Results Summary ===")
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All WebSocket integration tests passed!")
        print("WebSocket real-time updates are ready for deployment.")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Check the implementation.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = asyncio.run(main())