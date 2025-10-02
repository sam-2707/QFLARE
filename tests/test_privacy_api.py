"""
Test Differential Privacy API Endpoints
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import pytest
from fastapi.testclient import TestClient
from server.main import app

# Create test client
client = TestClient(app)

def test_privacy_status_endpoint():
    """Test privacy status endpoint."""
    print("Testing privacy status endpoint...")
    
    response = client.get("/api/privacy/status")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "success" in data
    assert "data" in data
    assert "timestamp" in data
    
    print("✓ Privacy status endpoint works")

def test_privacy_dashboard_endpoint():
    """Test privacy dashboard endpoint."""
    print("Testing privacy dashboard endpoint...")
    
    response = client.get("/api/privacy/dashboard")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "success" in data
    assert "dashboard_data" in data
    assert "timestamp" in data
    
    dashboard = data["dashboard_data"]
    assert "privacy_overview" in dashboard
    assert "privacy_parameters" in dashboard
    assert "privacy_budget" in dashboard
    
    print("✓ Privacy dashboard endpoint works")

def test_privacy_configure_endpoint():
    """Test privacy configuration endpoint."""
    print("Testing privacy configure endpoint...")
    
    config_data = {
        "privacy_level": "moderate"
    }
    
    response = client.post("/api/privacy/configure", json=config_data)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "success" in data
    assert data["success"] is True
    assert "message" in data
    
    print("✓ Privacy configure endpoint works")

def test_privacy_validate_parameters_endpoint():
    """Test privacy parameter validation endpoint."""
    print("Testing privacy parameter validation endpoint...")
    
    validation_data = {
        "epsilon": 1.0,
        "delta": 1e-5
    }
    
    response = client.post("/api/privacy/validate-parameters", json=validation_data)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "success" in data
    assert "validation" in data
    assert "parameters" in data
    
    validation = data["validation"]
    assert "valid" in validation
    assert "privacy_strength" in validation
    
    print("✓ Privacy parameter validation endpoint works")

def test_privacy_budget_endpoint():
    """Test privacy budget endpoint."""
    print("Testing privacy budget endpoint...")
    
    response = client.get("/api/privacy/budget")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "success" in data
    assert "budget" in data
    assert "budget_valid" in data
    assert "can_train" in data
    
    print("✓ Privacy budget endpoint works")

def test_privacy_history_endpoint():
    """Test privacy history endpoint."""
    print("Testing privacy history endpoint...")
    
    response = client.get("/api/privacy/history?limit=5")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "success" in data
    assert "history" in data
    assert "summary" in data
    
    summary = data["summary"]
    assert "total_privacy_rounds" in summary
    assert "privacy_level" in summary
    
    print("✓ Privacy history endpoint works")

def test_privacy_mechanisms_endpoint():
    """Test privacy mechanisms endpoint."""
    print("Testing privacy mechanisms endpoint...")
    
    response = client.get("/api/privacy/mechanisms")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "success" in data
    assert "mechanisms" in data
    assert "privacy_levels" in data
    assert "current_implementation" in data
    
    mechanisms = data["mechanisms"]
    assert "gaussian_mechanism" in mechanisms
    assert "gradient_clipping" in mechanisms
    assert "privacy_accounting" in mechanisms
    
    print("✓ Privacy mechanisms endpoint works")

def test_privacy_health_endpoint():
    """Test privacy health check endpoint."""
    print("Testing privacy health check endpoint...")
    
    response = client.get("/api/privacy/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "success" in data
    assert "health" in data
    
    health = data["health"]
    assert "privacy_engine_active" in health
    assert "privacy_level" in health
    assert "budget_valid" in health
    
    print("✓ Privacy health check endpoint works")

def test_privacy_training_round_endpoint():
    """Test privacy training round endpoint."""
    print("Testing privacy training round endpoint...")
    
    training_data = {
        "num_clients": 3,
        "client_fraction": 1.0,
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.01
    }
    
    response = client.post("/api/privacy/training-round", json=training_data)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "success" in data
    assert "message" in data
    assert "round_id" in data
    assert "configuration" in data
    
    print("✓ Privacy training round endpoint works")

def test_invalid_privacy_level():
    """Test invalid privacy level handling."""
    print("Testing invalid privacy level handling...")
    
    config_data = {
        "privacy_level": "invalid_level"
    }
    
    response = client.post("/api/privacy/configure", json=config_data)
    
    assert response.status_code == 400
    
    print("✓ Invalid privacy level handling works")

def test_invalid_privacy_parameters():
    """Test invalid privacy parameter handling."""
    print("Testing invalid privacy parameter handling...")
    
    validation_data = {
        "epsilon": -1.0,  # Invalid negative epsilon
        "delta": 1e-5
    }
    
    response = client.post("/api/privacy/validate-parameters", json=validation_data)
    
    assert response.status_code == 200
    data = response.json()
    
    validation = data["validation"]
    assert validation["valid"] is False
    assert len(validation["warnings"]) > 0
    
    print("✓ Invalid privacy parameter handling works")

def run_privacy_api_tests():
    """Run all privacy API tests."""
    print("="*60)
    print("RUNNING PRIVACY API TESTS")
    print("="*60)
    
    try:
        # Test basic endpoints
        test_privacy_status_endpoint()
        test_privacy_dashboard_endpoint()
        test_privacy_budget_endpoint()
        test_privacy_history_endpoint()
        test_privacy_mechanisms_endpoint()
        test_privacy_health_endpoint()
        
        # Test configuration endpoints
        test_privacy_configure_endpoint()
        test_privacy_validate_parameters_endpoint()
        
        # Test training endpoint
        test_privacy_training_round_endpoint()
        
        # Test error handling
        test_invalid_privacy_level()
        test_invalid_privacy_parameters()
        
        print("="*60)
        print("✅ ALL PRIVACY API TESTS PASSED!")
        print("="*60)
        print("Summary:")
        print("- Privacy Status & Dashboard: ✓")
        print("- Privacy Configuration: ✓")
        print("- Privacy Budget Management: ✓")
        print("- Privacy Training Integration: ✓")
        print("- Error Handling: ✓") 
        print("="*60)
        
        return True
        
    except Exception as e:
        print("="*60)
        print("❌ PRIVACY API TESTS FAILED!")
        print(f"Error: {str(e)}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run_privacy_api_tests()