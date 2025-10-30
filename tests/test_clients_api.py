#!/usr/bin/env python3
"""
Quick test to verify clients API endpoint
"""

import requests
import json

def test_clients_endpoint():
    """Test the /api/clients endpoint"""
    try:
        response = requests.get("http://localhost:8000/api/clients")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Response structure:")
            print(json.dumps(data, indent=2))
            
            if "clients" in data:
                clients = data["clients"]
                print(f"\nFound {len(clients)} clients")
                for i, client in enumerate(clients):
                    print(f"Client {i+1}: {client.get('client_id', 'unknown')} - Status: {client.get('status', 'unknown')}")
            else:
                print("No 'clients' key in response")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_clients_endpoint()