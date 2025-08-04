#!/usr/bin/env python3
"""
Quick test for QFLARE Vercel deployment
"""

import requests

SERVER_URL = "https://qflare-sam-2707s-projects.vercel.app"

def quick_test():
    print("🚀 Testing QFLARE Vercel Deployment")
    print("=" * 40)
    print(f"Server URL: {SERVER_URL}")
    print("=" * 40)
    
    try:
        # Test root endpoint
        print("\n🔍 Testing root endpoint...")
        response = requests.get(f"{SERVER_URL}/", timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS!")
            print(f"Response: {response.json()}")
        else:
            print("❌ FAILED")
            print(f"Error: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    quick_test() 