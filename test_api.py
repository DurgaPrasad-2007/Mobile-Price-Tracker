"""
Simple test script to verify Mobile Price Tracker is working
"""

import requests
import json
import time

def test_system():
    """Test the Mobile Price Tracker system"""
    print("🧪 Testing Mobile Price Tracker System...")
    
    # Test data
    test_phone = {
        "battery_power": 2000,
        "blue": 1,
        "clock_speed": 2.0,
        "dual_sim": 1,
        "fc": 8,
        "four_g": 1,
        "int_memory": 64,
        "m_deep": 0.8,
        "mobile_wt": 150,
        "n_cores": 4,
        "pc": 12,
        "px_height": 1920,
        "px_width": 1080,
        "ram": 4096,
        "sc_h": 15,
        "sc_w": 8,
        "talk_time": 20,
        "three_g": 1,
        "touch_screen": 1,
        "wifi": 1
    }
    
    base_url = "http://localhost:8000"
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Status: {response.json()['status']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test 2: Frontend
    print("\n2. Testing frontend...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200 and "Mobile Price Tracker" in response.text:
            print("✅ Frontend is accessible")
        else:
            print(f"❌ Frontend test failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Frontend test failed: {e}")
    
    # Test 3: Prediction
    print("\n3. Testing prediction...")
    try:
        response = requests.post(f"{base_url}/predict", json=test_phone, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"   Price Range: {result['price_range_label']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Processing Time: {result['processing_time']:.3f}s")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
    
    print("\n🎉 Testing completed!")
    print(f"🌐 Frontend: {base_url}")
    print(f"📚 API Docs: {base_url}/docs")

if __name__ == "__main__":
    test_system()
