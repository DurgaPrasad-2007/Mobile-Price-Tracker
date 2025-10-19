"""
Simple system test for Mobile Price Tracker
"""

import requests
import json
import time
from pathlib import Path


def test_api_health():
    """Test API health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("✅ Health check passed")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_frontend():
    """Test frontend is being served"""
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        assert response.status_code == 200
        content = response.text
        assert "Mobile Price Tracker" in content
        assert "mobilePriceTracker()" in content
        print("✅ Frontend is being served correctly")
        return True
    except Exception as e:
        print(f"❌ Frontend test failed: {e}")
        return False


def test_single_prediction():
    """Test single prediction endpoint"""
    test_data = {
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
    
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=test_data,
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert "price_range" in data
        assert "confidence" in data
        assert "price_range_label" in data
        print(f"✅ Single prediction passed: {data['price_range_label']} (confidence: {data['confidence']:.3f})")
        return True
    except Exception as e:
        print(f"❌ Single prediction failed: {e}")
        return False


def test_batch_prediction():
    """Test batch prediction endpoint"""
    test_data = [
        {
            "battery_power": 1500,
            "blue": 0,
            "clock_speed": 1.0,
            "dual_sim": 0,
            "fc": 2,
            "four_g": 0,
            "int_memory": 8,
            "m_deep": 1.0,
            "mobile_wt": 200,
            "n_cores": 2,
            "pc": 5,
            "px_height": 480,
            "px_width": 320,
            "ram": 512,
            "sc_h": 10,
            "sc_w": 6,
            "talk_time": 10,
            "three_g": 1,
            "touch_screen": 1,
            "wifi": 0
        },
        {
            "battery_power": 3000,
            "blue": 1,
            "clock_speed": 2.5,
            "dual_sim": 1,
            "fc": 16,
            "four_g": 1,
            "int_memory": 128,
            "m_deep": 0.7,
            "mobile_wt": 180,
            "n_cores": 8,
            "pc": 24,
            "px_height": 2160,
            "px_width": 1440,
            "ram": 8192,
            "sc_h": 16,
            "sc_w": 9,
            "talk_time": 25,
            "three_g": 1,
            "touch_screen": 1,
            "wifi": 1
        }
    ]
    
    try:
        response = requests.post(
            "http://localhost:8000/predict-batch",
            json=test_data,
            timeout=15
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        for result in data:
            assert "price_range" in result
            assert "confidence" in result
        print(f"✅ Batch prediction passed: {len(data)} predictions")
        return True
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
        return False


def test_metrics():
    """Test metrics endpoint"""
    try:
        response = requests.get("http://localhost:8000/metrics", timeout=5)
        assert response.status_code == 200
        print("✅ Metrics endpoint accessible")
        return True
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False


def test_stats():
    """Test stats endpoint"""
    try:
        response = requests.get("http://localhost:8000/stats", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        print("✅ Stats endpoint accessible")
        return True
    except Exception as e:
        print(f"❌ Stats test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Starting Mobile Price Tracker System Tests...")
    print("=" * 50)
    
    # Wait for API to be ready
    print("⏳ Waiting for API to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("✅ API is ready!")
                break
        except:
            if i == max_retries - 1:
                print("❌ API not ready after 30 seconds")
                return False
            time.sleep(1)
    
    # Run tests
    tests = [
        test_api_health,
        test_frontend,
        test_single_prediction,
        test_batch_prediction,
        test_metrics,
        test_stats
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)