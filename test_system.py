#!/usr/bin/env python3
"""
Mobile Price Tracker - Quick System Test
A simple script to test the basic functionality of the mobile price tracker system.
"""

import sys
import time
import requests
import pandas as pd
from pathlib import Path

# Add src to path
notebook_dir = Path.cwd()
src_path = notebook_dir / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Also add the current directory to path for absolute imports
if str(notebook_dir) not in sys.path:
    sys.path.insert(0, str(notebook_dir))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    try:
        from src.data.preprocessing import get_preprocessor
        from src.models.ensemble import get_model
        from src.utils.config import get_config
        print("All imports successful")
        return True
    except ImportError as e:
        print(f"Import failed: {e}")
        return False

def test_data_loading():
    """Test data loading and preprocessing"""
    print("\nTesting data loading...")
    try:
        from src.data.preprocessing import get_preprocessor
        preprocessor = get_preprocessor()
        df = preprocessor.load_mobile_dataset()
        df_engineered = preprocessor.engineer_features(df)

        print(f"Dataset loaded: {len(df)} samples")
        print(f"Features engineered: {len(df_engineered.columns)} total features")
        return True
    except Exception as e:
        print(f"Data loading failed: {e}")
        return False

def test_model_loading():
    """Test model loading"""
    print("\nTesting model loading...")
    try:
        from src.models.ensemble import get_model
        model = get_model()
        model.load_models()

        if model.is_trained:
            print(f"Models loaded successfully: {len(model.models)} models")
            return True
        else:
            print("Models not trained")
            return False
    except Exception as e:
        print(f"Model loading failed: {e}")
        return False

def test_api_endpoints(base_url="http://localhost:8000"):
    """Test API endpoints"""
    print(f"\nTesting API endpoints at {base_url}...")

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

    # Test health
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("Health check passed")
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False

    # Test prediction
    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/predict", json=test_phone, timeout=10)
        end_time = time.time()

        if response.status_code == 200:
            result = response.json()
            processing_time = end_time - start_time
            print(f"Prediction successful in {processing_time:.3f}s")
            print(f"   Predicted range: {result['price_range']} ({result['price_range_label']})")
            print(f"   Confidence: {result['confidence']:.3f}")
            return True
        else:
            print(f"Prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Prediction error: {e}")
        return False

def test_model_prediction():
    """Test direct model prediction"""
    print("\nTesting direct model prediction...")
    try:
        from src.data.preprocessing import get_preprocessor
        from src.models.ensemble import get_model

        # Load components
        preprocessor = get_preprocessor()
        model = get_model()
        model.load_models()

        if not model.is_trained:
            print("Model not trained")
            return False

        # Load data and make prediction
        df = preprocessor.load_mobile_dataset()
        df_engineered = preprocessor.engineer_features(df)

        # Test on first sample
        sample = df_engineered.iloc[0:1].drop('price_range', axis=1)
        prediction = model.predict(sample)[0]
        probabilities = model.predict_proba(sample)[0]
        confidence = max(probabilities)

        print("Direct prediction successful")
        print(f"   Predicted range: {prediction}")
        print(f"   Confidence: {confidence:.3f}")
        return True

    except Exception as e:
        print(f"Direct prediction failed: {e}")
        return False

def run_all_tests():
    """Run all system tests"""
    print("Mobile Price Tracker - System Test")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Model Loading", test_model_loading),
        ("Direct Model Prediction", test_model_prediction),
        ("API Endpoints", test_api_endpoints)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("All tests passed! System is working correctly.")
        print("\nTo start the full application:")
        print("  poetry run devrun")
        print("\nTo access the web interface:")
        print("  http://localhost:8000")
    else:
        print("Some tests failed. Check the output above for details.")
        print("\nTroubleshooting:")
        print("- Make sure models are trained: python main.py --mode train")
        print("- Make sure API is running: python main.py --mode serve")
        print("- Check data directory exists and contains processed dataset")

    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
