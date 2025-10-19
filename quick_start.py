"""
Quick start script for Mobile Price Tracker
Runs both backend API and frontend server
"""

import subprocess
import time
import webbrowser
import threading
from pathlib import Path

def run_backend():
    """Run the backend API server (now includes frontend)"""
    print("🚀 Starting Mobile Price Tracker (Backend + Frontend)...")
    subprocess.run(["poetry", "run", "python", "main.py", "--mode", "serve"])

def main():
    """Main function to start both servers"""
    print("=" * 60)
    print("📱 Mobile Price Tracker - Quick Start")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ Error: Please run this script from the Mobile-Price-Tracker directory")
        return
    
    print("🔧 Starting services...")
    print("🌐 Frontend UI: http://localhost:8000")
    print("📊 Backend API: http://localhost:8000/api")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("=" * 60)
    
    # Start backend (which now includes frontend)
    try:
        run_backend()
    except KeyboardInterrupt:
        print("\n👋 Shutting down Mobile Price Tracker...")
        print("Thank you for using Mobile Price Tracker!")

if __name__ == "__main__":
    main()
