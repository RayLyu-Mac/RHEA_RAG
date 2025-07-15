#!/usr/bin/env python3
"""
Simple script to run the Streamlit Paper Search & QA System
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app"""
    
    print("🚀 Starting Paper Search & QA System")
    print("=" * 50)
    
    # Check if we're in the GUI directory
    if not os.path.exists("app.py"):
        print("❌ app.py not found. Please run this script from the GUI directory.")
        return
    
    # Run the Streamlit app
    try:
        print("🌐 Starting Streamlit server...")
        print("📱 The app will open in your browser automatically")
        print("🔗 If it doesn't open, go to: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "false"])
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down the server...")
    except Exception as e:
        print(f"❌ Error running the app: {e}")

if __name__ == "__main__":
    main() 