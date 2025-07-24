#!/usr/bin/env python3
"""
Run script for the modular Paper Search & QA System.
"""

import subprocess
import sys
import os

def main():
    """Run the modular Streamlit app"""
    try:
        # Change to the directory containing this script
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Run the modular app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app_modular.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error running app: {e}")

if __name__ == "__main__":
    main() 