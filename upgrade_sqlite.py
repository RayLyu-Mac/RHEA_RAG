#!/usr/bin/env python3
"""
Script to help upgrade SQLite to a version compatible with Chroma
"""

import sys
import subprocess
import os

def check_current_sqlite():
    """Check current SQLite version"""
    try:
        import sqlite3
        version = sqlite3.sqlite_version
        print(f"📊 Current SQLite version: {version}")
        
        # Parse version
        parts = [int(x) for x in version.split('.')]
        if parts[0] > 3 or (parts[0] == 3 and parts[1] >= 35):
            print("✅ SQLite version is sufficient for Chroma (≥ 3.35.0)")
            return True
        else:
            print("❌ SQLite version is too old for Chroma")
            return False
    except Exception as e:
        print(f"❌ Error checking SQLite: {e}")
        return False

def try_pysqlite3_binary():
    """Try installing pysqlite3-binary to get a newer SQLite"""
    print("\n🔧 Attempting to install pysqlite3-binary...")
    try:
        # Try to install pysqlite3-binary
        subprocess.run([sys.executable, "-m", "pip", "install", "pysqlite3-binary"], 
                      check=True, capture_output=True, text=True)
        print("✅ pysqlite3-binary installed successfully")
        
        # Test if it works
        try:
            import pysqlite3
            print(f"✅ pysqlite3 version: {pysqlite3.sqlite_version}")
            return True
        except ImportError:
            print("❌ pysqlite3 import failed")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install pysqlite3-binary: {e}")
        return False

def try_conda_sqlite():
    """Try using conda to get a newer SQLite"""
    print("\n🔧 Attempting conda SQLite upgrade...")
    try:
        # Check if conda is available
        result = subprocess.run(["conda", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Conda not available")
            return False
            
        # Try to install newer SQLite via conda
        subprocess.run(["conda", "install", "-c", "conda-forge", "sqlite>=3.35.0", "-y"], 
                      check=True, capture_output=True, text=True)
        print("✅ Conda SQLite upgrade completed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Conda upgrade failed: {e}")
        return False

def try_system_upgrade():
    """Provide instructions for system-level SQLite upgrade"""
    print("\n🔧 System-level SQLite upgrade options:")
    
    import platform
    system = platform.system().lower()
    
    if system == "linux":
        print("📋 For Ubuntu/Debian:")
        print("   sudo apt update")
        print("   sudo apt install sqlite3")
        print("\n📋 For CentOS/RHEL/Fedora:")
        print("   sudo yum install sqlite  # or sudo dnf install sqlite")
        print("\n📋 For Arch Linux:")
        print("   sudo pacman -S sqlite")
        
    elif system == "darwin":  # macOS
        print("📋 Using Homebrew:")
        print("   brew install sqlite")
        print("\n📋 Using MacPorts:")
        print("   sudo port install sqlite3")
        
    elif system == "windows":
        print("📋 Download from SQLite website:")
        print("   https://www.sqlite.org/download.html")
        print("   Or use Chocolatey: choco install sqlite")
        
    print("\n💡 After system upgrade, restart your Python environment")

def create_sqlite_patch():
    """Create a patch to force Chroma to use a different SQLite"""
    print("\n🔧 Creating SQLite compatibility patch...")
    
    patch_code = '''
# SQLite compatibility patch
import os
import sys

# Try to use pysqlite3 if available
try:
    import pysqlite3
    # Replace the built-in sqlite3 with pysqlite3
    sys.modules['sqlite3'] = pysqlite3
    print("✅ Using pysqlite3 for SQLite compatibility")
except ImportError:
    print("⚠️ pysqlite3 not available, using system SQLite")

# This should be imported before any Chroma imports
'''
    
    try:
        with open("sqlite_patch.py", "w") as f:
            f.write(patch_code)
        print("✅ Created sqlite_patch.py")
        print("💡 Import this file at the top of your main script")
        return True
    except Exception as e:
        print(f"❌ Failed to create patch: {e}")
        return False

def main():
    """Main function"""
    print("🔬 SQLite Upgrade Helper for Chroma Compatibility")
    print("=" * 60)
    
    # Check current version
    if check_current_sqlite():
        print("\n🎉 Your SQLite version is already compatible!")
        return True
    
    print("\n" + "=" * 60)
    print("🔧 Attempting automatic fixes...")
    
    # Try pysqlite3-binary first
    if try_pysqlite3_binary():
        print("\n✅ pysqlite3-binary installed successfully!")
        print("💡 Restart your Python environment and try again")
        return True
    
    # Try conda if available
    if try_conda_sqlite():
        print("\n✅ Conda SQLite upgrade completed!")
        print("💡 Restart your Python environment and try again")
        return True
    
    # Create compatibility patch
    create_sqlite_patch()
    
    # Provide manual instructions
    print("\n" + "=" * 60)
    print("📋 Manual upgrade instructions:")
    try_system_upgrade()
    
    print("\n" + "=" * 60)
    print("💡 Alternative solutions:")
    print("1. Use a different vector store (FAISS, Pinecone, etc.)")
    print("2. Use Chroma in a Docker container with newer SQLite")
    print("3. Use a cloud-based vector store service")
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 