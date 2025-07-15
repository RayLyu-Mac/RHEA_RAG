#!/usr/bin/env python3
"""
Test script to verify the Streamlit app works correctly
"""

import sys
import os
import requests

def test_ollama_connection():
    """Test if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("✅ Ollama is running")
            print(f"   Available models: {[m['name'] for m in models]}")
            return True
        else:
            print("❌ Ollama is not responding properly")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return False

def test_file_structure():
    """Test if required files exist"""
    required_files = [
        "../vectorization_tracker.csv",
        "../VectorSpace",
        "../extracted_images"
    ]
    
    print("\n📁 Checking file structure:")
    all_good = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (missing)")
            all_good = False
    
    return all_good

def test_imports():
    """Test if all required packages are installed"""
    required_packages = [
        "streamlit",
        "langchain",
        "pandas",
        "PIL",
        "requests"
    ]
    
    print("\n📦 Checking required packages:")
    all_good = True
    
    for package in required_packages:
        try:
            if package == "PIL":
                import PIL
                print(f"   ✅ {package} (Pillow)")
            else:
                __import__(package)
                print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (not installed)")
            all_good = False
    
    return all_good

def main():
    """Run all tests"""
    print("🧪 Testing Streamlit App Setup")
    print("=" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test file structure
    files_ok = test_file_structure()
    
    # Test Ollama connection
    ollama_ok = test_ollama_connection()
    
    print("\n📊 Test Results:")
    print("=" * 40)
    print(f"   Imports: {'✅ OK' if imports_ok else '❌ FAIL'}")
    print(f"   Files: {'✅ OK' if files_ok else '❌ FAIL'}")
    print(f"   Ollama: {'✅ OK' if ollama_ok else '❌ FAIL'}")
    
    if imports_ok and files_ok and ollama_ok:
        print("\n🎉 All tests passed! You can run the app with:")
        print("   streamlit run app.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before running the app.")
        
        if not imports_ok:
            print("\n💡 To install missing packages:")
            print("   pip install -r requirements.txt")
        
        if not files_ok:
            print("\n💡 To create missing files:")
            print("   Run the vectorization process first")
        
        if not ollama_ok:
            print("\n💡 To start Ollama:")
            print("   ollama serve")

if __name__ == "__main__":
    main() 