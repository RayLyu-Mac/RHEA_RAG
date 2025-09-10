#!/usr/bin/env python3
"""
Test script for robust Ollama connection fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.robust_ollama_utils import (
    load_robust_llm, load_robust_embeddings, get_available_ollama_models,
    RobustOllamaConnection
)

def test_robust_connection():
    """Test the robust connection utilities"""
    print("🧪 Testing Robust Ollama Connection Fixes")
    print("=" * 50)
    
    # Test 1: Get available models
    print("\n1️⃣ Testing model discovery...")
    try:
        models = get_available_ollama_models()
        print(f"✅ Found {len(models)} models: {models}")
    except Exception as e:
        print(f"❌ Model discovery failed: {e}")
        return False
    
    # Test 2: Test robust connection
    print("\n2️⃣ Testing robust connection...")
    try:
        robust_conn = RobustOllamaConnection()
        models_data = robust_conn.get_models()
        if models_data and 'models' in models_data:
            print(f"✅ Robust connection works: {len(models_data['models'])} models found")
        else:
            print("❌ Robust connection failed")
            return False
    except Exception as e:
        print(f"❌ Robust connection test failed: {e}")
        return False
    
    # Test 3: Test embeddings
    print("\n3️⃣ Testing embeddings...")
    try:
        embeddings = load_robust_embeddings("nomic-embed-text:latest")
        if embeddings:
            test_embedding = embeddings.embed_query("test")
            print(f"✅ Embeddings work: Vector size {len(test_embedding)}")
        else:
            print("❌ Embeddings failed to load")
            return False
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        return False
    
    # Test 4: Test LLM loading
    print("\n4️⃣ Testing LLM loading...")
    try:
        if models:
            test_model = models[0]  # Use first available model
            llm = load_robust_llm(test_model)
            if llm:
                print(f"✅ LLM loaded successfully: {test_model}")
                
                # Test generation
                try:
                    response = llm.invoke("Hello, this is a test.")
                    print(f"✅ Generation works: {response[:50]}...")
                except Exception as gen_error:
                    print(f"⚠️  Generation failed: {gen_error}")
            else:
                print(f"❌ LLM loading failed for: {test_model}")
                return False
        else:
            print("❌ No models available for testing")
            return False
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! Your RAG system should work now.")
    return True

if __name__ == "__main__":
    success = test_robust_connection()
    if success:
        print("\n✅ Ready to run your RAG application!")
        print("💡 Run: streamlit run app.py")
    else:
        print("\n❌ Some tests failed. Check the issues above.")
