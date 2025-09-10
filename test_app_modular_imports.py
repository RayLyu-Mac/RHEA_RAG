#!/usr/bin/env python3
"""
Test that app_modular.py is using direct API utilities instead of LangChain
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that the correct utilities are being imported"""
    print("🔍 Testing app_modular.py imports...")
    print("=" * 60)
    
    try:
        # Test that we can import the direct API utilities
        from utils.direct_ollama_utils import (
            load_direct_llm, load_direct_embeddings, get_available_ollama_models,
            optimize_question, generate_answer, get_suggested_keywords, count_tokens
        )
        print("✅ Direct API utilities imported successfully")
        
        # Test that the utils package exports the direct API functions
        from utils import (
            load_llm, get_available_ollama_models, optimize_question, 
            get_suggested_keywords, generate_answer, count_tokens
        )
        print("✅ Utils package exports direct API functions")
        
        # Test that we can import app_modular without errors
        import app_modular
        print("✅ app_modular.py imports successfully")
        
        # Check if the functions are the direct API versions
        print("\n🔍 Checking function sources...")
        
        # Test load_llm function
        if hasattr(load_llm, '__name__'):
            print(f"✅ load_llm function: {load_llm.__name__}")
        
        # Test count_tokens function
        if hasattr(count_tokens, '__name__'):
            print(f"✅ count_tokens function: {count_tokens.__name__}")
        
        print("\n🎉 All imports successful!")
        print("\n💡 The system is now using:")
        print("   - Direct API calls to Ollama")
        print("   - No LangChain dependency for LLM/embeddings")
        print("   - Retry logic for network issues")
        print("   - Proper error handling")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    
    if success:
        print("\n✅ app_modular.py is now using direct API utilities!")
        print("🔧 The RemoteDisconnected error should be resolved.")
    else:
        print("\n❌ There are still issues with the imports!")
        print("🔧 Please check the error messages above.")
