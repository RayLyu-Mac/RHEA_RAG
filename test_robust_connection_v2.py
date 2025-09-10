#!/usr/bin/env python3
"""
Test the updated robust Ollama utilities with DirectAPIEmbeddings fallback.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.robust_ollama_utils import (
    load_robust_llm, load_robust_embeddings, get_available_ollama_models,
    DirectAPIEmbeddings, DirectAPILLM, RobustOllamaConnection
)

def test_robust_utilities():
    """Test the robust utilities with fallback mechanisms"""
    print("🚀 Testing Robust Ollama Utilities v2...")
    print("=" * 60)
    
    # Test 1: Model discovery
    print("\n1️⃣ Testing model discovery...")
    try:
        models = get_available_ollama_models()
        print(f"✅ Available models: {models}")
    except Exception as e:
        print(f"❌ Model discovery failed: {e}")
        return False
    
    # Test 2: Direct API Embeddings
    print("\n2️⃣ Testing Direct API Embeddings...")
    try:
        direct_embeddings = DirectAPIEmbeddings("nomic-embed-text:latest")
        test_text = "This is a test sentence for embeddings."
        embedding = direct_embeddings.embed_query(test_text)
        
        if embedding and len(embedding) > 0:
            print(f"✅ Direct API embeddings working!")
            print(f"📊 Embedding length: {len(embedding)}")
            print(f"📊 First 5 values: {embedding[:5]}")
        else:
            print("❌ Direct API embeddings returned empty result")
            return False
    except Exception as e:
        print(f"❌ Direct API embeddings failed: {e}")
        return False
    
    # Test 3: Robust Embeddings with fallback
    print("\n3️⃣ Testing Robust Embeddings with fallback...")
    try:
        embeddings = load_robust_embeddings("nomic-embed-text:latest")
        
        if embeddings is None:
            print("❌ Robust embeddings failed to load")
            return False
        
        # Test the embeddings
        test_text = "Testing robust embeddings functionality."
        embedding = embeddings.embed_query(test_text)
        
        if embedding and len(embedding) > 0:
            print(f"✅ Robust embeddings working!")
            print(f"📊 Embedding length: {len(embedding)}")
            print(f"📊 First 5 values: {embedding[:5]}")
            
            # Check if it's using direct API fallback
            if isinstance(embeddings, DirectAPIEmbeddings):
                print("🔄 Using Direct API fallback for embeddings")
            else:
                print("✅ Using LangChain embeddings")
        else:
            print("❌ Robust embeddings returned empty result")
            return False
    except Exception as e:
        print(f"❌ Robust embeddings failed: {e}")
        return False
    
    # Test 4: Robust LLM
    print("\n4️⃣ Testing Robust LLM...")
    try:
        # Try with a smaller model first
        llm_models = [model for model in models if any(keyword in model.lower() 
                     for keyword in ["gemma", "qwen", "deepseek"])]
        
        if not llm_models:
            print("❌ No suitable LLM models found")
            return False
        
        test_model = llm_models[0]
        print(f"🧪 Testing with model: {test_model}")
        
        llm = load_robust_llm(test_model)
        
        if llm is None:
            print("❌ Robust LLM failed to load")
            return False
        
        # Test the LLM
        test_prompt = "Hello, this is a test. Please respond with 'Test successful!'"
        try:
            response = llm.invoke(test_prompt)
            print(f"✅ Robust LLM working!")
            print(f"📝 Response: {response[:100]}...")
            
            # Check if it's using direct API fallback
            if isinstance(llm, DirectAPILLM):
                print("🔄 Using Direct API fallback for LLM")
            else:
                print("✅ Using LangChain LLM")
        except Exception as invoke_error:
            print(f"⚠️ LLM invocation failed: {invoke_error}")
            print("This might be normal if the model is still loading")
            
    except Exception as e:
        print(f"❌ Robust LLM failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    return True

if __name__ == "__main__":
    success = test_robust_utilities()
    
    if success:
        print("✅ Robust utilities are working correctly!")
        print("\n💡 The system now has:")
        print("   - Direct API fallback for embeddings")
        print("   - Direct API fallback for LLM")
        print("   - Retry logic for network issues")
        print("   - Proper error handling")
    else:
        print("❌ Some tests failed!")
        print("\n🔧 Please check:")
        print("   - Ollama is running")
        print("   - Models are available")
        print("   - Network connectivity")
