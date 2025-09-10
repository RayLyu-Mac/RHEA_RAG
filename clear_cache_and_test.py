#!/usr/bin/env python3
"""
Clear Streamlit cache and test direct API utilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def clear_streamlit_cache():
    """Clear Streamlit cache"""
    try:
        import streamlit as st
        
        # Clear cache_resource (for LLM objects)
        if hasattr(st, 'cache_resource'):
            st.cache_resource.clear()
            print("✅ Cleared st.cache_resource")
        
        # Clear cache_data (for data objects)
        if hasattr(st, 'cache_data'):
            st.cache_data.clear()
            print("✅ Cleared st.cache_data")
        
        # Clear legacy caches (if they exist)
        if hasattr(st, 'cache'):
            st.cache.clear()
            print("✅ Cleared st.cache")
            
        print("🔄 All Streamlit caches cleared!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to clear cache: {e}")
        return False

def test_fresh_llm_loading():
    """Test loading LLM with fresh cache"""
    print("\n🔍 Testing fresh LLM loading...")
    
    try:
        from utils.direct_ollama_utils import load_direct_llm, get_available_ollama_models
        
        # Get available models
        models = get_available_ollama_models()
        print(f"✅ Available models: {models}")
        
        if models:
            model_name = models[0]
            print(f"🔍 Loading fresh LLM: {model_name}")
            
            # Load LLM without cache first
            from utils.direct_ollama_utils import DirectOllamaLLM
            direct_llm = DirectOllamaLLM(model_name)
            
            print(f"✅ Direct LLM created: {type(direct_llm)}")
            
            # Test direct invoke
            test_prompt = "Say 'test successful' if you can hear me."
            try:
                response = direct_llm.invoke(test_prompt)
                print(f"✅ Direct LLM response: {response[:100]}...")
                
                # Now test cached version
                cached_llm = load_direct_llm(model_name)
                print(f"✅ Cached LLM created: {type(cached_llm)}")
                
                # Test cached invoke
                try:
                    cached_response = cached_llm.invoke(test_prompt)
                    print(f"✅ Cached LLM response: {cached_response[:100]}...")
                    return True
                except Exception as e:
                    print(f"❌ Cached LLM invoke failed: {e}")
                    print(f"❌ Error type: {type(e)}")
                    return False
                
            except Exception as e:
                print(f"❌ Direct LLM invoke failed: {e}")
                print(f"❌ Error type: {type(e)}")
                return False
        else:
            print("❌ No available models found")
            return False
            
    except Exception as e:
        print(f"❌ LLM loading failed: {e}")
        return False

def test_utils_after_cache_clear():
    """Test utils imports after cache clear"""
    print("\n🔍 Testing utils after cache clear...")
    
    try:
        # Force reimport of utils
        import importlib
        import utils
        importlib.reload(utils)
        
        from utils import load_llm, get_available_ollama_models
        
        models = get_available_ollama_models()
        if models:
            model_name = models[0]
            print(f"🔍 Loading LLM through utils: {model_name}")
            
            llm = load_llm(model_name)
            print(f"✅ Utils LLM type: {type(llm)}")
            print(f"✅ Utils LLM class: {llm.__class__.__name__}")
            
            # Test invoke
            try:
                response = llm.invoke("Test utils after cache clear")
                print(f"✅ Utils LLM response: {response[:50]}...")
                return True
            except Exception as e:
                print(f"❌ Utils LLM invoke failed: {e}")
                print(f"❌ Error type: {type(e)}")
                return False
        else:
            print("❌ No available models through utils")
            return False
            
    except Exception as e:
        print(f"❌ Utils test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Clearing Cache and Testing Direct API")
    print("=" * 60)
    
    # Clear cache
    cache_cleared = clear_streamlit_cache()
    
    # Test fresh loading
    fresh_test = test_fresh_llm_loading()
    
    # Test utils after cache clear
    utils_test = test_utils_after_cache_clear()
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print(f"✅ Cache cleared: {'PASS' if cache_cleared else 'FAIL'}")
    print(f"✅ Fresh LLM test: {'PASS' if fresh_test else 'FAIL'}")
    print(f"✅ Utils test: {'PASS' if utils_test else 'FAIL'}")
    
    if cache_cleared and fresh_test and utils_test:
        print("\n🎉 All tests passed! The cache issue should be resolved.")
        print("💡 Try running your app now - the RemoteDisconnected error should be gone.")
    else:
        print("\n❌ Some tests failed. The cache might not be the issue.")
        print("💡 Check the error messages above for more details.")
