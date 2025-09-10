#!/usr/bin/env python3
"""
Debug script to find the exact source of the RemoteDisconnected error
"""

import sys
import os
import traceback
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_direct_imports():
    """Test that all direct imports work correctly"""
    print("🔍 Testing direct imports...")
    
    try:
        # Test direct API imports
        from utils.direct_ollama_utils import (
            load_direct_llm, get_available_ollama_models, 
            optimize_question, generate_answer
        )
        print("✅ Direct API imports successful")
        
        # Test if we can create an LLM instance
        available_models = get_available_ollama_models()
        print(f"✅ Available models: {available_models}")
        
        if available_models:
            model_name = available_models[0]
            print(f"🔍 Testing LLM creation with {model_name}...")
            
            llm = load_direct_llm(model_name)
            print(f"✅ LLM created: {type(llm)}")
            
            # Test a simple invoke
            test_prompt = "Hello, can you say 'test successful'?"
            print(f"🔍 Testing LLM invoke with prompt: '{test_prompt}'")
            
            try:
                response = llm.invoke(test_prompt)
                print(f"✅ LLM response: {response[:100]}...")
                return True
            except Exception as e:
                print(f"❌ LLM invoke failed: {e}")
                print(f"❌ Error type: {type(e)}")
                print("❌ Full traceback:")
                traceback.print_exc()
                return False
        else:
            print("❌ No available models found")
            return False
            
    except Exception as e:
        print(f"❌ Import/setup failed: {e}")
        print("❌ Full traceback:")
        traceback.print_exc()
        return False

def test_utils_imports():
    """Test that utils imports work correctly"""
    print("\n🔍 Testing utils imports...")
    
    try:
        from utils import load_llm, get_available_ollama_models
        print("✅ Utils imports successful")
        
        # Check if these are the direct API versions
        print(f"✅ load_llm function: {load_llm}")
        print(f"✅ load_llm module: {load_llm.__module__}")
        
        # Test model loading through utils
        available_models = get_available_ollama_models()
        if available_models:
            model_name = available_models[0]
            print(f"🔍 Testing utils LLM creation with {model_name}...")
            
            llm = load_llm(model_name)
            print(f"✅ Utils LLM created: {type(llm)}")
            
            # Test invoke through utils
            test_prompt = "Test utils invoke"
            try:
                response = llm.invoke(test_prompt)
                print(f"✅ Utils LLM response: {response[:50]}...")
                return True
            except Exception as e:
                print(f"❌ Utils LLM invoke failed: {e}")
                print(f"❌ Error type: {type(e)}")
                print("❌ Full traceback:")
                traceback.print_exc()
                return False
        else:
            print("❌ No available models found through utils")
            return False
            
    except Exception as e:
        print(f"❌ Utils import/setup failed: {e}")
        print("❌ Full traceback:")
        traceback.print_exc()
        return False

def check_for_langchain_imports():
    """Check if any LangChain Ollama imports are still active"""
    print("\n🔍 Checking for LangChain imports in memory...")
    
    import sys
    langchain_modules = []
    
    for module_name in sys.modules:
        if 'langchain' in module_name and 'ollama' in module_name.lower():
            langchain_modules.append(module_name)
    
    if langchain_modules:
        print(f"⚠️ Found LangChain modules in memory: {langchain_modules}")
        return False
    else:
        print("✅ No LangChain Ollama modules found in memory")
        return True

if __name__ == "__main__":
    print("🚀 Debugging RemoteDisconnected Error Source")
    print("=" * 60)
    
    # Test direct imports
    direct_success = test_direct_imports()
    
    # Test utils imports
    utils_success = test_utils_imports()
    
    # Check for LangChain imports
    no_langchain = check_for_langchain_imports()
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print(f"✅ Direct imports: {'PASS' if direct_success else 'FAIL'}")
    print(f"✅ Utils imports: {'PASS' if utils_success else 'FAIL'}")
    print(f"✅ No LangChain: {'PASS' if no_langchain else 'FAIL'}")
    
    if direct_success and utils_success and no_langchain:
        print("\n🎉 All tests passed! The direct API system should work.")
        print("💡 If you're still getting RemoteDisconnected errors,")
        print("   it might be from cached imports or a different part of the system.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        print("💡 The RemoteDisconnected error is likely coming from the failed components.")
