#!/usr/bin/env python3
"""
Check what type of LLM object is being loaded to diagnose caching issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_llm_types():
    """Check what types of LLM objects are being created"""
    print("🔍 Checking LLM object types...")
    print("=" * 60)
    
    try:
        # Import utilities
        from utils import load_llm, get_available_ollama_models
        from utils.direct_ollama_utils import DirectOllamaLLM, load_direct_llm
        
        # Get available models
        models = get_available_ollama_models()
        print(f"✅ Available models: {models}")
        
        if not models:
            print("❌ No models available")
            return
        
        model_name = models[0]
        print(f"\n🔍 Testing with model: {model_name}")
        
        # Test 1: Direct LLM creation
        print("\n1️⃣ Creating DirectOllamaLLM directly...")
        try:
            direct_llm = DirectOllamaLLM(model_name)
            print(f"✅ Direct LLM type: {type(direct_llm)}")
            print(f"✅ Direct LLM class: {direct_llm.__class__.__name__}")
            print(f"✅ Direct LLM module: {direct_llm.__class__.__module__}")
        except Exception as e:
            print(f"❌ Direct LLM creation failed: {e}")
        
        # Test 2: Utils load_llm function
        print("\n2️⃣ Using utils.load_llm()...")
        try:
            utils_llm = load_llm(model_name)
            print(f"✅ Utils LLM type: {type(utils_llm)}")
            print(f"✅ Utils LLM class: {utils_llm.__class__.__name__}")
            print(f"✅ Utils LLM module: {utils_llm.__class__.__module__}")
            
            # Check if it's the same type as direct
            if type(utils_llm) == type(DirectOllamaLLM(model_name)):
                print("✅ Utils LLM is using DirectOllamaLLM (CORRECT)")
            else:
                print("❌ Utils LLM is NOT using DirectOllamaLLM (WRONG!)")
                
        except Exception as e:
            print(f"❌ Utils LLM creation failed: {e}")
        
        # Test 3: Direct API function
        print("\n3️⃣ Using load_direct_llm()...")
        try:
            direct_api_llm = load_direct_llm(model_name)
            print(f"✅ Direct API LLM type: {type(direct_api_llm)}")
            print(f"✅ Direct API LLM class: {direct_api_llm.__class__.__name__}")
            print(f"✅ Direct API LLM module: {direct_api_llm.__class__.__module__}")
        except Exception as e:
            print(f"❌ Direct API LLM creation failed: {e}")
        
        # Test 4: Check for LangChain imports
        print("\n4️⃣ Checking for LangChain Ollama in memory...")
        import sys
        langchain_ollama_modules = [name for name in sys.modules if 'langchain' in name and 'ollama' in name.lower()]
        if langchain_ollama_modules:
            print(f"⚠️ LangChain Ollama modules found: {langchain_ollama_modules}")
        else:
            print("✅ No LangChain Ollama modules in memory")
        
        # Test 5: Check what functions are exported by utils
        print("\n5️⃣ Checking utils exports...")
        import utils
        if hasattr(utils, 'load_llm'):
            print(f"✅ utils.load_llm: {utils.load_llm}")
            print(f"✅ utils.load_llm module: {utils.load_llm.__module__}")
        else:
            print("❌ utils.load_llm not found")
            
    except Exception as e:
        print(f"❌ Failed to check LLM types: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_llm_types()
