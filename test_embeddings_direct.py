#!/usr/bin/env python3
"""
Test embeddings directly with Ollama API to isolate the RemoteDisconnected issue.
"""

import requests
import json
import time

def test_embeddings_direct():
    """Test embeddings directly with Ollama API"""
    base_url = "http://127.0.0.1:11435"
    
    # Test text to embed
    test_text = "This is a test sentence for embeddings."
    
    print("🔍 Testing embeddings directly with Ollama API...")
    print(f"📝 Test text: '{test_text}'")
    
    # Test 1: Check if model is available
    print("\n1️⃣ Checking model availability...")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            print(f"✅ Available models: {model_names}")
            
            if "nomic-embed-text:latest" in model_names:
                print("✅ nomic-embed-text:latest is available")
            else:
                print("❌ nomic-embed-text:latest not found")
                return False
        else:
            print(f"❌ Failed to get models: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False
    
    # Test 2: Test embeddings generation using the correct endpoint
    print("\n2️⃣ Testing embeddings generation...")
    try:
        payload = {
            "model": "nomic-embed-text:latest",
            "prompt": test_text
        }
        
        print(f"📤 Sending request to {base_url}/api/embeddings")
        print(f"📦 Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            f"{base_url}/api/embeddings",
            json=payload,
            timeout=60,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📥 Response status: {response.status_code}")
        print(f"📥 Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Response: {json.dumps(result, indent=2)}")
            
            # Check if response contains embeddings
            if "embedding" in result:
                embedding = result["embedding"]
                print(f"✅ Embedding generated successfully!")
                print(f"📊 Embedding length: {len(embedding)}")
                print(f"📊 First 5 values: {embedding[:5]}")
                return True
            else:
                print("❌ No embedding found in response")
                return False
        else:
            print(f"❌ Failed to generate embeddings: {response.status_code}")
            print(f"❌ Response text: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return False
    except requests.exceptions.Timeout as e:
        print(f"❌ Timeout error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_embeddings_with_retry():
    """Test embeddings with retry logic"""
    print("🔄 Testing embeddings with retry logic...")
    
    max_retries = 3
    for attempt in range(max_retries):
        print(f"\n🔄 Attempt {attempt + 1}/{max_retries}")
        
        if test_embeddings_direct():
            print("✅ Embeddings test successful!")
            return True
        else:
            print(f"❌ Attempt {attempt + 1} failed")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
    
    print("❌ All attempts failed")
    return False

if __name__ == "__main__":
    print("🚀 Starting embeddings diagnostic...")
    print("=" * 50)
    
    success = test_embeddings_with_retry()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Embeddings test completed successfully!")
    else:
        print("💥 Embeddings test failed!")
    
    print("\n💡 This test helps isolate whether the issue is with:")
    print("   - Ollama API itself")
    print("   - LangChain's OllamaEmbeddings wrapper")
    print("   - Network connectivity")
