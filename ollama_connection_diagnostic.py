#!/usr/bin/env python3
"""
Ollama Connection Diagnostic Tool for RHEA RAG System
Diagnoses and fixes common connection issues with Ollama
"""

import requests
import json
import time
import subprocess
import sys
import os
from typing import Dict, List, Optional

class OllamaDiagnostic:
    def __init__(self, base_url: str = "http://127.0.0.1:11435"):
        self.base_url = base_url
        self.issues = []
        self.fixes = []
        
    def run_full_diagnostic(self):
        """Run complete diagnostic suite"""
        print("🔍 RHEA RAG Ollama Connection Diagnostic")
        print("=" * 60)
        
        # Test 1: Basic connectivity
        self.test_basic_connectivity()
        
        # Test 2: Model availability
        self.test_model_availability()
        
        # Test 3: API endpoints
        self.test_api_endpoints()
        
        # Test 4: Model loading
        self.test_model_loading()
        
        # Test 5: Generation test
        self.test_generation()
        
        # Test 6: LangChain integration
        self.test_langchain_integration()
        
        # Test 7: Resource usage
        self.test_resource_usage()
        
        # Generate report
        self.generate_report()
        
    def test_basic_connectivity(self):
        """Test basic network connectivity"""
        print("\n1️⃣ Testing Basic Connectivity...")
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                print("✅ Basic connectivity: OK")
                return True
            else:
                self.issues.append(f"HTTP {response.status_code} from Ollama API")
                print(f"❌ Basic connectivity: HTTP {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            self.issues.append("Connection refused - Ollama not running")
            print("❌ Basic connectivity: Connection refused")
            self.fixes.append("Start Ollama: ollama serve")
            return False
        except requests.exceptions.Timeout:
            self.issues.append("Connection timeout")
            print("❌ Basic connectivity: Timeout")
            self.fixes.append("Check Ollama performance and system resources")
            return False
        except Exception as e:
            self.issues.append(f"Unexpected error: {e}")
            print(f"❌ Basic connectivity: {e}")
            return False
    
    def test_model_availability(self):
        """Test if required models are available"""
        print("\n2️⃣ Testing Model Availability...")
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                models = [model['name'] for model in models_data.get('models', [])]
                
                print(f"📚 Available models: {len(models)}")
                for model in models:
                    print(f"   • {model}")
                
                # Check for required models
                required_models = ["nomic-embed-text:latest"]
                llm_models = [m for m in models if any(keyword in m.lower() 
                              for keyword in ["qwen", "gemma", "llama", "mistral", "deepseek", "gpt"])]
                
                if llm_models:
                    print(f"🤖 LLM models found: {len(llm_models)}")
                    required_models.extend(llm_models[:2])  # Add first 2 LLM models
                
                missing_models = []
                for model in required_models:
                    if model not in models:
                        missing_models.append(model)
                
                if missing_models:
                    self.issues.append(f"Missing models: {missing_models}")
                    print(f"⚠️  Missing models: {missing_models}")
                    for model in missing_models:
                        self.fixes.append(f"Pull model: ollama pull {model}")
                else:
                    print("✅ All required models available")
                    return True
                    
            else:
                self.issues.append("Cannot fetch model list")
                print("❌ Cannot fetch model list")
                return False
                
        except Exception as e:
            self.issues.append(f"Model availability test failed: {e}")
            print(f"❌ Model availability test failed: {e}")
            return False
    
    def test_api_endpoints(self):
        """Test various API endpoints"""
        print("\n3️⃣ Testing API Endpoints...")
        
        endpoints = [
            ("/api/tags", "Model listing"),
            ("/api/show", "Model info"),
            ("/api/ps", "Process status")
        ]
        
        for endpoint, description in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    print(f"✅ {description}: OK")
                else:
                    print(f"⚠️  {description}: HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ {description}: {e}")
    
    def test_model_loading(self):
        """Test model loading with different configurations"""
        print("\n4️⃣ Testing Model Loading...")
        
        try:
            # Test with a simple model first
            test_model = "gpt-oss:20b"  # Use the model we know exists
            
            # Test basic model loading
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": test_model,
                    "prompt": "Hello",
                    "stream": False,
                    "options": {
                        "num_predict": 10,
                        "temperature": 0.1
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Model loading test: OK (Response: {result.get('response', '')[:50]}...)")
                return True
            else:
                self.issues.append(f"Model loading failed: HTTP {response.status_code}")
                print(f"❌ Model loading test: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            self.issues.append("Model loading timeout")
            print("❌ Model loading test: Timeout")
            self.fixes.append("Increase timeout or check model performance")
            return False
        except Exception as e:
            self.issues.append(f"Model loading error: {e}")
            print(f"❌ Model loading test: {e}")
            return False
    
    def test_generation(self):
        """Test text generation with various settings"""
        print("\n5️⃣ Testing Text Generation...")
        
        try:
            test_model = "gpt-oss:20b"
            
            # Test with different generation parameters
            test_cases = [
                {"num_predict": 50, "temperature": 0.1},
                {"num_predict": 20, "temperature": 0.7},
                {"num_predict": 10, "temperature": 0.0}
            ]
            
            for i, options in enumerate(test_cases, 1):
                try:
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": test_model,
                            "prompt": "Test generation",
                            "stream": False,
                            "options": options
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        print(f"✅ Generation test {i}: OK")
                    else:
                        print(f"⚠️  Generation test {i}: HTTP {response.status_code}")
                        
                except Exception as e:
                    print(f"❌ Generation test {i}: {e}")
                    
        except Exception as e:
            self.issues.append(f"Generation test failed: {e}")
            print(f"❌ Generation test failed: {e}")
    
    def test_langchain_integration(self):
        """Test LangChain integration"""
        print("\n6️⃣ Testing LangChain Integration...")
        
        try:
            from langchain.llms import Ollama
            from langchain.embeddings import OllamaEmbeddings
            
            # Test LLM
            try:
                llm = Ollama(model="gpt-oss:20b", timeout=30)
                response = llm.invoke("Hello")
                print(f"✅ LangChain LLM: OK (Response: {response[:50]}...)")
            except Exception as e:
                self.issues.append(f"LangChain LLM error: {e}")
                print(f"❌ LangChain LLM: {e}")
            
            # Test Embeddings
            try:
                embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
                test_embedding = embeddings.embed_query("test")
                print(f"✅ LangChain Embeddings: OK (Vector size: {len(test_embedding)})")
            except Exception as e:
                self.issues.append(f"LangChain Embeddings error: {e}")
                print(f"❌ LangChain Embeddings: {e}")
                
        except ImportError:
            self.issues.append("LangChain not installed")
            print("❌ LangChain not installed")
            self.fixes.append("Install LangChain: pip install langchain")
        except Exception as e:
            self.issues.append(f"LangChain integration error: {e}")
            print(f"❌ LangChain integration: {e}")
    
    def test_resource_usage(self):
        """Test system resource usage"""
        print("\n7️⃣ Testing Resource Usage...")
        
        try:
            # Check Ollama process
            response = requests.get(f"{self.base_url}/api/ps", timeout=5)
            if response.status_code == 200:
                processes = response.json()
                print(f"📊 Active Ollama processes: {len(processes)}")
                
                total_memory = 0
                for proc in processes:
                    memory = proc.get('memory', 0)
                    total_memory += memory
                    print(f"   • {proc.get('model', 'Unknown')}: {memory / (1024**3):.2f} GB")
                
                print(f"💾 Total memory usage: {total_memory / (1024**3):.2f} GB")
                
                if total_memory > 16 * (1024**3):  # 16 GB
                    self.issues.append("High memory usage detected")
                    print("⚠️  High memory usage detected")
                    self.fixes.append("Consider using smaller models or freeing memory")
                    
            else:
                print("⚠️  Cannot fetch process information")
                
        except Exception as e:
            print(f"❌ Resource usage test: {e}")
    
    def generate_report(self):
        """Generate diagnostic report"""
        print("\n" + "=" * 60)
        print("📋 DIAGNOSTIC REPORT")
        print("=" * 60)
        
        if not self.issues:
            print("🎉 All tests passed! Ollama is working correctly.")
            print("\n💡 Recommendations:")
            print("   • Your RAG system should work fine")
            print("   • Monitor resource usage during heavy loads")
            print("   • Consider implementing connection retry logic")
        else:
            print(f"⚠️  Found {len(self.issues)} issue(s):")
            for i, issue in enumerate(self.issues, 1):
                print(f"   {i}. {issue}")
            
            print(f"\n🔧 Suggested fixes:")
            for i, fix in enumerate(self.fixes, 1):
                print(f"   {i}. {fix}")
            
            print("\n🚀 Quick fixes to try:")
            print("   1. Restart Ollama: ollama serve")
            print("   2. Check available models: ollama list")
            print("   3. Pull missing models: ollama pull <model_name>")
            print("   4. Check system resources (CPU, RAM)")
            print("   5. Increase timeout settings in your RAG app")

def create_connection_fix_script():
    """Create a script to fix common connection issues"""
    fix_script = '''#!/usr/bin/env python3
"""
Ollama Connection Fix Script for RHEA RAG
Implements retry logic and connection improvements
"""

import requests
import time
import random
from typing import Optional, Dict, Any
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings

class RobustOllamaConnection:
    def __init__(self, base_url: str = "http://127.0.0.1:11435", max_retries: int = 3):
        self.base_url = base_url
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'RHEA-RAG/1.0'
        })
    
    def _make_request_with_retry(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Make HTTP request with exponential backoff retry"""
        for attempt in range(self.max_retries):
            try:
                url = f"{self.base_url}{endpoint}"
                response = self.session.request(method, url, **kwargs)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    print(f"Endpoint not found: {endpoint}")
                    return None
                else:
                    print(f"HTTP {response.status_code} on attempt {attempt + 1}")
                    
            except requests.exceptions.ConnectionError as e:
                print(f"Connection error on attempt {attempt + 1}: {e}")
            except requests.exceptions.Timeout as e:
                print(f"Timeout on attempt {attempt + 1}: {e}")
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")
            
            if attempt < self.max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
        
        return None
    
    def get_models(self) -> Optional[Dict[str, Any]]:
        """Get available models with retry logic"""
        return self._make_request_with_retry('GET', '/api/tags')
    
    def generate_text(self, model: str, prompt: str, **options) -> Optional[Dict[str, Any]]:
        """Generate text with retry logic"""
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options
        }
        return self._make_request_with_retry('POST', '/api/generate', json=data)

class RobustOllamaLLM:
    """Enhanced Ollama LLM with retry logic"""
    
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.ollama = RobustOllamaConnection()
        self.max_retries = kwargs.get('max_retries', 3)
        self.timeout = kwargs.get('timeout', 60)
        
    def invoke(self, prompt: str) -> str:
        """Invoke with retry logic"""
        for attempt in range(self.max_retries):
            try:
                result = self.ollama.generate_text(
                    self.model, 
                    prompt,
                    num_predict=1000,
                    temperature=0.7
                )
                
                if result and 'response' in result:
                    return result['response']
                else:
                    raise Exception("No response in result")
                    
            except Exception as e:
                print(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep((2 ** attempt) + random.uniform(0, 1))
                else:
                    raise e
        
        raise Exception("All retry attempts failed")

# Usage example:
if __name__ == "__main__":
    # Test the robust connection
    robust_llm = RobustOllamaLLM("gpt-oss:20b")
    
    try:
        response = robust_llm.invoke("Hello, how are you?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
'''
    
    with open('ollama_connection_fix.py', 'w') as f:
        f.write(fix_script)
    
    print("✅ Created ollama_connection_fix.py with robust connection handling")

if __name__ == "__main__":
    diagnostic = OllamaDiagnostic()
    diagnostic.run_full_diagnostic()
    
    # Create fix script
    create_connection_fix_script()
    
    print("\n🎯 Next steps:")
    print("1. Review the diagnostic report above")
    print("2. Apply the suggested fixes")
    print("3. Use the robust connection script in your RAG app")
    print("4. Test your RAG system again")
