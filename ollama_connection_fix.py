#!/usr/bin/env python3
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
