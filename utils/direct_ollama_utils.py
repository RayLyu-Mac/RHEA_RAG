"""
Direct Ollama utilities for RHEA RAG System
Uses direct API calls to Ollama, completely bypassing LangChain
"""

import requests
import time
import random
import streamlit as st
from typing import List, Tuple, Optional, Dict, Any
from langchain.schema import Document
import tiktoken

class DirectOllamaConnection:
    """Direct API connection to Ollama with retry logic"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:11435", max_retries: int = 3):
        self.base_url = base_url
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'RHEA-RAG/1.0'
        })
    
    def make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with exponential backoff retry"""
        for attempt in range(self.max_retries):
            try:
                url = f"{self.base_url}{endpoint}"
                response = self.session.request(method, url, **kwargs)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 404:
                    print(f"Endpoint not found: {endpoint}")
                    return response
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
        
        # If all retries failed, raise the last exception
        raise Exception(f"All {self.max_retries} attempts failed")

class DirectOllamaLLM:
    """Direct API wrapper for Ollama LLM"""
    
    def __init__(self, model_name: str, base_url: str = "http://127.0.0.1:11435"):
        self.model_name = model_name
        self.base_url = base_url
        self.connection = DirectOllamaConnection(base_url)
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """Generate text using direct API call"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            # Add any additional parameters
            if "temperature" in kwargs:
                payload["options"] = {"temperature": kwargs["temperature"]}
            
            response = self.connection.make_request(
                "POST",
                "/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                raise Exception(f"API call failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Direct API LLM failed: {e}")

class DirectOllamaEmbeddings:
    """Direct API wrapper for Ollama embeddings"""
    
    def __init__(self, model_name: str = "nomic-embed-text:latest", base_url: str = "http://127.0.0.1:11435"):
        self.model_name = model_name
        self.base_url = base_url
        self.connection = DirectOllamaConnection(base_url)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents"""
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": text
            }
            
            response = self.connection.make_request(
                "POST",
                "/api/embeddings",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("embedding", [])
            else:
                raise Exception(f"Embeddings API call failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Direct API embeddings failed: {e}")

@st.cache_resource
def load_direct_llm_v2(model_name: str):
    """Load LLM using direct API calls"""
    try:
        llm = DirectOllamaLLM(model_name)
        
        # Test the connection
        test_response = llm.invoke("Test connection")
        if test_response:
            print(f"✅ Successfully loaded LLM: {model_name}")
            return llm
        else:
            print(f"❌ LLM test failed for: {model_name}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to load LLM {model_name}: {e}")
        return None

@st.cache_resource
def load_direct_embeddings(model_name: str = "nomic-embed-text:latest"):
    """Load embeddings using direct API calls"""
    try:
        embeddings = DirectOllamaEmbeddings(model_name)
        
        # Test the embeddings
        test_text = "Test embedding"
        test_embedding = embeddings.embed_query(test_text)
        if test_embedding and len(test_embedding) > 0:
            print(f"✅ Successfully loaded embeddings: {model_name}")
            return embeddings
        else:
            print(f"❌ Embeddings test failed for: {model_name}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to load embeddings {model_name}: {e}")
        return None

@st.cache_data
def get_available_ollama_models() -> List[str]:
    """Get list of available Ollama models using direct API"""
    try:
        connection = DirectOllamaConnection()
        response = connection.make_request("GET", "/api/tags")
        
        if response.status_code == 200:
            models_data = response.json()
            models = []
            for model in models_data.get("models", []):
                model_name = model.get("name", "")
                if model_name:
                    models.append(model_name)
            
            # Filter for common LLM models (exclude embedding models)
            llm_models = []
            for model in models:
                # Include models that are likely to be LLMs
                if any(keyword in model.lower() for keyword in [
                    "qwen", "gemma", "llama", "mistral", "codellama", "phi", 
                    "vicuna", "alpaca", "deepseek", "gpt"
                ]):
                    llm_models.append(model)
            
            return sorted(llm_models) if llm_models else ["qwen3:14b", "gemma3:4b"]
        else:
            return ["qwen3:14b", "gemma3:4b"]
            
    except Exception as e:
        print(f"Failed to fetch Ollama models: {e}")
        return ["qwen3:14b", "gemma3:4b"]

def optimize_question(llm, original_question: str) -> Tuple[str, List[str]]:
    """Use LLM to optimize the question for better retrieval"""
    if not llm:
        # Return original question and static keywords when LLM is not available
        static_keywords = [
            "precipitation strengthening", "dislocation density", "grain boundary", 
            "microstructure", "mechanical properties", "yield strength", "ductility"
        ]
        return original_question, static_keywords
    
    try:
        from .prompts import get_question_optimization_prompt
        optimization_prompt = get_question_optimization_prompt(original_question)
        
        # Use retry logic for the LLM call
        for attempt in range(3):
            try:
                response = llm.invoke(optimization_prompt)
                break
            except Exception as e:
                print(f"LLM optimization attempt {attempt + 1} failed: {e}")
                if attempt == 2:  # Last attempt
                    return original_question, []
                time.sleep(1)
        
        # Parse the response
        lines = response.strip().split('\n')
        optimized_question = original_question  # Default fallback
        keywords = []
        
        # Clean up common thinking process indicators
        cleaned_response = response
        thinking_indicators = [
            "Let me think about this", "I need to", "First, let me", "Let me analyze",
            "I should", "This is an interesting question", "To answer this",
            "Based on my understanding", "I'll help you", "Let me break this down"
        ]
        
        for indicator in thinking_indicators:
            if indicator in cleaned_response:
                cleaned_response = cleaned_response.replace(indicator, "")
        
        # Split the cleaned response
        lines = cleaned_response.strip().split('\n')
        
        # Parse the response
        for line in lines:
            line = line.strip()
            if line.startswith('OPTIMIZED QUESTION:'):
                optimized_question = line.replace('OPTIMIZED QUESTION:', '').strip()
            elif line.startswith('KEYWORDS:'):
                keyword_text = line.replace('KEYWORDS:', '').strip()
                keywords = [kw.strip() for kw in keyword_text.split(',') if kw.strip()]
        
        # Final validation
        if not optimized_question or optimized_question.strip() == "" or optimized_question == original_question:
            return "", []
        
        return optimized_question, keywords
        
    except Exception as e:
        print(f"Exception in optimize_question: {e}")
        return "", []

def get_suggested_keywords() -> List[str]:
    """Get suggested keywords based on the paper database content"""
    common_keywords = [
        "precipitation strengthening", "dislocation density", "grain boundary", 
        "microstructure", "mechanical properties", "yield strength", "ductility",
        "phase formation", "solid solution strengthening", "work hardening",
        "recrystallization", "texture", "fracture toughness", "creep resistance",
        "oxidation resistance", "high temperature", "BCC structure", "FCC structure",
        "intermetallic phases", "carbides", "nitrides", "strain hardening"
    ]
    return common_keywords

def count_tokens(text: str, model_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.get_encoding(model_name)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: approximate token count (roughly 4 characters per token)
        return len(text) // 4

def generate_answer(llm, question: str, search_results: List[Document], summarize: bool = False) -> Tuple[str, int]:
    """Generate answer using LLM based on search results"""
    if not llm:
        # Return a helpful message when LLM is not available
        if not search_results:
            fallback_answer = "No relevant documents found for your question."
            return fallback_answer, count_tokens(fallback_answer)
        
        # Create a simple summary of search results without LLM
        result_summary = []
        for i, doc in enumerate(search_results[:3]):  # Limit to first 3 results
            paper_name = doc.metadata.get('file_name', 'Unknown Paper')
            section = doc.metadata.get('section', 'Unknown Section')
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            
            result_summary.append(f"**{paper_name}** ({section}):\n{content_preview}\n")
        
        fallback_answer = f"""**Search Results Summary (LLM not available):**

Your question: "{question}"

Found {len(search_results)} relevant documents. Here are the top results:

{chr(10).join(result_summary)}

*Note: LLM-powered answer generation is not available. Please check your model configuration or try loading a different model.*"""
        
        return fallback_answer, count_tokens(fallback_answer)
    
    if not search_results:
        fallback_answer = "No relevant documents found for your question."
        return fallback_answer, count_tokens(fallback_answer)
    
    try:
        # Prepare context
        context_parts = []
        for i, doc in enumerate(search_results):
            paper_name = doc.metadata.get('file_name', 'Unknown Paper')
            doc_type = doc.metadata.get('document_type', 'unknown')
            section = doc.metadata.get('section', 'Unknown Section')
            
            context_parts.append(f"[Document {i+1}] ({doc_type.upper()}) {paper_name} - {section}")
            
            # Truncate content if too long
            content = doc.page_content[:1500] + "..." if len(doc.page_content) > 1500 else doc.page_content
            context_parts.append(content)
            context_parts.append("---")
        
        combined_context = "\n".join(context_parts)
        
        # Generate answer using centralized prompt
        from .prompts import get_answer_generation_prompt
        prompt = get_answer_generation_prompt(combined_context, question, summarize)
        
        # Use retry logic for the LLM call
        for attempt in range(3):
            try:
                answer = llm.invoke(prompt)
                token_count = count_tokens(answer)
                return answer, token_count
            except Exception as e:
                print(f"Answer generation attempt {attempt + 1} failed: {e}")
                if attempt == 2:  # Last attempt
                    error_answer = f"Error during answer generation: {str(e)}"
                    return error_answer, count_tokens(error_answer)
                time.sleep(2)
        
    except Exception as e:
        error_answer = f"Error during answer generation: {str(e)}"
        return error_answer, count_tokens(error_answer)
