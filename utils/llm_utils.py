"""
LLM utilities for the Paper Search & QA System.
Handles model loading, question optimization, and answer generation.
"""

import streamlit as st
import requests
from langchain.llms import Ollama
from typing import List, Tuple, Optional
from langchain.schema import Document


@st.cache_resource
def load_llm(model_name: str):
    """Load the LLM model"""
    try:
        return Ollama(model=model_name)
    except Exception as e:
        # Don't show error - just return None for graceful degradation
        return None


@st.cache_data
def get_available_ollama_models() -> List[str]:
    """Get list of available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
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
                if any(keyword in model.lower() for keyword in ["qwen", "gemma", "llama", "mistral", "codellama", "phi", "vicuna", "alpaca"]):
                    llm_models.append(model)
            
            return sorted(llm_models) if llm_models else ["qwen3:14b", "gemma3:4b"]
        else:
            return ["qwen3:14b", "gemma3:4b"]
    except Exception as e:
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
        optimization_prompt = f"""You are a materials science research expert. Optimize the following question for better search in a scientific paper database about Refractory High-Entropy Alloys (RHEA).

Original question: "{original_question}"

Tasks:
1. Rewrite the question to be more specific and technical for materials science literature search
2. Suggest 5-8 relevant keywords that would help retrieve relevant papers

Format your response as:
OPTIMIZED QUESTION: [your optimized question]
KEYWORDS: keyword1, keyword2, keyword3, keyword4, keyword5

Focus on materials science terminology like: microstructure, precipitation, dislocation, grain boundary, mechanical properties, strengthening mechanisms, phase formation, etc.

Response:"""
        
        response = llm.invoke(optimization_prompt)
        
        # Parse the response
        lines = response.strip().split('\n')
        optimized_question = original_question
        keywords = []
        
        for line in lines:
            if line.startswith('OPTIMIZED QUESTION:'):
                optimized_question = line.replace('OPTIMIZED QUESTION:', '').strip()
            elif line.startswith('KEYWORDS:'):
                keyword_text = line.replace('KEYWORDS:', '').strip()
                keywords = [kw.strip() for kw in keyword_text.split(',') if kw.strip()]
        
        return optimized_question, keywords
        
    except Exception as e:
        # Return original question and static keywords on error
        static_keywords = [
            "precipitation strengthening", "dislocation density", "grain boundary", 
            "microstructure", "mechanical properties", "yield strength", "ductility"
        ]
        return original_question, static_keywords


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


def generate_answer(llm, question: str, search_results: List[Document]) -> str:
    """Generate answer using LLM based on search results"""
    if not llm:
        # Return a helpful message when LLM is not available
        if not search_results:
            return "No relevant documents found for your question."
        
        # Create a simple summary of search results without LLM
        result_summary = []
        for i, doc in enumerate(search_results[:3]):  # Limit to first 3 results
            paper_name = doc.metadata.get('file_name', 'Unknown Paper')
            section = doc.metadata.get('section', 'Unknown Section')
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            
            result_summary.append(f"**{paper_name}** ({section}):\n{content_preview}\n")
        
        return f"""**Search Results Summary (LLM not available):**

Your question: "{question}"

Found {len(search_results)} relevant documents. Here are the top results:

{chr(10).join(result_summary)}

*Note: LLM-powered answer generation is not available. Please check your model configuration or try loading a different model.*"""
    
    if not search_results:
        return "No relevant documents found for your question."
    
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
        
        # Generate answer
        prompt = f"""You are a materials science research expert. Based on the following context from scientific papers, answer the user's question comprehensively and accurately.

Context from papers:
{combined_context}

Question: {question}

Instructions:
1. Provide a comprehensive answer based on the context provided
2. Focus on materials science concepts, mechanisms, and relationships
3. If figures are mentioned in the context, reference them appropriately
4. Use technical terminology appropriately
5. Structure your answer clearly with main points and supporting details
6. If specific papers are mentioned, cite them in your response

Answer:"""
        
        answer = llm.invoke(prompt)
        return answer
        
    except Exception as e:
        return f"Error during answer generation: {str(e)}" 