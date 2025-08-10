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
                if any(keyword in model.lower() for keyword in ["qwen", "gemma", "llama", "mistral", "codellama", "phi", "vicuna", "alpaca","deepseek","gpt"]):
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
        from .prompts import get_question_optimization_prompt
        optimization_prompt = get_question_optimization_prompt(original_question)
        
        response = llm.invoke(optimization_prompt)
        
        # Parse the response
        lines = response.strip().split('\n')
        optimized_question = original_question  # Default fallback
        keywords = []
        
        # Debug: Log the raw response for troubleshooting
        print(f"DEBUG: Raw LLM response: {response}")
        
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
                print(f"DEBUG: Removed thinking indicator: '{indicator}'")
        
        # Split the cleaned response
        lines = cleaned_response.strip().split('\n')
        
        # First pass: Look for the exact format markers
        for line in lines:
            line = line.strip()
            if line.startswith('OPTIMIZED QUESTION:'):
                optimized_question = line.replace('OPTIMIZED QUESTION:', '').strip()
                print(f"DEBUG: Found optimized question: '{optimized_question}'")
            elif line.startswith('KEYWORDS:'):
                keyword_text = line.replace('KEYWORDS:', '').strip()
                keywords = [kw.strip() for kw in keyword_text.split(',') if kw.strip()]
                print(f"DEBUG: Found keywords: {keywords}")
        
        # If we didn't find the exact format, try to extract meaningful content
        if not optimized_question or optimized_question.strip() == "" or optimized_question == original_question:
            print(f"DEBUG: Exact format not found, trying content extraction")
            
            # Look for lines that might contain the optimized question
            potential_questions = []
            for line in lines:
                line = line.strip()
                # Skip empty lines, headers, and obvious non-content
                if (line and 
                    not line.startswith('Original question:') and 
                    not line.startswith('Tasks:') and 
                    not line.startswith('**') and 
                    not line.startswith('Response:') and
                    not line.startswith('SYSTEM:') and
                    not line.startswith('TASK:') and
                    not line.startswith('CRITICAL:') and
                    not line.startswith('IMPORTANT:') and
                    not line.startswith('RESPONSE FORMAT:') and
                    not line.startswith('Focus on') and
                    not line.startswith('Example transformation:') and
                    not line.startswith('- Original:') and
                    not line.startswith('- Optimized:') and
                    not line.startswith('Let me') and
                    not line.startswith('I need to') and
                    not line.startswith('First,') and
                    not line.startswith('Based on') and
                    not line.startswith('I\'ll help') and
                    len(line) > 20):
                    
                    potential_questions.append(line)
                    print(f"DEBUG: Found potential content: '{line}'")
            
            # Use the first substantial line as the optimized question
            if potential_questions:
                optimized_question = potential_questions[0]
                print(f"DEBUG: Using first potential content: '{optimized_question}'")
            
            # If still no good result, try to find any meaningful content
            if not optimized_question or optimized_question.strip() == "" or optimized_question == original_question:
                print(f"DEBUG: Still no good result, looking for any meaningful content")
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 15 and not line.startswith('Original question:'):
                        optimized_question = line
                        print(f"DEBUG: Found fallback content: '{optimized_question}'")
                        break
            
            # Final attempt: Look for the longest meaningful sentence
            if not optimized_question or optimized_question.strip() == "" or optimized_question == original_question:
                print(f"DEBUG: Final attempt - looking for longest meaningful sentence")
                longest_line = ""
                for line in lines:
                    line = line.strip()
                    if (line and 
                        len(line) > 30 and 
                        not line.startswith('Original question:') and
                        not line.startswith('Let me') and
                        not line.startswith('I need to') and
                        not line.startswith('This is') and
                        not line.startswith('To answer')):
                        
                        if len(line) > len(longest_line):
                            longest_line = line
                            print(f"DEBUG: Found longer line: '{longest_line}'")
                
                if longest_line:
                    optimized_question = longest_line
                    print(f"DEBUG: Using longest meaningful line: '{optimized_question}'")
        
        # Final validation - if we still have nothing useful, return empty to trigger manual fallback
        if not optimized_question or optimized_question.strip() == "" or optimized_question == original_question:
            print(f"DEBUG: Final validation failed, returning empty to trigger manual fallback")
            return "", []
        
        return optimized_question, keywords
        
    except Exception as e:
        print(f"DEBUG: Exception in optimize_question: {e}")
        # Return empty to trigger manual fallback
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


def generate_answer(llm, question: str, search_results: List[Document], summarize: bool = False) -> str:
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
        
        # Generate answer using centralized prompt
        from .prompts import get_answer_generation_prompt
        prompt = get_answer_generation_prompt(combined_context, question, summarize)
        
        answer = llm.invoke(prompt)
        return answer
        
    except Exception as e:
        return f"Error during answer generation: {str(e)}" 