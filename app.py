import streamlit as st
import os
import sys
import pandas as pd
from pathlib import Path
from PIL import Image
import json
import requests

# Add parent directory to path to import vectorized module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

# Page configuration
st.set_page_config(
    page_title="Paper Search & QA System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for theme
if 'dark_theme' not in st.session_state:
    st.session_state.dark_theme = False

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'paper_list' not in st.session_state:
    st.session_state.paper_list = []
if 'tracker_df' not in st.session_state:
    st.session_state.tracker_df = None
if 'available_models' not in st.session_state:
    st.session_state.available_models = ["qwen3:14b", "gemma3:4b"]  # Default fallback
if 'optimized_question' not in st.session_state:
    st.session_state.optimized_question = ""
if 'suggested_keywords' not in st.session_state:
    st.session_state.suggested_keywords = []
if 'selected_keywords' not in st.session_state:
    st.session_state.selected_keywords = []

@st.cache_resource
def load_vectorstore():
    """Load the vector store"""
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        persist_directory = "../VectorSpace/paper_vector_db_nomic-embed-text_latest_parent_child"
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return None

@st.cache_resource
def load_llm(model_name):
    """Load the LLM model"""
    try:
        return Ollama(model=model_name)
    except Exception as e:
        st.error(f"Failed to load LLM model {model_name}: {e}")
        return None

@st.cache_data
def get_available_ollama_models():
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
            st.warning("Could not fetch Ollama models. Using default models.")
            return ["qwen3:14b", "gemma3:4b"]
    except Exception as e:
        st.warning(f"Failed to fetch Ollama models: {e}. Using default models.")
        return ["qwen3:14b", "gemma3:4b"]

@st.cache_data
def load_paper_list():
    """Load the list of papers from the tracker CSV"""
    try:
        tracker_path = "../vectorization_tracker.csv"
        if os.path.exists(tracker_path):
            df = pd.read_csv(tracker_path)
            # Filter only vectorized papers
            vectorized_papers = df[df['vectorized'] == True]
            paper_list = []
            for _, row in vectorized_papers.iterrows():
                paper_info = {
                    'file_name': row['file_name'],
                    'file_path': row['file_path'],
                    'figure_count': row.get('figure_count', 0),
                    'has_figures': row.get('has_figure_descriptions', False),
                    'folder': os.path.basename(os.path.dirname(row['file_path']))
                }
                paper_list.append(paper_info)
            return paper_list, df
        else:
            st.warning("Vectorization tracker not found. Please run the vectorization process first.")
            return [], None
    except Exception as e:
        st.error(f"Failed to load paper list: {e}")
        return [], None

def get_paper_figures(paper_name):
    """Get figures for a specific paper"""
    try:
        extracted_images_dir = "../extracted_images"
        if not os.path.exists(extracted_images_dir):
            return []
        
        # Clean paper name for matching
        clean_paper_name = paper_name.replace('.pdf', '')
        
        # Find all figures for this paper
        figures = []
        for img_file in os.listdir(extracted_images_dir):
            if img_file.startswith(clean_paper_name) and img_file.endswith('.png'):
                figures.append(os.path.join(extracted_images_dir, img_file))
        
        return sorted(figures)
    except Exception as e:
        st.error(f"Failed to load figures: {e}")
        return []

def get_paper_abstract_and_keywords(paper_name):
    """Get abstract (child document) and keywords for a specific paper"""
    try:
        if not st.session_state.vectorstore:
            return None, None
        
        # Search for child document (abstract) of this paper
        results = st.session_state.vectorstore.similarity_search(
            f"abstract {paper_name}", 
            k=10,
            filter={"file_name": paper_name}
        )
        
        abstract_content = None
        keywords = None
        
        # Find the child document (abstract)
        for doc in results:
            if (doc.metadata.get('file_name') == paper_name and 
                doc.metadata.get('document_type') == 'child'):
                abstract_content = doc.page_content
                keywords = doc.metadata.get('keywords', '')
                break
        
        # If no child document found, try to find any document from this paper
        if not abstract_content:
            for doc in results:
                if doc.metadata.get('file_name') == paper_name:
                    # Extract first 500 characters as abstract
                    abstract_content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                    keywords = doc.metadata.get('keywords', '')
                    break
        
        return abstract_content, keywords
        
    except Exception as e:
        st.error(f"Failed to load abstract and keywords: {e}")
        return None, None

def optimize_question(original_question):
    """Use LLM to optimize the question for better retrieval"""
    if not st.session_state.llm:
        return original_question, []
    
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
        
        response = st.session_state.llm.invoke(optimization_prompt)
        
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
        st.error(f"Failed to optimize question: {e}")
        return original_question, []

def get_suggested_keywords():
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

def search_papers(question, selected_papers=None, search_type="both", k=5):
    """Search papers and generate answer"""
    if not st.session_state.vectorstore:
        return None, []
    
    try:
        # Search the vector store
        search_results = st.session_state.vectorstore.similarity_search(question, k=k*2)

        # If user has selected papers, but none of the search results match, 
        # try to fetch all chunks from the selected papers as fallback
        if selected_papers and len(selected_papers) > 0:
            filtered_results = [
                doc for doc in search_results 
                if doc.metadata.get('file_name') in selected_papers
            ]
            # If no results after filtering, try to fetch all chunks from selected papers
            if not filtered_results:
                # Try to get all docs from vectorstore that match selected_papers
                # This assumes your vectorstore has a method to get all docs or you have access to the full doc list
                if hasattr(st.session_state.vectorstore, "docs"):
                    all_docs = st.session_state.vectorstore.docs
                elif hasattr(st.session_state.vectorstore, "get_all_documents"):
                    all_docs = st.session_state.vectorstore.get_all_documents()
                else:
                    all_docs = []
                filtered_results = [
                    doc for doc in all_docs
                    if doc.metadata.get('file_name') in selected_papers
                ][:k]
            search_results = filtered_results

        # Filter by document type
        if search_type == "parent":
            search_results = [doc for doc in search_results if doc.metadata.get('document_type') == 'parent']
        elif search_type == "child":
            search_results = [doc for doc in search_results if doc.metadata.get('document_type') == 'child']
        
        # Take only k results
        search_results = search_results[:k]
        
        if not search_results:
            # If user selected papers, but nothing found, give a more helpful message
            if selected_papers and len(selected_papers) > 0:
                return (
                    "No relevant documents found for your question in the selected papers. "
                    "Try broadening your selection or rephrasing your question.",
                    []
                )
            else:
                return "No relevant documents found for your question.", []
        
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
        if st.session_state.llm:
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
            
            answer = st.session_state.llm.invoke(prompt)
            return answer, search_results
        else:
            return "LLM not available. Please check your model selection.", search_results
            
    except Exception as e:
        st.error(f"Search failed: {e}")
        return f"Error during search: {str(e)}", []

def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Paper Search & QA System</h1>', unsafe_allow_html=True)
    
    # Apply theme-based CSS dynamically
    if st.session_state.dark_theme:
        # Dark theme CSS
        st.markdown("""
        <style>
            /* =========================
               DARK THEME STYLES
               ========================= */
            
            /* CSS Variables for Dark Theme */
            :root {
                --bg-primary: #1a1a1a;
                --bg-secondary: #2d2d2d;
                --bg-card: rgba(45, 45, 45, 0.95);
                --bg-glass: rgba(45, 45, 45, 0.85);
                --text-primary: #ffffff;
                --text-secondary: #e0e0e0;
                --text-muted: #a0a0a0;
                --border-color: rgba(255, 255, 255, 0.2);
                --shadow-color: rgba(0, 0, 0, 0.5);
                --accent-color: #4a9eff;
                --success-color: #4caf50;
                --warning-color: #ff9800;
                --error-color: #f44336;
            }
            
            /* Background */
            .stApp {
                background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%) !important;
                background-attachment: fixed;
                color: var(--text-primary) !important;
            }
            
            /* Main Header */
            .main-header {
                font-size: 2.5rem;
                color: var(--text-primary) !important;
                text-align: center;
                margin-bottom: 2rem;
                text-shadow: 0 2px 4px rgba(0,0,0,0.8);
                font-weight: 700;
            }
            
            /* Glass Cards */
            .glass-card {
                background: var(--bg-glass) !important;
                backdrop-filter: blur(15px);
                -webkit-backdrop-filter: blur(15px);
                border: 2px solid var(--border-color) !important;
                border-radius: 16px;
                padding: 1rem;
                margin-bottom: 1rem;
                box-shadow: 0 8px 32px 0 var(--shadow-color);
                transition: all 0.3s ease;
                color: var(--text-primary) !important;
            }
            .glass-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 40px 0 var(--shadow-color);
                border-color: var(--accent-color);
            }
            .glass-card h3, .glass-card h4 {
                color: var(--text-primary) !important;
                margin-top: 0;
                font-weight: 600;
            }
            
            /* Content Cards */
            .content-card {
                background: var(--bg-card) !important;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 1px solid var(--border-color) !important;
                border-radius: 12px;
                padding: 1rem;
                margin-bottom: 1rem;
                color: var(--text-primary) !important;
                box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.3);
            }
            
            /* Optimize Card */
            .optimize-card {
                background: linear-gradient(135deg, var(--accent-color) 0%, #357abd 100%) !important;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 2px solid rgba(255, 255, 255, 0.3) !important;
                border-radius: 12px;
                padding: 1rem;
                margin-bottom: 1rem;
                color: #000000 !important;
                box-shadow: 0 6px 24px 0 rgba(74, 158, 255, 0.4);
            }
            .optimize-card h4 {
                color: #000000 !important;
            }
            
            /* Buttons */
            .stButton > button {
                background: linear-gradient(135deg, var(--accent-color) 0%, #357abd 100%) !important;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                color: #ffffff !important;
                border: 2px solid rgba(255, 255, 255, 0.3) !important;
                border-radius: 12px !important;
                padding: 0.6rem 1.2rem !important;
                font-weight: 600 !important;
                box-shadow: 0 4px 16px 0 rgba(74, 158, 255, 0.3) !important;
                transition: all 0.3s ease !important;
            }
            .stButton > button:hover {
                background: linear-gradient(135deg, #5ba7ff 0%, #4a9eff 100%) !important;
                transform: translateY(-1px) !important;
                box-shadow: 0 6px 20px 0 rgba(74, 158, 255, 0.5) !important;
                color: #ffffff !important;
            }
            
            /* Sidebar */
            .stSidebar {
                background: var(--bg-secondary) !important;
                border-right: 1px solid var(--border-color) !important;
            }
            .stSidebar .stMarkdown {
                color: var(--text-primary) !important;
            }
            .stSidebar .stButton > button {
                background: var(--bg-card) !important;
                color: var(--text-primary) !important;
                border: 1px solid var(--border-color) !important;
            }
            .stSidebar .stButton > button:hover {
                background: var(--bg-glass) !important;
                color: var(--text-primary) !important;
            }
            
            /* Inputs */
            .stTextArea > div > div > textarea {
                background: var(--bg-card) !important;
                backdrop-filter: blur(8px);
                -webkit-backdrop-filter: blur(8px);
                border: 2px solid var(--border-color) !important;
                border-radius: 8px !important;
                color: var(--text-primary) !important;
            }
            .stTextArea > div > div > textarea:focus {
                border-color: var(--accent-color) !important;
                box-shadow: 0 0 0 2px rgba(74, 158, 255, 0.2) !important;
            }
            
            .stSelectbox > div > div > div {
                background: var(--bg-card) !important;
                backdrop-filter: blur(8px);
                -webkit-backdrop-filter: blur(8px);
                border: 2px solid var(--border-color) !important;
                border-radius: 8px !important;
                color: var(--text-primary) !important;
            }
            
            /* Text Colors */
            .stMarkdown, .stText, p, span, div {
                color: var(--text-primary) !important;
            }
            
            /* Expanders */
            .stExpander {
                background: var(--bg-card) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: 8px !important;
            }
            .stExpander summary {
                color: var(--text-primary) !important;
                background: var(--bg-glass) !important;
            }
            
            /* Checkboxes */
            .stCheckbox {
                color: var(--text-primary) !important;
            }
            .stCheckbox label {
                color: var(--text-primary) !important;
            }
            
            /* Metrics */
            .stMetric {
                background: var(--bg-card) !important;
                color: var(--text-primary) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: 8px !important;
                padding: 0.5rem !important;
            }
            
            /* Keyword Chips */
            .keyword-chip {
                display: inline-block;
                background: var(--bg-card) !important;
                backdrop-filter: blur(6px);
                -webkit-backdrop-filter: blur(6px);
                border: 1px solid var(--border-color) !important;
                border-radius: 20px;
                padding: 0.3rem 0.8rem;
                margin: 0.2rem;
                font-size: 0.85rem;
                color: #000000 !important;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            .keyword-chip:hover {
                background: var(--accent-color) !important;
                color: #000000 !important;
                transform: scale(1.05);
            }
            
            /* Fix Streamlit specific elements */
            .css-1d391kg {
                background-color: var(--bg-secondary) !important;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
            }
            .css-1d391kg .stMarkdown {
                color: var(--text-primary) !important;
            }
            .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {
                color: var(--text-primary) !important;
            }
            
            /* Success/Warning/Error messages */
            .stSuccess {
                background: rgba(76, 175, 80, 0.2) !important;
                border: 1px solid var(--success-color) !important;
                color: var(--success-color) !important;
            }
            .stWarning {
                background: rgba(255, 152, 0, 0.2) !important;
                border: 1px solid var(--warning-color) !important;
                color: var(--warning-color) !important;
            }
            .stError {
                background: rgba(244, 67, 54, 0.2) !important;
                border: 1px solid var(--error-color) !important;
                color: var(--error-color) !important;
            }
            
            /* Spinner */
            .stSpinner {
                color: var(--accent-color) !important;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Light theme CSS
        st.markdown("""
        <style>
            /* =========================
               LIGHT THEME STYLES
               ========================= */
            
            /* CSS Variables for Light Theme */
            :root {
                --bg-primary: #ffffff;
                --bg-secondary: #f8f9fa;
                --bg-card: rgba(255, 255, 255, 0.95);
                --bg-glass: rgba(255, 255, 255, 0.85);
                --text-primary: #000000;
                --text-secondary: #333333;
                --text-muted: #666666;
                --border-color: rgba(0, 0, 0, 0.1);
                --shadow-color: rgba(0, 0, 0, 0.15);
                --accent-color: #4a9eff;
                --success-color: #28a745;
                --warning-color: #ffc107;
                --error-color: #dc3545;
            }
            
            /* Background */
            .stApp {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
                background-attachment: fixed;
                color: var(--text-primary) !important;
            }
            
            /* Main Header */
            .main-header {
                font-size: 2.5rem;
                color: var(--text-primary) !important;
                text-align: center;
                margin-bottom: 2rem;
                text-shadow: 0 2px 4px rgba(255,255,255,0.8);
                font-weight: 700;
            }
            
            /* Glass Cards */
            .glass-card {
                background: var(--bg-glass) !important;
                backdrop-filter: blur(15px);
                -webkit-backdrop-filter: blur(15px);
                border: 2px solid var(--border-color) !important;
                border-radius: 16px;
                padding: 1rem;
                margin-bottom: 1rem;
                box-shadow: 0 8px 32px 0 var(--shadow-color);
                transition: all 0.3s ease;
                color: var(--text-primary) !important;
            }
            .glass-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 40px 0 var(--shadow-color);
                border-color: var(--accent-color);
            }
            .glass-card h3, .glass-card h4 {
                color: var(--text-primary) !important;
                margin-top: 0;
                font-weight: 600;
            }
            
            /* Content Cards */
            .content-card {
                background: var(--bg-card) !important;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 1px solid var(--border-color) !important;
                border-radius: 12px;
                padding: 1rem;
                margin-bottom: 1rem;
                color: var(--text-primary) !important;
                box-shadow: 0 4px 16px 0 var(--shadow-color);
            }
            
            /* Optimize Card */
            .optimize-card {
                background: linear-gradient(135deg, var(--accent-color) 0%, #357abd 100%) !important;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 2px solid rgba(255, 255, 255, 0.3) !important;
                border-radius: 12px;
                padding: 1rem;
                margin-bottom: 1rem;
                color: #ffffff !important;
                box-shadow: 0 6px 24px 0 rgba(74, 158, 255, 0.4);
            }
            
            /* Buttons */
            .stButton > button {
                background: linear-gradient(135deg, var(--accent-color) 0%, #357abd 100%) !important;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                color: #ffffff !important;
                border: 2px solid rgba(255, 255, 255, 0.3) !important;
                border-radius: 12px !important;
                padding: 0.6rem 1.2rem !important;
                font-weight: 600 !important;
                box-shadow: 0 4px 16px 0 rgba(74, 158, 255, 0.3) !important;
                transition: all 0.3s ease !important;
            }
            .stButton > button:hover {
                background: linear-gradient(135deg, #5ba7ff 0%, #4a9eff 100%) !important;
                transform: translateY(-1px) !important;
                box-shadow: 0 6px 20px 0 rgba(74, 158, 255, 0.5) !important;
                color: #ffffff !important;
            }
            
            /* Sidebar */
            .stSidebar {
                background: var(--bg-secondary) !important;
                border-right: 1px solid var(--border-color) !important;
            }
            .stSidebar .stMarkdown {
                color: var(--text-primary) !important;
            }
            .stSidebar .stButton > button {
                background: var(--bg-card) !important;
                color: var(--text-primary) !important;
                border: 1px solid var(--border-color) !important;
            }
            .stSidebar .stButton > button:hover {
                background: var(--bg-glass) !important;
                color: var(--text-primary) !important;
            }
            
            /* Inputs */
            .stTextArea > div > div > textarea {
                background: var(--bg-card) !important;
                backdrop-filter: blur(8px);
                -webkit-backdrop-filter: blur(8px);
                border: 2px solid var(--border-color) !important;
                border-radius: 8px !important;
                color: var(--text-primary) !important;
            }
            .stTextArea > div > div > textarea:focus {
                border-color: var(--accent-color) !important;
                box-shadow: 0 0 0 2px rgba(74, 158, 255, 0.2) !important;
            }
            
            .stSelectbox > div > div > div {
                background: var(--bg-card) !important;
                backdrop-filter: blur(8px);
                -webkit-backdrop-filter: blur(8px);
                border: 2px solid var(--border-color) !important;
                border-radius: 8px !important;
                color: var(--text-primary) !important;
            }
            
            /* Text Colors */
            .stMarkdown, .stText, p, span, div {
                color: var(--text-primary) !important;
            }
            
            /* Expanders */
            .stExpander {
                background: var(--bg-card) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: 8px !important;
            }
            .stExpander summary {
                color: var(--text-primary) !important;
                background: var(--bg-glass) !important;
            }
            
            /* Checkboxes */
            .stCheckbox {
                color: var(--text-primary) !important;
            }
            .stCheckbox label {
                color: var(--text-primary) !important;
            }
            
            /* Metrics */
            .stMetric {
                background: var(--bg-card) !important;
                color: var(--text-primary) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: 8px !important;
                padding: 0.5rem !important;
            }
            
            /* Keyword Chips */
            .keyword-chip {
                display: inline-block;
                background: var(--bg-card) !important;
                backdrop-filter: blur(6px);
                -webkit-backdrop-filter: blur(6px);
                border: 1px solid var(--border-color) !important;
                border-radius: 20px;
                padding: 0.3rem 0.8rem;
                margin: 0.2rem;
                font-size: 0.85rem;
                color: var(--text-primary) !important;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            .keyword-chip:hover {
                background: var(--accent-color) !important;
                color: #ffffff !important;
                transform: scale(1.05);
            }
            
            /* Fix Streamlit specific elements */
            .css-1d391kg {
                background-color: var(--bg-secondary) !important;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
            }
            .css-1d391kg .stMarkdown {
                color: var(--text-primary) !important;
            }
            .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {
                color: var(--text-primary) !important;
            }
            
            /* Success/Warning/Error messages */
            .stSuccess {
                background: rgba(40, 167, 69, 0.1) !important;
                border: 1px solid var(--success-color) !important;
                color: var(--success-color) !important;
            }
            .stWarning {
                background: rgba(255, 193, 7, 0.1) !important;
                border: 1px solid var(--warning-color) !important;
                color: #856404 !important;
            }
            .stError {
                background: rgba(220, 53, 69, 0.1) !important;
                border: 1px solid var(--error-color) !important;
                color: var(--error-color) !important;
            }
            
            /* Spinner */
            .stSpinner {
                color: var(--accent-color) !important;
            }
        </style>
        """, unsafe_allow_html=True)
    
    # Load initial data
    if not st.session_state.paper_list:
        st.session_state.paper_list, st.session_state.tracker_df = load_paper_list()
    
    # Load available models
    if not st.session_state.available_models or st.session_state.available_models == ["qwen3:14b", "gemma3:4b"]:
        st.session_state.available_models = get_available_ollama_models()
    
    # Sidebar - All controls and settings
    with st.sidebar:
        # Theme toggle
        st.markdown("### 🎨 Theme")
        theme_col1, theme_col2 = st.columns(2)
        with theme_col1:
            if st.button("🌙 Dark"):
                st.session_state.dark_theme = True
                st.rerun()
        with theme_col2:
            if st.button("☀️ Light"):
                st.session_state.dark_theme = False
                st.rerun()
        
        st.divider()
        
        # Load vector store
        if st.session_state.vectorstore is None:
            with st.spinner("Loading vector store..."):
                st.session_state.vectorstore = load_vectorstore()
        
        # Paper Selection Section
        st.header("📚 Paper Selection")
        
        selected_papers = []
        if st.session_state.paper_list:
            # Group papers by folder
            folders = {}
            for paper in st.session_state.paper_list:
                folder = paper['folder']
                if folder not in folders:
                    folders[folder] = []
                folders[folder].append(paper)
            
            # Define folder order and icons
            folder_order = ["dislocation", "grainBoundary", "Precipitation", "SSS"]
            folder_icons = {
                "dislocation": "🔧",
                "grainBoundary": "🧱", 
                "Precipitation": "💧",
                "SSS": "🔬"
            }
            
            # Display papers by folder in specific order with selection
            for folder in folder_order:
                if folder in folders:
                    papers = folders[folder]
                    icon = folder_icons.get(folder, "📁")
                    
                    with st.expander(f"{icon} {folder.upper()} ({len(papers)} papers)", expanded=False):
                        folder_selected = []
                        
                        # Select all/none buttons for each folder
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"Select All", key=f"select_all_{folder}"):
                                for paper in papers:
                                    st.session_state[f"paper_{paper['file_name']}"] = True
                        with col2:
                            if st.button(f"Select None", key=f"select_none_{folder}"):
                                for paper in papers:
                                    st.session_state[f"paper_{paper['file_name']}"] = False
                        
                        # Individual paper checkboxes
                        for paper in sorted(papers, key=lambda x: x['file_name']):
                            fig_info = f" (🖼️ {paper['figure_count']})" if paper['figure_count'] > 0 else ""
                            paper_name = paper['file_name'].replace('.pdf', '')
                            
                            if st.checkbox(
                                f"{paper_name}{fig_info}", 
                                key=f"paper_{paper['file_name']}",
                                help=f"File: {paper['file_name']}\nFigures: {paper['figure_count']}"
                            ):
                                selected_papers.append(paper['file_name'])
                                folder_selected.append(paper['file_name'])
                        
                        # Show folder selection summary
                        if folder_selected:
                            st.success(f"✅ {len(folder_selected)} papers selected from {folder}")
            
            # Show total selection summary
            if selected_papers:
                st.success(f"📊 Total: {len(selected_papers)} papers selected")
            else:
                st.info("ℹ️ No papers selected - LLM will search all papers automatically")
        else:
            st.warning("No papers found. Please run vectorization first.")
        
        st.divider()
        
        # Settings Section - Now in sidebar and collapsible
        with st.expander("⚙️ Settings", expanded=True):
            # LLM model selection
            llm_model = st.selectbox(
                "Select LLM Model:",
                st.session_state.available_models,
                help="Available Ollama models on your system"
            )
            
            # Refresh models button
            if st.button("🔄 Refresh Models", help="Refresh available models", key="refresh_models"):
                st.session_state.available_models = get_available_ollama_models()
                st.rerun()
            
            # Search type
            search_type = st.selectbox(
                "Search Type:",
                ["both", "parent", "child"],
                format_func=lambda x: {"both": "Both (Abstract + Full)", "parent": "Full Text + Figures", "child": "Abstract Only"}[x]
            )
            
            # Number of results
            num_results = st.slider("Number of Results:", 1, 10, 5)
        
        st.divider()
        
        
        # Upload new paper section - Now collapsible
        with st.expander("📤 Upload New Paper", expanded=False):
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
            if uploaded_file is not None:
                if st.button("Process Paper"):
                    # TODO: Implement paper processing
                    st.info("Paper processing feature coming soon!")
        st.divider()
         # Vector Store Status - Now collapsible
        with st.expander("📊 Vector Store Status", expanded=False):
            if st.session_state.vectorstore:
                st.success("✅ Vector store loaded")
                
                # Display vector store name
                persist_directory = "../VectorSpace/paper_vector_db_nomic-embed-text_latest_parent_child"
                vector_store_name = os.path.basename(persist_directory)
                st.info(f"📂 Vector store: {vector_store_name}")
                
                # Display paper statistics
                if st.session_state.tracker_df is not None:
                    total_papers = len(st.session_state.tracker_df)
                    vectorized_papers = st.session_state.tracker_df['vectorized'].sum()
                    total_figures = st.session_state.tracker_df['figure_count'].sum()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Papers", total_papers)
                        st.metric("Vectorized", vectorized_papers)
                    with col2:
                        st.metric("Figures", int(total_figures))
            else:
                st.error("❌ Failed to load vector store")
                return
        
    
    # Main content area - simplified layout
    if selected_papers:
        # Give more space to preview when papers are selected
        col1, col2 = st.columns([1.5, 1.5])
    else:
        # Default layout
        col1, col2 = st.columns([2, 1])
    
    with col1:
        # Question input card
        st.markdown("""
        <div class="glass-card" style="margin-bottom: 1rem;">
            <h3 style="margin-top: 0;">💬 Ask Questions</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Question input
        question = st.text_area(
            "Your Question:",
            placeholder="Ask about precipitation strengthening, microstructure, mechanical properties, etc.",
            height=100
        )
        
        # Question optimization section
        col_opt1, col_opt2 = st.columns([1, 1])
        
        with col_opt1:
            if st.button("🧠 Optimize Question", help="Let AI optimize your question for better search results"):
                if question.strip():
                    # Load LLM if not already loaded
                    if st.session_state.llm is None:
                        # Use the first available model for optimization
                        with st.spinner("Loading LLM for optimization..."):
                            st.session_state.llm = load_llm(st.session_state.available_models[0])
                    
                    if st.session_state.llm:
                        with st.spinner("Optimizing question..."):
                            optimized_q, keywords = optimize_question(question)
                            st.session_state.optimized_question = optimized_q
                            st.session_state.suggested_keywords = keywords
                    else:
                        st.error("Failed to load LLM for optimization")
                else:
                    st.warning("Please enter a question first")
        
        with col_opt2:
            if st.button("🔑 Show Keywords", help="Show suggested keywords for better search"):
                st.session_state.suggested_keywords = get_suggested_keywords()
        
        # Display optimized question if available
        if st.session_state.optimized_question:
            st.markdown("""
            <div class="optimize-card" style="margin-bottom: 0.5rem; padding: 0.5rem 0.75rem;">
                <h4 style="margin-top: 0; margin-bottom: 0.5rem;">🎯 Optimized Question</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(
                f'<div class="content-card" style="margin: 0.25rem 0 0.5rem 0; padding: 0.5rem 0.75rem;">{st.session_state.optimized_question}</div>',
                unsafe_allow_html=True
            )
            
            # Only show the "Ask Question (Optimized)" button, removing "Use Optimized Question"
            if st.button("🔍 Ask Question (Optimized)", key="ask_optimized_q", type="primary"):
                if st.session_state.optimized_question.strip():
                    # Load LLM if not already loaded or if model changed
                    if st.session_state.llm is None or st.session_state.get('current_model') != llm_model:
                        with st.spinner(f"Loading {llm_model}..."):
                            st.session_state.llm = load_llm(llm_model)
                            st.session_state.current_model = llm_model
                    if st.session_state.llm:
                        # Enhance question with selected keywords
                        enhanced_question = st.session_state.optimized_question
                        if st.session_state.selected_keywords:
                            keywords_text = " ".join(st.session_state.selected_keywords)
                            enhanced_question = f"{st.session_state.optimized_question} {keywords_text}"
                        with st.spinner("Searching and generating answer..."):
                            answer, search_results = search_papers(
                                enhanced_question, 
                                selected_papers if selected_papers else None,
                                search_type,
                                num_results
                            )
                        # Display answer card
                        st.markdown("""
                        <div class="glass-card" style="margin-bottom: 0.5rem; padding: 0.5rem 0.75rem;">
                            <h3 style="margin-top: 0;">📝 Answer</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(f"Generated using: {llm_model}")
                        st.markdown(f'<div class="content-card" style="margin: 0.25rem 0 0.5rem 0; padding: 0.5rem 0.75rem;">{answer}</div>', unsafe_allow_html=True)
                        # Display sources card
                        if search_results:
                            st.markdown("""
                            <div class="glass-card" style="margin-bottom: 0.5rem; padding: 0.5rem 0.75rem;">
                                <h3 style="margin-top: 0;">📚 Sources</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            for i, doc in enumerate(search_results):
                                with st.expander(f"Source {i+1}: {doc.metadata.get('file_name', 'Unknown')} [{doc.metadata.get('document_type', 'unknown').upper()}]"):
                                    st.markdown(f"""
                                    <div class="content-card" style="margin-bottom: 0.25rem; padding: 0.5rem 0.75rem;">
                                        <strong>Section:</strong> {doc.metadata.get('section', 'Unknown')}<br>
                                        <strong>Content Length:</strong> {len(doc.page_content)} characters<br>
                                        {'<strong>Figures:</strong> ' + str(doc.metadata.get('figure_count', 0)) + '<br>' if doc.metadata.get('figure_count', 0) > 0 else ''}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.markdown("**Preview:**")
                                    preview_text = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                                    st.markdown(f'<div class="content-card" style="font-size: 0.9em; margin: 0.25rem 0; padding: 0.5rem 0.75rem;">{preview_text}</div>', unsafe_allow_html=True)
                    else:
                        st.error("Failed to load LLM model")
                else:
                    st.warning("Please enter a question")
        
        # Keyword selection section
        if st.session_state.suggested_keywords:
            st.markdown("""
            <div class="glass-card" style="margin-bottom: 0.5rem; padding: 0.5rem 0.75rem;">
                <h4 style="margin-top: 0; margin-bottom: 0.5rem;">🏷️ Select Keywords</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Select keywords to enhance your search:**")
            cols = st.columns(3)
            for i, keyword in enumerate(st.session_state.suggested_keywords):
                with cols[i % 3]:
                    if st.checkbox(keyword, key=f"keyword_{keyword}"):
                        if keyword not in st.session_state.selected_keywords:
                            st.session_state.selected_keywords.append(keyword)
                    else:
                        if keyword in st.session_state.selected_keywords:
                            st.session_state.selected_keywords.remove(keyword)
            
            # Clear keywords button
            col_clear1, col_clear2 = st.columns([1, 1])
            with col_clear1:
                if st.button("🗑️ Clear Keywords"):
                    st.session_state.selected_keywords = []
                    st.rerun()
            with col_clear2:
                if st.button("🔄 Refresh Keywords"):
                    st.session_state.suggested_keywords = get_suggested_keywords()
                    st.rerun()
        
        # Show selected keywords
        if st.session_state.selected_keywords:
            st.markdown("**Selected Keywords:**")
            selected_keywords_text = " • ".join(st.session_state.selected_keywords)
            st.markdown(f'<div class="content-card" style="background: rgba(0, 0, 0, 0.1); margin: 0.25rem 0 0.5rem 0; padding: 0.5rem 0.75rem;">{selected_keywords_text}</div>', unsafe_allow_html=True)
        
        # Ask button - now prominent and standalone (only if not showing optimized question)
        if not st.session_state.optimized_question:
            st.markdown("---")
            if st.button("🔍 Ask Question", type="primary", use_container_width=True):
                if question.strip():
                    # Load LLM if not already loaded or if model changed
                    if st.session_state.llm is None or st.session_state.get('current_model') != llm_model:
                        with st.spinner(f"Loading {llm_model}..."):
                            st.session_state.llm = load_llm(llm_model)
                            st.session_state.current_model = llm_model
                    
                    if st.session_state.llm:
                        # Enhance question with selected keywords
                        enhanced_question = question
                        if st.session_state.selected_keywords:
                            keywords_text = " ".join(st.session_state.selected_keywords)
                            enhanced_question = f"{question} {keywords_text}"
                        
                        with st.spinner("Searching and generating answer..."):
                            answer, search_results = search_papers(
                                enhanced_question, 
                                selected_papers if selected_papers else None,
                                search_type,
                                num_results
                            )
                        
                        # Display answer card - full width for optimized question
                        st.markdown("""
                        <div class="glass-card" style="margin-bottom: 0.5rem; padding: 0.5rem 0.75rem; width: 100%; box-sizing: border-box;">
                            <h3 style="margin-top: 0;">📝 Answer</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.caption(f"Generated using: {llm_model}")
                        st.markdown(f'''
                        <div class="content-card" style="
                            margin: 0.25rem 0 0.5rem 0;
                            padding: 0.5rem 0.75rem;
                            width: 100%;
                            box-sizing: border-box;
                            background: rgba(0,0,0,0.1);
                        ">{answer}</div>
                        ''', unsafe_allow_html=True)
                        
                        # Display sources card
                        if search_results:
                            st.markdown("""
                            <div class="glass-card" style="margin-bottom: 0.5rem; padding: 0.5rem 0.75rem; width: 100%; box-sizing: border-box;">
                                <h3 style="margin-top: 0;">📚 Sources</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for i, doc in enumerate(search_results):
                                with st.expander(f"Source {i+1}: {doc.metadata.get('file_name', 'Unknown')} [{doc.metadata.get('document_type', 'unknown').upper()}]"):
                                    st.markdown(f"""
                                    <div class="content-card" style="margin-bottom: 0.25rem; padding: 0.5rem 0.75rem; width: 100%; box-sizing: border-box;">
                                        <strong>Section:</strong> {doc.metadata.get('section', 'Unknown')}<br>
                                        <strong>Content Length:</strong> {len(doc.page_content)} characters<br>
                                        {'<strong>Figures:</strong> ' + str(doc.metadata.get('figure_count', 0)) + '<br>' if doc.metadata.get('figure_count', 0) > 0 else ''}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    st.markdown("**Preview:**")
                                    preview_text = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                                    st.markdown(f'''
                                    <div class="content-card" style="
                                        font-size: 0.9em;
                                        margin: 0.25rem 0;
                                        padding: 0.5rem 0.75rem;
                                        width: 100%;
                                        box-sizing: border-box;
                                    ">{preview_text}</div>
                                    ''', unsafe_allow_html=True)
                    else:
                        st.error("Failed to load LLM model")
                else:
                    st.warning("Please enter a question")
    
    with col2:
        st.header("🖼️ Paper Preview")
        
        # Adjust column width based on selection
        if selected_papers:
           
            
            # Show detailed preview for selected papers
            for paper_name in selected_papers:
                paper_info = next((p for p in st.session_state.paper_list if p['file_name'] == paper_name), None)
                if paper_info:
                    with st.expander(f"📄 {paper_name.replace('.pdf', '')}", expanded=True):
                        
                        # Get paper abstract and metadata
                        abstract_content, keywords = get_paper_abstract_and_keywords(paper_name)
                        
                        # Display abstract
                        if abstract_content:
                            st.markdown("**📝 Abstract:**")
                            st.markdown(f'<div class="content-card" style="font-size: 0.9em; max-height: 200px; overflow-y: auto;">{abstract_content[:500]}{"..." if len(abstract_content) > 500 else ""}</div>', unsafe_allow_html=True)
                        
                        # Display keywords
                        if keywords:
                            st.markdown("**🔑 Keywords:**")
                            st.markdown(f'<div class="content-card" style="font-size: 0.85em; background: rgba(0, 0, 0, 0.1); color: #000000;">{keywords}</div>', unsafe_allow_html=True)
                        
                        # Display figures
                        figures = get_paper_figures(paper_name)
                        if figures:
                            st.markdown("**🖼️ Figures:**")
                            for fig_path in figures[:3]:  # Show max 3 figures per paper
                                try:
                                    image = Image.open(fig_path)
                                    st.image(image, caption=os.path.basename(fig_path), use_container_width=True)
                                except Exception as e:
                                    st.error(f"Failed to load image: {e}")
                        else:
                            st.info("No figures available for this paper")
                        
                        # Paper stats
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Figures", paper_info['figure_count'])
                        with col2:
                            st.metric("Folder", paper_info['folder'])
        
        elif 'search_results' in locals() and search_results:
            # Show figures from search results
            st.markdown("### 🔍 Search Results Preview")
            papers_in_results = list(set([doc.metadata.get('file_name') for doc in search_results]))
            
            for paper_name in papers_in_results:
                if paper_name and paper_name != 'Unknown':
                    with st.expander(f"📄 {paper_name.replace('.pdf', '')}", expanded=False):
                        
                        # Get paper abstract and metadata
                        abstract_content, keywords = get_paper_abstract_and_keywords(paper_name)
                        
                        # Display abstract
                        if abstract_content:
                            st.markdown("**📝 Abstract:**")
                            st.markdown(f'<div class="content-card" style="font-size: 0.9em; max-height: 150px; overflow-y: auto;">{abstract_content[:300]}{"..." if len(abstract_content) > 300 else ""}</div>', unsafe_allow_html=True)
                        
                        # Display keywords
                        if keywords:
                            st.markdown("**🔑 Keywords:**")
                            st.markdown(f'<div class="content-card" style="font-size: 0.85em; background: rgba(0, 0, 0, 0.1); color: #000000;">{keywords}</div>', unsafe_allow_html=True)
                        
                        # Display figures
                        figures = get_paper_figures(paper_name)
                        if figures:
                            st.markdown("**🖼️ Figures:**")
                            for fig_path in figures[:2]:  # Show max 2 figures per paper
                                try:
                                    image = Image.open(fig_path)
                                    st.image(image, caption=os.path.basename(fig_path), use_container_width=True)
                                except Exception as e:
                                    st.error(f"Failed to load image: {e}")
        else:
            st.info("📋 Select papers or ask questions to see previews here")
            
            # Show sample instruction
            st.markdown("""
            **How to use:**
            1. Select papers from the sidebar, or
            2. Ask a question to see relevant papers
            3. View abstracts, keywords, and figures here
            """)

if __name__ == "__main__":
    main() 