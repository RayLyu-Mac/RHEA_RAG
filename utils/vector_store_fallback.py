"""
Fallback vector store utilities using FAISS instead of Chroma
Use this when Chroma fails due to SQLite version issues
"""

import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from typing import List, Optional, Tuple, Any
from langchain.schema import Document
import os
import pickle


@st.cache_resource
def load_vectorstore_fallback(persist_directory: str = "./VectorSpace/paper_vector_db_nomic-embed-text_latest_parent_child"):
    """Load vector store using FAISS as fallback when Chroma fails"""
    try:
        # Try to load embeddings
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        except Exception as emb_error:
            st.error(f"❌ Failed to load embeddings: {emb_error}")
            st.info("💡 Please ensure Ollama is running and the 'nomic-embed-text:latest' model is available.")
            return None
        
        # Check if FAISS index exists
        faiss_index_path = os.path.join(persist_directory, "faiss_index")
        if os.path.exists(faiss_index_path):
            try:
                vectorstore = FAISS.load_local(faiss_index_path, embeddings)
                st.success("✅ FAISS vector store loaded successfully!")
                return vectorstore
            except Exception as e:
                st.warning(f"⚠️ Failed to load existing FAISS index: {e}")
        
        # If no FAISS index exists, try to convert from Chroma data
        st.info("🔄 No FAISS index found. Attempting to convert from Chroma data...")
        
        # This would require implementing a conversion function
        # For now, return None and suggest manual conversion
        st.error("❌ FAISS index not found and conversion not implemented")
        st.info("💡 You'll need to re-vectorize your papers using FAISS")
        return None
        
    except Exception as e:
        st.error(f"❌ Unexpected error loading FAISS vector store: {e}")
        return None


def search_papers_fallback(vectorstore, question: str, selected_papers: Optional[List[str]] = None, 
                          search_type: str = "both", k: int = 5) -> Tuple[List[Document], bool]:
    """
    Search papers in the FAISS vector store
    """
    if not vectorstore:
        return [], False
    
    try:
        # FAISS search
        search_results = vectorstore.similarity_search(question, k=k*2)
        
        # Filter by selected papers if specified
        if selected_papers and len(selected_papers) > 0:
            search_results = [
                doc for doc in search_results 
                if doc.metadata.get('file_name') in selected_papers
            ]
        
        # Filter by document type
        if search_type == "parent":
            search_results = [doc for doc in search_results if doc.metadata.get('document_type') == 'parent']
        elif search_type == "child":
            search_results = [doc for doc in search_results if doc.metadata.get('document_type') == 'child']
        
        # Take only k results
        search_results = search_results[:k]
        
        return search_results, True
        
    except Exception as e:
        st.error(f"FAISS search failed: {e}")
        return [], False


def get_paper_abstract_and_keywords_fallback(vectorstore, paper_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Get abstract and keywords using FAISS"""
    try:
        if not vectorstore:
            return None, None
        
        # Search for documents from this paper
        results = vectorstore.similarity_search(
            f"abstract {paper_name}", 
            k=10
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
        st.error(f"Failed to load abstract and keywords from FAISS: {e}")
        return None, None


def create_faiss_from_chroma_data(chroma_directory: str, output_directory: str = "./VectorSpace/faiss_index"):
    """
    Convert Chroma data to FAISS format
    This is a placeholder function - would need to be implemented
    """
    st.info("🔄 FAISS conversion function not yet implemented")
    st.info("💡 You would need to:")
    st.info("1. Extract documents from Chroma")
    st.info("2. Create embeddings")
    st.info("3. Build FAISS index")
    st.info("4. Save to disk")
    
    return None 