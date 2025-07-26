"""
Fallback vector store utilities using FAISS instead of Chroma
Use this when Chroma fails due to SQLite version issues
"""

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from typing import List, Optional, Tuple, Any
from langchain.schema import Document
import os
import pickle


def add_system_message(message_type: str, message: str):
    """Add a system message to the session state for display in sidebar"""
    if 'system_messages' not in st.session_state:
        st.session_state.system_messages = []
    
    st.session_state.system_messages.append({
        'type': message_type,
        'message': message
    })

@st.cache_resource
def load_vectorstore_fallback(persist_directory: str = "./VectorSpace/paper_vector_db_faiss"):
    """Load FAISS vector store as fallback"""
    try:
        # Try to load embeddings
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        except Exception as emb_error:
            add_system_message('error', f"❌ Failed to load embeddings: {emb_error}")
            add_system_message('info', "💡 Please ensure Ollama is running and the 'nomic-embed-text:latest' model is available.")
            return None
        
        # Check if FAISS index exists
        import os
        faiss_index_path = os.path.join(persist_directory, "index.faiss")
        if not os.path.exists(faiss_index_path):
            add_system_message('info', "🔄 No FAISS index found. Attempting to convert from Chroma data...")
            # Here you would implement conversion from Chroma to FAISS
            # For now, just create an empty FAISS store
            add_system_message('info', "💡 You'll need to re-vectorize your papers using FAISS")
            return None
        
        # Load FAISS vector store
        try:
            vectorstore = FAISS.load_local(persist_directory, embeddings)
            add_system_message('success', "✅ FAISS vector store loaded successfully!")
            return vectorstore
        except Exception as faiss_error:
            add_system_message('error', f"❌ Failed to load FAISS vector store: {faiss_error}")
            return None
            
    except Exception as e:
        add_system_message('error', f"❌ Unexpected error loading FAISS vector store: {e}")
        return None


def search_papers_fallback(vectorstore, query: str, selected_papers: Optional[List[str]] = None, 
                          search_type: str = "both", num_results: int = 5) -> Tuple[List[Document], bool]:
    """Search papers using FAISS fallback"""
    try:
        if vectorstore is None:
            return [], False
        
        # Perform similarity search
        results = vectorstore.similarity_search(query, k=num_results)
        
        # Filter by selected papers if specified
        if selected_papers:
            filtered_results = []
            for doc in results:
                file_name = doc.metadata.get('file_name', '')
                if file_name in selected_papers:
                    filtered_results.append(doc)
            results = filtered_results
        
        return results, True
        
    except Exception as e:
        add_system_message('error', f"❌ Error during FAISS search: {e}")
        return [], False


def get_paper_abstract_and_keywords_fallback(vectorstore, file_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Get paper abstract and keywords using FAISS fallback"""
    try:
        if vectorstore is None:
            return None, None
        
        # Search for the specific paper
        results = vectorstore.similarity_search(f"file_name:{file_name}", k=1)
        
        if results:
            doc = results[0]
            # Extract abstract from content (assuming it's in the first part)
            content = doc.page_content
            abstract = content[:1000] if len(content) > 1000 else content  # First 1000 chars as abstract
            
            # For keywords, we could implement keyword extraction here
            keywords = "Keywords extraction not implemented in FAISS fallback"
            
            return abstract, keywords
        else:
            return None, None
            
    except Exception as e:
        add_system_message('error', f"❌ Error getting paper data from FAISS: {e}")
        return None, None


def convert_chroma_to_faiss(chroma_persist_directory: str, faiss_persist_directory: str):
    """Convert Chroma vector store to FAISS format"""
    add_system_message('info', "🔄 FAISS conversion function not yet implemented")
    add_system_message('info', "💡 You would need to:")
    add_system_message('info', "1. Extract documents from Chroma")
    add_system_message('info', "2. Create embeddings")
    add_system_message('info', "3. Build FAISS index")
    add_system_message('info', "4. Save to disk")
    
    # This is a placeholder for the conversion logic
    # In a real implementation, you would:
    # 1. Load the Chroma collection
    # 2. Extract all documents and their embeddings
    # 3. Create a FAISS index
    # 4. Save the FAISS index to disk
    
    return False 