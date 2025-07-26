"""
Vector store utilities for the Paper Search & QA System.
Handles loading, searching, and managing the Chroma vector database.
"""

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from typing import List, Optional, Tuple, Any
from langchain.schema import Document


def add_system_message(message_type: str, message: str):
    """Add a system message to the session state for display in sidebar"""
    if 'system_messages' not in st.session_state:
        st.session_state.system_messages = []
    
    st.session_state.system_messages.append({
        'type': message_type,
        'message': message
    })

@st.cache_resource
def load_vectorstore(persist_directory: str = "./VectorSpace/paper_vector_db_nomic-embed-text_latest_parent_child"):
    """Load the vector store with enhanced error handling for SQLite issues"""
    try:
        # First, check SQLite version
        import sqlite3
        sqlite_version = sqlite3.sqlite_version
        add_system_message('info', f"📊 SQLite version detected: {sqlite_version}")
        
        # Check if the persist directory exists
        import os
        if not os.path.exists(persist_directory):
            add_system_message('error', f"❌ Vector store directory not found: {persist_directory}")
            add_system_message('info', "💡 Please ensure the vector store has been created and the path is correct.")
            return None
        
        # Try to load embeddings
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        except Exception as emb_error:
            add_system_message('error', f"❌ Failed to load embeddings: {emb_error}")
            add_system_message('info', "💡 Please ensure Ollama is running and the 'nomic-embed-text:latest' model is available.")
            return None
        
        # Try to load Chroma with specific error handling
        try:
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            
            # Test the vector store by trying to get collection info
            try:
                collection = vectorstore._collection
                if collection:
                    count = collection.count()
                    add_system_message('success', f"✅ Vector store loaded successfully! Found {count} documents.")
                else:
                    add_system_message('warning', "⚠️ Vector store loaded but collection appears empty.")
            except Exception as test_error:
                add_system_message('warning', f"⚠️ Vector store loaded but couldn't verify contents: {test_error}")
            
            return vectorstore
            
        except Exception as chroma_error:
            error_str = str(chroma_error)
            
            # Handle specific SQLite version error
            if "sqlite3" in error_str.lower() and "3.35.0" in error_str:
                add_system_message('error', "❌ SQLite version compatibility issue detected!")
                add_system_message('error', f"Current SQLite version: {sqlite_version}")
                add_system_message('error', "Chroma requires SQLite ≥ 3.35.0")
                
                # Provide solutions
                add_system_message('info', "**Solutions:**")
                add_system_message('info', "1. **Update Python environment** (recommended):")
                add_system_message('info', "   ```bash")
                add_system_message('info', "   pip install --upgrade chromadb")
                add_system_message('info', "   pip install --upgrade pysqlite3-binary  # Optional")
                add_system_message('info', "   ```")
                
                add_system_message('info', "2. **Use a different vector store backend:**")
                add_system_message('info', "   - Consider using FAISS or other backends")
                add_system_message('info', "   - Or use in-memory Chroma")
                
                # Try in-memory fallback
                add_system_message('info', "🔄 Attempting to use in-memory Chroma as fallback...")
                try:
                    vectorstore = Chroma(embedding_function=embeddings)
                    add_system_message('success', "✅ Successfully loaded in-memory Chroma vector store!")
                    add_system_message('warning', "⚠️ Note: This is an empty in-memory store. You'll need to re-vectorize your papers.")
                    return vectorstore
                except Exception as mem_error:
                    add_system_message('error', f"❌ In-memory fallback also failed: {mem_error}")
                
                add_system_message('info', "3. **Check for multiple Python installations:**")
                add_system_message('info', "   - Ensure you're using the correct Python environment")
                add_system_message('info', "   - Check if conda/pyenv is affecting SQLite detection")
                
                return None
            
            # Handle other Chroma errors
            else:
                add_system_message('error', f"❌ Failed to load Chroma vector store: {chroma_error}")
                add_system_message('info', "💡 This might be due to:")
                add_system_message('info', "- Corrupted vector store files")
                add_system_message('info', "- Missing or incompatible embeddings model")
                add_system_message('info', "- Permission issues with the persist directory")
                return None
                
    except Exception as e:
        add_system_message('error', f"❌ Unexpected error loading vector store: {e}")
        return None


def search_papers(vectorstore, question: str, selected_papers: Optional[List[str]] = None, 
                 search_type: str = "both", k: int = 5) -> Tuple[List[Document], bool]:
    """
    Search papers in the vector store with enhanced logic for selected papers.
    
    Args:
        vectorstore: The loaded Chroma vector store
        question: The search query
        selected_papers: List of selected paper filenames (optional)
        search_type: "both", "parent", or "child"
        k: Number of results to return
    
    Returns:
        Tuple of (search_results, success_flag)
    """
    if not vectorstore:
        # Fallback: Return papers from CSV data if vector store is not available
        try:
            add_system_message('warning', "⚠️ Vector store not available, using CSV fallback for search")
            
            import streamlit as st
            from langchain.schema import Document
            
            if hasattr(st, 'session_state') and 'tracker_df' in st.session_state and st.session_state.tracker_df is not None:
                df = st.session_state.tracker_df
                vectorized_papers = df[df['vectorized'] == True]
                
                # Filter by selected papers if specified
                if selected_papers and len(selected_papers) > 0:
                    vectorized_papers = vectorized_papers[vectorized_papers['file_name'].isin(selected_papers)]
                
                # Create simple Document objects from CSV data
                search_results = []
                for _, row in vectorized_papers.head(k).iterrows():
                    # Create more detailed content from CSV data
                    paper_title = row['file_name'].replace('.pdf', '').replace('_', ' ')
                    folder = row.get('folder', 'Unknown')
                    figure_count = row.get('figure_count', 0)
                    vectorized_date = row.get('vectorized_date', 'Unknown')
                    vectorized_model = row.get('vectorized_model', 'Unknown')
                    chunk_count = row.get('chunk_count', 0)
                    
                    content = f"""**Paper Title:** {paper_title}

**Folder:** {folder}
**Figures:** {figure_count}
**Vectorized Date:** {vectorized_date}
**Vectorization Model:** {vectorized_model}
**Content Chunks:** {chunk_count}

This paper has been processed and vectorized in the database. The full content and abstract are available through the vector store when Ollama is running.

**Note:** To access the complete paper content, ensure Ollama is running and the vector store is accessible."""
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            'file_name': row['file_name'],
                            'document_type': 'parent',
                            'content_type': 'research_paper',
                            'title': paper_title,
                            'folder': folder,
                            'figure_count': figure_count,
                            'vectorized_date': vectorized_date,
                            'vectorized_model': vectorized_model,
                            'chunk_count': chunk_count
                        }
                    )
                    search_results.append(doc)
                
                add_system_message('info', f"✅ Used CSV fallback, found {len(search_results)} papers")
                return search_results, True
            else:
                add_system_message('error', "❌ No CSV data available for fallback")
                return [], False
                
        except Exception as fallback_error:
            add_system_message('error', f"❌ CSV fallback failed: {fallback_error}")
            return [], False
    
    try:
        # If specific papers are selected, search with a broader query and larger k
        if selected_papers and len(selected_papers) > 0:
            # Search with larger k to increase chances of finding selected papers
            search_results = vectorstore.similarity_search(question, k=k*4)
            
            # Filter by selected papers
            filtered_results = [
                doc for doc in search_results 
                if doc.metadata.get('file_name') in selected_papers
            ]
            
            # If we still don't have enough results, try different search strategies
            if len(filtered_results) < k:
                # Try searching for each paper individually with broader terms
                additional_results = []
                for paper_name in selected_papers:
                    # Search specifically within this paper
                    try:
                        # Try with filter first (if supported)
                        paper_results = vectorstore.similarity_search(
                            f"{question} {paper_name.replace('.pdf', '')}", 
                            k=k,
                            filter={"file_name": paper_name}
                        )
                    except Exception:
                        # Fallback: search with broader query and filter manually
                        paper_results = vectorstore.similarity_search(
                            f"{question} {paper_name.replace('.pdf', '')}", 
                            k=k*2
                        )
                        paper_results = [
                            doc for doc in paper_results 
                            if doc.metadata.get('file_name') == paper_name
                        ]
                    additional_results.extend(paper_results)
                
                # Combine and deduplicate results
                all_results = filtered_results.copy()
                seen_ids = set([doc.metadata.get('document_id', '') for doc in filtered_results])
                
                for doc in additional_results:
                    doc_id = doc.metadata.get('document_id', '')
                    if doc_id not in seen_ids:
                        all_results.append(doc)
                        seen_ids.add(doc_id)
                
                filtered_results = all_results
            
            # If still no results, try a very broad search within selected papers
            if not filtered_results:
                for paper_name in selected_papers:
                    # Try searching just for the paper name
                    try:
                        broad_results = vectorstore.similarity_search(
                            paper_name.replace('.pdf', ''),
                            k=2,
                            filter={"file_name": paper_name}
                        )
                    except Exception:
                        # Fallback: search broadly and filter manually
                        broad_results = vectorstore.similarity_search(
                            paper_name.replace('.pdf', ''),
                            k=4
                        )
                        broad_results = [
                            doc for doc in broad_results 
                            if doc.metadata.get('file_name') == paper_name
                        ]
                    filtered_results.extend(broad_results)
            
            search_results = filtered_results
        else:
            # No specific papers selected, search normally
            search_results = vectorstore.similarity_search(question, k=k*2)

        # Filter by document type
        if search_type == "parent":
            search_results = [doc for doc in search_results if doc.metadata.get('document_type') == 'parent']
        elif search_type == "child":
            search_results = [doc for doc in search_results if doc.metadata.get('document_type') == 'child']
        
        # Take only k results
        search_results = search_results[:k]
        
        return search_results, True
        
    except Exception as e:
        # Fallback: Try CSV data if vector store search fails
        try:
            add_system_message('warning', f"⚠️ Vector store search failed, trying CSV fallback")
            
            import streamlit as st
            from langchain.schema import Document
            
            if hasattr(st, 'session_state') and 'tracker_df' in st.session_state and st.session_state.tracker_df is not None:
                df = st.session_state.tracker_df
                vectorized_papers = df[df['vectorized'] == True]
                
                # Filter by selected papers if specified
                if selected_papers and len(selected_papers) > 0:
                    vectorized_papers = vectorized_papers[vectorized_papers['file_name'].isin(selected_papers)]
                
                # Create simple Document objects from CSV data
                search_results = []
                for _, row in vectorized_papers.head(k).iterrows():
                    # Create more detailed content from CSV data
                    paper_title = row['file_name'].replace('.pdf', '').replace('_', ' ')
                    folder = row.get('folder', 'Unknown')
                    figure_count = row.get('figure_count', 0)
                    vectorized_date = row.get('vectorized_date', 'Unknown')
                    vectorized_model = row.get('vectorized_model', 'Unknown')
                    chunk_count = row.get('chunk_count', 0)
                    
                    content = f"""**Paper Title:** {paper_title}

**Folder:** {folder}
**Figures:** {figure_count}
**Vectorized Date:** {vectorized_date}
**Vectorization Model:** {vectorized_model}
**Content Chunks:** {chunk_count}

This paper has been processed and vectorized in the database. The full content and abstract are available through the vector store when Ollama is running.

**Note:** To access the complete paper content, ensure Ollama is running and the vector store is accessible."""
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            'file_name': row['file_name'],
                            'document_type': 'parent',
                            'content_type': 'research_paper',
                            'title': paper_title,
                            'folder': folder,
                            'figure_count': figure_count,
                            'vectorized_date': vectorized_date,
                            'vectorized_model': vectorized_model,
                            'chunk_count': chunk_count
                        }
                    )
                    search_results.append(doc)
                
                add_system_message('info', f"✅ Used CSV fallback, found {len(search_results)} papers")
                return search_results, True
            else:
                add_system_message('error', "❌ No CSV data available for fallback")
                return [], False
                
        except Exception as fallback_error:
            add_system_message('error', f"❌ CSV fallback failed: {fallback_error}")
        
        st.error(f"Search failed: {e}")
        return [], False


def get_paper_abstract_and_keywords(vectorstore, paper_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Get abstract (child document) and keywords for a specific paper"""
    try:
        if not vectorstore:
            print(f"🔍 Debug: Vector store is None for {paper_name}")
            return None, None
        
        print(f"🔍 Debug: Attempting to search vector store for {paper_name}")
        
        # Search for child document (abstract) of this paper
        results = vectorstore.similarity_search(
            f"abstract {paper_name}", 
            k=10,
            filter={"file_name": paper_name}
        )
        
        print(f"🔍 Debug: Found {len(results)} results for {paper_name}")
        
        abstract_content = None
        keywords = None
        
        # Find the child document (abstract)
        for i, doc in enumerate(results):
            print(f"🔍 Debug: Result {i+1} for {paper_name}:")
            print(f"   - file_name: {doc.metadata.get('file_name')}")
            print(f"   - document_type: {doc.metadata.get('document_type')}")
            print(f"   - title: {doc.metadata.get('title', 'None')}")
            print(f"   - content_length: {len(doc.page_content)}")
            print(f"   - content_preview: {doc.page_content[:100]}...")
            
            if (doc.metadata.get('file_name') == paper_name and 
                doc.metadata.get('document_type') == 'child'):
                abstract_content = doc.page_content
                keywords = doc.metadata.get('keywords', '')
                print(f"✅ Found child document (abstract) for {paper_name}")
                break
        
        # If no child document found, try to find any document from this paper
        if not abstract_content:
            print(f"🔍 Debug: No child document found, looking for any document from {paper_name}")
            for i, doc in enumerate(results):
                if doc.metadata.get('file_name') == paper_name:
                    # Extract first 500 characters as abstract
                    abstract_content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                    keywords = doc.metadata.get('keywords', '')
                    print(f"✅ Found any document for {paper_name}, using first 500 chars")
                    break
        
        print(f"🔍 Debug: Final result for {paper_name}:")
        print(f"   - abstract_content: {len(abstract_content) if abstract_content else 0} chars")
        print(f"   - keywords: {keywords}")
        
        return abstract_content, keywords
        
    except Exception as e:
        print(f"❌ Error in get_paper_abstract_and_keywords for {paper_name}: {e}")
        # Fallback: Try to get paper info from CSV data if vector store fails
        try:
            add_system_message('warning', f"⚠️ Vector store search failed for {paper_name}, trying CSV fallback")
            
            # Try to access the tracker_df from session state
            import streamlit as st
            if hasattr(st, 'session_state') and 'tracker_df' in st.session_state and st.session_state.tracker_df is not None:
                df = st.session_state.tracker_df
                paper_row = df[df['file_name'] == paper_name]
                
                if not paper_row.empty:
                    # Create a more detailed abstract from the file name and metadata
                    paper_title = paper_name.replace('.pdf', '').replace('_', ' ')
                    folder = paper_row.iloc[0].get('folder', 'Unknown')
                    figure_count = paper_row.iloc[0].get('figure_count', 0)
                    vectorized_date = paper_row.iloc[0].get('vectorized_date', 'Unknown')
                    vectorized_model = paper_row.iloc[0].get('vectorized_model', 'Unknown')
                    chunk_count = paper_row.iloc[0].get('chunk_count', 0)
                    
                    abstract_content = f"""**Paper Title:** {paper_title}

**Folder:** {folder}
**Figures:** {figure_count}
**Vectorized Date:** {vectorized_date}
**Vectorization Model:** {vectorized_model}
**Content Chunks:** {chunk_count}

This paper has been processed and vectorized in the database. The full content and abstract are available through the vector store when Ollama is running. 

**Note:** To access the complete paper content, ensure Ollama is running and the vector store is accessible."""
                    
                    keywords = f"RHEA, materials science, {folder}, research paper, vectorized"
                    
                    add_system_message('info', f"✅ Used CSV fallback for {paper_name}")
                    return abstract_content, keywords
                else:
                    add_system_message('warning', f"⚠️ Paper {paper_name} not found in CSV data")
            else:
                add_system_message('warning', f"⚠️ No CSV data available for fallback")
                
        except Exception as fallback_error:
            add_system_message('error', f"❌ Both vector store and CSV fallback failed for {paper_name}")
        
        st.error(f"Failed to load abstract and keywords: {e}")
        return None, None 