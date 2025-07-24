"""
Vector store utilities for the Paper Search & QA System.
Handles loading, searching, and managing the Chroma vector database.
"""

import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from typing import List, Optional, Tuple, Any
from langchain.schema import Document


@st.cache_resource
def load_vectorstore(persist_directory: str = "../VectorSpace/paper_vector_db_nomic-embed-text_latest_parent_child"):
    """Load the vector store"""
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
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
        st.error(f"Search failed: {e}")
        return [], False


def get_paper_abstract_and_keywords(vectorstore, paper_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Get abstract (child document) and keywords for a specific paper"""
    try:
        if not vectorstore:
            return None, None
        
        # Search for child document (abstract) of this paper
        results = vectorstore.similarity_search(
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