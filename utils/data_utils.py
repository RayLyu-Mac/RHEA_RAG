"""
Data utilities for the Paper Search & QA System.
Handles loading paper lists, managing figures, and file operations.
"""

import streamlit as st
import os
import pandas as pd
from PIL import Image
from typing import List, Dict, Tuple, Optional


def add_system_message(message_type: str, message: str):
    """Add a system message to the session state for display in sidebar"""
    if 'system_messages' not in st.session_state:
        st.session_state.system_messages = []
    
    st.session_state.system_messages.append({
        'type': message_type,
        'message': message
    })


@st.cache_data
def load_paper_list(tracker_path: str = "./vectorization_tracker.csv") -> Tuple[List[Dict], Optional[pd.DataFrame]]:
    """Load the list of papers from the vector store instead of CSV"""
    try:
        # Try to get papers from vector store first (if available)
        if 'vectorstore' in st.session_state and st.session_state.vectorstore is not None:
            try:
                # Get all documents from the vector store
                collection = st.session_state.vectorstore._collection
                if collection:
                    # Get all documents from the collection
                    results = collection.get()
                    
                    if results and 'metadatas' in results and results['metadatas']:
                        paper_list = []
                        seen_papers = set()  # To avoid duplicates
                        
                        # Debug: show metadata structure for first few items
                        if len(results['metadatas']) > 0:
                            sample_metadata = results['metadatas'][0]
                            add_system_message('info', f"📊 Vector store metadata sample keys: {list(sample_metadata.keys())}")
                        
                        for metadata in results['metadatas']:
                            file_name = metadata.get('file_name', '')
                            
                            # Skip if we've already processed this paper or if it's a meeting note
                            if not file_name or file_name in seen_papers or metadata.get('content_type') == 'meeting_notes':
                                continue
                            
                            seen_papers.add(file_name)
                            
                            # Extract folder information from metadata
                            folder_name = "unknown"
                            
                            # Try different metadata fields for folder information
                            if metadata.get('folder'):
                                folder_name = metadata.get('folder')
                            elif metadata.get('section'):
                                folder_name = metadata.get('section')
                            elif metadata.get('file_path'):
                                # Extract from file path if available
                                file_path = metadata.get('file_path')
                                path_parts = file_path.replace('\\', '/').split('/')
                                papers_index = -1
                                for i, part in enumerate(path_parts):
                                    if part == 'Papers':
                                        papers_index = i
                                        break
                                
                                if papers_index >= 0 and papers_index + 1 < len(path_parts):
                                    folder_parts = path_parts[papers_index + 1:]
                                    if len(folder_parts) >= 2:
                                        folder_name = folder_parts[0]
                            
                            # Create paper info
                            paper_info = {
                                'file_name': file_name,
                                'file_path': metadata.get('file_path', ''),
                                'figure_count': metadata.get('figure_count', 0),
                                'has_figures': metadata.get('has_figure_descriptions', False),
                                'folder': folder_name,
                                'folder_path': folder_name,  # Simplified for now
                                'top_level_folder': folder_name,
                                'rel_folder_path': folder_name,
                                'abstract': metadata.get('abstract', ''),
                                'title': metadata.get('title', file_name),
                            }
                            paper_list.append(paper_info)
                        
                        if paper_list:
                            add_system_message('success', f"✅ Loaded {len(paper_list)} papers from vector store")
                            return paper_list, None
                        else:
                            add_system_message('warning', "No papers found in vector store")
                    else:
                        add_system_message('warning', "Vector store appears to be empty")
                        
            except Exception as e:
                add_system_message('warning', f"Failed to load papers from vector store: {e}")
        
        # Fallback to CSV if vector store method fails
        if os.path.exists(tracker_path):
            df = pd.read_csv(tracker_path)
            # Filter only vectorized papers
            vectorized_papers = df[df['vectorized'] == True]
            paper_list = []
            
            for _, row in vectorized_papers.iterrows():
                # Extract folder information from the file path
                file_path = row['file_path']
                file_name = row['file_name']
                
                # Handle both Windows and Unix paths
                path_parts = file_path.replace('\\', '/').split('/')
                
                # Find the 'Papers' directory in the path
                papers_index = -1
                for i, part in enumerate(path_parts):
                    if part == 'Papers':
                        papers_index = i
                        break
                
                if papers_index >= 0 and papers_index + 1 < len(path_parts):
                    # Extract folder information after 'Papers'
                    folder_parts = path_parts[papers_index + 1:]
                    if len(folder_parts) >= 2:  # Should have at least folder and filename
                        folder_name = folder_parts[0]  # The immediate folder (e.g., 'dislocation')
                        rel_folder_path = '/'.join(folder_parts[:-1])  # All folders except filename
                        top_level_folder = folder_name
                    else:
                        # Fallback if path structure is unexpected
                        folder_name = "unknown"
                        rel_folder_path = "unknown"
                        top_level_folder = "unknown"
                else:
                    # Fallback if 'Papers' not found in path
                    folder_name = "unknown"
                    rel_folder_path = "unknown"
                    top_level_folder = "unknown"
                
                paper_info = {
                    'file_name': file_name,
                    'file_path': file_path,  # Keep original path for reference
                    'figure_count': row.get('figure_count', 0),
                    'has_figures': row.get('has_figure_descriptions', False),
                    'folder': folder_name,
                    'folder_path': rel_folder_path,
                    'top_level_folder': top_level_folder,
                    'rel_folder_path': rel_folder_path,
                }
                paper_list.append(paper_info)
            
            # Add success message for paper loading
            if paper_list:
                add_system_message('success', f"✅ Loaded {len(paper_list)} papers from vectorization tracker (fallback)")
            
            return paper_list, df
        else:
            # Add system message instead of st.warning for deployment
            add_system_message('warning', "Vectorization tracker not found. Please run the vectorization process first.")
            return [], None
    except Exception as e:
        # Add system message instead of st.error for deployment
        add_system_message('error', f"Failed to load paper list: {e}")
        return [], None


def get_paper_figures(paper_name: str, extracted_images_dir: str = "../extracted_images") -> List[str]:
    """Get figures for a specific paper"""
    try:
        if not os.path.exists(extracted_images_dir):
            # In deployment, images might not be available
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
        # Add system message instead of st.error for deployment
        add_system_message('warning', f"Failed to load figures for {paper_name}: {e}")
        return []


def group_papers_by_folder(paper_list: List[Dict]) -> Dict[str, List[Dict]]:
    """Group papers by their folder"""
    folders = {}
    for paper in paper_list:
        folder = paper['folder']
        if folder not in folders:
            folders[folder] = []
        folders[folder].append(paper)
    return folders


def get_folder_config() -> Tuple[List[str], Dict[str, str]]:
    """Get folder configuration with order and icons"""
    folder_order = ["dislocation", "grainBoundary", "Precipitation", "SSS"]
    folder_icons = {
        "dislocation": "🔧",
        "grainBoundary": "🧱", 
        "Precipitation": "💧",
        "SSS": "🔬"
    }
    return folder_order, folder_icons


def display_image_safely(image_path: str, caption: str = None, use_container_width: bool = True) -> bool:
    """Safely display an image with error handling"""
    try:
        image = Image.open(image_path)
        st.image(image, caption=caption or os.path.basename(image_path), use_container_width=use_container_width)
        return True
    except Exception as e:
        st.error(f"Failed to load image {os.path.basename(image_path)}: {e}")
        return False


def get_paper_stats(paper_list: List[Dict]) -> Dict[str, int]:
    """Get statistics about the paper collection"""
    stats = {
        'total_papers': len(paper_list),
        'total_figures': sum(paper.get('figure_count', 0) for paper in paper_list),
        'papers_with_figures': sum(1 for paper in paper_list if paper.get('figure_count', 0) > 0)
    }
    
    # Count by folder
    folders = group_papers_by_folder(paper_list)
    for folder, papers in folders.items():
        stats[f'{folder}_count'] = len(papers)
    
    return stats 