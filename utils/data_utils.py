"""
Data utilities for the Paper Search & QA System.
Handles loading paper lists, managing figures, and file operations.
"""

import streamlit as st
import os
import pandas as pd
from PIL import Image
from typing import List, Dict, Tuple, Optional


@st.cache_data
def load_paper_list(tracker_path: str = "./vectorization_tracker.csv") -> Tuple[List[Dict], Optional[pd.DataFrame]]:
    """Load the list of papers from the tracker CSV"""
    try:
        if os.path.exists(tracker_path):
            df = pd.read_csv(tracker_path)
            
            # Show loading progress
            total_papers = len(df)
            vectorized_papers = df[df['vectorized'] == True]
            vectorized_count = len(vectorized_papers)
            
            paper_list = []
            for _, row in vectorized_papers.iterrows():
                # Debug: Show the raw file path first
                if len(paper_list) < 3:
                    print(f"🔍 Raw file_path for {row['file_name']}: '{row['file_path']}'")
                
                # Get folder name directly from file path (more reliable)
                folder_name = os.path.basename(os.path.dirname(row['file_path']))
                
                # If folder_name is '..' or unexpected, try to extract from the full path
                if folder_name in ['..', '.', ''] or len(folder_name) > 50:
                    # Try to find the actual folder name in the path
                    path_parts = row['file_path'].replace('\\', '/').split('/')
                    print(f"🔍 Path parts for {row['file_name']}: {path_parts}")
                    
                    # Look for known folder names in the path
                    for part in path_parts:
                        if part in ['dislocation', 'grainBoundary', 'Precipitation', 'SSS']:
                            folder_name = part
                            print(f"✅ Found folder '{part}' in path for {row['file_name']}")
                            break
                    else:
                        # If no known folder found, use the last directory before the file
                        if len(path_parts) >= 2:
                            folder_name = path_parts[-2]  # Second to last part
                            print(f"⚠️ Using fallback folder '{folder_name}' for {row['file_name']}")
                        else:
                            folder_name = "unknown"
                            print(f"❌ Could not determine folder for {row['file_name']}")
                
                # Compute full folder path relative to Papers root (for compatibility)
                abs_paper_path = os.path.abspath(row['file_path'])
                abs_root = os.path.abspath('../Papers')
                rel_folder_path = os.path.relpath(os.path.dirname(abs_paper_path), abs_root).replace('\\', '/')
                top_level_folder = rel_folder_path.split('/')[0] if '/' in rel_folder_path else rel_folder_path
                
                # Debug: Print path information for first few papers
                if len(paper_list) < 3:
                    print(f"🔍 Debug path for {row['file_name']}:")
                    print(f"   file_path: {row['file_path']}")
                    print(f"   folder_name: {folder_name}")
                    print(f"   abs_paper_path: {abs_paper_path}")
                    print(f"   abs_root: {abs_root}")
                    print(f"   rel_folder_path: {rel_folder_path}")
                    print(f"   top_level_folder: {top_level_folder}")
                
                # Use folder_name as the primary source for top_level_folder
                # This should work regardless of the deployment environment
                if folder_name in ['dislocation', 'grainBoundary', 'Precipitation', 'SSS']:
                    top_level_folder = folder_name
                    rel_folder_path = folder_name
                    print(f"✅ Using folder name: {folder_name} for {row['file_name']}")
                else:
                    print(f"⚠️ Unknown folder: {folder_name} for {row['file_name']}")
                
                paper_info = {
                    'file_name': row['file_name'],
                    'file_path': row['file_path'],
                    'figure_count': row.get('figure_count', 0),
                    'has_figures': row.get('has_figure_descriptions', False),
                    'folder': folder_name,
                    'folder_path': rel_folder_path,
                    'top_level_folder': top_level_folder,
                    'rel_folder_path': rel_folder_path,
                    'vectorized_date': row.get('vectorized_date', ''),
                    'vectorized_model': row.get('vectorized_model', ''),
                    'chunk_count': row.get('chunk_count', 0),
                }
                paper_list.append(paper_info)
            
            return paper_list, df
        else:
            return [], None
    except Exception as e:
        return [], None


def get_paper_figures(paper_name: str, extracted_images_dir: str = "../extracted_images") -> List[str]:
    """Get figures for a specific paper"""
    try:
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
    
    # Debug: Print folder configuration
    print(f"📁 Folder config - Order: {folder_order}")
    print(f"📁 Folder config - Icons: {folder_icons}")
    
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