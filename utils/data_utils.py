"""
Data utilities for the Paper Search & QA System.
Handles loading paper lists, managing figures, and file operations.
"""

import streamlit as st
import os
import pandas as pd
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any


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


def get_folder_config() -> Tuple[List[str], Dict[str, str], Dict[str, List[str]]]:
    """Get hierarchical folder configuration with order, icons, and subfolder mapping"""
    # Define the main research areas (top level)
    main_folders = ["RHEA_Strengthening", "Other_Research_Areas"]
    
    # Define subfolders for each main area
    folder_hierarchy = {
        "RHEA_Strengthening": ["SSS", "dislocation", "grainBoundary", "Precipitation"],
        "Other_Research_Areas": []  # Placeholder for future research areas
    }
    
    # Icons for main folders
    main_folder_icons = {
        "RHEA_Strengthening": "🔬",
        "Other_Research_Areas": "📚"
    }
    
    # Icons for subfolders
    subfolder_icons = {
        "SSS": "🔬",
        "dislocation": "🔧", 
        "grainBoundary": "🧱",
        "Precipitation": "💧"
    }
    
    # Debug: Print folder configuration
    print(f"📁 Main folders: {main_folders}")
    print(f"📁 Folder hierarchy: {folder_hierarchy}")
    print(f"📁 Main folder icons: {main_folder_icons}")
    print(f"📁 Subfolder icons: {subfolder_icons}")
    
    return main_folders, main_folder_icons, folder_hierarchy


def get_all_folder_icons() -> Dict[str, str]:
    """Get all folder icons (main folders + subfolders)"""
    _, main_icons, hierarchy = get_folder_config()
    subfolder_icons = {
        "SSS": "🔬",
        "dislocation": "🔧", 
        "grainBoundary": "🧱",
        "Precipitation": "💧"
    }
    
    # Combine main and subfolder icons
    all_icons = {**main_icons, **subfolder_icons}
    return all_icons


def get_flat_folder_order() -> List[str]:
    """Get flat list of all folders for backward compatibility"""
    _, _, hierarchy = get_folder_config()
    flat_order = []
    for main_folder, subfolders in hierarchy.items():
        flat_order.extend(subfolders)
    return flat_order


def get_paper_hierarchy(paper_list: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    """Organize papers into hierarchical structure"""
    hierarchy = {}
    
    for paper in paper_list:
        # Get the subfolder (e.g., "SSS", "dislocation")
        subfolder = paper.get('folder', 'Unknown')
        
        # Determine which main folder this subfolder belongs to
        main_folder = "RHEA_Strengthening"  # Default for current subfolders
        
        # Initialize main folder if not exists
        if main_folder not in hierarchy:
            hierarchy[main_folder] = {}
        
        # Initialize subfolder if not exists
        if subfolder not in hierarchy[main_folder]:
            hierarchy[main_folder][subfolder] = []
        
        # Add paper to subfolder
        hierarchy[main_folder][subfolder].append(paper)
    
    return hierarchy


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


def add_research_area(main_folder_name: str, subfolders: List[str], icon: str = "📚") -> None:
    """
    Add a new research area to the folder hierarchy.
    This function allows dynamic addition of new research areas without modifying the core code.
    
    Args:
        main_folder_name: Name of the main research area (e.g., "RHEA_Corrosion")
        subfolders: List of subfolder names for this research area
        icon: Icon to use for the main folder
    """
    # Get current configuration
    main_folders, main_folder_icons, folder_hierarchy = get_folder_config()
    
    # Add new research area
    if main_folder_name not in main_folders:
        main_folders.append(main_folder_name)
        main_folder_icons[main_folder_name] = icon
        folder_hierarchy[main_folder_name] = subfolders
        
        print(f"✅ Added new research area: {main_folder_name} with subfolders: {subfolders}")
    else:
        print(f"⚠️ Research area {main_folder_name} already exists")


def get_research_area_info(main_folder_name: str) -> Dict[str, Any]:
    """
    Get information about a specific research area.
    
    Args:
        main_folder_name: Name of the main research area
        
    Returns:
        Dictionary containing folder info, subfolders, and icon
    """
    main_folders, main_folder_icons, folder_hierarchy = get_folder_config()
    
    if main_folder_name in main_folders:
        return {
            'name': main_folder_name,
            'icon': main_folder_icons.get(main_folder_name, '📁'),
            'subfolders': folder_hierarchy.get(main_folder_name, []),
            'total_subfolders': len(folder_hierarchy.get(main_folder_name, []))
        }
    else:
        return None


def list_all_research_areas() -> List[Dict[str, Any]]:
    """
    List all research areas with their information.
    
    Returns:
        List of dictionaries containing research area information
    """
    main_folders, main_folder_icons, folder_hierarchy = get_folder_config()
    
    research_areas = []
    for main_folder in main_folders:
        info = get_research_area_info(main_folder)
        if info:
            research_areas.append(info)
    
    return research_areas 