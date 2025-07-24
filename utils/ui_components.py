"""
UI Components for the Paper Search & QA System.
Contains reusable UI components and styling functions.
"""

import streamlit as st
from typing import List, Dict


def apply_theme_css(dark_theme: bool = False):
    """Apply theme-based CSS styling"""
    if dark_theme:
        st.markdown("""
        <style>
        .main-header {
            color: #ffffff;
            text-align: center;
            margin-bottom: 2rem;
            font-size: 2.5rem;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .stApp {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        }
        .stSidebar {
            background: rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }
        .content-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        .optimize-card {
            background: linear-gradient(135deg, rgba(255, 215, 0, 0.2), rgba(255, 165, 0, 0.2));
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 215, 0, 0.3);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .main-header {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 2rem;
            font-size: 2.5rem;
            font-weight: bold;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .stSidebar {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
        }
        .content-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 0, 0, 0.1);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 0, 0, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .optimize-card {
            background: linear-gradient(135deg, rgba(255, 215, 0, 0.1), rgba(255, 165, 0, 0.1));
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 215, 0, 0.3);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        </style>
        """, unsafe_allow_html=True)


def create_glass_card(title: str) -> str:
    """Create a glass card HTML string"""
    return f"""
    <div class="glass-card">
        <h3 style="margin: 0 0 1rem 0; color: {'#ffffff' if st.session_state.get('dark_theme', False) else '#2c3e50'};">
            {title}
        </h3>
    </div>
    """


def create_content_card(content: str, additional_style: str = "") -> str:
    """Create a content card HTML string"""
    return f"""
    <div class="content-card" style="{additional_style}">
        {content}
    </div>
    """


def create_optimize_card(title: str) -> str:
    """Create an optimization card HTML string"""
    return f"""
    <div class="optimize-card">
        <h3 style="margin: 0 0 1rem 0; color: {'#ffffff' if st.session_state.get('dark_theme', False) else '#2c3e50'};">
            {title}
        </h3>
    </div>
    """


def display_theme_toggle():
    """Display theme toggle in sidebar"""
    st.markdown("### 🎨 Theme")
    if st.button("🌙 Dark" if not st.session_state.dark_theme else "☀️ Light"):
        st.session_state.dark_theme = not st.session_state.dark_theme
        st.rerun()


def display_paper_selection(paper_list: List[Dict], folder_order: List[str], folder_icons: Dict[str, str]) -> List[str]:
    """Display paper selection interface with folder grouping"""
    selected_papers = []
    
    # Group papers by folder
    papers_by_folder = {}
    for paper in paper_list:
        folder = paper['folder']
        if folder not in papers_by_folder:
            papers_by_folder[folder] = []
        papers_by_folder[folder].append(paper)
    
    # Display papers by folder in specified order
    for folder in folder_order:
        if folder in papers_by_folder:
            folder_papers = papers_by_folder[folder]
            if folder_papers:
                st.markdown(f"**{folder_icons.get(folder, '📁')} {folder}**")
                
                for paper in folder_papers:
                    col_paper, col_view = st.columns([7, 1])
                    
                    with col_paper:
                        if st.checkbox(
                            paper['file_name'].replace('.pdf', ''),
                            key=f"paper_{paper['file_name']}",
                            help=f"Figures: {paper['figure_count']}"
                        ):
                            selected_papers.append(paper['file_name'])
                    
                    with col_view:
                        # Add inline CSS to reduce button padding
                        st.markdown('''<style>.stButton button {padding: 0.1rem 0.3rem !important; font-size: 1.1em !important;}</style>''', unsafe_allow_html=True)
                        if st.button("👁️", key=f"view_{paper['file_name']}", help="View paper"):
                            st.session_state.view_paper_pdf = paper['file_path']
                            st.rerun()
                
                st.divider()
    
    return selected_papers


def display_keyword_selection(suggested_keywords: List[str]) -> List[str]:
    """Display keyword selection interface"""
    selected_keywords = []
    
    if suggested_keywords:
        st.markdown("**Select keywords to enhance your search:**")
        cols = st.columns(3)
        for i, keyword in enumerate(suggested_keywords):
            with cols[i % 3]:
                if st.checkbox(keyword, key=f"keyword_{keyword}"):
                    selected_keywords.append(keyword)
    
    return selected_keywords 