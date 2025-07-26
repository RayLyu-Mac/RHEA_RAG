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
            margin-bottom: 1rem;
            font-size: 2.2rem;
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
            border-radius: 8px;
            padding: 0.5rem 0.75rem;
            margin: 0.25rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
        }
        .optimize-card {
            background: linear-gradient(135deg, rgba(255, 215, 0, 0.2), rgba(255, 165, 0, 0.2));
            border-radius: 10px;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 215, 0, 0.3);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
        }
        .stButton > button {
            border: 2px solid #222 !important;
            box-shadow: none !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .main-header {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 1rem;
            font-size: 2.2rem;
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
            border-radius: 8px;
            padding: 0.5rem 0.75rem;
            margin: 0.25rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 0, 0, 0.1);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.07);
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 0, 0, 0.1);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.07);
        }
        .optimize-card {
            background: linear-gradient(135deg, rgba(255, 215, 0, 0.1), rgba(255, 165, 0, 0.1));
            border-radius: 10px;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 215, 0, 0.3);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.07);
        }
        .stButton > button {
            border: 2px solid #222 !important;
            box-shadow: none !important;
        }
        </style>
        """, unsafe_allow_html=True)


def create_glass_card(title: str) -> str:
    """Create a glass card HTML string (compact)"""
    return f"""
    <div class="glass-card">
        <h3 style="margin: 0 0 0.5rem 0; color: {'#ffffff' if st.session_state.get('dark_theme', False) else '#2c3e50'}; font-size: 1.1em;">
            {title}
        </h3>
    </div>
    """


def create_content_card(content: str, additional_style: str = "") -> str:
    """Create a content card HTML string (compact)"""
    return f"""
    <div class="content-card" style="{additional_style}; padding: 0.5rem 0.75rem; margin: 0.25rem 0;">
        {content}
    </div>
    """


def create_optimize_card(title: str) -> str:
    """Create an optimization card HTML string (compact)"""
    return f"""
    <div class="optimize-card">
        <h3 style="margin: 0 0 0.5rem 0; color: {'#ffffff' if st.session_state.get('dark_theme', False) else '#2c3e50'}; font-size: 1.1em;">
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


def display_folder_tree(papers, parent_path=""):
    """Recursively display papers in collapsible folders/subfolders."""
    import os
    from collections import defaultdict
    selected_papers = []
    folders = defaultdict(list)
    files = []
    for paper in papers:
        # Defensive: compute folder_path if missing
        if 'folder_path' not in paper:
            abs_paper_path = os.path.abspath(paper['file_path'])
            abs_root = os.path.abspath('../Papers')
            paper['folder_path'] = os.path.relpath(os.path.dirname(abs_paper_path), abs_root).replace('\\', '/')
        rel_path = os.path.relpath(paper['folder_path'], parent_path).replace('\\', '/')
        if '/' in rel_path and rel_path != '.':
            next_folder = rel_path.split('/', 1)[0]
            folders[next_folder].append(paper)
        elif rel_path == '' or rel_path == '.' or '/' not in rel_path:
            files.append(paper)
        else:
            files.append(paper)
    # Show files in this folder
    for paper in sorted(files, key=lambda x: x['file_name']):
        col_paper, col_view = st.columns([7, 1])
        with col_paper:
            if st.checkbox(
                paper['file_name'].replace('.pdf', ''),
                key=f"paper_{paper['file_name']}",
                help=f"Figures: {paper['figure_count']}"
            ):
                selected_papers.append(paper['file_name'])
        with col_view:
            st.markdown('''<style>.stButton button {padding: 0.1rem 0.3rem !important; font-size: 1.1em !important;}</style>''', unsafe_allow_html=True)
            if st.button("👁️", key=f"view_{paper['file_name']}", help="View paper"):
                st.session_state.view_paper_pdf = paper['file_path']
                st.rerun()
    # Show subfolders
    for folder in sorted(folders.keys()):
        with st.expander(folder, expanded=False):
            selected_papers += display_folder_tree(folders[folder], os.path.join(parent_path, folder))
    return selected_papers

def display_paper_selection(paper_list, folder_order, folder_icons):
    from collections import defaultdict
    selected_papers = []
    
    # Debug information
    if not paper_list:
        st.error("❌ No papers provided to display_paper_selection")
        return selected_papers
    
    st.caption(f"🔍 Debug: Processing {len(paper_list)} papers for display")
    
    papers_by_top = defaultdict(list)
    for paper in paper_list:
        top_level = paper.get('top_level_folder')
        if not top_level:
            import os
            if 'folder_path' in paper:
                top_level = paper['folder_path'].split('/')[0]
            else:
                abs_paper_path = os.path.abspath(paper['file_path'])
                abs_root = os.path.abspath('../Papers')
                rel_folder_path = os.path.relpath(os.path.dirname(abs_paper_path), abs_root).replace('\\', '/')
                top_level = rel_folder_path.split('/')[0] if '/' in rel_folder_path else rel_folder_path
            paper['top_level_folder'] = top_level
        papers_by_top[top_level].append(paper)
    
    st.caption(f"📁 Debug: Papers grouped into {len(papers_by_top)} top-level folders: {list(papers_by_top.keys())}")
    
    for folder in folder_order:
        if folder in papers_by_top:
            icon = folder_icons.get(folder, '📁')
            with st.expander(f"{icon} {folder}", expanded=False):
                folder_papers = papers_by_top[folder]
                st.caption(f"📄 Debug: {len(folder_papers)} papers in {folder}")
                
                # Select All / Deselect All buttons
                col_sel, col_desel = st.columns([1, 1])
                if col_sel.button(f"Select All {folder}", key=f"select_all_{folder}"):
                    for paper in folder_papers:
                        st.session_state[f"paper_{paper['file_name']}"] = True
                if col_desel.button(f"Deselect All {folder}", key=f"deselect_all_{folder}"):
                    for paper in folder_papers:
                        st.session_state[f"paper_{paper['file_name']}"] = False
                for paper in sorted(folder_papers, key=lambda x: x['file_name']):
                    col_paper, col_view = st.columns([7, 1])
                    with col_paper:
                        checked = st.session_state.get(f"paper_{paper['file_name']}", False)
                        
                        # Create enhanced tooltip with vectorization info
                        tooltip_parts = [f"Figures: {paper['figure_count']}"]
                        if paper.get('vectorized_model'):
                            tooltip_parts.append(f"Model: {paper['vectorized_model']}")
                        if paper.get('vectorized_date'):
                            # Format the date for display
                            try:
                                from datetime import datetime
                                date_str = paper['vectorized_date']
                                if 'T' in date_str:
                                    date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                                    formatted_date = date_obj.strftime('%Y-%m-%d')
                                    tooltip_parts.append(f"Vectorized: {formatted_date}")
                            except:
                                tooltip_parts.append(f"Vectorized: {paper['vectorized_date'][:10]}")
                        
                        tooltip_text = " | ".join(tooltip_parts)
                        
                        if st.checkbox(
                            paper['file_name'].replace('.pdf', ''),
                            key=f"paper_{paper['file_name']}",
                            value=checked,
                            help=tooltip_text
                        ):
                            selected_papers.append(paper['file_name'])
                    with col_view:
                        st.markdown('''<style>.stButton button {padding: 0.1rem 0.3rem !important; font-size: 1.1em !important;}</style>''', unsafe_allow_html=True)
                        if st.button("👁️", key=f"view_{paper['file_name']}", help="View paper"):
                            st.session_state.view_paper_pdf = paper['file_path']
                            st.rerun()
        else:
            st.caption(f"⚠️ Debug: No papers found for folder '{folder}'")
    
    st.caption(f"✅ Debug: {len(selected_papers)} papers selected")
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