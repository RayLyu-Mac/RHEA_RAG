"""
Modular Paper Search & QA System
A Streamlit application for searching and querying research papers about 
Refractory High-Entropy Alloys (RHEA) using vector embeddings and LLM.
"""

# Import SQLite patch first (before any Chroma imports)
try:
    import sqlite_patch
except ImportError:
    # Fallback: Apply patch directly if sqlite_patch module is not available
    try:
        __import__('pysqlite3')
        import sys
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        print("✅ Applied direct SQLite patch")
    except ImportError:
        print("⚠️ pysqlite3-binary not available, using system SQLite")
        pass  # Continue with system SQLite

import streamlit as st
import os
import sys
import datetime
import html
from urllib.parse import quote
import re
from typing import List, Optional

# Try to import graphviz, but make it optional
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    graphviz = None

# Add parent directory to path to import vectorized module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions
from utils import (
    # Vector store utilities
    load_vectorstore, search_papers, get_paper_abstract_and_keywords,
    # LLM utilities  
    load_llm, get_available_ollama_models, optimize_question, get_suggested_keywords, generate_answer,
    # Data utilities
    load_paper_list, get_paper_figures, get_folder_config, display_image_safely, get_paper_stats,
    # UI components
    apply_theme_css, create_glass_card, create_content_card, create_optimize_card,
    display_theme_toggle, display_paper_selection, display_keyword_selection,
    # Notes utilities
    display_notes_section,
    # Prompt management
    get_research_gap_prompt, get_follow_up_prompt, get_scholar_summary_prompt,
    get_llm_grouping_refinement_prompt, get_rag_flowchart_prompt
)

# Import clean UI components
from utils.clean_ui_components import (
    apply_clean_theme, create_clean_card, create_clean_badge, create_clean_metric,
    create_clean_divider, display_clean_header, display_clean_section_header, display_clean_stats,
    display_clean_search_box, display_clean_loading_spinner, display_clean_empty_state,
    create_clean_content_area, create_clean_navigation, apply_clean_tab_style
)

# Modern UI (incremental replacement of components)
from utils.modern_ui_components import apply_modern_tab_style, create_modern_card

# Import paper keywords
from utils.paper_keywords import get_paper_keywords, get_folder_keywords

# Import logo utilities
from utils.logo_display import display_logo_header, display_logo_in_sidebar

# Import React components (optional)
try:
    from react_header_component import display_react_header, handle_header_message, send_header_update  # type: ignore
    REACT_HEADER_AVAILABLE = True
except Exception:
    REACT_HEADER_AVAILABLE = False
    def display_react_header():
        return None
    def handle_header_message():
        return None
    def send_header_update(*args, **kwargs):
        return None

# Page configuration
st.set_page_config(
    page_title="Paper Search & QA System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'dark_theme': False,
        'vectorstore': None,
        'llm': None,
        'paper_list': [],
        'tracker_df': None,
        'available_models': ["qwen3:14b", "gemma3:4b"],  # Default fallback
        'optimized_question': "",
        'suggested_keywords': [],
        'selected_keywords': [],
        'current_model': None,
        'selected_notes_for_qa': [],
        'view_paper_pdf': None,
        'suggested_followup': [], # Added for suggested follow-up reading
        'follow_up_answer': None,  # Added for follow-up question answers
        'follow_up_question': None,  # Added for follow-up questions
        'latest_follow_up_question': None,  # Store latest follow-up question (separate from widget key)
        'current_selected_papers': [],  # Track currently selected papers for debug display
        'suggestion_active_papers': [],  # Track active papers for follow-up suggestion context
        'summarize_answers': False,  # Toggle for summarizing answers to 3-5 sentences
        'design_outline': False,  # Toggle for adding experimental design outline to prompts
        'follow_up_history': [],  # Store all follow-up Q&A pairs
        'selected_context_answers': [],  # Track which follow-up answers are selected as context
        'current_answer_as_context': False,  # Toggle for using current answer as context
        'original_question': "",  # Store original question for export
        'followup_year_limit': 'All',  # Year filter for follow-up suggestions
        'answer_generation_time': None,  # Store time taken to generate answer
        # React dashboard integration
        'search_query': "",
        'search_options': {},
        'selected_paper': None,
        'dashboard_data': None,
        'dashboard_update': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def load_initial_data():
    """Load initial data if not already loaded"""
    # Initialize system status messages
    if 'system_messages' not in st.session_state:
        st.session_state.system_messages = []
    
    # Load paper list from vectorization tracker CSV
    if not st.session_state.paper_list:
        # Add loading message
        st.session_state.system_messages.append({
            'type': 'info',
            'message': "📊 Loading papers from vectorization tracker CSV..."
        })
        
        with st.spinner("Loading papers from vectorization tracker..."):
            st.session_state.paper_list, st.session_state.tracker_df = load_paper_list()
        
        # Remove loading message and add result
        st.session_state.system_messages = [msg for msg in st.session_state.system_messages if "Loading papers" not in msg['message']]
        
        # Add paper loading status to system messages
        if st.session_state.paper_list:
            paper_count = len(st.session_state.paper_list)
            st.session_state.system_messages.append({
                'type': 'success',
                'message': f"✅ Loaded {paper_count} papers from vectorization tracker CSV"
            })
            
            # Add folder distribution info
            folders = {}
            for paper in st.session_state.paper_list:
                folder = paper['folder']
                if folder not in folders:
                    folders[folder] = 0
                folders[folder] += 1
            
            folder_info = ", ".join([f"{folder}: {count}" for folder, count in sorted(folders.items())])
            st.session_state.system_messages.append({
                'type': 'info',
                'message': f"📁 Papers organized by folders: {folder_info}"
            })
        else:
            st.session_state.system_messages.append({
                'type': 'error',
                'message': "❌ Failed to load papers from vectorization tracker CSV"
            })
    
    # Load available models
    if not st.session_state.available_models or st.session_state.available_models == ["qwen3:14b", "gemma3:4b"]:
        st.session_state.available_models = get_available_ollama_models()
    
    # Load vector store
    if st.session_state.vectorstore is None:
        with st.spinner("Loading vector store..."):
            st.session_state.vectorstore = load_vectorstore()
    
    # Try to load LLM (but don't fail if it doesn't work)
    if st.session_state.llm is None and st.session_state.available_models:
        with st.spinner("Attempting to load LLM..."):
            st.session_state.llm = load_llm(st.session_state.available_models[0])
            st.session_state.current_model = st.session_state.available_models[0]
    
    # Store LLM status message in session state instead of displaying directly
    if st.session_state.llm is None:
        st.session_state.system_messages.append({
            'type': 'info',
            'message': "🤖 **LLM Mode**: Paper Management & Note-Taking Only - LLM features are disabled. You can still search papers, view previews, and take notes."
        })
    else:
        st.session_state.system_messages.append({
            'type': 'success',
            'message': f"🤖 **LLM Mode**: Full AI Features Enabled - Using {st.session_state.current_model}"
        })


def display_sidebar() -> tuple:
    """Display sidebar and return selected papers and settings"""
    # Initialize default values
    selected_papers = []
    llm_model = st.session_state.available_models[0] if st.session_state.available_models else "qwen3:14b"
    search_type = "both"
    num_results = 5
    
    with st.sidebar:
        # Logo in sidebar
        display_logo_in_sidebar()
        
        # Theme toggle
        display_theme_toggle()
        
        # Create tabs for main sections
        tab1, tab2 = st.tabs(["📚 Papers & Search", "📝 Meeting Notes"])
        
        with tab1:
            # Paper Selection Section (now collapsible)
            with st.expander("📚 Paper Selection", expanded=True):
                if st.session_state.paper_list:
                    folder_order, folder_icons = get_folder_config()
                    selected_papers = display_paper_selection(st.session_state.paper_list, folder_order, folder_icons)
                    # Store selected papers in session state for debug display
                    st.session_state.current_selected_papers = selected_papers
                    # Initialize suggestion_active_papers to current selection if empty or if selection changed
                    current_active = set(st.session_state.get('suggestion_active_papers', []))
                    current_selected = set(selected_papers)
                    if not current_active or current_active != current_selected:
                        st.session_state['suggestion_active_papers'] = list(selected_papers)
                else:
                    st.error("❌ No papers in session state")
                    st.caption("This means papers were loaded but not stored properly in session state")
                    selected_papers = []
                    st.session_state.current_selected_papers = []
            
            # Settings Section
            with st.expander("⚙️ Settings", expanded=True):
                # LLM model selection
                llm_model = st.selectbox(
                    "Select LLM Model:",
                    st.session_state.available_models,
                    help="Available Ollama models on your system"
                )
                
                # Check if model selection changed and reload LLM if needed
                if llm_model != st.session_state.current_model:
                    with st.spinner(f"Loading {llm_model}..."):
                        st.session_state.llm = load_llm(llm_model)
                        st.session_state.current_model = llm_model
                        if st.session_state.llm is None:
                            st.error(f"Failed to load {llm_model}")
                        else:
                            st.success(f"Successfully loaded {llm_model}")
                
                # Load LLM button (if not loaded)
                if st.session_state.llm is None:
                    if st.button("🤖 Load LLM", help="Attempt to load the selected LLM model", key="load_llm"):
                        with st.spinner(f"Loading {llm_model}..."):
                            st.session_state.llm = load_llm(llm_model)
                            st.session_state.current_model = llm_model
                            if st.session_state.llm is None:
                                st.error(f"Failed to load {llm_model}")
                            else:
                                st.success(f"Successfully loaded {llm_model}")
                                st.rerun()
                
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
            
            # Upload new paper section
            with st.expander("📤 Upload New Paper", expanded=False):
                uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
                if uploaded_file is not None:
                    if st.button("Process Paper"):
                        st.info("Paper processing feature coming soon!")
            
            # System Status
            with st.expander("📊 System Status", expanded=False):
                # Display system messages from session state
                if st.session_state.system_messages:
                    for i, msg in enumerate(st.session_state.system_messages):
                        if msg['type'] == 'info':
                            st.info(msg['message'])
                        elif msg['type'] == 'success':
                            st.success(msg['message'])
                        elif msg['type'] == 'warning':
                            st.warning(msg['message'])
                        elif msg['type'] == 'error':
                            st.error(msg['message'])
                    
                    # Add clear button for system messages
                    if st.button("🗑️ Clear Messages", key="clear_system_messages"):
                        st.session_state.system_messages = []
                        st.rerun()
                else:
                    st.info("No system messages to display")
                
                st.divider()
                
                # Paper Loading Status
                if st.session_state.paper_list:
                    st.success("✅ Papers loaded from vectorization tracker")
                    
                    # Show paper statistics
                    paper_count = len(st.session_state.paper_list)
                    total_figures = sum(paper.get('figure_count', 0) for paper in st.session_state.paper_list)
                    
                    # Display clean statistics
                    stats_data = [
                        {"label": "Total Papers", "value": str(paper_count), "change": "", "change_type": "neutral"},
                        {"label": "Total Figures", "value": str(total_figures), "change": "", "change_type": "neutral"}
                    ]
                    display_clean_stats(stats_data)
                    
                    # Show folder distribution
                    folders = {}
                    for paper in st.session_state.paper_list:
                        folder = paper['folder']
                        if folder not in folders:
                            folders[folder] = 0
                        folders[folder] += 1
                    
                    st.markdown("**Folder Distribution:**")
                    for folder, count in sorted(folders.items()):
                        st.caption(f"• {folder}: {count} papers")
                    
                    # Paper Selection Debug Information
                    st.markdown("**📚 Paper Selection Status:**")
                    st.caption(f"📊 Papers loaded in session state: {len(st.session_state.paper_list)}")
                    
                    # Show first few papers for debugging
                    if len(st.session_state.paper_list) > 0:
                        sample_papers = [p['file_name'] for p in st.session_state.paper_list[:3]]
                        st.caption(f"📄 Sample papers: {sample_papers}")
                    
                    # Show folder configuration
                    folder_order, folder_icons = get_folder_config()
                    st.caption(f"📁 Folder order: {folder_order}")
                    st.caption(f"📁 Folder icons: {folder_icons}")
                    
                    # Show current paper selection status
                    current_selected = st.session_state.get('current_selected_papers', [])
                    if current_selected:
                        st.caption(f"✅ Currently selected: {len(current_selected)} papers")
                        if len(current_selected) <= 5:
                            st.caption(f"📄 Selected papers: {current_selected}")
                        else:
                            st.caption(f"📄 Selected papers: {current_selected[:3]} ... and {len(current_selected)-3} more")
                    else:
                        st.caption("ℹ️ No papers currently selected")
                    
                    # Debug: Show raw paper data
                    with st.expander("🔍 Raw Paper Data (Debug)", expanded=False):
                        st.markdown("**First 3 papers data:**")
                        for i, paper in enumerate(st.session_state.paper_list[:3]):
                            st.json(paper)
                        
                        # Show raw CSV data for debugging
                        if st.session_state.tracker_df is not None:
                            st.markdown("**Raw CSV data (first 3 rows):**")
                            st.dataframe(st.session_state.tracker_df[['file_name', 'file_path']].head(3))
                            
                            # Show unique folder names found
                            st.markdown("**Unique folder names in CSV:**")
                            folder_names = []
                            for _, row in st.session_state.tracker_df.iterrows():
                                folder = os.path.basename(os.path.dirname(row['file_path']))
                                folder_names.append(folder)
                            
                            unique_folders = list(set(folder_names))
                            st.write(f"Found folders: {unique_folders}")
                            
                            # Show path structure
                            st.markdown("**Path structure analysis:**")
                            sample_path = st.session_state.tracker_df.iloc[0]['file_path']
                            path_parts = sample_path.replace('\\', '/').split('/')
                            st.write(f"Sample path: {sample_path}")
                            st.write(f"Path parts: {path_parts}")
                else:
                    st.error("❌ No papers loaded")
                    
                    # Debug information for deployment issues
                    with st.expander("🔍 Debug Info", expanded=False):
                        st.markdown("**File Paths:**")
                        st.code(f"Current working directory: {os.getcwd()}")
                        st.code(f"CSV path: {os.path.abspath('./vectorization_tracker.csv')}")
                        st.code(f"CSV exists: {os.path.exists('./vectorization_tracker.csv')}")
                        
                        if os.path.exists('./vectorization_tracker.csv'):
                            try:
                                import pandas as pd
                                df = pd.read_csv('./vectorization_tracker.csv')
                                st.code(f"CSV rows: {len(df)}")
                                st.code(f"Vectorized papers: {len(df[df['vectorized'] == True])}")
                            except Exception as e:
                                st.code(f"CSV read error: {e}")
                        else:
                            st.code("CSV file not found")
                
                st.divider()
                
                # Vector Store Status
                if st.session_state.vectorstore:
                    st.success("✅ Vector store loaded")
                    
                    # Display vector store info
                    persist_directory = "./VectorSpace/paper_vector_db_nomic-embed-text_latest_parent_child"
                    vector_store_name = os.path.basename(persist_directory)
                    st.info(f"📂 Vector store: {vector_store_name}")
                    
                    # Display paper statistics
                    if st.session_state.tracker_df is not None:
                        stats = get_paper_stats(st.session_state.paper_list)
                        vectorized_count = len(st.session_state.tracker_df[st.session_state.tracker_df['vectorized'] == True])
                        
                        # Display clean statistics
                        stats_data = [
                            {"label": "Papers", "value": str(stats['total_papers']), "change": "", "change_type": "neutral"},
                            {"label": "Vectorized", "value": str(vectorized_count), "change": "", "change_type": "neutral"},
                            {"label": "Figures", "value": str(stats['total_figures']), "change": "", "change_type": "neutral"}
                        ]
                        display_clean_stats(stats_data)
                else:
                    st.error("❌ Failed to load vector store")
        
        with tab2:
            # Meeting Notes Section
            display_notes_section()
    
    return selected_papers, llm_model, search_type, num_results


def display_controls_main() -> tuple:
    """Render minimalist controls in the main area (no sidebar) and return selections."""
    selected_papers: List[str] = []
    llm_model = st.session_state.available_models[0] if st.session_state.available_models else "qwen3:14b"
    search_type = "both"
    num_results = 5

    # Top controls row
    col_a, col_b, col_c, col_d = st.columns([2, 2, 1, 1])
    with col_a:
        st.markdown("**LLM Model**")
        llm_model = st.selectbox(
            "",
            st.session_state.available_models,
            key="main_llm_model",
            label_visibility="collapsed",
            help="Available Ollama models on your system"
        )
    with col_b:
        st.markdown("**Search Type**")
        search_type = st.selectbox(
            "",
            ["both", "parent", "child"],
            format_func=lambda x: {"both": "Both (Abstract + Full)", "parent": "Full Text + Figures", "child": "Abstract Only"}[x],
            key="main_search_type",
            label_visibility="collapsed"
        )
    with col_c:
        st.markdown("**Results**")
        num_results = st.slider("", 1, 10, 5, key="main_num_results", label_visibility="collapsed")
    with col_d:
        st.markdown("**Actions**")
        if st.button("🔄 Refresh Models", key="main_refresh_models"):
            st.session_state.available_models = get_available_ollama_models()
            st.rerun()

    # Load LLM if needed
    if llm_model != st.session_state.get('current_model'):
        with st.spinner(f"Loading {llm_model}..."):
            st.session_state.llm = load_llm(llm_model)
            st.session_state.current_model = llm_model

    # Paper selection
    with st.expander("📚 Paper Selection", expanded=True):
        if st.session_state.paper_list:
            folder_order, folder_icons = get_folder_config()
            selected_papers = display_paper_selection(st.session_state.paper_list, folder_order, folder_icons)
            st.session_state.current_selected_papers = selected_papers
            st.caption(f"Selected: {len(selected_papers)}")
            if not st.session_state.get('suggestion_active_papers'):
                st.session_state['suggestion_active_papers'] = list(selected_papers)
        else:
            st.warning("No papers loaded")
            selected_papers = []
            st.session_state.current_selected_papers = []

    # System status
    with st.expander("📊 System Status", expanded=False):
        if st.session_state.paper_list:
            st.success("✅ Papers loaded from vectorization tracker")
        if st.session_state.vectorstore:
            st.success("✅ Vector store loaded")
            if st.session_state.tracker_df is not None:
                stats = get_paper_stats(st.session_state.paper_list)
                vectorized_count = len(st.session_state.tracker_df[st.session_state.tracker_df['vectorized'] == True])
                st.caption(f"Papers: {stats['total_papers']} | Vectorized: {vectorized_count} | Figures: {stats['total_figures']}")
        else:
            st.error("❌ Failed to load vector store")

    return selected_papers, llm_model, search_type, num_results


def display_ask_combined_card(llm_model: str, selected_papers: List[str], search_type: str, num_results: int):
    """Ask UI that is full-width initially; after optimization, show 2-column layout."""
    has_optimized = bool(st.session_state.get('optimized_question'))

    # Full-width form before optimization
    if not has_optimized:
            # Live follow-up toggle + year limit (outside form for instant rerun)
            col_sf1, col_sf2 = st.columns([1, 1])
            with col_sf1:
                follow_live = st.toggle(
                    "Suggest Follow-up",
                    value=st.session_state.get("followup_toggle", False),
                    key="followup_toggle_live",
                    help="Show Google Scholar suggestions next to the answer"
                )
                st.session_state["followup_toggle"] = follow_live
            with col_sf2:
                if follow_live:
                    current_year = datetime.datetime.now().year
                    years = ["All"] + [str(y) for y in range(current_year, 1999, -1)]
                    st.session_state['followup_year_limit'] = st.selectbox(
                        "Year limit",
                        years,
                        index=0 if st.session_state.get('followup_year_limit', 'All') == 'All' else years.index(st.session_state['followup_year_limit']),
                        key="followup_year_limit_select_live",
                    )

            # Show selected papers as text buttons (outside form)
            current = list(st.session_state.get('current_selected_papers', []))
            active = set(st.session_state.get('suggestion_active_papers', []))
            
            # Initialize suggestion_active_papers to current selection if empty
            if not active and current:
                active = set(current)
                st.session_state['suggestion_active_papers'] = list(active)
            
            if current:
                st.markdown("**📚 Selected Papers:**")
                # Display as text buttons in a row
                cols = st.columns(len(current))
                for i, fname in enumerate(current):
                    with cols[i]:
                        checked = fname in active
                        button_text = fname.replace('_', ' ').replace('.pdf', '')
                        if st.button(
                            button_text, 
                            key=f"paper_btn_{fname}", 
                            help=f"Toggle {fname} for suggestions",
                            type="primary" if checked else "secondary",
                            use_container_width=True
                        ):
                            if checked:
                                active.discard(fname)
                            else:
                                active.add(fname)
                            st.session_state['suggestion_active_papers'] = list(active)
                            st.rerun()

            with st.form("ask_combined_form_full", clear_on_submit=False):
                # Toggles row (gaps, summarize, design outline inside form)
                col_gap, col_summary, col_design = st.columns([1, 1, 1])
                with col_gap:
                    try:
                        _gap_val = st.toggle(
                            "Identify Research Gaps",
                            value=st.session_state.get("gap_toggle", False),
                            key="ask_gap_toggle",
                            help="LLM required for research gap analysis",
                            disabled=st.session_state.llm is None,
                        )
                    except Exception:
                        _gap_val = st.checkbox(
                            "Identify Research Gaps",
                            value=st.session_state.get("gap_toggle", False),
                            key="ask_gap_toggle_cb",
                            help="LLM required for research gap analysis",
                        )
                    st.session_state["gap_toggle"] = _gap_val
                with col_summary:
                    try:
                        _sum_val = st.toggle(
                            "Summarize Answers",
                            value=st.session_state.get("summarize_answers", False),
                            key="ask_summarize_toggle",
                            help="Summarize answers to 3-5 sentences",
                        )
                    except Exception:
                        _sum_val = st.checkbox(
                            "Summarize Answers",
                            value=st.session_state.get("summarize_answers", False),
                            key="ask_summarize_toggle_cb",
                            help="Summarize answers to 3-5 sentences",
                        )
                    st.session_state["summarize_answers"] = _sum_val
                with col_design:
                    try:
                        _design_val = st.toggle(
                            "Design Outline",
                            value=st.session_state.get("design_outline", False),
                            key="ask_design_outline_toggle",
                            help="Add a detailed experimental outline to the answer",
                        )
                    except Exception:
                        _design_val = st.checkbox(
                            "Design Outline",
                            value=st.session_state.get("design_outline", False),
                            key="ask_design_outline_toggle_cb",
                            help="Add a detailed experimental outline to the answer",
                        )
                    st.session_state["design_outline"] = _design_val

                # Full-width input
                question = st.text_area(
                    label="Your Question",
                    value=st.session_state.get("original_question", ""),
                    placeholder="Ask about precipitation strengthening, microstructure, mechanical properties, etc.",
                    height=130,
                    key="combined_question_area",
                    label_visibility="visible",
                )

                col_ask, col_opt = st.columns([3, 1])
                with col_ask:
                    ask_pressed = st.form_submit_button("🔍 Ask Question", type="primary", use_container_width=True)
                with col_opt:
                    optimize_pressed = st.form_submit_button("🧠 Optimize", use_container_width=True, disabled=st.session_state.llm is None)

            # Get question from session state for logic outside form
            question = st.session_state.get("combined_question_area", "")
            
            if optimize_pressed and question.strip() and st.session_state.llm:
                with st.spinner("Optimizing question..."):
                    try:
                        # Preserve the user's original input so the text area keeps its content after rerun
                        st.session_state['original_question'] = question
                        optimized_q, keywords = optimize_question(st.session_state.llm, question)
                        st.session_state.optimized_question = optimized_q
                        st.session_state.suggested_keywords = keywords
                        st.rerun()
                    except Exception as e:
                        st.error(f"Optimization failed: {e}")

            if ask_pressed:
                if question.strip():
                    st.session_state['original_question'] = question
                    use_gap = st.session_state.get('gap_toggle', False)
                    if use_gap:
                        if st.session_state.llm is None:
                            st.error("LLM is required for research gap analysis. Please load an LLM model first.")
                        else:
                            abstracts = []
                            for paper in st.session_state.paper_list:
                                if paper['file_name'] in selected_papers:
                                    abstract = paper.get('abstract', None)
                                    if not abstract:
                                        abstract, _ = get_paper_abstract_and_keywords(st.session_state.vectorstore, paper['file_name'])
                                    if abstract:
                                        abstracts.append(abstract)
                            if abstracts and st.session_state.llm:
                                gap_prompt = get_research_gap_prompt(abstracts, st.session_state.get('summarize_answers', False))
                                # Append design outline instruction if requested
                                if st.session_state.get('design_outline'):
                                    gap_prompt += "\n\nAdditionally, generate detailed experimental outline, with specific experiment procedure and outline the challenges and expected result."
                                start_time = datetime.datetime.now()
                                with st.spinner("LLM is analyzing research gaps..."):
                                    try:
                                        gap_response = st.session_state.llm.invoke(gap_prompt)
                                        end_time = datetime.datetime.now()
                                        st.session_state['qa_answer'] = gap_response
                                        st.session_state['answer_generation_time'] = (end_time - start_time).total_seconds()
                                    except Exception as e:
                                        st.session_state['qa_answer'] = f"Error: {e}"
                                        st.session_state['answer_generation_time'] = None
                            else:
                                st.session_state['qa_answer'] = "No abstracts found or LLM not loaded."
                                st.session_state['answer_generation_time'] = None
                    else:
                        handle_question_submission(question, llm_model, selected_papers, search_type, num_results)
                    st.rerun()
                else:
                    st.warning("Please enter a question")

    # Split view after optimization
    else:
        col_left, col_right = st.columns([2, 1])
        with col_left:
            # Live follow-up toggle + year limit (outside form for instant rerun)
            col_sf1, col_sf2 = st.columns([1, 1])
            with col_sf1:
                follow_live = st.toggle(
                    "Suggest Follow-up",
                    value=st.session_state.get("followup_toggle", False),
                    key="followup_toggle_live_split",
                    help="Show Google Scholar suggestions next to the answer"
                )
                st.session_state["followup_toggle"] = follow_live
            with col_sf2:
                if follow_live:
                    current_year = datetime.datetime.now().year
                    years = ["All"] + [str(y) for y in range(current_year, 1999, -1)]
                    st.session_state['followup_year_limit'] = st.selectbox(
                        "Year limit",
                        years,
                        index=0 if st.session_state.get('followup_year_limit', 'All') == 'All' else years.index(st.session_state['followup_year_limit']),
                        key="followup_year_limit_select_live_split",
                    )

            # Show selected papers as text buttons (outside form)
            current = list(st.session_state.get('current_selected_papers', []))
            active = set(st.session_state.get('suggestion_active_papers', []))
            
            # Initialize suggestion_active_papers to current selection if empty
            if not active and current:
                active = set(current)
                st.session_state['suggestion_active_papers'] = list(active)
            
            if current:
                st.markdown("**📚 Selected Papers:**")
                # Display as text buttons in a row
                cols = st.columns(len(current))
                for i, fname in enumerate(current):
                    with cols[i]:
                        checked = fname in active
                        button_text = fname.replace('_', ' ').replace('.pdf', '')
                        if st.button(
                            button_text, 
                            key=f"paper_btn_split_{fname}", 
                            help=f"Toggle {fname} for suggestions",
                            type="primary" if checked else "secondary",
                            use_container_width=True
                        ):
                            if checked:
                                active.discard(fname)
                            else:
                                active.add(fname)
                            st.session_state['suggestion_active_papers'] = list(active)
                            st.rerun()

            with st.form("ask_combined_form_split", clear_on_submit=False):
                col_gap, col_summary, col_design = st.columns([1, 1, 1])
                with col_gap:
                    try:
                        _gap_val = st.toggle(
                            "Identify Research Gaps",
                            value=st.session_state.get("gap_toggle", False),
                            key="ask_gap_toggle",
                            help="LLM required for research gap analysis",
                            disabled=st.session_state.llm is None,
                        )
                    except Exception:
                        _gap_val = st.checkbox(
                            "Identify Research Gaps",
                            value=st.session_state.get("gap_toggle", False),
                            key="ask_gap_toggle_cb",
                            help="LLM required for research gap analysis",
                        )
                    st.session_state["gap_toggle"] = _gap_val
                with col_summary:
                    try:
                        _sum_val = st.toggle(
                            "Summarize Answers",
                            value=st.session_state.get("summarize_answers", False),
                            key="ask_summarize_toggle",
                            help="Summarize answers to 3-5 sentences",
                        )
                    except Exception:
                        _sum_val = st.checkbox(
                            "Summarize Answers",
                            value=st.session_state.get("summarize_answers", False),
                            key="ask_summarize_toggle_cb",
                            help="Summarize answers to 3-5 sentences",
                        )
                    st.session_state["summarize_answers"] = _sum_val
                with col_design:
                    try:
                        _design_val = st.toggle(
                            "Design Outline",
                            value=st.session_state.get("design_outline", False),
                            key="ask_design_outline_toggle",
                            help="Add a detailed experimental outline to the answer",
                        )
                    except Exception:
                        _design_val = st.checkbox(
                            "Design Outline",
                            value=st.session_state.get("design_outline", False),
                            key="ask_design_outline_toggle_cb",
                            help="Add a detailed experimental outline to the answer",
                        )
                    st.session_state["design_outline"] = _design_val

                question = st.text_area(
                    label="Your Question",
                    value=st.session_state.get("original_question", ""),
                    placeholder="Ask about precipitation strengthening, microstructure, mechanical properties, etc.",
                    height=120,
                    key="combined_question_area",
                    label_visibility="visible",
                )

                col_ask, col_opt = st.columns([3, 1])
                with col_ask:
                    ask_pressed = st.form_submit_button("🔍 Ask Question", type="primary", use_container_width=True)
                with col_opt:
                    optimize_pressed = st.form_submit_button("🧠 Optimize", use_container_width=True, disabled=st.session_state.llm is None)

            # Get question from session state for logic outside form
            question = st.session_state.get("combined_question_area", "")
            
            if optimize_pressed and question.strip() and st.session_state.llm:
                with st.spinner("Optimizing question..."):
                    try:
                        # Keep original input intact
                        st.session_state['original_question'] = question
                        st.caption(f"🔍 Debug: Starting optimization for: '{question}'")
                        
                        optimized_q, keywords = optimize_question(st.session_state.llm, question)
                        
                        # Debug: Show what we got from the LLM
                        st.caption(f"🔍 Debug: LLM returned - Question: '{optimized_q}', Keywords: {keywords}")
                        st.caption(f"🔍 Debug: Question type: {type(optimized_q)}, Length: {len(str(optimized_q)) if optimized_q else 0}")
                        
                        # Check if optimization actually improved the question or returned empty
                        if not optimized_q or optimized_q.strip() == "" or optimized_q == question:
                            st.warning("LLM returned empty or same question. Applying manual optimization...")
                            # Manual optimization as fallback - generate based on user's question
                            if "atomic size" in question.lower() or "size difference" in question.lower():
                                manual_optimized = f"Investigate the relationship between atomic size differences and microstructural evolution in refractory high-entropy alloys, specifically examining: (1) lattice distortion effects on dislocation behavior, (2) phase stability changes due to atomic size mismatch, and (3) mechanical property correlations with microstructural modifications"
                                manual_keywords = ["atomic size difference", "lattice distortion", "dislocation behavior", "phase stability", "microstructural evolution"]
                            elif "temperature" in question.lower() or "heat" in question.lower():
                                manual_optimized = f"Investigate the relationship between thermal processing parameters and microstructural evolution in refractory high-entropy alloys, specifically examining: (1) temperature-dependent phase transformations, (2) grain growth kinetics, and (3) mechanical property correlations with thermal history"
                                manual_keywords = ["thermal processing", "phase transformation", "grain growth", "mechanical properties", "temperature effects"]
                            elif "strength" in question.lower() or "mechanical" in question.lower():
                                manual_optimized = f"Analyze the strengthening mechanisms and mechanical properties of refractory high-entropy alloys, focusing on: (1) dislocation density evolution, (2) precipitation behavior, and (3) grain boundary effects on yield strength and ductility"
                                manual_keywords = ["strengthening mechanisms", "dislocation density", "precipitation", "grain boundaries", "mechanical properties"]
                            elif "microstructure" in question.lower() or "structure" in question.lower():
                                manual_optimized = f"Examine the microstructural evolution and phase stability in refractory high-entropy alloys, investigating: (1) phase formation mechanisms, (2) grain size distribution, and (3) interface characteristics and their influence on properties"
                                manual_keywords = ["microstructure", "phase stability", "grain size", "interfaces", "phase formation"]
                            else:
                                # Generic but relevant optimization
                                manual_optimized = f"Investigate the fundamental relationships between composition, microstructure, and properties in refractory high-entropy alloys, specifically examining: (1) atomic size effects, (2) phase stability, and (3) property correlations"
                                manual_keywords = ["composition", "microstructure", "atomic size", "phase stability", "property correlations"]
                            
                            st.session_state.optimized_question = manual_optimized
                            st.session_state.suggested_keywords = manual_keywords
                            st.success("Manual optimization applied!")
                        else:
                            st.session_state.optimized_question = optimized_q
                            st.session_state.suggested_keywords = keywords
                            st.success("Question optimized successfully!")
                        
                        st.caption(f"🔍 Debug: Final optimization result - Question: '{st.session_state.optimized_question}', Keywords: {st.session_state.suggested_keywords}")
                    except Exception as e:
                        st.error(f"Optimization failed: {e}")
                        st.caption(f"🔍 Debug: Exception details: {str(e)}")

            if ask_pressed:
                if question.strip():
                    st.session_state['original_question'] = question
                    
                    # Check if we have an optimized question and use it instead of the original
                    question_to_use = question
                    if st.session_state.get("optimized_question") and st.session_state.get("optimized_question").strip():
                        question_to_use = st.session_state["optimized_question"]
                        st.info(f"Using optimized question: {question_to_use}")
                    else:
                        st.info(f"Using original question: {question_to_use}")
                    
                    # Debug: Show what question is being used
                    st.caption(f"🔍 Debug: Question to use: '{question_to_use}'")
                    st.caption(f"🔍 Debug: Original question: '{question}'")
                    st.caption(f"🔍 Debug: Optimized question exists: {bool(st.session_state.get('optimized_question'))}")
                    if st.session_state.get('optimized_question'):
                        st.caption(f"🔍 Debug: Optimized question value: '{st.session_state.get('optimized_question')}'")
                    
                    use_gap = st.session_state.get('gap_toggle', False)
                    if use_gap:
                        if st.session_state.llm is None:
                            st.error("LLM is required for research gap analysis. Please load an LLM model first.")
                        else:
                            abstracts = []
                            for paper in st.session_state.paper_list:
                                if paper['file_name'] in selected_papers:
                                    abstract = paper.get('abstract', None)
                                    if not abstract:
                                        abstract, _ = get_paper_abstract_and_keywords(st.session_state.vectorstore, paper['file_name'])
                                    if abstract:
                                        abstracts.append(abstract)
                            if abstracts and st.session_state.llm:
                                gap_prompt = get_research_gap_prompt(abstracts, st.session_state.get('summarize_answers', False))
                                if st.session_state.get('design_outline'):
                                    gap_prompt += "\n\nAdditionally, generate detailed experimental outline, with specific experiment procedure and outline the challenges and expected result."
                                start_time = datetime.datetime.now()
                                with st.spinner("LLM is analyzing research gaps..."):
                                    try:
                                        gap_response = st.session_state.llm.invoke(gap_prompt)
                                        end_time = datetime.datetime.now()
                                        st.session_state['qa_answer'] = gap_response
                                        st.session_state['answer_generation_time'] = (end_time - start_time).total_seconds()
                                    except Exception as e:
                                        st.session_state['qa_answer'] = f"Error: {e}"
                                        st.session_state['answer_generation_time'] = None
                            else:
                                st.session_state['qa_answer'] = "No abstracts found or LLM not loaded."
                                st.session_state['answer_generation_time'] = None
                    else:
                        handle_question_submission(question_to_use, llm_model, selected_papers, search_type, num_results)
                    st.rerun()
                else:
                    st.warning("Please enter a question")

        with col_right:
            # Display optimized question with proper HTML rendering
            optimized_content = st.session_state.optimized_question
            if optimized_content:
                # Clean any HTML tags that might be in the content
                import re
                clean_content = re.sub(r'<[^>]+>', '', optimized_content)
                
                st.markdown(create_clean_card(
                    title="Optimized Question",
                    content=clean_content,
                    icon="✨",
                    variant="success"
                ), unsafe_allow_html=True)
            # Ask using optimized question
            ask_opt_btn = st.button("🔍 Ask Optimized Question", use_container_width=True, key="ask_optimized_btn")
            if ask_opt_btn:
                opt_q = st.session_state.get('optimized_question', '').strip()
                if not opt_q:
                    st.warning("No optimized question available.")
                else:
                    # Store the optimized question as the question to use
                    st.session_state['original_question'] = opt_q
                    st.info(f"Using optimized question: {opt_q}")
                    
                    use_gap = st.session_state.get('gap_toggle', False)
                    if use_gap:
                        if st.session_state.llm is None:
                            st.error("LLM is required for research gap analysis. Please load an LLM model first.")
                        else:
                            abstracts = []
                            for paper in st.session_state.paper_list:
                                if paper['file_name'] in selected_papers:
                                    abstract = paper.get('abstract', None)
                                    if not abstract:
                                        abstract, _ = get_paper_abstract_and_keywords(st.session_state.vectorstore, paper['file_name'])
                                    if abstract:
                                        abstracts.append(abstract)
                            if abstracts and st.session_state.llm:
                                gap_prompt = get_research_gap_prompt(abstracts, st.session_state.get('summarize_answers', False))
                                with st.spinner("LLM is analyzing research gaps..."):
                                    try:
                                        gap_response = st.session_state.llm.invoke(gap_prompt)
                                        st.session_state['qa_answer'] = gap_response
                                    except Exception as e:
                                        st.session_state['qa_answer'] = f"Error: {e}"
                            else:
                                st.session_state['qa_answer'] = "No abstracts found or LLM not loaded."
                    else:
                        handle_question_submission(opt_q, llm_model, selected_papers, search_type, num_results)
                    st.rerun()


def handle_dashboard_message():
    """
    Handle messages from the React dashboard component.
    This mirrors the logic used in the React-enabled app.
    """
    if 'dashboard_action' in st.session_state:
        action = st.session_state.dashboard_action
        if action == 'search':
            query = st.session_state.get('dashboard_query', '')
            options = st.session_state.get('dashboard_options', {})
            st.session_state.search_query = query
            st.session_state.search_options = options
            st.rerun()
        elif action == 'optimize_question':
            query = st.session_state.get('dashboard_query', '')
            if query and st.session_state.llm:
                optimized_q, keywords = optimize_question(st.session_state.llm, query)
                st.session_state.optimized_question = optimized_q
                st.session_state.suggested_keywords = keywords
                send_dashboard_update('set_optimized_question', optimized_q)
            st.rerun()
        elif action == 'select_paper':
            paper = st.session_state.get('dashboard_paper', {})
            st.session_state.selected_paper = paper
            st.rerun()
        del st.session_state.dashboard_action


def send_dashboard_data(data_type: str, data: dict):
    """Send data to the React dashboard component via session state."""
    st.session_state.dashboard_data = {'type': data_type, 'data': data}


def send_dashboard_update(action: str, data):
    """Send updates to the React dashboard component via session state."""
    st.session_state.dashboard_update = {'action': action, 'data': data}


def process_search_query(query: str, selected_papers: List[str], llm_model: str, search_type: str, num_results: int):
    """Process search query from React dashboard."""
    if not query.strip():
        return None
    options = st.session_state.get('search_options', {})
    identify_gaps = options.get('identify_gaps', False)
    summarize_answers = options.get('summarize_answers', False)
    st.session_state.summarize_answers = summarize_answers

    if st.session_state.vectorstore:
        with st.spinner("Searching papers..."):
            if identify_gaps:
                abstracts = []
                for paper in st.session_state.paper_list:
                    if paper['file_name'] in selected_papers:
                        abstract = paper.get('abstract', None)
                        if not abstract:
                            abstract, _ = get_paper_abstract_and_keywords(st.session_state.vectorstore, paper['file_name'])
                        if abstract:
                            abstracts.append(abstract)
                if abstracts and st.session_state.llm:
                    gap_prompt = get_research_gap_prompt(abstracts, summarize_answers)
                    try:
                        answer = st.session_state.llm.invoke(gap_prompt)
                        return {'query': query, 'answer': answer, 'papers': [], 'total_results': 0, 'type': 'research_gaps'}
                    except Exception as e:
                        return {'query': query, 'answer': f"Error analyzing research gaps: {e}", 'papers': [], 'total_results': 0, 'type': 'error'}
            else:
                search_results, success = search_papers(
                    st.session_state.vectorstore,
                    query,
                    selected_papers if selected_papers else None,
                    search_type,
                    num_results
                )
                if success and search_results:
                    answer = generate_answer(st.session_state.llm, query, search_results, summarize_answers)
                    papers_data = []
                    for doc in search_results:
                        paper_info = {
                            'title': doc.metadata.get('title', doc.metadata.get('file_name', 'Unknown')),
                            'authors': doc.metadata.get('authors', ['Unknown']),
                            'abstract': doc.page_content[:300] + '...' if len(doc.page_content) > 300 else doc.page_content,
                            'year': doc.metadata.get('year', 'Unknown'),
                            'citations': doc.metadata.get('citations', 0),
                            'venue': doc.metadata.get('venue', 'Unknown'),
                            'tags': doc.metadata.get('tags', []),
                            'file_name': doc.metadata.get('file_name', 'Unknown')
                        }
                        papers_data.append(paper_info)
                    return {'query': query, 'answer': answer, 'papers': papers_data, 'total_results': len(search_results), 'type': 'normal_search'}
    return None


def display_dashboard_controls():
    """Toggles/buttons in the same fashion as the modular UI for the React dashboard flow."""
    display_clean_section_header("Ask & Options", "Use toggles below to control dashboard behavior")
    if 'search_options' not in st.session_state or not isinstance(st.session_state.search_options, dict):
        st.session_state.search_options = {}
    col_gap, col_summary, col_follow = st.columns([1, 1, 1])
    with col_gap:
        identify_gaps_default = bool(st.session_state.search_options.get('identify_gaps', False))
        identify_gaps_val = st.checkbox(
            "Identify Research Gaps",
            value=identify_gaps_default,
            key="identify_gaps_toggle",
            disabled=st.session_state.llm is None,
            help="LLM required for research gap analysis"
        )
        st.session_state.search_options['identify_gaps'] = identify_gaps_val
    with col_summary:
        summarize_default = bool(st.session_state.get('summarize_answers', False))
        summarize_val = st.checkbox(
            "Summarize Answers",
            value=summarize_default,
            key="summarize_answers_toggle",
            help="Summarize answers to 3-5 sentences"
        )
        st.session_state.summarize_answers = summarize_val
        st.session_state.search_options['summarize_answers'] = summarize_val
    with col_follow:
        suggest_default = bool(st.session_state.search_options.get('suggest_followup', False))
        suggest_val = st.checkbox(
            "Suggest Follow-up Reading",
            value=suggest_default,
            key="suggest_followup_toggle",
            help="Recommend related papers from results"
        )
        st.session_state.search_options['suggest_followup'] = suggest_val
    if st.session_state.get('optimized_question'):
        st.markdown(create_clean_card(
            title="Optimized Question",
            content=st.session_state.optimized_question,
            icon="✨",
            variant="success"
        ), unsafe_allow_html=True)


def display_question_section(llm_model: str, selected_papers: List[str], search_type: str, num_results: int, gap_toggle: bool = False):
    """Display the question input and optimization section"""
    # Question input area (no text, just the input)
    
    question = st.text_area(
        "Your Question:",
        placeholder="Ask about precipitation strengthening, microstructure, mechanical properties, etc.",
        height=100
    )
    
    # Question optimization section
    col_opt1, col_opt2 = st.columns([1, 1])
    with col_opt1:
        if st.button("🧠 Optimize Question", help="Let AI optimize your question for better search results", disabled=st.session_state.llm is None):
            if question.strip():
                if st.session_state.llm:
                    with st.spinner("Optimizing question..."):
                        optimized_q, keywords = optimize_question(st.session_state.llm, question)
                        st.session_state.optimized_question = optimized_q
                        st.session_state.suggested_keywords = keywords
                else:
                    st.error("LLM not available for optimization")
            else:
                st.warning("Please enter a question first")
        # Display optimized question if available (in left column)
        if st.session_state.optimized_question:
            st.markdown(create_clean_card(
                title="Optimized Question",
                content=st.session_state.optimized_question,
                icon="✨",
                variant="success"
            ), unsafe_allow_html=True)
    with col_opt2:
        if st.button("🔑 Show Keywords", help="Show suggested keywords for better search"):
            st.session_state.suggested_keywords = get_suggested_keywords()
        # Keyword selection section (in right column)
        if st.session_state.suggested_keywords:
            st.markdown(create_clean_card(
                title="Select Keywords",
                content="Choose relevant keywords to enhance your search results.",
                icon="🔑",
                variant="info"
            ), unsafe_allow_html=True)
            
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
    # Show selected keywords (below both columns)
    # Keyword UI removed per request
    # Ask button - always show it, regardless of optimization status
    if st.button("🔍 Ask Question", type="primary", use_container_width=True):
        # Debug information
        st.caption(f"🔍 Debug: Button clicked! Question: '{question}'")
        st.caption(f"🔍 Debug: gap_toggle: {gap_toggle}")
        st.caption(f"🔍 Debug: optimized_question exists: {'optimized_question' in st.session_state}")
        if 'optimized_question' in st.session_state:
            st.caption(f"🔍 Debug: optimized_question value: {st.session_state.optimized_question}")
        
        if question.strip():
            # Store original question for export
            st.session_state['original_question'] = question
            
            if gap_toggle:
                if st.session_state.llm is None:
                    st.error("LLM is required for research gap analysis. Please load an LLM model first.")
                else:
                    # GAP IDENTIFIER LOGIC
                    abstracts = []
                    for paper in st.session_state.paper_list:
                        if paper['file_name'] in selected_papers:
                            abstract = paper.get('abstract', None)
                            if not abstract:
                                abstract, _ = get_paper_abstract_and_keywords(st.session_state.vectorstore, paper['file_name'])
                            if abstract:
                                abstracts.append(abstract)
                    if abstracts and st.session_state.llm:
                        # Use centralized prompt for research gap analysis
                        gap_prompt = get_research_gap_prompt(abstracts, st.session_state.get('summarize_answers', False))
                        with st.spinner("LLM is analyzing research gaps..."):
                            try:
                                gap_response = st.session_state.llm.invoke(gap_prompt)
                                st.session_state['qa_answer'] = gap_response
                                st.caption(f"🔍 Debug: Gap analysis completed, stored answer length: {len(gap_response)}")
                            except Exception as e:
                                st.session_state['qa_answer'] = f"Error: {e}"
                                st.caption(f"🔍 Debug: Gap analysis failed with error: {e}")
                    else:
                        st.session_state['qa_answer'] = "No abstracts found or LLM not loaded."
                        st.caption("🔍 Debug: No abstracts or LLM not available for gap analysis")
            else:
                st.caption("🔍 Debug: Calling handle_question_submission for normal question")
                handle_question_submission(question, llm_model, selected_papers, search_type, num_results)
            
            # Rerun to display the answer in the full-width section below
            st.rerun()
        else:
            st.warning("Please enter a question")


def handle_question_submission(question: str, llm_model: str, selected_papers: List[str], search_type: str, num_results: int):
    """Handle question submission and display results"""
    # Load LLM if not already loaded or if model changed
    if st.session_state.llm is None or st.session_state.get('current_model') != llm_model:
        with st.spinner(f"Attempting to load {llm_model}..."):
            st.session_state.llm = load_llm(llm_model)
            st.session_state.current_model = llm_model
    
    # Enhance question with selected keywords
    enhanced_question = question
    if st.session_state.selected_keywords:
        keywords_text = " ".join(st.session_state.selected_keywords)
        enhanced_question = f"{question} {keywords_text}"
    
    # Use LLM to extract keywords for Scholar (if available)
    if st.session_state.llm:
        optimized_q, _ = optimize_question(st.session_state.llm, question)
        st.session_state.optimized_question = optimized_q

    start_time = datetime.datetime.now()
    with st.spinner("Searching papers..."):
        # Search papers
        search_results, success = search_papers(
            st.session_state.vectorstore,
            enhanced_question, 
            selected_papers if selected_papers else None,
            search_type,
            num_results
        )
        
        if success and search_results:
            # Generate answer (with or without LLM)
            answer = generate_answer(st.session_state.llm, question, search_results, st.session_state.get('summarize_answers', False))
            # If design outline requested, append instruction to the answer via a follow-up enhancement call when LLM is available
            if st.session_state.llm and st.session_state.get('design_outline'):
                try:
                    design_suffix = "\n\nAdditionally, generate detailed experimental outline, with specific experiment procedure and outline the challenges and expected result."
                    answer = st.session_state.llm.invoke(answer + design_suffix)
                except Exception:
                    pass
        else:
            if selected_papers and len(selected_papers) > 0:
                answer = "No relevant documents found for your question in the selected papers. Try broadening your selection or rephrasing your question."
            else:
                answer = "No relevant documents found for your question."
            search_results = []
    
    end_time = datetime.datetime.now()
    # Store answer and timing in session state
    st.session_state['qa_answer'] = answer
    st.session_state['answer_generation_time'] = (end_time - start_time).total_seconds()
    
    # Display sources card
    if search_results:
        # Count meeting notes vs papers in results
        meeting_notes = [doc for doc in search_results if doc.metadata.get('content_type') == 'meeting_notes']
        research_papers = [doc for doc in search_results if doc.metadata.get('content_type') != 'meeting_notes']
        
        sources_title = "📚 Sources"
        if meeting_notes:
            sources_title += f" ({len(research_papers)} papers, {len(meeting_notes)} meeting notes)"
        
        st.markdown(create_glass_card(sources_title), unsafe_allow_html=True)
        
        for i, doc in enumerate(search_results):
            # Determine source type and icon
            is_meeting_note = doc.metadata.get('content_type') == 'meeting_notes'
            source_icon = "📝" if is_meeting_note else "📄"
            source_type = "Meeting Note" if is_meeting_note else "Research Paper"
            
            file_display = doc.metadata.get('file_name', 'Unknown')
            if is_meeting_note:
                # Show meeting title instead of filename for notes
                file_display = doc.metadata.get('title', 'Unknown Meeting Note')
            
            with st.expander(f"{source_icon} Source {i+1}: {file_display} [{doc.metadata.get('document_type', 'unknown').upper()}] - {source_type}"):
                # Show different metadata based on source type
                if is_meeting_note:
                    st.markdown(f"""
                    <div class="content-card" style="margin-bottom: 0.25rem; padding: 0.5rem 0.75rem;">
                        <strong>Meeting Date:</strong> {doc.metadata.get('meeting_date', 'Unknown')}<br>
                        <strong>Section:</strong> {doc.metadata.get('section', 'Unknown')}<br>
                        <strong>Content Length:</strong> {len(doc.page_content)} characters<br>
                        {'<strong>Papers Discussed:</strong> ' + ', '.join(doc.metadata.get('papers_discussed', [])) + '<br>' if doc.metadata.get('papers_discussed') else ''}
                        {'<strong>Tags:</strong> ' + ', '.join(doc.metadata.get('tags', [])) + '<br>' if doc.metadata.get('tags') else ''}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="content-card" style="margin-bottom: 0.25rem; padding: 0.5rem 0.75rem;">
                        <strong>Section:</strong> {doc.metadata.get('section', 'Unknown')}<br>
                        <strong>Content Length:</strong> {len(doc.page_content)} characters<br>
                        {'<strong>Figures:</strong> ' + str(doc.metadata.get('figure_count', 0)) + '<br>' if doc.metadata.get('figure_count', 0) > 0 else ''}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("**Preview:**")
                preview_text = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                st.markdown(
                    create_content_card(preview_text, "font-size: 0.9em; margin: 0.25rem 0; padding: 0.5rem 0.75rem;"),
                    unsafe_allow_html=True
                )
    
    # Suggested Follow Up Reading setup remains; keyword UI removed above


def display_preview_section(selected_papers: List[str]):
    """Display the paper preview section using exactly two boxes (expanders)."""
    # Box 1: Browse & Select
    with st.expander("📚 Browse & Select Papers", expanded=True):
        if st.session_state.paper_list:
            folder_order, folder_icons = get_folder_config()
            papers_by_folder = {}
            for paper in st.session_state.paper_list:
                folder = paper.get('folder', 'Unknown')
                papers_by_folder.setdefault(folder, []).append(paper)

            if 'selected_folder' not in st.session_state:
                st.session_state.selected_folder = None

            folder_cols = st.columns(len(folder_order))
            for i, folder in enumerate(folder_order):
                if folder in papers_by_folder:
                    icon = folder_icons.get(folder, '📁')
                    paper_count = len(papers_by_folder[folder])
                    with folder_cols[i]:
                        if st.button(
                            f"{icon} {folder}\n({paper_count} papers)",
                            key=f"folder_btn_{folder}",
                            use_container_width=True,
                            type="primary" if st.session_state.selected_folder == folder else "secondary"
                        ):
                            st.session_state.selected_folder = folder
                            st.rerun()

            # Small selection summary
            current_selected = st.session_state.get('current_selected_papers', [])
            st.caption(f"Selected papers: {len(current_selected)}")
            if st.session_state.selected_folder:
                st.caption(f"Active folder: {st.session_state.selected_folder}")
        else:
            display_clean_empty_state(
                icon="📋",
                title="No Papers Available",
                description="No papers found in the system. Please check the vector store configuration.",
                action_text="Check System Status",
                action_func=None
            )

    # Box 2: Papers grid (card design, two per row)
    with st.expander("📄 Papers", expanded=True):
        if not st.session_state.paper_list:
            st.info("No papers to display.")
            return

        folder_order, _ = get_folder_config()
        papers_by_folder = {}
        for paper in st.session_state.paper_list:
            folder = paper.get('folder', 'Unknown')
            papers_by_folder.setdefault(folder, []).append(paper)

        def render_cards(papers: list[dict]):
            cols = st.columns(2)
            for i, paper in enumerate(papers):
                with cols[i % 2]:
                    display_paper_card(paper, is_selected=False)

        # Selected papers view
        if selected_papers:
            st.markdown("**📚 Selected Papers**")
            selected_infos = [
                next((p for p in st.session_state.paper_list if p['file_name'] == name), None)
                for name in selected_papers
            ]
            selected_infos = [p for p in selected_infos if p]
            if selected_infos:
                render_cards(selected_infos)
            else:
                st.info("No details found for selected papers.")
        # Active folder view
        elif st.session_state.selected_folder and st.session_state.selected_folder in papers_by_folder:
            st.markdown(f"**📚 Papers in {st.session_state.selected_folder}**")
            render_cards(papers_by_folder[st.session_state.selected_folder])
        # Sample view
        else:
            st.info("👆 Select a research area above or choose papers from the sidebar.")
            sample_papers = []
            for folder in folder_order:
                if folder in papers_by_folder:
                    sample_papers.extend(papers_by_folder[folder][:2])
            if sample_papers:
                render_cards(sample_papers[:8])


def display_paper_card(paper_info: dict, is_selected: bool = False):
    """Display a single paper in card format using clean UI components"""
    # Get paper metadata
    paper_name = paper_info['file_name']
    folder = paper_info.get('folder', 'Unknown')
    figure_count = paper_info.get('figure_count', 0)
    vectorized_model = paper_info.get('vectorized_model', None)
    
    # Get abstract from vector store
    abstract_content, _ = get_paper_abstract_and_keywords(st.session_state.vectorstore, paper_name)
    
    # Get keywords from our keywords file
    keywords = get_paper_keywords(paper_name, folder)
    
    # Get actual title from vector store or use filename
    actual_title = None
    if st.session_state.vectorstore:
        try:
            results = st.session_state.vectorstore.similarity_search(
                paper_name, 
                k=5,
                filter={"file_name": paper_name}
            )
            for doc in results:
                if doc.metadata.get('title'):
                    actual_title = doc.metadata.get('title')
                    break
        except Exception:
            pass
    
    display_title = actual_title if actual_title else paper_name.replace('.pdf', '').replace('_', ' ')
    
    # Create Google Scholar link
    scholar_query = display_title.replace(' ', '+')
    scholar_url = f"https://scholar.google.com/scholar?q={scholar_query}"
    
    # Card variant based on selection status
    card_variant = "success" if is_selected else "default"
    
    # Prepare content for the card
    content_parts = []
    
    # Add metadata row
    metadata_text = f"📁 {folder} | 🖼️ {figure_count} figures | {'✅ Vectorized' if vectorized_model else '❌ Not Vectorized'}"
    content_parts.append(metadata_text)
    
    # Add abstract
    abstract_text = abstract_content[:300] + '...' if abstract_content and len(abstract_content) > 300 else (abstract_content or 'Abstract not available')
    content_parts.append(f"\n**Abstract:** {abstract_text}")
    
    # Add keywords in a compact row layout without card design
    if keywords:
        # Create keyword badges in a row (limit to 2-3 keywords)
        keyword_badges = []
        for kw in keywords[:3]:  # Show only 2-3 keywords to save space
            keyword_badges.append(f"**{kw}**")
        
        # Join keywords with spaces for compact row layout
        keyword_row = " ".join(keyword_badges)
        content_parts.append(f"\n**Keywords:** {keyword_row}")
    
    # Add Google Scholar link
    content_parts.append(f"\n🔗 **[View on Google Scholar]({scholar_url})**")
    
    # Combine all content
    card_content = "\n".join(content_parts)
    
    # Create the paper card using clean UI components
    st.markdown(create_clean_card(
        title=display_title,
        content=card_content,
        icon="📄",
        variant=card_variant,
        padding="medium"
    ), unsafe_allow_html=True)


def display_network_section(selected_papers: List[str]):
    """Display the paper network visualization section"""
    display_clean_section_header("Paper Network", "Visualize relationships between selected papers")
    
    if selected_papers:
        # Get paper metadata for selected papers
        selected_metadata = []
        for paper_name in selected_papers:
            paper_info = next((p for p in st.session_state.paper_list if p['file_name'] == paper_name), None)
            if paper_info:
                # Extract title and authors from filename for now
                title = paper_name.replace('.pdf', '').replace('_', ' ')
                authors = "Unknown"  # Could be enhanced to extract from actual paper content
                selected_metadata.append({'title': title, 'authors': authors})
        
        if selected_metadata:
            # Create demo similarity matrix (replace with real similarity calculation)
            n = len(selected_metadata)
            import numpy as np
            # Create a demo similarity matrix (random for now)
            np.random.seed(42)
            demo_matrix = np.random.uniform(0.1, 0.9, (n, n))
            # Make it symmetric
            demo_matrix = (demo_matrix + demo_matrix.T) / 2
            # Set diagonal to 1
            np.fill_diagonal(demo_matrix, 1.0)
            
           
        else:
            st.warning("No valid paper metadata found")
    else:
        display_clean_empty_state(
            icon="🕸️",
            title="No Papers Selected",
            description="Select papers from the sidebar to visualize their network relationships.",
            action_text="Select Papers",
            action_func=None
        )


def display_scholar_section():
    """Display the Google Scholar scraper section"""
    display_clean_section_header("Scholar Abstract Scraper", "Fetch abstracts from Google Scholar")
    
    st.markdown(create_clean_card(
        title="Important Note",
        content="This is for research/prototyping. For production, consider SerpAPI.",
        icon="⚠️",
        variant="warning"
    ), unsafe_allow_html=True)
    
    # Year selection
    import datetime
    current_year = datetime.datetime.now().year
    years = ["All"] + [str(y) for y in range(current_year, 1999, -1)]
    selected_year = st.selectbox("Select publication year (optional):", years, index=0)
    
    query = st.text_input("Enter paper title or search query:")
    
    if st.button("🔍 Scrape Abstract"):
        if not query.strip():
            st.warning("Please enter a query.")
            return
        
        # Try to import scholarly
        try:
            from scholarly import scholarly
            SCHOLARLY_AVAILABLE = True
        except ImportError:
            SCHOLARLY_AVAILABLE = False
        
        if not SCHOLARLY_AVAILABLE:
            st.error("The 'scholarly' package is not installed. Please install it with 'pip install scholarly'.")
            return
        
        try:
            search_iter = scholarly.search_pubs(query)
            filtered_result = None
            for result in search_iter:
                bib = result.get('bib', {})
                year = str(bib.get('pub_year', bib.get('year', '')))
                if selected_year == "All" or year == selected_year:
                    filtered_result = result
                    break
            
            if not filtered_result:
                st.warning(f"No results found for year {selected_year}." if selected_year != "All" else "No results found.")
                return
            
            bib = filtered_result.get('bib', {})
            title = bib.get('title', '(No title found)')
            authors = bib.get('author', '(No authors found)')
            year = bib.get('pub_year', bib.get('year', ''))
            venue = bib.get('venue', bib.get('journal', ''))
            abstract = bib.get('abstract', '(No abstract found)')
            url = bib.get('url', '')
            num_citations = filtered_result.get('num_citations', None)
            
            # Display formatted result
            st.markdown(f"""
**<span style='font-size:1.3em'>{title}</span>**

**Authors:** {authors}

**Year:** {year if year else 'N/A'}

**Venue:** {venue if venue else 'N/A'}

**Abstract:**
> {abstract}

{f'**URL:** [{url}]({url})' if url else ''}

{f'**Citations:** {num_citations}' if num_citations is not None else ''}
""", unsafe_allow_html=True)
            
            # Raw result as expandable debug
            with st.expander("Show raw result (debug)"):
                st.write(filtered_result)
                
        except Exception as e:
            st.error(f"Error during scholarly search: {e}")


def scholar_search_and_display():
    st.markdown('### 📖 Suggested Follow Up Reading')
    keywords = st.session_state.get('suggested_keywords', [])[:4]
    if not keywords:
        st.info("No keywords found yet. Ask a question to get recommendations!")
        return
    query = ' '.join(keywords)
    st.markdown(f"**Keywords used:** `{query}`")
    try:
        from scholarly import scholarly
        search_iter = scholarly.search_pubs(query)
        count = 0
        paper_links = []
        abstracts = []
        years = []
        for result in search_iter:
            if count >= 5:
                break
            bib = result.get('bib', {})
            title = bib.get('title', '(No title found)')
            abstract = bib.get('abstract', '')
            url = bib.get('url', '')
            year = bib.get('pub_year', bib.get('year', ''))
            scholar_link = url if url else f'https://scholar.google.com/scholar?q={title.replace(' ', '+')}'
            paper_links.append({'title': title, 'link': scholar_link, 'year': year})
            if abstract:
                abstracts.append(abstract)
            years.append(year)
            count += 1
        # LLM summary of all abstracts (if available)
        if abstracts and st.session_state.llm:
            summary_prompt = get_scholar_summary_prompt(abstracts)
            try:
                summary = st.session_state.llm.invoke(summary_prompt)
                st.markdown(f"**Summary of Suggested Readings:**\n{summary}")
            except Exception as e:
                st.warning(f"Could not generate summary: {e}")
        elif abstracts:
            # Simple summary without LLM
            st.markdown("**Summary of Suggested Readings:**")
            st.markdown("*Note: LLM not available for AI-powered summary. Showing paper links below.*")
        else:
            st.info("No abstracts available to summarize.")
        # Display paper links and years
        for paper in paper_links:
            year_str = f" ({paper['year']})" if paper['year'] else ''
            st.markdown(f"- [{paper['title']}]({paper['link']}){year_str}")
        if count == 0:
            st.info("No results found on Google Scholar.")
    except ImportError:
        st.error("The 'scholarly' package is not installed. Please install it with 'pip install scholarly'.")
    except Exception as e:
        st.error(f"Error during scholarly search: {e}")


def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Initialize system messages if not exists (but don't clear existing ones)
    if 'system_messages' not in st.session_state:
        st.session_state.system_messages = []
    
    # Apply clean theme and styling (keep existing)
    apply_clean_theme()
    apply_clean_tab_style()
    # Incrementally apply modern tab style for better visuals
    apply_modern_tab_style()
    
    # Add custom CSS for larger toggles, buttons, and text
    st.markdown("""
    <style>
    /* Larger toggles */
    .stToggle > label {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        line-height: 1.4 !important;
    }
    
    /* Larger toggle switch size */
    .stToggle > div > div {
        transform: scale(1.2) !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Larger checkboxes */
    .stCheckbox > label {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        line-height: 1.4 !important;
    }
    
    /* Larger checkbox size */
    .stCheckbox > div > div {
        transform: scale(1.2) !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Larger buttons */
    .stButton > button {
        font-size: 1.3rem !important;
        padding: 1rem 2rem !important;
        font-weight: 600 !important;
    }
    
    /* Larger text areas */
    .stTextArea > div > div > textarea {
        font-size: 1.3rem !important;
        line-height: 1.5 !important;
    }
    
    /* Larger text area labels */
    .stTextArea > label {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }
    
    /* Larger select boxes */
    .stSelectbox > div > div > div {
        font-size: 1.3rem !important;
    }
    
    /* Larger select box labels */
    .stSelectbox > label {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }
    
    /* Larger form submit buttons */
    .stFormSubmitButton > button {
        font-size: 1.3rem !important;
        padding: 1rem 2rem !important;
        font-weight: 600 !important;
    }
    
    /* Larger markdown text */
    .stMarkdown > div > div {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }
    
    /* Larger form labels */
    .stForm > div > div > div > label {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # React header (optional) and main title
    display_react_header()
    handle_header_message()
    # Left-aligned header with controlled top whitespace (~1/8 viewport) without animations
    st.markdown(
        """
        <div style="margin-top:12.5vh;"></div>
        <div class="clean-card" style="margin-top:0; margin-bottom:0.75rem; border-left: 3px solid var(--accent-color); padding: var(--spacing-md); text-align:left;">
            <h1 style="margin:0; font-size:1.6rem; color: var(--text-primary); font-weight:600;">🔬 Material Research RAG System</h1>
            <p style="margin:0.25rem 0 0 0; color: var(--text-secondary);">Modern paper search and LLM Q&A with streamlined UI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Load initial data
    load_initial_data()
    
    # Check if vector store loaded successfully
    if not st.session_state.vectorstore:
        st.error("❌ Failed to load vector store. Please check the configuration.")
        st.info("💡 Check the sidebar under 'System Status' for detailed error messages and troubleshooting steps.")
        # Don't return - let the app continue so user can see error details in sidebar
    
    # Use sidebar controls and paper selection
    selected_papers, llm_model, search_type, num_results = display_sidebar()
    
    # Note: Keeping core modular UI intact; React dashboard intentionally not injected to preserve functionality

    # Main content area with tabs - conditionally create tabs based on LLM availability
    if st.session_state.llm is not None:
        # LLM is available - show all tabs including Paper Network
        tab1, tab2, tab3, tab4 = st.tabs([
            "💬 Ask Question", 
            "🖼️ Paper Preview", 
            "🕸️ Paper Network", 
            "🌐 Scholar Abstract Scraper"
        ])
    else:
        # LLM is not available - hide Paper Network tab
        tab1, tab2, tab4 = st.tabs([
            "💬 Ask Question", 
            "🖼️ Paper Preview", 
            "🌐 Scholar Abstract Scraper"
        ])

    with tab1:
        display_ask_combined_card(llm_model, selected_papers, search_type, num_results)
        
  
        
        # Display answer from session state (if exists)
        if st.session_state.get('qa_answer'):
            use_gap = st.session_state.get('gap_toggle', False)
            if use_gap:
                # Add timing info to the title
                timing_info = ""
                if st.session_state.get('answer_generation_time'):
                    timing_info = f" ⏱️ {st.session_state['answer_generation_time']:.1f}s"
                
                st.markdown(create_clean_card(
                    title=f"Research Gaps Identified{timing_info}",
                    content=st.session_state['qa_answer'],
                    icon="🔍",
                    variant="warning"
                ), unsafe_allow_html=True)
                # Show summarize indicator if enabled
                if st.session_state.get('summarize_answers', False):
                    st.caption("📝 **Summarized to 3-5 sentences**")
            else:
                # Answer + optional follow-up suggestions in 7:3 layout
                if st.session_state.get('followup_toggle', False):
                    col_ans, col_suggest = st.columns([7, 3])
                    with col_ans:
                        # Add timing info to the title
                        timing_info = ""
                        if st.session_state.get('answer_generation_time'):
                            timing_info = f" ⏱️ {st.session_state['answer_generation_time']:.1f}s"
                        
                        st.markdown(create_clean_card(
                            title=f"AI-Generated Answer{timing_info}",
                            content=st.session_state['qa_answer'],
                            icon="🤖",
                            variant="default"
                        ), unsafe_allow_html=True)
                    with col_suggest:
                        # Show suggested follow-up reading results
                        st.markdown(
                            "**📖 Suggested Follow-up Reading**"
                        )

                        # Build query from optimized question or original question
                        query_for_followup = st.session_state.get('optimized_question') or st.session_state.get('original_question', '')
                        # Optionally append active paper titles to bias the scholar query
                        if st.session_state.get('suggestion_active_papers'):
                            titles = []
                            for name in st.session_state['suggestion_active_papers']:
                                titles.append(name.replace('_', ' ').replace('.pdf', ''))
                            query_for_followup = (query_for_followup + " " + " ".join(titles)).strip()
                        year_filter = st.session_state.get('followup_year_limit', 'All')

                        # Try fetching up to 3 suggestions via scholarly directly
                        suggestions: list[dict] = []
                        try:
                            from scholarly import scholarly  # type: ignore
                            search_iter = scholarly.search_pubs(query_for_followup)
                            for result in search_iter:
                                bib = result.get('bib', {})
                                year = str(bib.get('pub_year', bib.get('year', '')))
                                if year_filter == 'All' or year == year_filter:
                                    suggestions.append({
                                        'title': bib.get('title', '(No title found)'),
                                        'year': year,
                                        'venue': bib.get('venue', bib.get('journal', '')),
                                        'abstract': bib.get('abstract', '(No abstract found)'),
                                        'url': bib.get('url', '')
                                    })
                                if len(suggestions) >= 3:
                                    break
                        except Exception:
                            # Fallback to utility (may return 1)
                            try:
                                from utils.scholar_scraper_tab import search_scholar_followup
                                single = search_scholar_followup(query_for_followup, year_filter)
                                if single:
                                    suggestions.append(single)
                            except Exception:
                                pass

                        if suggestions:
                            # Header
                            st.markdown(
                                """
                                <div class="clean-card" style="border-left: 3px solid var(--accent-color); padding: var(--spacing-sm);">
                                  <div style="display:flex; align-items:center; gap:0.5rem; font-weight:600; color: var(--accent-color);">🔎 Suggested Follow-up</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            for s in suggestions:
                                title_html = html.escape(s.get('title', '(No title)'))
                                venue_html = html.escape(s.get('venue', ''))
                                abstract_html = html.escape(s.get('abstract', ''))
                                year_html = html.escape(s.get('year', ''))
                                url = s.get('url')
                                if not url:
                                    url = f"https://scholar.google.com/scholar?q={quote(s.get('title',''))}"
                                item_html = (
                                    f"<div class=\"clean-card\" style=\"border-left: 3px solid var(--primary-color); padding: var(--spacing-sm);\">"
                                    f"  <div style='font-weight:600; color: var(--text-primary);'>{title_html}</div>"
                                    f"  <div style='color: var(--text-muted); font-size: 0.9em'>{year_html}{' • ' + venue_html if venue_html else ''}</div>"
                                    f"  <div style='margin-top:0.5rem; line-height:1.6; color: var(--text-secondary);'>{abstract_html}</div>"
                                    f"  <div style='margin-top:0.5rem'><a href='{url}' target='_blank'>Open</a></div>"
                                    f"</div>"
                                )
                                st.markdown(item_html, unsafe_allow_html=True)
                        else:
                            st.info("No follow-up suggestions found.")
                else:
                    # Add timing info to the title
                    timing_info = ""
                    if st.session_state.get('answer_generation_time'):
                        timing_info = f" ⏱️ {st.session_state['answer_generation_time']:.1f}s"
                    
                    st.markdown(create_clean_card(
                        title=f"AI-Generated Answer{timing_info}",
                        content=st.session_state['qa_answer'],
                        icon="🤖",
                        variant="default"
                    ), unsafe_allow_html=True)
                
                # Minimal answer presentation; remove extra captions
            
            # Follow-up (single combined form) without extra spacer box
            with st.form("followup_form", clear_on_submit=False):
                # Header with timing information
                col_header, col_timing = st.columns([3, 1])
                with col_header:
                    st.markdown("**💭 Follow-up Question**")
                    st.caption("Ask a follow-up question based on the previous answer.")
                with col_timing:
                    current_time = datetime.datetime.now().strftime("%H:%M:%S")
                    st.caption(f"⏰ {current_time}")
                
                follow_up_question = st.text_area(
                    "",
                    placeholder="Ask a follow-up question based on the previous answer...",
                    height=80,
                    key="follow_up_question",
                    label_visibility="collapsed"
                )
                col_followup1, col_followup2 = st.columns([1, 1])
                with col_followup1:
                    ask_followup_pressed = st.form_submit_button("🔍 Ask Follow-up", use_container_width=True)
                with col_followup2:
                    clear_followup_pressed = st.form_submit_button("🗑️ Clear Follow-up", use_container_width=True)

            if ask_followup_pressed:
                    if follow_up_question.strip():
                        if st.session_state.llm:
                            # Build context from selected follow-up answers and current answer
                            additional_context = ""
                            context_parts = []
                            
                            # Include current answer as context if toggle is enabled
                            if st.session_state.get('current_answer_as_context', False) and st.session_state.get('follow_up_answer'):
                                # Use the latest follow-up question from session state
                                latest_q = st.session_state.get('latest_follow_up_question', follow_up_question)
                                context_parts.append(f"Current Follow-up Q&A:\nQ: {latest_q}\nA: {st.session_state['follow_up_answer']}")
                            
                            # Include selected historical follow-up answers
                            if st.session_state.selected_context_answers:
                                for idx in st.session_state.selected_context_answers:
                                    if idx < len(st.session_state.follow_up_history):
                                        question, answer, _ = st.session_state.follow_up_history[idx]
                                        context_parts.append(f"Previous Q&A {idx+1}:\nQ: {question}\nA: {answer}")
                            
                            if context_parts:
                                additional_context = "\n\nAdditional Context from Follow-ups:\n" + "\n\n".join(context_parts)
                            
                            # Create a simplified follow-up prompt to avoid token limits
                            # Truncate context if it's too long
                            max_context_length = 2000  # Limit context to avoid token overflow
                            truncated_answer = st.session_state['qa_answer'][:max_context_length] + "..." if len(st.session_state['qa_answer']) > max_context_length else st.session_state['qa_answer']
                            truncated_context = additional_context[:max_context_length] + "..." if len(additional_context) > max_context_length else additional_context
                            
                            # Use centralized prompt for follow-up questions
                            follow_up_prompt = get_follow_up_prompt(
                                previous_answer=truncated_answer,
                                additional_context=truncated_context,
                                follow_up_question=follow_up_question,
                                summarize=st.session_state.get('summarize_answers', False)
                            )
                            if st.session_state.get('design_outline'):
                                follow_up_prompt += "\n\nAdditionally, generate detailed experimental outline, with specific experiment procedure and outline the challenges and expected result."
                            
                            with st.spinner("Generating follow-up answer..."):
                                try:
                                    # Debug: Check if we have a valid previous answer
                                    if not st.session_state.get('qa_answer'):
                                        st.error("No previous answer found. Please ask a main question first.")
                                        return
                                    
                                    follow_up_answer = st.session_state.llm.invoke(follow_up_prompt)
                                    
                                    # Debug: Check if the answer is empty
                                    if not follow_up_answer or follow_up_answer.strip() == "":
                                        st.warning("LLM returned an empty response. Trying with a simplified prompt...")
                                        
                                        # Try with a much simpler prompt
                                        simple_prompt = f"""Based on the previous answer, please respond to this follow-up question:

Previous Answer: {truncated_answer[:1000]}...

Follow-up Question: {follow_up_question}

Please provide a comprehensive response:"""
                                        
                                        try:
                                            follow_up_answer = st.session_state.llm.invoke(simple_prompt)
                                            if not follow_up_answer or follow_up_answer.strip() == "":
                                                st.error("LLM still returned an empty response. This might indicate a model issue.")
                                                st.error(f"Debug - follow_up_question: '{follow_up_question}'")
                                                st.error(f"Debug - previous_answer length: {len(st.session_state.get('qa_answer', ''))}")
                                                st.error(f"Debug - additional_context length: {len(additional_context)}")
                                                return
                                        except Exception as e2:
                                            st.error(f"Error with simplified prompt: {e2}")
                                            return
                                    
                                    # Store the follow-up Q&A in history
                                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    
                                    # Clean up the answer to remove thinking process indicators
                                    clean_follow_up_answer = follow_up_answer
                                    thinking_indicators = [
                                        "<think>", "</think>", "Let me think about this", "I need to", 
                                        "First, let me", "Let me analyze", "I should", 
                                        "This is an interesting question", "To answer this",
                                        "Based on my understanding", "I'll help you", "Let me break this down"
                                    ]
                                    
                                    for indicator in thinking_indicators:
                                        if indicator in clean_follow_up_answer:
                                            clean_follow_up_answer = clean_follow_up_answer.replace(indicator, "")
                                    
                                    # Clean up any remaining HTML-like tags
                                    import re
                                    clean_follow_up_answer = re.sub(r'<[^>]+>', '', clean_follow_up_answer)
                                    
                                    st.session_state.follow_up_history.append((follow_up_question, clean_follow_up_answer, timestamp))
                                    # Store the current follow-up answer for display
                                    st.session_state['follow_up_answer'] = clean_follow_up_answer
                                    # Store the current follow-up question for context
                                    st.session_state['latest_follow_up_question'] = follow_up_question
                                    # Clear selected context after using it
                                    st.session_state.selected_context_answers = []
                                    st.session_state.current_answer_as_context = False
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error generating follow-up answer: {e}")
                                    # Add debug information
                                    st.error(f"Debug - follow_up_question: '{follow_up_question}'")
                                    st.error(f"Debug - follow_up_prompt length: {len(follow_up_prompt)}")
                                    st.error(f"Debug - previous_answer exists: {bool(st.session_state.get('qa_answer'))}")
                        else:
                            st.error("LLM not available for follow-up questions. Please load an LLM model first.")
                    else:
                        st.warning("Please enter a follow-up question.")
            
            if clear_followup_pressed:
                # Clear follow-up related session state
                if 'follow_up_answer' in st.session_state:
                    del st.session_state['follow_up_answer']
                # Do not delete the widget value; clear only our latest copy
                if 'latest_follow_up_question' in st.session_state:
                    del st.session_state['latest_follow_up_question']
                # Clear history and selected context
                st.session_state.follow_up_history = []
                st.session_state.selected_context_answers = []
                st.session_state.current_answer_as_context = False
                st.rerun()
            
            # Display current follow-up answer if it exists (before it gets added to history)
            if st.session_state.get('follow_up_answer') and st.session_state.get('latest_follow_up_question'):
                st.markdown(create_clean_divider(), unsafe_allow_html=True)
                
                # Get current timestamp for display
                current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Display current follow-up with timing information
                col_title, col_timing = st.columns([3, 1])
                with col_title:
                    st.markdown("**💭 Current Follow-up Answer**")
                with col_timing:
                    st.caption(f"⏱️ {current_timestamp}")
                
                # Show the question and answer
                st.markdown(f"**Q:** {st.session_state['latest_follow_up_question']}")
                
                # Clean up the answer for display
                clean_current_answer = st.session_state['follow_up_answer']
                thinking_indicators = [
                    "<think>", "</think>", "Let me think about this", "I need to", 
                    "First, let me", "Let me analyze", "I should", 
                    "This is an interesting question", "To answer this",
                    "Based on my understanding", "I'll help you", "Let me break this down"
                ]
                
                for indicator in thinking_indicators:
                    if indicator in clean_current_answer:
                        clean_current_answer = clean_current_answer.replace(indicator, "")
                
                # Clean up any remaining HTML-like tags
                import re
                clean_current_answer = re.sub(r'<[^>]+>', '', clean_current_answer)
                
                st.markdown(f"**A:** {clean_current_answer}")
                st.caption("This answer will be added to your follow-up history.")
            
            # Show follow-up history as brief, two-column cards with inline full-content toggle
            if st.session_state.get('follow_up_history'):
                items = list(reversed(list(enumerate(st.session_state.follow_up_history))))
                for row_start in range(0, len(items), 2):
                    cols = st.columns(2)
                    for col_idx, (orig_idx, (fq, fa, ts)) in enumerate(items[row_start:row_start+2]):
                        with cols[col_idx]:
                            # Header row: title + timing + context toggle
                            hcol_title, hcol_timing, hcol_toggle = st.columns([2, 2, 2])
                            with hcol_title:
                                st.markdown(f"**Follow-up {orig_idx+1}**")
                            with hcol_timing:
                                # Show timing information
                                st.caption(f"⏱️ {ts}")
                            with hcol_toggle:
                                try:
                                    use_ctx = st.toggle(
                                        "",
                                        key=f"context_toggle_{orig_idx}",
                                        value=(orig_idx in st.session_state.selected_context_answers),
                                        label_visibility="collapsed",
                                        help="Use as context for next follow-up"
                                    )
                                except Exception:
                                    use_ctx = st.checkbox(
                                        "",
                                        key=f"context_toggle_{orig_idx}",
                                        value=(orig_idx in st.session_state.selected_context_answers),
                                        label_visibility="collapsed",
                                        help="Use as context for next follow-up"
                                    )
                                if use_ctx and orig_idx not in st.session_state.selected_context_answers:
                                    st.session_state.selected_context_answers.append(orig_idx)
                                if not use_ctx and orig_idx in st.session_state.selected_context_answers:
                                    st.session_state.selected_context_answers.remove(orig_idx)

                            # Summary: Show the question instead of thinking process
                            # Clean up the answer to remove thinking process indicators
                            clean_answer = fa or ""
                            thinking_indicators = [
                                "<think>", "</think>", "Let me think about this", "I need to", 
                                "First, let me", "Let me analyze", "I should", 
                                "This is an interesting question", "To answer this",
                                "Based on my understanding", "I'll help you", "Let me break this down"
                            ]
                            
                            for indicator in thinking_indicators:
                                if indicator in clean_answer:
                                    clean_answer = clean_answer.replace(indicator, "")
                            
                            # Clean up any remaining HTML-like tags
                            import re
                            clean_answer = re.sub(r'<[^>]+>', '', clean_answer)
                            
                            # Show the question in the summary card instead of the answer
                            st.markdown(
                                f"""
                                <div class=\"clean-card\" style=\"border-left: 3px solid var(--success-color); padding: var(--spacing-md);\"> 
                                  <div style=\"color: var(--text-primary); font-weight: 500; margin-bottom: 8px;\">Q: {fq}</div>
                                  <div style=\"white-space: pre-wrap; color: var(--text-secondary); font-size: 0.9em;\">{html.escape(clean_answer[:200])}{'...' if len(clean_answer) > 200 else ''}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            # Full content directly below the card using native Streamlit
                            with st.expander("Show full content", expanded=False):
                                st.markdown(f"**Q:** {fq}")
                                st.markdown(f"**A:** {clean_answer}")
                                st.caption(f"Time: {ts}")
            
            # Export Q&A session (single card)
            st.markdown(create_clean_divider(), unsafe_allow_html=True)
            st.markdown(create_clean_card(
                title="Export Q&A Session",
                content="Download the complete Q&A session including main question, answer, and recent follow-ups as a markdown file.",
                icon="📥",
                variant="info"
            ), unsafe_allow_html=True)

            md_content = "# Q&A Session Report\n\n"
            md_content += f"**Generated on:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            if st.session_state.get('qa_answer'):
                md_content += "## Main Question & Answer\n\n"
                md_content += f"**Question:** {st.session_state.get('original_question', 'Unknown question')}\n\n"
                md_content += f"**Answer:**\n{st.session_state['qa_answer']}\n\n"
            if st.session_state.get('follow_up_history'):
                md_content += "## Follow-up Questions & Answers\n\n"
                for i, (question, answer, timestamp) in enumerate(st.session_state.follow_up_history, 1):
                    md_content += f"### Follow-up {i}\n\n"
                    md_content += f"**Question:** {question}\n\n"
                    md_content += f"**Answer:**\n{answer}\n\n"
                    md_content += f"**Timestamp:** {timestamp}\n\n"
                    md_content += "---\n\n"
            if st.session_state.get('follow_up_answer'):
                md_content += "## Latest Follow-up\n\n"
                md_content += f"**Question:** {st.session_state.get('latest_follow_up_question', 'Unknown question')}\n\n"
                md_content += f"**Answer:**\n{st.session_state['follow_up_answer']}\n\n"
                md_content += f"**Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

            filename = f"qa_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            st.download_button(
                label="📥 Download Q&A Session as Markdown",
                data=md_content,
                file_name=filename,
                mime="text/markdown",
                help="Download the Q&A session"
            )
            
            # Remove debug box after answer
        else:
            # Show placeholder when no answer exists
            st.info("💡 Enter a question and click 'Ask Question' to see the answer here.")

    with tab2:
        display_preview_section(selected_papers)
    
    # Only show tab3 (Paper Network) if LLM is available
    if st.session_state.llm is not None:
        with tab3:
            st.markdown('### 🤖 LLM-Powered Paper Grouping Table')
            group_question = st.text_input('Enter a grouping question for the table (e.g., "What type of precipitate is present in the paper?")', value='')
            if 'llm_grouped_table' not in st.session_state:
                st.session_state['llm_grouped_table'] = None
            if 'llm_grouped_table_refined' not in st.session_state:
                st.session_state['llm_grouped_table_refined'] = None
            if st.button('Group Papers by LLM', disabled=st.session_state.llm is None, help='LLM must be loaded to group papers.'):
                # Only now fetch abstracts and run LLM
                selected_paper_objs = []
                for paper in st.session_state.paper_list:
                    if paper['file_name'] in selected_papers:
                        abstract = paper.get('abstract', None)
                        if not abstract:
                            abstract, _ = get_paper_abstract_and_keywords(st.session_state.vectorstore, paper['file_name'])
                        selected_paper_objs.append({'file_name': paper['file_name'], 'abstract': abstract or ''})
                from utils.paper_network_viz import llm_grouped_network_interactive
                # Get group labels for each paper
                group_labels = []
                for paper in selected_paper_objs:
                    _, group_to_color = llm_grouped_network_interactive([paper], st.session_state.llm, group_question)
                    group_label = list(group_to_color.keys())[0] if group_to_color else 'Unknown'
                    group_labels.append(group_label)
                # Build table data (no abstract)
                table_data = []
                for paper, group_label in zip(selected_paper_objs, group_labels):
                    table_data.append({
                        'Paper Title': paper['file_name'],
                        'Group': group_label
                    })
                st.session_state['llm_grouped_table'] = table_data

                # LLM refinement step
                table_str = "\n".join([f"{row['Paper Title']} | {row['Group']}" for row in table_data])
                refine_prompt = get_llm_grouping_refinement_prompt(table_str)
                try:
                    refined_output = st.session_state.llm.invoke(refine_prompt)
                    # Try to parse the LLM's output into a table
                    import pandas as pd
                    import io
                    # Find the start of the table in the output
                    lines = [line for line in refined_output.splitlines() if '|' in line]
                    if lines:
                        # Assume the first line is header, rest are data
                        header = lines[0]
                        data_lines = lines[1:]
                        csv_str = header.replace('|', ',') + '\n' + '\n'.join([l.replace('|', ',') for l in data_lines])
                        df_refined = pd.read_csv(io.StringIO(csv_str))
                        st.session_state['llm_grouped_table_refined'] = df_refined
                    else:
                        st.session_state['llm_grouped_table_refined'] = None
                except Exception as e:
                    st.warning(f"Could not refine groups: {e}")
                    st.session_state['llm_grouped_table_refined'] = None
            # Show the initial and refined tables if available
            table_data = st.session_state.get('llm_grouped_table', None)
            if table_data:
                import pandas as pd
                df = pd.DataFrame(table_data)
                st.markdown("**Initial LLM Grouping:**")
                st.dataframe(df)
                df_refined = st.session_state.get('llm_grouped_table_refined', None)
                if df_refined is not None:
                    st.markdown("**LLM-Refined Grouping:**")
                    st.dataframe(df_refined)
            else:
                st.info('Select papers and click the button to group and view them by LLM-extracted mechanism/type/conclusion.')

            st.markdown('---')
            st.markdown('### 🗺️ RAG Pipeline Flowchart Generator')
            st.write('Select papers on the left, then generate a RAG pipeline flowchart using LLM.')
            
            # Check if graphviz is available
            if not GRAPHVIZ_AVAILABLE:
                st.warning("⚠️ **Graphviz not available**: The `graphviz` package is not installed. DOT code generation will work, but visualization will be disabled.")
                st.info("To enable visualization, install graphviz: `pip install graphviz`")
            
            dot_code = st.session_state.get('rag_dot_code', None)
            col_dot, col_graph = st.columns([1, 2])
            with col_dot:
                st.markdown('**DOT Code:**')
                if dot_code:
                    st.code(dot_code, language='dot')
                else:
                    st.info('No DOT code generated yet.')
            with col_graph:
                button_disabled = st.session_state.llm is None
                button_help = 'LLM must be loaded to generate DOT code.' if button_disabled else 'Generate a RAG pipeline flowchart using LLM.'
                if st.button('Generate RAG Flowchart (DOT) with LLM', disabled=button_disabled, help=button_help):
                    paper_titles = ', '.join([p['file_name'] for p in st.session_state.paper_list if p['file_name'] in selected_papers])
                    prompt = get_rag_flowchart_prompt(paper_titles if paper_titles else 'None')
                    try:
                        dot_code = st.session_state.llm.invoke(prompt)
                        st.session_state['rag_dot_code'] = dot_code
                    except Exception as e:
                        st.error(f"LLM failed to generate DOT code: {e}")
                # Show DOT code and render if available
                dot_code = st.session_state.get('rag_dot_code', None)
                if dot_code:
                    if GRAPHVIZ_AVAILABLE:
                        try:
                            graph = graphviz.Source(dot_code)
                            st.graphviz_chart(graph.source)
                        except Exception as e:
                            st.error(f"Failed to render DOT graph: {e}")
                    else:
                        st.info("📋 DOT code generated successfully! Install graphviz to visualize the flowchart.")
                        st.markdown("**Installation command:** `pip install graphviz`")
    
    with tab4:
        display_scholar_section()


if __name__ == "__main__":
    main()