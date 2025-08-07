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

# Import paper keywords
from utils.paper_keywords import get_paper_keywords, get_folder_keywords

# Import logo utilities
from utils.logo_display import display_logo_header, display_logo_in_sidebar

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
        'current_selected_papers': [],  # Track currently selected papers for debug display
        'summarize_answers': False,  # Toggle for summarizing answers to 3-5 sentences
        'follow_up_history': [],  # Store all follow-up Q&A pairs
        'selected_context_answers': [],  # Track which follow-up answers are selected as context
        'current_answer_as_context': False,  # Toggle for using current answer as context
        'original_question': ""  # Store original question for export
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
    if st.session_state.selected_keywords:
        st.markdown("**Selected Keywords:**")
        selected_keywords_text = " • ".join(st.session_state.selected_keywords)
        st.markdown(create_clean_card(
            title="Selected Keywords",
            content=selected_keywords_text,
            icon="✅",
            variant="success"
        ), unsafe_allow_html=True)
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
        optimized_q, keywords = optimize_question(st.session_state.llm, question)
        st.session_state.optimized_question = optimized_q
        st.session_state.suggested_keywords = keywords[:4] if keywords else []

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
        else:
            if selected_papers and len(selected_papers) > 0:
                answer = "No relevant documents found for your question in the selected papers. Try broadening your selection or rephrasing your question."
            else:
                answer = "No relevant documents found for your question."
            search_results = []
    
    # Store answer in session state for display_question_section to use
    st.session_state['qa_answer'] = answer
    
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
    
    # Suggested Follow Up Reading (right column)
    if search_results:
        # Only show research papers (not meeting notes)
        research_papers = [doc for doc in search_results if doc.metadata.get('content_type') != 'meeting_notes']
        if research_papers:
            # Build a mapping from file_name to file_path
            file_map = {p['file_name']: p['file_path'] for p in st.session_state.paper_list}
            # Use a session variable to store for right column
            st.session_state.suggested_followup = [
                {
                    'file_name': doc.metadata.get('file_name', 'Unknown'),
                    'file_path': file_map.get(doc.metadata.get('file_name', ''), None),
                    'title': doc.metadata.get('title', doc.metadata.get('file_name', 'Unknown')),
                    'abstract': doc.metadata.get('abstract') or (doc.page_content[:300] + '...' if len(doc.page_content) > 300 else doc.page_content)
                }
                for doc in research_papers
            ]
        else:
            st.session_state.suggested_followup = []
    else:
        st.session_state.suggested_followup = []


def display_preview_section(selected_papers: List[str]):
    """Display the paper preview section with card-based UI and folder buttons"""
    display_clean_section_header("Paper Preview", "Browse and explore research papers")
    
    if selected_papers:
        # Show selected papers in card format
        st.markdown("### 📚 Selected Papers")
        for paper_name in selected_papers:
            paper_info = next((p for p in st.session_state.paper_list if p['file_name'] == paper_name), None)
            if paper_info:
                display_paper_card(paper_info, is_selected=True)
    else:
        # Display all papers grouped by folders with buttons
        if st.session_state.paper_list:
            # Group papers by folder
            folder_order, folder_icons = get_folder_config()
            papers_by_folder = {}
            
            for paper in st.session_state.paper_list:
                folder = paper.get('folder', 'Unknown')
                if folder not in papers_by_folder:
                    papers_by_folder[folder] = []
                papers_by_folder[folder].append(paper)
            
            # Initialize session state for folder selection
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
            
            # Display papers for selected folder
            if st.session_state.selected_folder and st.session_state.selected_folder in papers_by_folder:
                st.markdown(f"### 📚 Papers in {st.session_state.selected_folder}")
                
                # Create a grid layout for papers (2 columns)
                papers = papers_by_folder[st.session_state.selected_folder]
                cols = st.columns(2)
                
                for i, paper in enumerate(papers):
                    with cols[i % 2]:
                        display_paper_card(paper, is_selected=False)
            
            # Show "All Papers" option
            elif st.session_state.selected_folder is None:
                st.markdown("### 📚 All Papers")
                st.info("👆 Select a research area above to view papers")
                
                # Show a few sample papers from all folders
                sample_papers = []
                for folder in folder_order:
                    if folder in papers_by_folder:
                        sample_papers.extend(papers_by_folder[folder][:2])  # Show 2 papers per folder
                
                if sample_papers:
                    cols = st.columns(2)
                    for i, paper in enumerate(sample_papers[:6]):  # Show max 6 sample papers
                        with cols[i % 2]:
                            display_paper_card(paper, is_selected=False)
        else:
            display_clean_empty_state(
                icon="📋",
                title="No Papers Available",
                description="No papers found in the system. Please check the vector store configuration.",
                action_text="Check System Status",
                action_func=None
            )


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
    
    # Apply clean theme and styling
    apply_clean_theme()
    apply_clean_tab_style()
    
    # Clean header with logo in left corner
    display_logo_header()
    
    # Load initial data
    load_initial_data()
    
    # Check if vector store loaded successfully
    if not st.session_state.vectorstore:
        st.error("❌ Failed to load vector store. Please check the configuration.")
        st.info("💡 Check the sidebar under 'System Status' for detailed error messages and troubleshooting steps.")
        # Don't return - let the app continue so user can see error details in sidebar
    
    # Display sidebar and get settings
    selected_papers, llm_model, search_type, num_results = display_sidebar()
    
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
        # Left column: Question input and optimization
        col_ask, col_suggest = st.columns([6, 4])
        with col_ask:
            st.markdown(create_clean_card(
                title="Ask Questions",
                content="Use AI to ask questions about your research papers and get intelligent answers.",
                icon="🤖",
                variant="default"
            ), unsafe_allow_html=True)
            
            # Toggles row
            col_gap, col_summary = st.columns([1, 1])
            with col_gap:
                gap_toggle = st.checkbox("Identify Research Gaps", value=False, key="gap_toggle", disabled=st.session_state.llm is None, help="LLM required for research gap analysis")
            with col_summary:
                st.session_state.summarize_answers = st.checkbox("Summarize Answers", value=st.session_state.summarize_answers, key="summarize_toggle", help="Summarize answers to 3-5 sentences")
            
            display_question_section(llm_model, selected_papers, search_type, num_results, gap_toggle=gap_toggle)
        
        # Right column: Follow-up reading (with max height constraint)
        with col_suggest:
            # Create a container with max height constraint
            with st.container():
                # At the very top of the right column: toggle, year, and results
                import datetime
                current_year = datetime.datetime.now().year
                years = ["All"] + [str(y) for y in range(current_year, 1999, -1)]
                col_year, col_toggle = st.columns([1, 1])
                with col_year:
                    scholar_year = st.selectbox("Year limit", years, index=0, key="year_limit", label_visibility="visible")
                with col_toggle:
                    scholar_toggle = st.checkbox("Suggest follow-up reading", value=False)
                
                # Follow-up reading section with max height
                scholar_search_and_display()
        
  
        
        # Display answer from session state (if exists)
        if st.session_state.get('qa_answer'):
            if gap_toggle:
                st.markdown(create_clean_card(
                    title="Research Gaps Identified",
                    content=st.session_state['qa_answer'],
                    icon="🔍",
                    variant="warning"
                ), unsafe_allow_html=True)
                # Show summarize indicator if enabled
                if st.session_state.get('summarize_answers', False):
                    st.caption("📝 **Summarized to 3-5 sentences**")
            else:
                st.markdown(create_clean_card(
                    title="AI-Generated Answer",
                    content=st.session_state['qa_answer'],
                    icon="🤖",
                    variant="default"
                ), unsafe_allow_html=True)
                
                if st.session_state.llm:
                    st.caption(f"Generated using: {llm_model}")
                else:
                    st.caption("LLM not available - showing search results summary")
                
                # Show summarize indicator if enabled
                if st.session_state.get('summarize_answers', False):
                    st.caption("📝 **Summarized to 3-5 sentences**")
            
            # Follow-up question section
            st.markdown(create_clean_divider(), unsafe_allow_html=True)
            st.markdown(create_clean_card(
                title="Follow-up Question",
                content="Ask a follow-up question based on the previous answer. You can select previous follow-up answers as additional context.",
                icon="💭",
                variant="info"
            ), unsafe_allow_html=True)

            
            # Display previous follow-up answers as selectable context cards
            if st.session_state.get('follow_up_history'):
                st.markdown("**📚 Previous Follow-up Answers (Select as Context):**")
                context_container = st.container()
                with context_container:
                    # Create a grid layout for context cards
                    cols = st.columns(2)
                    for i, (question, answer, timestamp) in enumerate(st.session_state.follow_up_history):
                        with cols[i % 2]:
                            with st.expander(f"Q{i+1}: {question[:50]}...", expanded=False):
                                st.markdown(f"**Question:** {question}")
                                st.markdown(f"**Answer:** {answer[:200]}...")
                                st.markdown(f"**Time:** {timestamp}")
                                
                                # Row for actions
                                col_check, col_delete = st.columns([3, 1])
                                with col_check:
                                    # Checkbox to select this answer as context
                                    if st.checkbox(f"Use as context", key=f"context_{i}"):
                                        if i not in st.session_state.selected_context_answers:
                                            st.session_state.selected_context_answers.append(i)
                                    else:
                                        if i in st.session_state.selected_context_answers:
                                            st.session_state.selected_context_answers.remove(i)
                                
                                with col_delete:
                                    # Button to remove this specific follow-up
                                    if st.button("🗑️", key=f"delete_{i}", help="Remove this follow-up"):
                                        st.session_state.follow_up_history.pop(i)
                                        # Adjust selected context indices
                                        st.session_state.selected_context_answers = [
                                            idx if idx < i else idx - 1 
                                            for idx in st.session_state.selected_context_answers 
                                            if idx != i
                                        ]
                                        st.rerun()
                
                # Show selected context summary
                context_summary = []
                
                # Add current answer if selected
                if st.session_state.get('current_answer_as_context', False):
                    context_summary.append("Current Answer")
                
                # Add selected historical answers
                for idx in st.session_state.selected_context_answers:
                    if idx < len(st.session_state.follow_up_history):
                        question, answer, _ = st.session_state.follow_up_history[idx]
                        context_summary.append(f"Q{idx+1}: {question[:30]}...")
                
                if context_summary:
                    st.caption(f"✅ Selected context: {', '.join(context_summary)}")
            
            follow_up_question = st.text_area(
                "Follow-up Question:",
                placeholder="Ask a follow-up question based on the previous answer...",
                height=80,
                key="follow_up_question"
            )
            
            col_followup1, col_followup2 = st.columns([1, 1])
            with col_followup1:
                if st.button("🔍 Ask Follow-up", type="secondary", use_container_width=True):
                    if follow_up_question.strip():
                        if st.session_state.llm:
                            # Build context from selected follow-up answers and current answer
                            additional_context = ""
                            context_parts = []
                            
                            # Include current answer as context if toggle is enabled
                            if st.session_state.get('current_answer_as_context', False) and st.session_state.get('follow_up_answer'):
                                context_parts.append(f"Current Follow-up Q&A:\nQ: {st.session_state.get('follow_up_question', 'Unknown')}\nA: {st.session_state['follow_up_answer']}")
                            
                            # Include selected historical follow-up answers
                            if st.session_state.selected_context_answers:
                                for idx in st.session_state.selected_context_answers:
                                    if idx < len(st.session_state.follow_up_history):
                                        question, answer, _ = st.session_state.follow_up_history[idx]
                                        context_parts.append(f"Previous Q&A {idx+1}:\nQ: {question}\nA: {answer}")
                            
                            if context_parts:
                                additional_context = "\n\nAdditional Context from Follow-ups:\n" + "\n\n".join(context_parts)
                            
                            # Use centralized prompt for follow-up questions
                            follow_up_prompt = get_follow_up_prompt(
                                previous_answer=st.session_state['qa_answer'],
                                additional_context=additional_context,
                                follow_up_question=follow_up_question,
                                summarize=st.session_state.get('summarize_answers', False)
                            )
                            
                            with st.spinner("Generating follow-up answer..."):
                                try:
                                    follow_up_answer = st.session_state.llm.invoke(follow_up_prompt)
                                    
                                    # Store the follow-up Q&A in history
                                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    st.session_state.follow_up_history.append((follow_up_question, follow_up_answer, timestamp))
                                    
                                    # Store the current follow-up answer for display
                                    st.session_state['follow_up_answer'] = follow_up_answer
                                    st.session_state['follow_up_question'] = follow_up_question
                                    
                                    # Clear selected context after using it
                                    st.session_state.selected_context_answers = []
                                    st.session_state.current_answer_as_context = False
                                    
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error generating follow-up answer: {e}")
                        else:
                            st.error("LLM not available for follow-up questions. Please load an LLM model first.")
                    else:
                        st.warning("Please enter a follow-up question.")
            
            with col_followup2:
                if st.button("🗑️ Clear Follow-up", type="secondary", use_container_width=True):
                    # Clear follow-up related session state
                    if 'follow_up_answer' in st.session_state:
                        del st.session_state['follow_up_answer']
                    if 'follow_up_question' in st.session_state:
                        del st.session_state['follow_up_question']
                    # Clear history and selected context
                    st.session_state.follow_up_history = []
                    st.session_state.selected_context_answers = []
                    st.session_state.current_answer_as_context = False
                    st.rerun()
            
            # Display current follow-up answer if exists
            if st.session_state.get('follow_up_answer'):
                st.markdown(create_clean_divider(), unsafe_allow_html=True)
                st.markdown(create_clean_card(
                    title="Latest Follow-up Answer",
                    content=st.session_state['follow_up_answer'],
                    icon="💡",
                    variant="success"
                ), unsafe_allow_html=True)
                st.caption(f"Follow-up to: {st.session_state.get('follow_up_question', 'Unknown question')}")
                
                # Show summarize indicator if enabled
                if st.session_state.get('summarize_answers', False):
                    st.caption("📝 **Summarized to 3-5 sentences**")
                
                # Display current follow-up as a selectable context card
                with st.expander("📋 Current Follow-up Answer", expanded=True):
                    st.markdown(f"**Question:** {st.session_state.get('follow_up_question', 'Unknown question')}")
                    st.markdown(f"**Answer:** {st.session_state['follow_up_answer']}")
                    st.markdown(f"**Time:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Toggle to select this answer as context for next question
                    st.markdown("---")
                    if st.checkbox("✅ Use this answer as context for next follow-up question", key="current_context_toggle"):
                        st.session_state['current_answer_as_context'] = True
                        st.caption("✅ This answer will be included as context for your next follow-up question")
                    else:
                        st.session_state['current_answer_as_context'] = False
                        st.caption("ℹ️ This answer will not be included as context")
            
            # Export Q&A session to markdown
            st.markdown(create_clean_divider(), unsafe_allow_html=True)
            st.markdown(create_clean_card(
                title="Export Q&A Session",
                content="Download the complete Q&A session including main question, answer, and all follow-ups as a markdown file.",
                icon="📥",
                variant="info"
            ), unsafe_allow_html=True)
            
            # Generate markdown content
            if st.session_state.get('qa_answer') or st.session_state.get('follow_up_history'):
                # Create markdown content
                md_content = "# Q&A Session Report\n\n"
                md_content += f"**Generated on:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                # Add main question and answer
                if st.session_state.get('qa_answer'):
                    md_content += "## Main Question & Answer\n\n"
                    md_content += f"**Question:** {st.session_state.get('original_question', 'Unknown question')}\n\n"
                    md_content += f"**Answer:**\n{st.session_state['qa_answer']}\n\n"
                
                # Add follow-up history
                if st.session_state.get('follow_up_history'):
                    md_content += "## Follow-up Questions & Answers\n\n"
                    for i, (question, answer, timestamp) in enumerate(st.session_state.follow_up_history, 1):
                        md_content += f"### Follow-up {i}\n\n"
                        md_content += f"**Question:** {question}\n\n"
                        md_content += f"**Answer:**\n{answer}\n\n"
                        md_content += f"**Timestamp:** {timestamp}\n\n"
                        md_content += "---\n\n"
                
                # Add current follow-up if exists
                if st.session_state.get('follow_up_answer'):
                    md_content += "## Latest Follow-up\n\n"
                    md_content += f"**Question:** {st.session_state.get('follow_up_question', 'Unknown question')}\n\n"
                    md_content += f"**Answer:**\n{st.session_state['follow_up_answer']}\n\n"
                    md_content += f"**Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                # Create download button
                filename = f"qa_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                st.download_button(
                    label="📥 Download Q&A Session as Markdown",
                    data=md_content,
                    file_name=filename,
                    mime="text/markdown",
                    help="Download the complete Q&A session including main question, answer, and all follow-ups"
                )
                
                # Show preview
                with st.expander("👀 Preview Markdown Content", expanded=False):
                    st.code(md_content, language="markdown")
            else:
                st.info("No Q&A content available to export. Ask a question first!")
            
            # Debug information in collapsible section
            with st.expander("🔍 Debug Information", expanded=False):
                st.caption(f"🔍 Debug: Answer stored in session state. Length: {len(st.session_state['qa_answer']) if st.session_state['qa_answer'] else 0}")
                st.caption(f"🔍 Debug: Answer preview: {st.session_state['qa_answer'][:100] + '...' if st.session_state['qa_answer'] and len(st.session_state['qa_answer']) > 100 else st.session_state['qa_answer']}")
                st.caption(f"🔍 Debug: gap_toggle: {gap_toggle}")
                st.caption(f"🔍 Debug: optimized_question exists: {'optimized_question' in st.session_state}")
                if 'optimized_question' in st.session_state:
                    st.caption(f"🔍 Debug: optimized_question value: {st.session_state.optimized_question}")
                st.caption(f"🔍 Debug: selected_papers count: {len(selected_papers) if selected_papers else 0}")
                if selected_papers:
                    st.caption(f"🔍 Debug: selected_papers: {selected_papers}")
                # Follow-up debug info
                if st.session_state.get('follow_up_answer'):
                    st.caption(f"🔍 Debug: Follow-up answer length: {len(st.session_state['follow_up_answer'])}")
                    st.caption(f"🔍 Debug: Follow-up question: {st.session_state.get('follow_up_question', 'Unknown')}")
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