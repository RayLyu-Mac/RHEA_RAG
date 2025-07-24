"""
Modular Paper Search & QA System
A Streamlit application for searching and querying research papers about 
Refractory High-Entropy Alloys (RHEA) using vector embeddings and LLM.
"""

import streamlit as st
import os
import sys
from typing import List, Optional

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
    display_notes_section
)

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
        'view_paper_pdf': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def load_initial_data():
    """Load initial data if not already loaded"""
    # Load paper list
    if not st.session_state.paper_list:
        st.session_state.paper_list, st.session_state.tracker_df = load_paper_list()
    
    # Load available models
    if not st.session_state.available_models or st.session_state.available_models == ["qwen3:14b", "gemma3:4b"]:
        st.session_state.available_models = get_available_ollama_models()
    
    # Load vector store
    if st.session_state.vectorstore is None:
        with st.spinner("Loading vector store..."):
            st.session_state.vectorstore = load_vectorstore()


def display_sidebar() -> tuple:
    """Display sidebar and return selected papers and settings"""
    # Initialize default values
    selected_papers = []
    llm_model = st.session_state.available_models[0] if st.session_state.available_models else "qwen3:14b"
    search_type = "both"
    num_results = 5
    
    with st.sidebar:
        # Theme toggle
        display_theme_toggle()
        st.divider()
        
        # Create tabs for main sections
        tab1, tab2 = st.tabs(["📚 Papers & Search", "📝 Meeting Notes"])
        
        with tab1:
            # Paper Selection Section (now collapsible)
            with st.expander("📚 Paper Selection", expanded=True):
                folder_order, folder_icons = get_folder_config()
                selected_papers = display_paper_selection(st.session_state.paper_list, folder_order, folder_icons)
            
            st.divider()
            
            # Settings Section
            with st.expander("⚙️ Settings", expanded=True):
                # LLM model selection
                llm_model = st.selectbox(
                    "Select LLM Model:",
                    st.session_state.available_models,
                    help="Available Ollama models on your system"
                )
                
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
            
            st.divider()
            
            # Upload new paper section
            with st.expander("📤 Upload New Paper", expanded=False):
                uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
                if uploaded_file is not None:
                    if st.button("Process Paper"):
                        st.info("Paper processing feature coming soon!")
            
            st.divider()
            
            # Vector Store Status
            with st.expander("📊 Vector Store Status", expanded=False):
                if st.session_state.vectorstore:
                    st.success("✅ Vector store loaded")
                    
                    # Display vector store info
                    persist_directory = "../VectorSpace/paper_vector_db_nomic-embed-text_latest_parent_child"
                    vector_store_name = os.path.basename(persist_directory)
                    st.info(f"📂 Vector store: {vector_store_name}")
                    
                    # Display paper statistics
                    if st.session_state.tracker_df is not None:
                        stats = get_paper_stats(st.session_state.paper_list)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Papers", stats['total_papers'])
                            st.metric("Vectorized", len(st.session_state.tracker_df[st.session_state.tracker_df['vectorized'] == True]))
                        with col2:
                            st.metric("Figures", stats['total_figures'])
                else:
                    st.error("❌ Failed to load vector store")
        
        with tab2:
            # Meeting Notes Section
            display_notes_section()
    
    return selected_papers, llm_model, search_type, num_results


def display_question_section(llm_model: str, selected_papers: List[str], search_type: str, num_results: int):
    """Display the question input and optimization section"""
    # Question input card
    st.markdown(create_glass_card("💬 Ask Questions"), unsafe_allow_html=True)
    
    # Question input
    question = st.text_area(
        "Your Question:",
        placeholder="Ask about precipitation strengthening, microstructure, mechanical properties, etc.",
        height=100
    )
    
    # Question optimization section
    col_opt1, col_opt2 = st.columns([1, 1])
    
    with col_opt1:
        if st.button("🧠 Optimize Question", help="Let AI optimize your question for better search results"):
            if question.strip():
                # Load LLM if not already loaded
                if st.session_state.llm is None:
                    with st.spinner("Loading LLM for optimization..."):
                        st.session_state.llm = load_llm(st.session_state.available_models[0])
                
                if st.session_state.llm:
                    with st.spinner("Optimizing question..."):
                        optimized_q, keywords = optimize_question(st.session_state.llm, question)
                        st.session_state.optimized_question = optimized_q
                        st.session_state.suggested_keywords = keywords
                else:
                    st.error("Failed to load LLM for optimization")
            else:
                st.warning("Please enter a question first")
    
    with col_opt2:
        if st.button("🔑 Show Keywords", help="Show suggested keywords for better search"):
            st.session_state.suggested_keywords = get_suggested_keywords()
    
    # Display optimized question if available
    if st.session_state.optimized_question:
        st.markdown(create_optimize_card("🎯 Optimized Question"), unsafe_allow_html=True)
        st.markdown(
            create_content_card(st.session_state.optimized_question, "margin: 0.25rem 0 0.5rem 0; padding: 0.5rem 0.75rem;"),
            unsafe_allow_html=True
        )
        
        # Ask optimized question button
        if st.button("🔍 Ask Question (Optimized)", key="ask_optimized_q", type="primary"):
            handle_question_submission(st.session_state.optimized_question, llm_model, selected_papers, search_type, num_results)
    
    # Keyword selection section
    if st.session_state.suggested_keywords:
        st.markdown(create_glass_card("🏷️ Select Keywords"), unsafe_allow_html=True)
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
    
    # Show selected keywords
    if st.session_state.selected_keywords:
        st.markdown("**Selected Keywords:**")
        selected_keywords_text = " • ".join(st.session_state.selected_keywords)
        st.markdown(
            create_content_card(selected_keywords_text, "background: rgba(0, 0, 0, 0.1); margin: 0.25rem 0 0.5rem 0; padding: 0.5rem 0.75rem;"),
            unsafe_allow_html=True
        )
    
    # Ask button - only if not showing optimized question
    if not st.session_state.optimized_question:
        st.markdown("---")
        if st.button("🔍 Ask Question", type="primary", use_container_width=True):
            if question.strip():
                handle_question_submission(question, llm_model, selected_papers, search_type, num_results)
            else:
                st.warning("Please enter a question")


def handle_question_submission(question: str, llm_model: str, selected_papers: List[str], search_type: str, num_results: int):
    """Handle question submission and display results"""
    # Load LLM if not already loaded or if model changed
    if st.session_state.llm is None or st.session_state.get('current_model') != llm_model:
        with st.spinner(f"Loading {llm_model}..."):
            st.session_state.llm = load_llm(llm_model)
            st.session_state.current_model = llm_model
    
    if st.session_state.llm:
        # Enhance question with selected keywords
        enhanced_question = question
        if st.session_state.selected_keywords:
            keywords_text = " ".join(st.session_state.selected_keywords)
            enhanced_question = f"{question} {keywords_text}"
        
        with st.spinner("Searching and generating answer..."):
            # Search papers
            search_results, success = search_papers(
                st.session_state.vectorstore,
                enhanced_question, 
                selected_papers if selected_papers else None,
                search_type,
                num_results
            )
            
            if success and search_results:
                # Generate answer
                answer = generate_answer(st.session_state.llm, question, search_results)
            else:
                if selected_papers and len(selected_papers) > 0:
                    answer = "No relevant documents found for your question in the selected papers. Try broadening your selection or rephrasing your question."
                else:
                    answer = "No relevant documents found for your question."
                search_results = []
        
        # Display answer card
        st.markdown(create_glass_card("📝 Answer"), unsafe_allow_html=True)
        st.caption(f"Generated using: {llm_model}")
        st.markdown(
            create_content_card(answer, "margin: 0.25rem 0 0.5rem 0; padding: 0.5rem 0.75rem; background: rgba(0,0,0,0.1);"),
            unsafe_allow_html=True
        )
        
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
    else:
        st.error("Failed to load LLM model")


def display_preview_section(selected_papers: List[str]):
    """Display the paper preview section"""
    st.header("🖼️ Paper Preview")
    
    if selected_papers:
        # Show detailed preview for selected papers
        for paper_name in selected_papers:
            paper_info = next((p for p in st.session_state.paper_list if p['file_name'] == paper_name), None)
            if paper_info:
                with st.expander(f"📄 {paper_name.replace('.pdf', '')}", expanded=True):
                    
                    # Get paper abstract and metadata
                    abstract_content, keywords = get_paper_abstract_and_keywords(st.session_state.vectorstore, paper_name)
                    
                    # Display abstract
                    if abstract_content:
                        st.markdown("**📝 Abstract:**")
                        st.markdown(
                            create_content_card(
                                abstract_content[:500] + ("..." if len(abstract_content) > 500 else ""),
                                "font-size: 0.9em; max-height: 200px; overflow-y: auto;"
                            ),
                            unsafe_allow_html=True
                        )
                    
                    # Display keywords
                    if keywords:
                        st.markdown("**🔑 Keywords:**")
                        st.markdown(
                            create_content_card(keywords, "font-size: 0.85em; background: rgba(0, 0, 0, 0.1); color: #000000;"),
                            unsafe_allow_html=True
                        )
                    
                    # Display figures
                    figures = get_paper_figures(paper_name)
                    if figures:
                        st.markdown("**🖼️ Figures:**")
                        for fig_path in figures[:3]:  # Show max 3 figures per paper
                            display_image_safely(fig_path)
                    else:
                        st.info("No figures available for this paper")
                    
                    # Paper stats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Figures", paper_info['figure_count'])
                    with col2:
                        st.metric("Folder", paper_info['folder'])
    
    else:
        st.info("📋 Select papers or ask questions to see previews here")
        
        # Show sample instruction
        st.markdown("""
        **How to use:**
        1. Select papers from the sidebar, or
        2. Ask a question to see relevant papers
        3. View abstracts, keywords, and figures here
        """)


def display_network_section(selected_papers: List[str]):
    """Display the paper network visualization section"""
    st.header("🕸️ Paper Network")
    
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
            
            # Use the paper network visualization function
            from utils.paper_network_viz import plot_paper_network_interactive
            fig = plot_paper_network_interactive(selected_metadata, demo_matrix)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No connections found with current threshold")
        else:
            st.warning("No valid paper metadata found")
    else:
        st.info("📋 Select papers from the sidebar to visualize their network relationships")


def display_scholar_section():
    """Display the Google Scholar scraper section"""
    st.header("🌐 Scholar Abstract Scraper")
    st.info("Enter a paper title or query to fetch the abstract using the scholarly package (Google Scholar API wrapper).\n\n⚠️ This is for research/prototyping. For production, consider SerpAPI.")
    
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


def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Paper Search & QA System</h1>', unsafe_allow_html=True)
    
    # Apply theme-based CSS
    apply_theme_css(st.session_state.dark_theme)
    
    # Load initial data
    load_initial_data()
    
    # Check if vector store loaded successfully
    if not st.session_state.vectorstore:
        st.error("❌ Failed to load vector store. Please check the configuration.")
        return
    
    # Display sidebar and get settings
    selected_papers, llm_model, search_type, num_results = display_sidebar()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Ask Question", "🖼️ Paper Preview", "🕸️ Paper Network", "🌐 Scholar Abstract Scraper"])
    
    with tab1:
        # Scholar follow-up controls
        import datetime
        current_year = datetime.datetime.now().year
        years = ["All"] + [str(y) for y in range(current_year, 1999, -1)]
        scholar_year = st.selectbox("Year limit for follow-up reading (optional):", years, index=0)
        scholar_toggle = st.checkbox("Suggest follow-up reading from Google Scholar", value=False)
        
        col_llm, col_scholar = st.columns([2, 1])
        
        with col_llm:
            display_question_section(llm_model, selected_papers, search_type, num_results)
        
        with col_scholar:
            # Scholar follow-up logic (after LLM answer)
            if (st.session_state.get('optimized_question') or st.session_state.get('qa_answer') or st.session_state.get('optimized_question', '').strip() or st.session_state.get('qa_question', '').strip()) and scholar_toggle:
                # Try to import scholarly
                try:
                    from scholarly import scholarly
                    SCHOLARLY_AVAILABLE = True
                except ImportError:
                    SCHOLARLY_AVAILABLE = False
                
                if not SCHOLARLY_AVAILABLE:
                    st.error("The 'scholarly' package is not installed. Please install it with 'pip install scholarly'.")
                else:
                    # Use the last asked question as the query
                    query = st.session_state.get('optimized_question') or st.session_state.get('qa_question') or ''
                    if query.strip():
                        st.markdown("### 📚 Scholar Follow-up")
                        try:
                            search_iter = scholarly.search_pubs(query)
                            filtered_result = None
                            for result in search_iter:
                                bib = result.get('bib', {})
                                year = str(bib.get('pub_year', bib.get('year', '')))
                                if scholar_year == "All" or year == scholar_year:
                                    filtered_result = result
                                    break
                            
                            if not filtered_result:
                                st.warning(f"No results found for year {scholar_year}." if scholar_year != "All" else "No results found.")
                            else:
                                bib = filtered_result.get('bib', {})
                                title = bib.get('title', '(No title found)')
                                authors = bib.get('author', '(No authors found)')
                                year = bib.get('pub_year', bib.get('year', ''))
                                venue = bib.get('venue', bib.get('journal', ''))
                                abstract = bib.get('abstract', '(No abstract found)')
                                url = bib.get('url', '')
                                num_citations = filtered_result.get('num_citations', None)
                                
                                st.markdown(f"""
**<span style='font-size:1.1em'>{title}</span>**

**Authors:** {authors}

**Year:** {year if year else 'N/A'}

**Venue:** {venue if venue else 'N/A'}

**Abstract:**
> {abstract}

{f'**URL:** [{url}]({url})' if url else ''}

{f'**Citations:** {num_citations}' if num_citations is not None else ''}
""", unsafe_allow_html=True)
                                
                                with st.expander("Show raw result (debug)"):
                                    st.write(filtered_result)
                                    
                        except Exception as e:
                            st.error(f"Error during scholarly search: {e}")
    
    with tab2:
        display_preview_section(selected_papers)
    
    with tab3:
        display_network_section(selected_papers)
    
    with tab4:
        display_scholar_section()


if __name__ == "__main__":
    main() 