"""
Streamlit-React Bridge
Enables React components to communicate with Streamlit backend
"""

import streamlit as st
import json
import requests
from typing import Dict, Any, List
import os

class StreamlitReactBridge:
    """Bridge between React frontend and Streamlit backend"""
    
    def __init__(self):
        self.streamlit_port = 8501  # Default Streamlit port
        self.api_endpoints = {
            'search': '/api/search',
            'papers': '/api/papers',
            'correlations': '/api/correlations',
            'network': '/api/network'
        }
    
    def create_api_endpoints(self):
        """Create API endpoints for React frontend"""
        
        # Search endpoint
        @st.experimental_api
        def search_papers_api(question: str, selected_papers: List[str] = None, 
                            search_type: str = "both", k: int = 5):
            """API endpoint for paper search"""
            try:
                # Use existing search logic from your app
                from app import search_papers
                answer, results = search_papers(question, selected_papers, search_type, k)
                
                # Format results for React
                formatted_results = []
                for doc in results:
                    formatted_results.append({
                        'file_name': doc.metadata.get('file_name', 'Unknown'),
                        'section': doc.metadata.get('section', 'Unknown'),
                        'content': doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        'document_type': doc.metadata.get('document_type', 'unknown'),
                        'figure_count': doc.metadata.get('figure_count', 0)
                    })
                
                return {
                    'answer': answer,
                    'results': formatted_results,
                    'total_results': len(results)
                }
            except Exception as e:
                return {'error': str(e)}
        
        # Papers endpoint
        @st.experimental_api
        def get_papers_api():
            """API endpoint for getting paper list"""
            try:
                from app import load_paper_list
                paper_list, _ = load_paper_list()
                
                # Group papers by folder
                papers_by_folder = {}
                for paper in paper_list:
                    folder = paper['folder']
                    if folder not in papers_by_folder:
                        papers_by_folder[folder] = []
                    papers_by_folder[folder].append({
                        'file_name': paper['file_name'],
                        'figure_count': paper['figure_count'],
                        'has_figures': paper['has_figures'],
                        'folder': paper['folder']
                    })
                
                return {
                    'papers': papers_by_folder,
                    'total_papers': len(paper_list)
                }
            except Exception as e:
                return {'error': str(e)}
        
        # Correlations endpoint
        @st.experimental_api
        def get_correlations_api(topic: str = None):
            """API endpoint for getting paper correlations"""
            try:
                if 'correlation_manager' in st.session_state:
                    if topic:
                        df = st.session_state.correlation_manager.export_to_dataframe(topic)
                    else:
                        df = st.session_state.correlation_manager.export_to_dataframe()
                    
                    return {
                        'correlations': df.to_dict('records'),
                        'topics': list(st.session_state.correlation_manager.topics.keys())
                    }
                else:
                    return {'error': 'Correlation manager not initialized'}
            except Exception as e:
                return {'error': str(e)}
        
        # Network endpoint
        @st.experimental_api
        def get_network_data_api(topic: str = None):
            """API endpoint for network visualization data"""
            try:
                if 'correlation_manager' in st.session_state:
                    if topic:
                        G = st.session_state.correlation_manager.create_network_graph(topic)
                    else:
                        G = st.session_state.correlation_manager.create_network_graph()
                    
                    # Convert networkx graph to JSON-serializable format
                    nodes = [{'id': node, 'connections': len(list(G.neighbors(node)))} 
                            for node in G.nodes()]
                    edges = [{'source': edge[0], 'target': edge[1], 'data': edge[2]} 
                            for edge in G.edges(data=True)]
                    
                    return {
                        'nodes': nodes,
                        'edges': edges,
                        'total_nodes': G.number_of_nodes(),
                        'total_edges': G.number_of_edges()
                    }
                else:
                    return {'error': 'Correlation manager not initialized'}
            except Exception as e:
                return {'error': str(e)}
    
    def embed_react_component(self, component_name: str, props: Dict[str, Any] = None):
        """Embed a React component in Streamlit"""
        
        # Create HTML wrapper for React component
        html_code = f"""
        <div id="react-{component_name}" data-props='{json.dumps(props or {})}'>
            <div class="react-loading">Loading {component_name}...</div>
        </div>
        <script>
            // React component will be loaded here
            // This would typically load from your React build
            console.log('React component {component_name} should load here');
        </script>
        """
        
        st.components.v1.html(html_code, height=400)
    
    def create_react_wrapper(self):
        """Create a wrapper that loads React components"""
        
        # Add React and dependencies
        st.markdown("""
        <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
        <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
        <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
        """, unsafe_allow_html=True)
        
        # Load your React components
        react_components = """
        <script type="text/babel">
            // Your React components would go here
            const { useState, useEffect } = React;
            
            function MaterialResearchDashboard() {
                const [activeTab, setActiveTab] = useState('search');
                const [papers, setPapers] = useState([]);
                const [searchResults, setSearchResults] = useState([]);
                
                useEffect(() => {
                    // Load papers from Streamlit API
                    fetch('/api/papers')
                        .then(response => response.json())
                        .then(data => setPapers(data.papers || []))
                        .catch(error => console.error('Error loading papers:', error));
                }, []);
                
                const handleSearch = async (question) => {
                    try {
                        const response = await fetch('/api/search', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ question })
                        });
                        const data = await response.json();
                        setSearchResults(data.results || []);
                    } catch (error) {
                        console.error('Error searching:', error);
                    }
                };
                
                return (
                    <div className="material-research-dashboard">
                        <h1>Material Research RAG</h1>
                        {/* Your React components here */}
                    </div>
                );
            }
            
            // Render the component
            ReactDOM.render(<MaterialResearchDashboard />, document.getElementById('react-root'));
        </script>
        """
        
        st.components.v1.html(react_components, height=600)

# Usage example
def integrate_react_with_streamlit():
    """Example of how to integrate React with your existing Streamlit app"""
    
    st.title("🔬 Material Research RAG - React Integration")
    
    # Create bridge
    bridge = StreamlitReactBridge()
    
    # Create API endpoints
    bridge.create_api_endpoints()
    
    # Add React component
    tab1, tab2 = st.tabs(["Streamlit UI", "React UI"])
    
    with tab1:
        st.write("Your existing Streamlit interface goes here")
        # Your existing app logic
    
    with tab2:
        st.write("React-based interface")
        bridge.create_react_wrapper()

if __name__ == "__main__":
    integrate_react_with_streamlit()
