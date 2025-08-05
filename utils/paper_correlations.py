"""
Paper Correlation Management System

This module provides functionality to store, manage, and visualize correlations between papers
in different research areas. It supports hierarchical organization of research topics and
their relationships.
"""

import json
import os
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path


@dataclass
class PaperCorrelation:
    """Represents a correlation between papers or concepts."""
    source: str
    target: str
    relationship_type: str
    description: str
    strength: float = 1.0  # 0.0 to 1.0
    evidence: Optional[str] = None
    date_added: Optional[str] = None
    
    def __post_init__(self):
        if self.date_added is None:
            self.date_added = datetime.now().isoformat()


@dataclass
class ResearchTopic:
    """Represents a research topic with its papers and correlations."""
    name: str
    description: str
    papers: List[str]
    correlations: List[PaperCorrelation]
    parent_topic: Optional[str] = None
    sub_topics: List[str] = None
    
    def __post_init__(self):
        if self.sub_topics is None:
            self.sub_topics = []


class PaperCorrelationManager:
    """Manages paper correlations across different research areas."""
    
    def __init__(self, data_file: str = "paper_correlations.json"):
        self.data_file = data_file
        self.topics: Dict[str, ResearchTopic] = {}
        self.load_data()
    
    def load_data(self):
        """Load correlation data from JSON file."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.topics = {
                        name: ResearchTopic(**topic_data) 
                        for name, topic_data in data.get('topics', {}).items()
                    }
            except Exception as e:
                st.error(f"Error loading correlation data: {e}")
                self.topics = {}
        else:
            self.topics = {}
    
    def save_data(self):
        """Save correlation data to JSON file."""
        try:
            data = {
                'topics': {
                    name: asdict(topic) 
                    for name, topic in self.topics.items()
                },
                'last_updated': datetime.now().isoformat()
            }
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Error saving correlation data: {e}")
    
    def add_topic(self, name: str, description: str, parent_topic: Optional[str] = None):
        """Add a new research topic."""
        if name not in self.topics:
            self.topics[name] = ResearchTopic(
                name=name,
                description=description,
                papers=[],
                correlations=[],
                parent_topic=parent_topic
            )
            if parent_topic and parent_topic in self.topics:
                self.topics[parent_topic].sub_topics.append(name)
            self.save_data()
            return True
        return False
    
    def add_paper_to_topic(self, topic_name: str, paper_name: str):
        """Add a paper to a specific topic."""
        if topic_name in self.topics:
            if paper_name not in self.topics[topic_name].papers:
                self.topics[topic_name].papers.append(paper_name)
                self.save_data()
                return True
        return False
    
    def add_correlation(self, topic_name: str, correlation: PaperCorrelation):
        """Add a correlation to a specific topic."""
        if topic_name in self.topics:
            self.topics[topic_name].correlations.append(correlation)
            self.save_data()
            return True
        return False
    
    def get_topic_papers(self, topic_name: str) -> List[str]:
        """Get all papers in a topic."""
        if topic_name in self.topics:
            return self.topics[topic_name].papers
        return []
    
    def get_topic_correlations(self, topic_name: str) -> List[PaperCorrelation]:
        """Get all correlations in a topic."""
        if topic_name in self.topics:
            return self.topics[topic_name].correlations
        return []
    
    def get_all_papers(self) -> Set[str]:
        """Get all papers across all topics."""
        papers = set()
        for topic in self.topics.values():
            papers.update(topic.papers)
        return papers
    
    def search_correlations(self, paper_name: str) -> List[PaperCorrelation]:
        """Search for correlations involving a specific paper."""
        correlations = []
        for topic in self.topics.values():
            for corr in topic.correlations:
                if paper_name in [corr.source, corr.target]:
                    correlations.append(corr)
        return correlations
    
    def create_network_graph(self, topic_name: Optional[str] = None) -> nx.DiGraph:
        """Create a network graph for visualization."""
        G = nx.DiGraph()
        
        if topic_name and topic_name in self.topics:
            # Single topic
            topic = self.topics[topic_name]
            for paper in topic.papers:
                G.add_node(paper, type='paper')
            
            for corr in topic.correlations:
                G.add_edge(
                    corr.source, corr.target,
                    relationship=corr.relationship_type,
                    description=corr.description,
                    strength=corr.strength
                )
        else:
            # All topics
            for topic in self.topics.values():
                for paper in topic.papers:
                    G.add_node(paper, type='paper', topic=topic.name)
                
                for corr in topic.correlations:
                    G.add_edge(
                        corr.source, corr.target,
                        relationship=corr.relationship_type,
                        description=corr.description,
                        strength=corr.strength,
                        topic=topic.name
                    )
        
        return G
    
    def export_to_dataframe(self, topic_name: Optional[str] = None) -> pd.DataFrame:
        """Export correlations to a pandas DataFrame."""
        data = []
        
        if topic_name and topic_name in self.topics:
            topic = self.topics[topic_name]
            for corr in topic.correlations:
                data.append({
                    'source': corr.source,
                    'target': corr.target,
                    'relationship_type': corr.relationship_type,
                    'description': corr.description,
                    'strength': corr.strength,
                    'evidence': corr.evidence,
                    'topic': topic_name
                })
        else:
            for topic_name, topic in self.topics.items():
                for corr in topic.correlations:
                    data.append({
                        'source': corr.source,
                        'target': corr.target,
                        'relationship_type': corr.relationship_type,
                        'description': corr.description,
                        'strength': corr.strength,
                        'evidence': corr.evidence,
                        'topic': topic_name
                    })
        
        return pd.DataFrame(data)


# Initialize the SSS (Solid Solution Strengthening) correlations
def initialize_sss_correlations() -> PaperCorrelationManager:
    """Initialize the SSS correlations with the provided data."""
    manager = PaperCorrelationManager()
    
    # Add SSS topic
    manager.add_topic(
        name="Solid Solution Strengthening (SSS)",
        description="Research on solid solution strengthening mechanisms in RHEAs"
    )
    
    # Add papers to SSS topic
    sss_papers = [
        "zhou2025strategies", "zheng2022development", "dou2024modulus",
        "wang2025dual", "wang2025hf", "fang2024composition", "chen2025low",
        "ko2025boron", "ji2022effect", "liu2025effect"
    ]
    
    for paper in sss_papers:
        manager.add_paper_to_topic("Solid Solution Strengthening (SSS)", paper)
    
    # Add correlations
    correlations = [
        # Shear modulus mismatch correlations
        PaperCorrelation(
            source="zhou2025strategies",
            target="zheng2022development",
            relationship_type="shear_modulus_mismatch",
            description="Shear modulus mismatch induces lattice distortion, improving yield and ductility",
            strength=0.9
        ),
        PaperCorrelation(
            source="zheng2022development",
            target="dou2024modulus",
            relationship_type="shear_modulus_mismatch",
            description="Shear modulus mismatch induces lattice distortion, improving yield and ductility",
            strength=0.9
        ),
        PaperCorrelation(
            source="dou2024modulus",
            target="zhou2025strategies",
            relationship_type="shear_modulus_mismatch",
            description="Shear modulus mismatch induces lattice distortion, improving yield and ductility",
            strength=0.9
        ),
        
        # Atomic size mismatch - RT correlations
        PaperCorrelation(
            source="wang2025dual",
            target="wang2025hf",
            relationship_type="atomic_size_mismatch_rt",
            description="Atomic size mismatch induces lattice distortion, improving yield but reducing ductility in RT",
            strength=0.8
        ),
        PaperCorrelation(
            source="wang2025hf",
            target="fang2024composition",
            relationship_type="atomic_size_mismatch_rt",
            description="Atomic size mismatch induces lattice distortion, improving yield but reducing ductility in RT",
            strength=0.8
        ),
        PaperCorrelation(
            source="fang2024composition",
            target="chen2025low",
            relationship_type="atomic_size_mismatch_rt",
            description="Atomic size mismatch induces lattice distortion, improving yield but reducing ductility in RT",
            strength=0.8
        ),
        
        # Atomic size mismatch - HT correlations
        PaperCorrelation(
            source="ko2025boron",
            target="fang2024composition",
            relationship_type="atomic_size_mismatch_ht",
            description="Atomic size mismatch induces lattice distortion, improving yield but reducing ductility in HT",
            strength=0.7
        ),
        
        # Atomic size mismatch - RT with comparable ductility
        PaperCorrelation(
            source="ji2022effect",
            target="fang2024composition",
            relationship_type="atomic_size_mismatch_rt_comparable",
            description="Atomic size mismatch induces lattice distortion, improving yield while maintaining comparable ductility in RT",
            strength=0.8
        ),
        
        # Solid solution treatment correlations
        PaperCorrelation(
            source="liu2025effect",
            target="fang2024composition",
            relationship_type="solid_solution_treatment_positive",
            description="Heat treatment dissolves secondary phase, increasing solute concentration and enhancing SSS",
            strength=0.9
        ),
        PaperCorrelation(
            source="fang2024composition",
            target="liu2025effect",
            relationship_type="solid_solution_treatment_negative",
            description="Heat treatment temperature too high and time too long leads to secondary phase formation",
            strength=0.8
        )
    ]
    
    for corr in correlations:
        manager.add_correlation("Solid Solution Strengthening (SSS)", corr)
    
    return manager


def display_correlation_interface():
    """Display the correlation management interface in Streamlit."""
    st.header("📊 Paper Correlation Management")
    
    # Initialize manager
    manager = PaperCorrelationManager()
    
    # Initialize SSS if not exists
    if "Solid Solution Strengthening (SSS)" not in manager.topics:
        manager = initialize_sss_correlations()
    
    # Sidebar for navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Add Correlation", "View Correlations", "Network Visualization", "Export Data"]
    )
    
    if page == "Overview":
        display_overview(manager)
    elif page == "Add Correlation":
        display_add_correlation(manager)
    elif page == "View Correlations":
        display_view_correlations(manager)
    elif page == "Network Visualization":
        display_network_visualization(manager)
    elif page == "Export Data":
        display_export_data(manager)


def display_overview(manager: PaperCorrelationManager):
    """Display overview of all topics and correlations."""
    st.subheader("📈 Research Topics Overview")
    
    for topic_name, topic in manager.topics.items():
        with st.expander(f"🔬 {topic_name} ({len(topic.papers)} papers, {len(topic.correlations)} correlations)"):
            st.write(f"**Description:** {topic.description}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Papers:**")
                for paper in topic.papers:
                    st.write(f"• {paper}")
            
            with col2:
                st.write("**Correlation Types:**")
                corr_types = set(corr.relationship_type for corr in topic.correlations)
                for corr_type in corr_types:
                    count = len([c for c in topic.correlations if c.relationship_type == corr_type])
                    st.write(f"• {corr_type}: {count}")


def display_add_correlation(manager: PaperCorrelationManager):
    """Display interface for adding new correlations."""
    st.subheader("➕ Add New Correlation")
    
    # Topic selection
    topic_name = st.selectbox("Select Topic", list(manager.topics.keys()))
    
    if topic_name:
        topic = manager.topics[topic_name]
        
        col1, col2 = st.columns(2)
        
        with col1:
            source = st.selectbox("Source Paper", topic.papers)
            relationship_type = st.text_input("Relationship Type", placeholder="e.g., shear_modulus_mismatch")
            strength = st.slider("Correlation Strength", 0.0, 1.0, 0.5, 0.1)
        
        with col2:
            target = st.selectbox("Target Paper", topic.papers)
            description = st.text_area("Description", placeholder="Describe the relationship...")
            evidence = st.text_area("Evidence (Optional)", placeholder="Supporting evidence...")
        
        if st.button("Add Correlation"):
            if source != target and relationship_type and description:
                correlation = PaperCorrelation(
                    source=source,
                    target=target,
                    relationship_type=relationship_type,
                    description=description,
                    strength=strength,
                    evidence=evidence if evidence else None
                )
                
                if manager.add_correlation(topic_name, correlation):
                    st.success("Correlation added successfully!")
                else:
                    st.error("Failed to add correlation.")
            else:
                st.error("Please fill in all required fields and ensure source and target are different.")


def display_view_correlations(manager: PaperCorrelationManager):
    """Display all correlations in a table format."""
    st.subheader("📋 View Correlations")
    
    topic_filter = st.selectbox("Filter by Topic", ["All"] + list(manager.topics.keys()))
    
    if topic_filter == "All":
        df = manager.export_to_dataframe()
    else:
        df = manager.export_to_dataframe(topic_filter)
    
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Correlations", len(df))
        with col2:
            st.metric("Unique Papers", len(set(df['source'].tolist() + df['target'].tolist())))
        with col3:
            st.metric("Avg Strength", f"{df['strength'].mean():.2f}")
    else:
        st.info("No correlations found.")


def display_network_visualization(manager: PaperCorrelationManager):
    """Display network visualization of correlations."""
    st.subheader("🕸️ Network Visualization")
    
    topic_filter = st.selectbox("Select Topic for Visualization", ["All"] + list(manager.topics.keys()))
    
    if topic_filter == "All":
        G = manager.create_network_graph()
    else:
        G = manager.create_network_graph(topic_filter)
    
    if G.number_of_nodes() > 0:
        # Create network layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=2, color='gray'),
                hoverinfo='none',
                mode='lines'
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=20,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                )
            )
        )
        
        # Color nodes by number of connections
        node_adjacencies = []
        for node in G.nodes():
            node_adjacencies.append(len(list(G.neighbors(node))))
        node_trace.marker.color = node_adjacencies
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace],
                       layout=go.Layout(
                           title=f'Paper Correlation Network - {topic_filter}',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Network statistics
        st.write(f"**Network Statistics:**")
        st.write(f"• Nodes (Papers): {G.number_of_nodes()}")
        st.write(f"• Edges (Correlations): {G.number_of_edges()}")
        st.write(f"• Average Degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    else:
        st.info("No network data available.")


def display_export_data(manager: PaperCorrelationManager):
    """Display export options for correlation data."""
    st.subheader("📤 Export Data")
    
    export_format = st.selectbox("Export Format", ["CSV", "JSON"])
    topic_filter = st.selectbox("Filter by Topic", ["All"] + list(manager.topics.keys()))
    
    if st.button("Export Data"):
        if topic_filter == "All":
            df = manager.export_to_dataframe()
        else:
            df = manager.export_to_dataframe(topic_filter)
        
        if not df.empty:
            if export_format == "CSV":
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"paper_correlations_{topic_filter.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            else:  # JSON
                json_data = df.to_json(orient='records', indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"paper_correlations_{topic_filter.lower().replace(' ', '_')}.json",
                    mime="application/json"
                )
        else:
            st.warning("No data to export.")


if __name__ == "__main__":
    # Test the system
    manager = initialize_sss_correlations()
    print(f"Initialized SSS correlations with {len(manager.topics)} topics")
    for topic_name, topic in manager.topics.items():
        print(f"Topic: {topic_name}")
        print(f"  Papers: {len(topic.papers)}")
        print(f"  Correlations: {len(topic.correlations)}") 