"""
Paper Network Visualization utilities for the Paper Search & QA System.
Provides interactive network diagram visualization for research papers.
"""

import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import numpy as np


def plot_paper_network_interactive(paper_metadata, similarity_matrix, min_threshold=0.2, max_threshold=1.0, default_threshold=0.5):
    """
    Plots an interactive network diagram of research papers based on a similarity matrix.
    
    Args:
        paper_metadata (list of dict): Each dict must have 'title' and 'authors' keys.
        similarity_matrix (2D list or np.ndarray): Symmetric matrix of pairwise similarities.
        min_threshold (float): Minimum value for similarity threshold slider.
        max_threshold (float): Maximum value for similarity threshold slider.
        default_threshold (float): Default value for similarity threshold slider.
    
    Returns:
        plotly.graph_objects.Figure: The Plotly figure for display in Streamlit.
    """
    try:
        # Validate inputs
        n = len(paper_metadata)
        if n == 0:
            st.warning("No papers to visualize.")
            return None
        
        if not isinstance(similarity_matrix, np.ndarray):
            similarity_matrix = np.array(similarity_matrix)
        
        if similarity_matrix.shape != (n, n):
            st.error(f"Similarity matrix shape {similarity_matrix.shape} does not match number of papers {n}.")
            return None
        
        # Streamlit slider for threshold
        threshold = st.slider(
            "Similarity threshold for connections",
            min_value=float(min_threshold),
            max_value=float(max_threshold),
            value=float(default_threshold),
            step=0.01
        )
        
        # Build NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for i, meta in enumerate(paper_metadata):
            G.add_node(i, title=meta.get('title', f'Paper {i+1}'), authors=meta.get('authors', 'Unknown'))
        
        # Add edges based on threshold
        for i in range(n):
            for j in range(i+1, n):
                sim = similarity_matrix[i, j]
                if sim >= threshold:
                    G.add_edge(i, j, weight=sim)
        
        if G.number_of_edges() == 0:
            st.info("No connections above the selected threshold.")
            return None
        
        # Layout for Plotly (spring layout)
        pos = nx.spring_layout(G, seed=42)
        
        # Node positions
        node_x, node_y = [], []
        node_text = []
        node_degree = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            meta = G.nodes[node]
            node_text.append(f"<b>{meta['title']}</b><br>Authors: {meta['authors']}")
            node_degree.append(G.degree[node])
        
        # Edge positions
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        
        # Plotly traces
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            textposition='top center',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                color=node_degree,
                size=18,
                colorbar=dict(
                    thickness=15,
                    title='Node Degree',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2
            ),
            text=[meta['title'] for meta in paper_metadata],
            hovertext=node_text
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='<b>Paper Relationship Network</b>',
                titlefont_size=20,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Node color = degree (number of connections)",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error generating network visualization: {e}")
        return None 