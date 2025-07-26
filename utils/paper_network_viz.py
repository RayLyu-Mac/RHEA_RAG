"""
Paper Network Visualization utilities for the Paper Search & QA System.
Provides interactive network diagram visualization for research papers.
"""

import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import tempfile
import os

# Try to import graphviz, but make it optional
try:
    from graphviz import Source
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    Source = None

def plot_mechanism_network_interactive(paper_metadata):
    """
    Plots an interactive network diagram of research papers grouped by mechanism/conclusion.
    Args:
        paper_metadata (list of dict): Each dict must have 'title', 'mechanism', 'color'.
    Returns:
        plotly.graph_objects.Figure: The Plotly figure for display in Streamlit.
    """
    try:
        n = len(paper_metadata)
        if n == 0:
            st.warning("No papers to visualize.")
            return None
        # Build fully connected graph
        G = nx.Graph()
        for i, meta in enumerate(paper_metadata):
            G.add_node(i, title=meta.get('title', f'Paper {i+1}'), mechanism=meta.get('mechanism', 'Unknown'), color=meta.get('color', '#888'))
        for i in range(n):
            for j in range(i+1, n):
                G.add_edge(i, j)
        pos = nx.spring_layout(G, seed=42)
        node_x, node_y, node_text, node_color = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            meta = G.nodes[node]
            node_text.append(f"<b>{meta['title']}</b><br>Mechanism: {meta['mechanism']}")
            node_color.append(meta['color'])
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
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
                color=node_color,
                size=18,
                line_width=2
            ),
            text=[meta['title'] for meta in paper_metadata],
            hovertext=node_text
        )
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='<b>Mechanism/Conclusion Network</b>',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        return fig
    except Exception as e:
        st.error(f"Error generating mechanism network visualization: {e}")
        return None

def render_dot_flowchart(dot_code):
    """
    Render a Graphviz DOT flowchart and display it in Streamlit.
    Args:
        dot_code (str): DOT code string.
    """
    if not GRAPHVIZ_AVAILABLE:
        st.warning("⚠️ **Graphviz not available**: Cannot render DOT flowchart. Install graphviz with `pip install graphviz`")
        st.code(dot_code, language='dot')
        return
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            dot_path = os.path.join(tmpdir, "flowchart.dot")
            png_path = os.path.join(tmpdir, "flowchart.png")
            with open(dot_path, "w") as f:
                f.write(dot_code)
            graph = Source.from_file(dot_path)
            graph.render(filename="flowchart", directory=tmpdir, format="png", cleanup=True)
            st.image(png_path)
    except Exception as e:
        st.error(f"Error rendering DOT flowchart: {e}")

def llm_grouped_network_interactive(selected_papers, llm, user_question=None):
    """
    LLM-powered grouping network: groups papers by LLM-extracted label (mechanism/type/conclusion) based on user_question.
    Args:
        selected_papers (list of dict): Each dict must have 'file_name', 'abstract'.
        llm: The loaded LLM (must have .invoke(prompt)).
        user_question (str): The user's question for grouping (optional).
    Returns:
        fig: plotly.graph_objects.Figure
        group_legend: dict mapping group label to color
    """
    import plotly.graph_objects as go
    import networkx as nx
    color_palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
    paper_metadata = []
    group_to_color = {}
    color_idx = 0
    if not user_question:
        user_question = "What is the main mechanism or conclusion discussed in this paper? Summarize in one sentence."
    for paper in selected_papers:
        abstract = paper.get('abstract', '')
        prompt = (
            f"Given the following abstract and the user's question:\n\n"
            f"Question: {user_question}\n\n"
            f"Abstract:\n{abstract}\n\n"
            f"What is the main mechanism, type, or conclusion discussed in this paper relevant to the question? Summarize in one sentence."
        )
        try:
            group_label = llm.invoke(prompt)
        except Exception as e:
            group_label = "Unknown"
        if group_label not in group_to_color:
            group_to_color[group_label] = color_palette[color_idx % len(color_palette)]
            color_idx += 1
        paper_metadata.append({
            'title': paper['file_name'],
            'group_label': group_label,
            'color': group_to_color[group_label]
        })
    # Build fully connected graph
    n = len(paper_metadata)
    G = nx.Graph()
    for i, meta in enumerate(paper_metadata):
        G.add_node(i, title=meta['title'], group_label=meta['group_label'], color=meta['color'])
    for i in range(n):
        for j in range(i+1, n):
            G.add_edge(i, j)
    pos = nx.spring_layout(G, seed=42)
    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        meta = G.nodes[node]
        node_text.append(f"<b>{meta['title']}</b><br>Group: {meta['group_label']}")
        node_color.append(meta['color'])
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
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
            color=node_color,
            size=18,
            line_width=2
        ),
        text=[meta['title'] for meta in paper_metadata],
        hovertext=node_text
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='<b>LLM-Grouped Paper Network</b>',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    return fig, group_to_color 