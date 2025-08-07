"""
Logo display utilities for the Material Research RAG System.
Provides functions to display the logo in Streamlit applications.
"""

import streamlit as st
import os
from pathlib import Path


def display_logo(size: int = 64, color: str = "#60B57D"):
    """
    Display the Material Research RAG logo in Streamlit.
    
    Args:
        size (int): Size of the logo in pixels (default: 64)
        color (str): Color of the logo (default: "#60B57D" - green)
    """
    # Get the path to the logo file
    current_dir = Path(__file__).parent
    logo_path = current_dir.parent / "material_research_logo.svg"
    
    if logo_path.exists():
        # Read the SVG content
        with open(logo_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        
        # Replace the size and color in the SVG
        svg_content = svg_content.replace('width="256"', f'width="{size}"')
        svg_content = svg_content.replace('height="256"', f'height="{size}"')
        svg_content = svg_content.replace('fill="#60B57D"', f'fill="{color}"')
        
        # Display the SVG
        st.markdown(svg_content, unsafe_allow_html=True)
    else:
        # Fallback: display a simple text-based logo
        st.markdown(f"""
        <div style="text-align: center; margin: 20px 0;">
            <div style="font-size: {size//2}px; color: {color}; font-weight: bold;">
                ⚛️
            </div>
            <div style="font-size: {size//4}px; color: #666; margin-top: 10px;">
                Material Research RAG
            </div>
        </div>
        """, unsafe_allow_html=True)


def display_logo_with_title(title: str = "Material Research RAG System", size: int = 64):
    """
    Display the logo with a title in a clean layout.
    
    Args:
        title (str): Title to display next to the logo
        size (int): Size of the logo in pixels
    """
    col1, col2 = st.columns([1, 4])
    
    with col1:
        display_logo(size=size)
    
    with col2:
        st.markdown(f"""
        <div style="margin-top: 10px;">
            <h1 style="color: #2c3e50; font-size: 1.8rem; margin: 0; font-weight: bold;">
                {title}
            </h1>
        </div>
        """, unsafe_allow_html=True)


def display_logo_in_sidebar(size: int = 48):
    """
    Display a smaller logo in the sidebar.
    
    Args:
        size (int): Size of the logo in pixels (default: 48 for sidebar)
    """
    with st.sidebar:
        # Simple logo display without extra headers
        display_logo(size=size)
        st.caption("Material Research RAG")
        st.divider()


def get_logo_html(size: int = 64, color: str = "#60B57D") -> str:
    """
    Get the logo as HTML string for custom use.
    
    Args:
        size (int): Size of the logo in pixels
        color (str): Color of the logo
        
    Returns:
        str: HTML string containing the logo
    """
    # Get the path to the logo file
    current_dir = Path(__file__).parent
    logo_path = current_dir.parent / "material_research_logo.svg"
    
    if logo_path.exists():
        # Read the SVG content
        with open(logo_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        
        # Replace the size and color in the SVG
        svg_content = svg_content.replace('width="256"', f'width="{size}"')
        svg_content = svg_content.replace('height="256"', f'height="{size}"')
        svg_content = svg_content.replace('fill="#60B57D"', f'fill="{color}"')
        
        return svg_content
    else:
        # Fallback HTML
        return f"""
        <div style="text-align: center; margin: 20px 0;">
            <div style="font-size: {size//2}px; color: {color}; font-weight: bold;">
                ⚛️
            </div>
            <div style="font-size: {size//4}px; color: #666; margin-top: 10px;">
                Material Research RAG
            </div>
        </div>
        """


def display_logo_header():
    """
    Display the logo as a clean header with title in the left corner.
    """
    # Create a simple layout with logo on left and title on right
    col1, col2 = st.columns([1, 4])
    
    with col1:
        # Display the logo
        display_logo(size=60, color="#60B57D")
    
    with col2:
        st.markdown("""
        <div style="margin-top: 10px;">
            <h1 style="color: #2c3e50; font-size: 2.2rem; margin: 0; font-weight: bold;">
                🔬 Material Research RAG System
            </h1>
            <p style="color: #666; font-size: 1.1rem; margin: 5px 0 0 0;">
                Advanced Paper Search & Question Answering for Materials Science
            </p>
        </div>
        """, unsafe_allow_html=True)


def display_logo_footer():
    """
    Display the logo as a footer.
    """
    st.markdown("""
    <div style="text-align: center; margin: 40px 0 20px 0; padding: 20px; border-top: 1px solid #eee;">
        <div style="display: inline-block; margin-bottom: 10px;">
    """, unsafe_allow_html=True)
    
    # Display a smaller logo
    display_logo(size=40, color="#60B57D")
    
    st.markdown("""
        </div>
        <p style="color: #666; font-size: 0.9rem; margin: 0;">
            Powered by Material Research RAG System
        </p>
    </div>
    """, unsafe_allow_html=True)
