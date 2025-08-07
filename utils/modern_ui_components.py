"""
Modern UI Components for Material Research RAG System
Provides consistent, modern styling across all application tabs.
"""

import streamlit as st
from typing import List, Dict, Optional, Any
import datetime


def apply_modern_theme():
    """Apply modern theme CSS with consistent design system"""
    st.markdown("""
    <style>
    /* Modern Design System Variables */
    :root {
        --primary-color: #2563eb;
        --primary-light: #3b82f6;
        --primary-dark: #1d4ed8;
        --secondary-color: #64748b;
        --accent-color: #06b6d4;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --background-light: #ffffff;
        --background-dark: #0f172a;
        --surface-light: #f8fafc;
        --surface-dark: #1e293b;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --text-muted: #94a3b8;
        --border-color: #e2e8f0;
        --border-radius: 12px;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Dark theme overrides */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }

    .dark [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }

    /* Modern Card Components */
    .modern-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-md);
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: var(--transition);
    }

    .dark .modern-card {
        background: rgba(30, 41, 59, 0.95);
        border-color: #334155;
    }

    .modern-card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }

    /* Modern Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-light));
        border: none;
        border-radius: var(--border-radius);
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: var(--transition);
        box-shadow: var(--shadow-sm);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, var(--primary-dark), var(--primary-color));
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Secondary Button */
    .stButton > button[data-variant="secondary"] {
        background: linear-gradient(135deg, var(--secondary-color), #94a3b8);
    }

    /* Modern Input Fields */
    .stTextInput > div > div > input {
        border: 2px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 0.75rem 1rem;
        font-size: 0.95rem;
        transition: var(--transition);
        background: rgba(255, 255, 255, 0.9);
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        outline: none;
    }

    /* Modern Text Area */
    .stTextArea > div > div > textarea {
        border: 2px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 0.75rem 1rem;
        font-size: 0.95rem;
        transition: var(--transition);
        background: rgba(255, 255, 255, 0.9);
        resize: vertical;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        outline: none;
    }

    /* Modern Select Box */
    .stSelectbox > div > div > div {
        border: 2px solid var(--border-color);
        border-radius: var(--border-radius);
        background: rgba(255, 255, 255, 0.9);
    }

    /* Modern Slider */
    .stSlider > div > div > div > div {
        background: var(--primary-color);
    }

    /* Modern Checkbox */
    .stCheckbox > div > div > div {
        border: 2px solid var(--border-color);
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.9);
    }

    /* Modern Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 1rem;
        font-weight: 600;
        transition: var(--transition);
    }

    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.95);
        box-shadow: var(--shadow-sm);
    }

    /* Modern Tabs */
    .stTabs > div > div > div > div {
        background: rgba(255, 255, 255, 0.8);
        border-radius: var(--border-radius);
        border: 1px solid var(--border-color);
    }

    .stTabs > div > div > div > div > div {
        border-radius: var(--border-radius);
        transition: var(--transition);
    }

    .stTabs > div > div > div > div > div[aria-selected="true"] {
        background: var(--primary-color);
        color: white;
    }

    /* Modern Sidebar */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid var(--border-color);
    }

    /* Modern Metrics */
    .stMetric > div > div > div {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 1rem;
        box-shadow: var(--shadow-sm);
    }

    /* Modern Dataframe */
    .stDataFrame > div {
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }

    /* Modern Success/Error Messages */
    .stAlert {
        border-radius: var(--border-radius);
        border: none;
        box-shadow: var(--shadow-sm);
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.1);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-dark);
    }

    /* Modern Loading Spinner */
    .stSpinner > div {
        border: 3px solid var(--border-color);
        border-top: 3px solid var(--primary-color);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Modern Badge/Tag */
    .modern-badge {
        display: inline-block;
        background: linear-gradient(135deg, var(--primary-color), var(--primary-light));
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Modern Divider */
    .modern-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 1.5rem 0;
    }

    /* Modern Code Block */
    .modern-code {
        background: rgba(0, 0, 0, 0.05);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 1rem;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 0.875rem;
        overflow-x: auto;
    }

    .dark .modern-code {
        background: rgba(255, 255, 255, 0.05);
    }

    /* Modern Tooltip */
    .modern-tooltip {
        position: relative;
        display: inline-block;
    }

    .modern-tooltip:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        font-size: 0.75rem;
        white-space: nowrap;
        z-index: 1000;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .modern-card {
            padding: 1rem;
            margin: 0.25rem 0;
        }
        
        .stButton > button {
            padding: 0.5rem 1rem;
            font-size: 0.875rem;
        }
    }

    /* Animation Classes */
    .fade-in {
        animation: fadeIn 0.5s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .slide-in {
        animation: slideIn 0.3s ease-out;
    }

    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    /* Modern Status Indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }

    .status-online { background: var(--success-color); }
    .status-offline { background: var(--error-color); }
    .status-warning { background: var(--warning-color); }
    .status-info { background: var(--accent-color); }

    </style>
    """, unsafe_allow_html=True)


def create_modern_card(title: str = "", content: str = "", icon: str = "", 
                      variant: str = "default", padding: str = "1.5rem") -> str:
    """
    Create a modern card component with consistent styling.
    
    Args:
        title: Card title
        content: Card content
        icon: Icon emoji or symbol
        variant: Card variant (default, success, warning, error, info)
        padding: Custom padding
    
    Returns:
        HTML string for the modern card
    """
    variant_colors = {
        "default": "var(--primary-color)",
        "success": "var(--success-color)",
        "warning": "var(--warning-color)",
        "error": "var(--error-color)",
        "info": "var(--accent-color)"
    }
    
    color = variant_colors.get(variant, variant_colors["default"])
    
    return f"""
    <div class="modern-card fade-in" style="border-left: 4px solid {color}; padding: {padding};">
        {f'<div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">{icon} <h3 style="margin: 0; color: {color}; font-weight: 600;">{title}</h3></div>' if title else ''}
        <div style="line-height: 1.6; color: var(--text-secondary);">
            {content}
        </div>
    </div>
    """


def create_modern_badge(text: str, variant: str = "default") -> str:
    """
    Create a modern badge/tag component.
    
    Args:
        text: Badge text
        variant: Badge variant (default, success, warning, error, info)
    
    Returns:
        HTML string for the modern badge
    """
    variant_colors = {
        "default": "var(--primary-color)",
        "success": "var(--success-color)",
        "warning": "var(--warning-color)",
        "error": "var(--error-color)",
        "info": "var(--accent-color)"
    }
    
    color = variant_colors.get(variant, variant_colors["default"])
    
    return f"""
    <span class="modern-badge" style="background: linear-gradient(135deg, {color}, {color}dd);">
        {text}
    </span>
    """


def create_modern_metric(label: str, value: str, change: str = "", 
                        change_type: str = "neutral") -> str:
    """
    Create a modern metric component.
    
    Args:
        label: Metric label
        value: Metric value
        change: Change indicator (e.g., "+5.2%")
        change_type: Change type (positive, negative, neutral)
    
    Returns:
        HTML string for the modern metric
    """
    change_color = {
        "positive": "var(--success-color)",
        "negative": "var(--error-color)",
        "neutral": "var(--text-muted)"
    }.get(change_type, "var(--text-muted)")
    
    change_html = f'<div style="color: {change_color}; font-size: 0.875rem; margin-top: 0.25rem;">{change}</div>' if change else ""
    
    return f"""
    <div class="modern-card" style="text-align: center; padding: 1rem;">
        <div style="color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 0.5rem;">{label}</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: var(--text-primary);">{value}</div>
        {change_html}
    </div>
    """


def create_modern_divider() -> str:
    """Create a modern divider component."""
    return '<div class="modern-divider"></div>'


def create_modern_code_block(code: str, language: str = "python") -> str:
    """
    Create a modern code block component.
    
    Args:
        code: Code content
        language: Programming language for syntax highlighting
    
    Returns:
        HTML string for the modern code block
    """
    return f"""
    <div class="modern-code">
        <div style="color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">{language}</div>
        <pre style="margin: 0; white-space: pre-wrap; color: var(--text-primary);">{code}</pre>
    </div>
    """


def create_modern_status_indicator(status: str, text: str = "") -> str:
    """
    Create a modern status indicator component.
    
    Args:
        status: Status type (online, offline, warning, info)
        text: Status text
    
    Returns:
        HTML string for the modern status indicator
    """
    return f"""
    <div style="display: flex; align-items: center;">
        <span class="status-indicator status-{status}"></span>
        {f'<span style="color: var(--text-secondary);">{text}</span>' if text else ''}
    </div>
    """


def create_modern_tooltip(element: str, tooltip: str) -> str:
    """
    Create a modern tooltip component.
    
    Args:
        element: HTML element to attach tooltip to
        tooltip: Tooltip text
    
    Returns:
        HTML string with tooltip
    """
    return f'<div class="modern-tooltip" data-tooltip="{tooltip}">{element}</div>'


def display_modern_header(title: str, subtitle: str = "", icon: str = "🔬") -> None:
    """
    Display a modern header component.
    
    Args:
        title: Header title
        subtitle: Header subtitle
        icon: Header icon
    """
    st.markdown(f"""
    <div style="text-align: center; margin: 2rem 0; padding: 2rem; background: rgba(255, 255, 255, 0.9); border-radius: var(--border-radius); border: 1px solid var(--border-color); box-shadow: var(--shadow-md);">
        <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
        <h1 style="margin: 0; color: var(--text-primary); font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">{title}</h1>
        {f'<p style="margin: 0; color: var(--text-secondary); font-size: 1.1rem;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def display_modern_stats(stats: List[Dict[str, Any]]) -> None:
    """
    Display modern statistics cards.
    
    Args:
        stats: List of statistics dictionaries with 'label', 'value', 'change', 'change_type' keys
    """
    cols = st.columns(len(stats))
    
    for i, stat in enumerate(stats):
        with cols[i]:
            st.markdown(create_modern_metric(
                stat.get('label', ''),
                stat.get('value', ''),
                stat.get('change', ''),
                stat.get('change_type', 'neutral')
            ), unsafe_allow_html=True)


def display_modern_search_box(placeholder: str = "Search papers...", key: str = "search") -> str:
    """
    Display a modern search input box.
    
    Args:
        placeholder: Search placeholder text
        key: Streamlit key for the input
    
    Returns:
        Search query string
    """
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        <div style="position: relative;">
            <div style="position: absolute; left: 1rem; top: 50%; transform: translateY(-50%); color: var(--text-muted);">🔍</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    return st.text_input("", placeholder=placeholder, key=key, label_visibility="collapsed")


def display_modern_loading_spinner(text: str = "Loading...") -> None:
    """
    Display a modern loading spinner.
    
    Args:
        text: Loading text
    """
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 1rem; padding: 2rem;">
        <div class="stSpinner">
            <div style="width: 2rem; height: 2rem;"></div>
        </div>
        <span style="color: var(--text-secondary); font-weight: 500;">{text}</span>
    </div>
    """, unsafe_allow_html=True)


def display_modern_empty_state(icon: str, title: str, description: str, action_text: str = "", action_func = None) -> None:
    """
    Display a modern empty state component.
    
    Args:
        icon: Empty state icon
        title: Empty state title
        description: Empty state description
        action_text: Action button text
        action_func: Action button function
    """
    st.markdown(f"""
    <div style="text-align: center; padding: 4rem 2rem; color: var(--text-secondary);">
        <div style="font-size: 4rem; margin-bottom: 1rem;">{icon}</div>
        <h3 style="margin: 0 0 0.5rem 0; color: var(--text-primary); font-size: 1.5rem;">{title}</h3>
        <p style="margin: 0; font-size: 1rem; line-height: 1.6;">{description}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if action_text and action_func:
        if st.button(action_text, type="primary"):
            action_func()


def create_modern_tab_style() -> str:
    """Create modern tab styling CSS."""
    return """
    <style>
    .stTabs > div > div > div > div {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        border: 1px solid var(--border-color);
        padding: 0.5rem;
        gap: 0.5rem;
    }
    
    .stTabs > div > div > div > div > div {
        border-radius: 8px;
        transition: all 0.2s ease;
        font-weight: 500;
    }
    
    .stTabs > div > div > div > div > div[aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-light));
        color: white;
        box-shadow: var(--shadow-sm);
    }
    
    .stTabs > div > div > div > div > div:hover:not([aria-selected="true"]) {
        background: rgba(37, 99, 235, 0.1);
    }
    </style>
    """


def apply_modern_tab_style():
    """Apply modern tab styling."""
    st.markdown(create_modern_tab_style(), unsafe_allow_html=True)


# Initialize modern theme when imported
apply_modern_theme()
