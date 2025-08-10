"""
Clean & Minimalist UI Components for Material Research RAG System
Provides a clean, open, and minimalist design with plenty of whitespace.
"""

import streamlit as st
from typing import List, Dict, Optional, Any
import datetime


def apply_clean_theme():
    """Apply clean, minimalist theme CSS with open design"""
    st.markdown("""
    <style>
    /* Clean Design System Variables */
    :root {
        --primary-color: #3b82f6;
        --primary-light: #60a5fa;
        --primary-dark: #2563eb;
        --secondary-color: #64748b;
        --accent-color: #06b6d4;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --background: #ffffff;
        --surface: #f8fafc;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --text-muted: #94a3b8;
        --border-color: #e2e8f0;
        --border-radius: 8px;
        --shadow-sm: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --transition: all 0.15s ease-in-out;
        --spacing-xs: 0.5rem;
        --spacing-sm: 1rem;
        --spacing-md: 1.5rem;
        --spacing-lg: 2rem;
        --spacing-xl: 3rem;
    }

    /* Clean Background */
    [data-testid="stAppViewContainer"] {
        background: var(--background);
    }

    .main .block-container {
        padding-top: 0.25rem;
        padding-bottom: var(--spacing-lg);
        max-width: 1200px;
    }

    /* Clean Card Components */
    .clean-card {
        background: var(--background);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-sm);
        padding: var(--spacing-md);
        margin: var(--spacing-sm) 0;
        transition: var(--transition);
    }

    .clean-card:hover {
        box-shadow: var(--shadow-md);
        border-color: var(--primary-light);
    }

    /* Clean Button Styles */
    .stButton > button {
        background: var(--primary-color);
        border: none;
        border-radius: var(--border-radius);
        color: white;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        transition: var(--transition);
        box-shadow: var(--shadow-sm);
    }

    .stButton > button:hover {
        background: var(--primary-dark);
        box-shadow: var(--shadow-md);
    }

    .stButton > button:active {
        /* no motion on click */
    }

    /* Clean Input Fields */
    .stTextInput > div > div > input {
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 0.75rem 1rem;
        font-size: 0.95rem;
        transition: var(--transition);
        background: var(--background);
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        outline: none;
    }

    /* Clean Text Area */
    .stTextArea > div > div > textarea {
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 0.75rem 1rem;
        font-size: 0.95rem;
        transition: var(--transition);
        background: var(--background);
        resize: vertical;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        outline: none;
    }

    /* Clean Select Box */
    .stSelectbox > div > div > div {
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        background: var(--background);
    }

    /* Clean Slider */
    .stSlider > div > div > div > div {
        background: var(--primary-color);
    }

    /* Clean Checkbox */
    .stCheckbox > div > div > div {
        border: 1px solid var(--border-color);
        border-radius: 4px;
        background: var(--background);
    }

    /* Clean Expander */
    .streamlit-expanderHeader {
        background: var(--surface);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: var(--spacing-sm);
        font-weight: 500;
        transition: var(--transition);
    }

    .streamlit-expanderHeader:hover {
        background: var(--background);
        border-color: var(--primary-light);
    }

    /* Clean Tabs */
    .stTabs > div > div > div > div {
        background: var(--surface);
        border-radius: var(--border-radius);
        border: 1px solid var(--border-color);
        padding: 0.25rem;
        gap: 0.25rem;
    }

    .stTabs > div > div > div > div > div {
        border-radius: 6px;
        transition: var(--transition);
        font-weight: 500;
        padding: 0.5rem 1rem;
    }

    .stTabs > div > div > div > div > div[aria-selected="true"] {
        background: var(--primary-color);
        color: white;
        box-shadow: var(--shadow-sm);
    }

    .stTabs > div > div > div > div > div:hover:not([aria-selected="true"]) {
        background: rgba(59, 130, 246, 0.1);
    }

    /* Clean Sidebar */
    .css-1d391kg {
        background: var(--surface);
        border-right: 1px solid var(--border-color);
    }

    /* Clean Metrics */
    .stMetric > div > div > div {
        background: var(--background);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: var(--spacing-sm);
        box-shadow: var(--shadow-sm);
    }

    /* Clean Dataframe */
    .stDataFrame > div {
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }

    /* Clean Success/Error Messages */
    .stAlert {
        border-radius: var(--border-radius);
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }

    ::-webkit-scrollbar-track {
        background: var(--surface);
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-dark);
    }

    /* Clean Loading Spinner - no rotation */
    .stSpinner > div {
        border: 2px solid var(--border-color);
        border-top: 2px solid var(--primary-color);
        border-radius: 50%;
        animation: none !important;
    }

    /* Clean Badge/Tag */
    .clean-badge {
        display: inline-block;
        background: var(--primary-color);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 16px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    /* Clean Divider */
    .clean-divider {
        height: 1px;
        background: var(--border-color);
        margin: var(--spacing-md) 0;
    }

    /* Clean Code Block */
    .clean-code {
        background: var(--surface);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: var(--spacing-sm);
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 0.875rem;
        overflow-x: auto;
    }

    /* Clean Tooltip */
    .clean-tooltip {
        position: relative;
        display: inline-block;
    }

    .clean-tooltip:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        background: var(--text-primary);
        color: white;
        padding: 0.5rem 0.75rem;
        border-radius: 4px;
        font-size: 0.75rem;
        white-space: nowrap;
        z-index: 1000;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .clean-card {
            padding: var(--spacing-sm);
            margin: var(--spacing-xs) 0;
        }
        
        .stButton > button {
            padding: 0.5rem 1rem;
            font-size: 0.875rem;
        }
    }

    /* Disable animations */
    .fade-in, .slide-in {
        animation: none !important;
    }

    /* Clean Status Indicators */
    .status-indicator {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }

    .status-online { background: var(--success-color); }
    .status-offline { background: var(--error-color); }
    .status-warning { background: var(--warning-color); }
    .status-info { background: var(--accent-color); }

    /* Clean Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: var(--spacing-sm) 0 var(--spacing-sm) 0;
        padding-bottom: var(--spacing-sm);
        border-bottom: 2px solid var(--border-color);
    }

    .section-subheader {
        font-size: 1.1rem;
        font-weight: 500;
        color: var(--text-secondary);
        margin: var(--spacing-md) 0 var(--spacing-sm) 0;
    }

    /* Clean Content Areas */
    .content-area {
        background: var(--background);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: var(--spacing-md);
        margin: var(--spacing-sm) 0;
        box-shadow: var(--shadow-sm);
    }

    /* Clean Navigation */
    .nav-item {
        padding: var(--spacing-sm);
        border-radius: var(--border-radius);
        transition: var(--transition);
        cursor: pointer;
    }

    .nav-item:hover {
        background: var(--surface);
    }

    .nav-item.active {
        background: var(--primary-color);
        color: white;
    }

    </style>
    """, unsafe_allow_html=True)


def create_clean_card(title: str = "", content: str = "", icon: str = "", 
                     variant: str = "default", padding: str = "var(--spacing-md)") -> str:
    """
    Create a clean card component with minimalist styling.
    
    Args:
        title: Card title
        content: Card content
        icon: Icon emoji or symbol
        variant: Card variant (default, success, warning, error, info)
        padding: Custom padding
    
    Returns:
        HTML string for the clean card
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
    <div class="clean-card fade-in" style="border-left: 3px solid {color}; padding: {padding};">
        {f'<div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: var(--spacing-sm);">{icon} <h3 style="margin: 0; color: {color}; font-weight: 600; font-size: 1.1rem;">{title}</h3></div>' if title else ''}
        <div style="line-height: 1.6; color: var(--text-secondary);">
            {content}
        </div>
    </div>
    """


def create_clean_badge(text: str, variant: str = "default") -> str:
    """
    Create a clean badge/tag component.
    
    Args:
        text: Badge text
        variant: Badge variant (default, success, warning, error, info)
    
    Returns:
        HTML string for the clean badge
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
    <span class="clean-badge" style="background: {color};">
        {text}
    </span>
    """


def create_clean_metric(label: str, value: str, change: str = "", 
                       change_type: str = "neutral") -> str:
    """
    Create a clean metric component.
    
    Args:
        label: Metric label
        value: Metric value
        change: Change indicator (e.g., "+5.2%")
        change_type: Change type (positive, negative, neutral)
    
    Returns:
        HTML string for the clean metric
    """
    change_color = {
        "positive": "var(--success-color)",
        "negative": "var(--error-color)",
        "neutral": "var(--text-muted)"
    }.get(change_type, "var(--text-muted)")
    
    change_html = f'<div style="color: {change_color}; font-size: 0.875rem; margin-top: 0.25rem;">{change}</div>' if change else ""
    
    return f"""
    <div class="clean-card" style="text-align: center; padding: var(--spacing-sm);">
        <div style="color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 0.5rem;">{label}</div>
        <div style="font-size: 1.5rem; font-weight: 600; color: var(--text-primary);">{value}</div>
        {change_html}
    </div>
    """


def create_clean_divider() -> str:
    """Create a clean divider component."""
    return '<div class="clean-divider"></div>'


def create_clean_code_block(code: str, language: str = "python") -> str:
    """
    Create a clean code block component.
    
    Args:
        code: Code content
        language: Programming language for syntax highlighting
    
    Returns:
        HTML string for the clean code block
    """
    return f"""
    <div class="clean-code">
        <div style="color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">{language}</div>
        <pre style="margin: 0; white-space: pre-wrap; color: var(--text-primary);">{code}</pre>
    </div>
    """


def create_clean_status_indicator(status: str, text: str = "") -> str:
    """
    Create a clean status indicator component.
    
    Args:
        status: Status type (online, offline, warning, info)
        text: Status text
    
    Returns:
        HTML string for the clean status indicator
    """
    return f"""
    <div style="display: flex; align-items: center;">
        <span class="status-indicator status-{status}"></span>
        {f'<span style="color: var(--text-secondary);">{text}</span>' if text else ''}
    </div>
    """


def create_clean_tooltip(element: str, tooltip: str) -> str:
    """
    Create a clean tooltip component.
    
    Args:
        element: HTML element to attach tooltip to
        tooltip: Tooltip text
    
    Returns:
        HTML string with tooltip
    """
    return f'<div class="clean-tooltip" data-tooltip="{tooltip}">{element}</div>'


def display_clean_header(title: str, subtitle: str = "", icon: str = "🔬") -> None:
    """
    Display a clean header component.
    
    Args:
        title: Header title
        subtitle: Header subtitle
        icon: Header icon
    """
    st.markdown(f"""
    <div style="text-align: center; margin: var(--spacing-md) 0; padding: var(--spacing-md); background: var(--surface); border-radius: var(--border-radius); border: 1px solid var(--border-color);">
        <div style="font-size: 2rem; margin-bottom: var(--spacing-sm);">{icon}</div>
        <h1 style="margin: 0; color: var(--text-primary); font-size: 1.75rem; font-weight: 600; margin-bottom: 0.5rem;">{title}</h1>
        {f'<p style="margin: 0; color: var(--text-secondary); font-size: 1rem;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def display_clean_section_header(title: str, subtitle: str = "") -> None:
    """
    Display a clean section header.
    
    Args:
        title: Section title
        subtitle: Section subtitle
    """
    st.markdown(f"""
    <div class="section-header">
        {title}
        {f'<div class="section-subheader">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def display_clean_stats(stats: List[Dict[str, Any]]) -> None:
    """
    Display clean statistics cards.
    
    Args:
        stats: List of statistics dictionaries with 'label', 'value', 'change', 'change_type' keys
    """
    cols = st.columns(len(stats))
    
    for i, stat in enumerate(stats):
        with cols[i]:
            st.markdown(create_clean_metric(
                stat.get('label', ''),
                stat.get('value', ''),
                stat.get('change', ''),
                stat.get('change_type', 'neutral')
            ), unsafe_allow_html=True)


def display_clean_search_box(placeholder: str = "Search papers...", key: str = "search") -> str:
    """
    Display a clean search input box.
    
    Args:
        placeholder: Search placeholder text
        key: Streamlit key for the input
    
    Returns:
        Search query string
    """
    return st.text_input("", placeholder=placeholder, key=key, label_visibility="collapsed")


def display_clean_loading_spinner(text: str = "Loading...") -> None:
    """
    Display a clean loading spinner.
    
    Args:
        text: Loading text
    """
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: var(--spacing-sm); padding: var(--spacing-xl);">
        <div class="stSpinner">
            <div style="width: 1.5rem; height: 1.5rem;"></div>
        </div>
        <span style="color: var(--text-secondary); font-weight: 500;">{text}</span>
    </div>
    """, unsafe_allow_html=True)


def display_clean_empty_state(icon: str, title: str, description: str, action_text: str = "", action_func = None) -> None:
    """
    Display a clean empty state component.
    
    Args:
        icon: Empty state icon
        title: Empty state title
        description: Empty state description
        action_text: Action button text
        action_func: Action button function
    """
    st.markdown(f"""
    <div style="text-align: center; padding: var(--spacing-xl) var(--spacing-lg); color: var(--text-secondary);">
        <div style="font-size: 3rem; margin-bottom: var(--spacing-md);">{icon}</div>
        <h3 style="margin: 0 0 0.5rem 0; color: var(--text-primary); font-size: 1.25rem; font-weight: 600;">{title}</h3>
        <p style="margin: 0; font-size: 0.95rem; line-height: 1.6;">{description}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if action_text and action_func:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(action_text, type="primary"):
                action_func()


def create_clean_content_area(content: str, title: str = "") -> str:
    """
    Create a clean content area.
    
    Args:
        content: Content to display
        title: Optional title
    
    Returns:
        HTML string for the clean content area
    """
    title_html = f'<h3 style="margin: 0 0 var(--spacing-sm) 0; color: var(--text-primary); font-weight: 600;">{title}</h3>' if title else ""
    
    return f"""
    <div class="content-area">
        {title_html}
        <div style="color: var(--text-secondary); line-height: 1.6;">
            {content}
        </div>
    </div>
    """


def create_clean_navigation(items: List[Dict[str, str]]) -> str:
    """
    Create a clean navigation component.
    
    Args:
        items: List of navigation items with 'text', 'icon', 'active' keys
    
    Returns:
        HTML string for the clean navigation
    """
    nav_items = []
    for item in items:
        active_class = "active" if item.get('active', False) else ""
        nav_items.append(f"""
        <div class="nav-item {active_class}">
            {item.get('icon', '')} {item.get('text', '')}
        </div>
        """)
    
    return f"""
    <div style="display: flex; flex-direction: column; gap: 0.25rem; margin: var(--spacing-sm) 0;">
        {''.join(nav_items)}
    </div>
    """


def apply_clean_tab_style():
    """Apply clean tab styling."""
    st.markdown("""
    <style>
    .stTabs > div > div > div > div {
        background: var(--surface);
        border-radius: var(--border-radius);
        border: 1px solid var(--border-color);
        padding: 0.25rem;
        gap: 0.25rem;
    }
    
    .stTabs > div > div > div > div > div {
        position: relative;
        border-radius: 6px;
        transition: var(--transition);
        font-weight: 500;
        padding: 0.5rem 1rem;
        overflow: hidden;
    }
    
    .stTabs > div > div > div > div > div[aria-selected="true"] {
        background: var(--primary-color);
        color: white !important;
        box-shadow: var(--shadow-sm);
    }
    
    /* Ensure background fully covers text area */
    .stTabs > div > div > div > div > div[aria-selected="true"] * {
        color: white !important;
    }
    
    .stTabs > div > div > div > div > div:hover:not([aria-selected="true"]) {
        background: rgba(59, 130, 246, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)


# Initialize clean theme when imported
apply_clean_theme()
