"""
Utils package for the Paper Search & QA System.
"""

from .vector_store import load_vectorstore, search_papers, get_paper_abstract_and_keywords
# from .llm_utils import load_llm, get_available_ollama_models, optimize_question, get_suggested_keywords, generate_answer
from .direct_ollama_utils import (
    load_direct_llm_v2 as load_llm, load_direct_embeddings, get_available_ollama_models,
    optimize_question, generate_answer, get_suggested_keywords, count_tokens,
    DirectOllamaConnection, DirectOllamaLLM, DirectOllamaEmbeddings
)
from .data_utils import load_paper_list, get_paper_figures, group_papers_by_folder, get_folder_config, get_all_folder_icons, get_paper_hierarchy, add_research_area, get_research_area_info, list_all_research_areas, display_image_safely, get_paper_stats
from .ui_components import apply_theme_css, create_glass_card, create_content_card, create_optimize_card, display_theme_toggle, display_paper_selection, display_keyword_selection
from .notes_utils import load_meeting_notes, save_meeting_notes, add_note_to_vectorstore, display_notes_section, ask_question_about_notes, sync_selected_notes_to_vectorstore, display_selective_sync_interface
from .paper_network_viz import plot_mechanism_network_interactive, render_dot_flowchart
from .paper_correlations import (
    PaperCorrelation, ResearchTopic, PaperCorrelationManager,
    initialize_sss_correlations, display_correlation_interface
)
from .scholar_scraper_tab import display_scholar_scraper_tab, search_scholar_followup
from .prompts import (
    PromptTemplates, format_prompt, add_summary_requirement,
    get_question_optimization_prompt, get_answer_generation_prompt,
    get_research_gap_prompt, get_follow_up_prompt, get_meeting_notes_prompt,
    get_paper_grouping_prompt, get_rag_flowchart_prompt,
    get_llm_grouping_refinement_prompt, get_scholar_summary_prompt
)

__all__ = [
    # Vector store utilities
    'load_vectorstore',
    'search_papers', 
    'get_paper_abstract_and_keywords',
    
    # LLM utilities
    'load_llm',
    'get_available_ollama_models',
    'optimize_question',
    'get_suggested_keywords',
    'generate_answer',
    
    # Data utilities
    'load_paper_list',
    'get_paper_figures',
    'group_papers_by_folder',
    'get_folder_config',
    'get_all_folder_icons',
    'get_paper_hierarchy',
    'add_research_area',
    'get_research_area_info',
    'list_all_research_areas',
    'display_image_safely',
    'get_paper_stats',
    
    # UI components
    'apply_theme_css',
    'create_glass_card',
    'create_content_card', 
    'create_optimize_card',
    'display_theme_toggle',
    'display_paper_selection',
    'display_keyword_selection',
    
    # Notes utilities
    'load_meeting_notes',
    'save_meeting_notes',
    'add_note_to_vectorstore',
    'display_notes_section',
    'ask_question_about_notes',
    'sync_selected_notes_to_vectorstore',
    'display_selective_sync_interface',
    
    # Paper network visualization
    'plot_mechanism_network_interactive',
    'render_dot_flowchart',
    
    # Paper correlations
    'PaperCorrelation',
    'ResearchTopic', 
    'PaperCorrelationManager',
    'initialize_sss_correlations',
    'display_correlation_interface',
    
    # Scholar scraper
    'display_scholar_scraper_tab',
    'search_scholar_followup',
    
    # Prompt management
    'PromptTemplates',
    'format_prompt',
    'add_summary_requirement',
    'get_question_optimization_prompt',
    'get_answer_generation_prompt',
    'get_research_gap_prompt',
    'get_follow_up_prompt',
    'get_meeting_notes_prompt',
    'get_paper_grouping_prompt',
    'get_rag_flowchart_prompt',
    'get_llm_grouping_refinement_prompt',
    'get_scholar_summary_prompt'
] 