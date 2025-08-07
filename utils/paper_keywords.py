"""
Paper Keywords Configuration
This file contains keywords for different research areas and papers.
You can modify these keywords to improve search and categorization.
"""

# General research area keywords
RESEARCH_AREA_KEYWORDS = {
    "dislocation": [
        "dislocation", "dislocation motion", "dislocation density", "dislocation structure",
        "dislocation multiplication", "dislocation annihilation", "dislocation interaction",
        "dislocation pile-up", "dislocation network", "dislocation dynamics"
    ],
    "grainBoundary": [
        "grain boundary", "grain boundary strengthening", "grain boundary sliding",
        "grain boundary migration", "grain boundary character", "grain boundary energy",
        "grain boundary segregation", "grain boundary engineering", "grain boundary structure"
    ],
    "Precipitation": [
        "precipitation", "precipitation strengthening", "precipitate", "precipitation hardening",
        "precipitate formation", "precipitate growth", "precipitate coarsening",
        "precipitate morphology", "precipitate distribution", "precipitate evolution"
    ],
    "SSS": [
        "solid solution strengthening", "solid solution", "alloying elements",
        "substitutional atoms", "interstitial atoms", "solution hardening",
        "alloy strengthening", "composition effects", "elemental additions"
    ]
}

# Specific paper keywords (can be expanded based on your papers)
PAPER_SPECIFIC_KEYWORDS = {
    # Example structure - you can add specific papers here
    # "paper_filename.pdf": ["keyword1", "keyword2", "keyword3"],
    
    # Dislocation papers
    "baruffi2022screw.pdf": ["screw dislocation", "dislocation core", "dislocation structure"],
    "hu2025.pdf": ["dislocation motion", "dislocation dynamics", "plastic deformation"],
    "jiang20253d.pdf": ["3D dislocation", "dislocation network", "dislocation interaction"],
    
    # Grain boundary papers
    "dou2024.pdf": ["grain boundary", "grain boundary strengthening", "microstructure"],
    "ko2025.pdf": ["grain boundary engineering", "grain boundary character"],
    "lee2025bulging.pdf": ["grain boundary sliding", "creep deformation"],
    
    # Precipitation papers
    "dai2025.pdf": ["precipitation", "precipitate formation", "aging"],
    "jin2023high.pdf": ["high temperature", "precipitation strengthening"],
    "jin2024.pdf": ["precipitate evolution", "microstructure evolution"],
    
    # SSS papers
    "chen2025low.pdf": ["low temperature", "solid solution", "strengthening"],
    "dou2024modulus.pdf": ["elastic modulus", "mechanical properties"],
    "fang2024composition.pdf": ["composition", "alloying", "composition effects"]
}

# Default keywords for papers not in the specific list
DEFAULT_KEYWORDS = [
    "RHEA", "refractory high entropy alloy", "mechanical properties",
    "microstructure", "strengthening mechanisms", "materials science",
    "alloy design", "phase stability", "deformation mechanisms"
]

def get_paper_keywords(paper_filename: str, folder: str = None) -> list:
    """
    Get keywords for a specific paper.
    
    Args:
        paper_filename: Name of the paper file
        folder: Folder/category of the paper
    
    Returns:
        List of keywords for the paper
    """
    # First check for paper-specific keywords
    if paper_filename in PAPER_SPECIFIC_KEYWORDS:
        return PAPER_SPECIFIC_KEYWORDS[paper_filename]
    
    # If no specific keywords, use folder-based keywords
    if folder and folder in RESEARCH_AREA_KEYWORDS:
        return RESEARCH_AREA_KEYWORDS[folder]
    
    # Fallback to default keywords
    return DEFAULT_KEYWORDS

def get_folder_keywords(folder: str) -> list:
    """
    Get keywords for a specific research folder/area.
    
    Args:
        folder: Folder/category name
    
    Returns:
        List of keywords for the folder
    """
    return RESEARCH_AREA_KEYWORDS.get(folder, DEFAULT_KEYWORDS)

def add_paper_keywords(paper_filename: str, keywords: list):
    """
    Add or update keywords for a specific paper.
    
    Args:
        paper_filename: Name of the paper file
        keywords: List of keywords to add
    """
    PAPER_SPECIFIC_KEYWORDS[paper_filename] = keywords

def add_folder_keywords(folder: str, keywords: list):
    """
    Add or update keywords for a specific folder/area.
    
    Args:
        folder: Folder/category name
        keywords: List of keywords to add
    """
    RESEARCH_AREA_KEYWORDS[folder] = keywords
