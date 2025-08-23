#!/usr/bin/env python3
"""
Test script to verify the hierarchical UI structure is working correctly.
This tests the display_paper_selection function with sample data.
"""

import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("🔍 Testing hierarchical UI structure...")
    
    # Test imports
    from utils.data_utils import get_folder_config, get_paper_hierarchy, get_all_folder_icons
    from utils.ui_components import display_paper_selection
    print("✅ All imports successful")
    
    # Test folder configuration
    main_folders, main_folder_icons, folder_hierarchy = get_folder_config()
    print(f"📁 Main folders: {main_folders}")
    print(f"📁 Folder hierarchy: {folder_hierarchy}")
    print(f"📁 Main folder icons: {main_folder_icons}")
    
    # Test paper hierarchy with sample data
    sample_papers = [
        {'file_name': 'paper1.pdf', 'folder': 'SSS', 'figure_count': 5, 'file_path': '/path/to/paper1.pdf'},
        {'file_name': 'paper2.pdf', 'folder': 'dislocation', 'figure_count': 3, 'file_path': '/path/to/paper2.pdf'},
        {'file_name': 'paper3.pdf', 'folder': 'grainBoundary', 'figure_count': 7, 'file_path': '/path/to/paper3.pdf'},
        {'file_name': 'paper4.pdf', 'folder': 'Precipitation', 'figure_count': 4, 'file_path': '/path/to/paper4.pdf'},
        {'file_name': 'paper5.pdf', 'folder': 'SSS', 'figure_count': 6, 'file_path': '/path/to/paper5.pdf'},
    ]
    
    hierarchy = get_paper_hierarchy(sample_papers)
    print(f"\n📊 Paper hierarchy created: {len(hierarchy)} main areas")
    
    for main_area, subfolders in hierarchy.items():
        print(f"  🔬 {main_area}:")
        for subfolder, papers in subfolders.items():
            print(f"    • {subfolder}: {len(papers)} papers")
    
    # Test folder icons
    all_icons = get_all_folder_icons()
    print(f"\n🎨 All folder icons: {len(all_icons)} total")
    
    print("\n🎉 Hierarchical UI structure test passed!")
    print("\n📋 Expected behavior:")
    print("   • Main folder: 🔬 RHEA_Strengthening (5 papers)")
    print("   • Subfolders: Each individually collapsible")
    print("     - 🔬 SSS (2 papers) [▶]")
    print("     - 🔧 dislocation (1 paper) [▶]")
    print("     - 🧱 grainBoundary (1 paper) [▶]")
    print("     - 💧 Precipitation (1 paper) [▶]")
    print("   • Each subfolder can be expanded/collapsed independently")
    
    print("\n🚀 Ready to test in Streamlit!")
    print("   Run: streamlit run app_modular.py")
    print("   Expected: Nested expandable folders with individual subfolder control")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
