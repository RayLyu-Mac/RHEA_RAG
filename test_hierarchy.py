#!/usr/bin/env python3
"""
Test script to verify the hierarchical paper structure is working correctly.
"""

import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("🔍 Testing hierarchical paper structure...")
    
    # Test imports
    from utils.data_utils import get_folder_config, get_paper_hierarchy, get_all_folder_icons
    print("✅ All imports successful")
    
    # Test folder configuration
    main_folders, main_folder_icons, folder_hierarchy = get_folder_config()
    print(f"📁 Main folders: {main_folders}")
    print(f"📁 Folder hierarchy: {folder_hierarchy}")
    print(f"📁 Main folder icons: {main_folder_icons}")
    
    # Test paper hierarchy with sample data
    sample_papers = [
        {'file_name': 'paper1.pdf', 'folder': 'SSS'},
        {'file_name': 'paper2.pdf', 'folder': 'dislocation'},
        {'file_name': 'paper3.pdf', 'folder': 'grainBoundary'},
        {'file_name': 'paper4.pdf', 'folder': 'Precipitation'},
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
    
    print("\n🎉 Hierarchical structure test passed!")
    print("\n📋 Expected behavior:")
    print("   • Main folder: 🔬 RHEA_Strengthening")
    print("   • Subfolders: SSS, dislocation, grainBoundary, Precipitation")
    print("   • Papers organized under appropriate subfolders")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
