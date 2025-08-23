#!/usr/bin/env python3
"""
Simple test script to verify that all hierarchical functions can be imported correctly.
"""

try:
    print("🔍 Testing imports...")
    
    # Test basic imports
    from utils import get_folder_config, get_all_folder_icons, get_paper_hierarchy
    print("✅ Basic imports successful")
    
    # Test helper function imports
    from utils import add_research_area, get_research_area_info, list_all_research_areas
    print("✅ Helper function imports successful")
    
    # Test function calls
    print("\n🔬 Testing function calls...")
    
    # Test get_folder_config
    main_folders, main_folder_icons, folder_hierarchy = get_folder_config()
    print(f"✅ get_folder_config: {len(main_folders)} main folders")
    
    # Test get_all_folder_icons
    all_icons = get_all_folder_icons()
    print(f"✅ get_all_folder_icons: {len(all_icons)} icons")
    
    # Test get_paper_hierarchy with empty list
    empty_hierarchy = get_paper_hierarchy([])
    print(f"✅ get_paper_hierarchy: {len(empty_hierarchy)} main areas")
    
    # Test add_research_area
    add_research_area("Test_Area", ["Sub1", "Sub2"], "🧪")
    print("✅ add_research_area: Test area added")
    
    # Test get_research_area_info
    info = get_research_area_info("Test_Area")
    if info:
        print(f"✅ get_research_area_info: {info['name']} with {info['total_subfolders']} subfolders")
    
    # Test list_all_research_areas
    areas = list_all_research_areas()
    print(f"✅ list_all_research_areas: {len(areas)} areas total")
    
    print("\n🎉 All tests passed! The hierarchical system is working correctly.")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("This means the function is not properly exported from the utils package.")
except Exception as e:
    print(f"❌ Error: {e}")
    print("This means there's an issue with the function implementation.")
