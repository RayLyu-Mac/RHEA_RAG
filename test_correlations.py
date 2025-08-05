#!/usr/bin/env python3
"""
Test script for the Paper Correlation Management System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.paper_correlations import (
    PaperCorrelation, ResearchTopic, PaperCorrelationManager,
    initialize_sss_correlations
)

def test_basic_functionality():
    """Test basic functionality of the correlation system."""
    print("🧪 Testing Paper Correlation Management System")
    print("=" * 50)
    
    # Initialize SSS correlations
    manager = initialize_sss_correlations()
    
    # Test basic properties
    print(f"✅ Topics created: {len(manager.topics)}")
    print(f"✅ SSS topic papers: {len(manager.get_topic_papers('Solid Solution Strengthening (SSS)'))}")
    print(f"✅ SSS correlations: {len(manager.get_topic_correlations('Solid Solution Strengthening (SSS)'))}")
    
    # Test search functionality
    correlations = manager.search_correlations("fang2024composition")
    print(f"✅ Correlations for fang2024composition: {len(correlations)}")
    
    # Test network creation
    G = manager.create_network_graph("Solid Solution Strengthening (SSS)")
    print(f"✅ Network nodes: {G.number_of_nodes()}")
    print(f"✅ Network edges: {G.number_of_edges()}")
    
    # Test DataFrame export
    df = manager.export_to_dataframe("Solid Solution Strengthening (SSS)")
    print(f"✅ DataFrame shape: {df.shape}")
    
    # Display some sample data
    print("\n📊 Sample Correlations:")
    print(df.head().to_string())
    
    print("\n🎯 Test completed successfully!")

def test_add_new_correlation():
    """Test adding a new correlation."""
    print("\n🧪 Testing Add New Correlation")
    print("=" * 30)
    
    manager = PaperCorrelationManager()
    
    # Add a new correlation
    new_corr = PaperCorrelation(
        source="test_paper_1",
        target="test_paper_2", 
        relationship_type="test_relationship",
        description="Test correlation for demonstration",
        strength=0.7
    )
    
    # Add test papers to topic
    manager.add_paper_to_topic("Solid Solution Strengthening (SSS)", "test_paper_1")
    manager.add_paper_to_topic("Solid Solution Strengthening (SSS)", "test_paper_2")
    
    # Add correlation
    success = manager.add_correlation("Solid Solution Strengthening (SSS)", new_corr)
    print(f"✅ Added new correlation: {success}")
    
    # Verify it was added
    correlations = manager.get_topic_correlations("Solid Solution Strengthening (SSS)")
    test_correlations = [c for c in correlations if c.source == "test_paper_1"]
    print(f"✅ Found test correlation: {len(test_correlations) > 0}")

if __name__ == "__main__":
    test_basic_functionality()
    test_add_new_correlation() 