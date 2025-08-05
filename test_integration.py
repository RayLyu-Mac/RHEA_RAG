#!/usr/bin/env python3
"""
Test script for the integrated Paper Correlation Management System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.paper_correlations import (
    PaperCorrelation, ResearchTopic, PaperCorrelationManager,
    initialize_sss_correlations
)

def test_integration():
    """Test the integration of correlation system with search functionality."""
    print("🧪 Testing Integrated Paper Correlation System")
    print("=" * 60)
    
    # Initialize SSS correlations
    manager = initialize_sss_correlations()
    
    # Test 1: Basic functionality
    print("✅ Test 1: Basic functionality")
    print(f"   Topics: {len(manager.topics)}")
    print(f"   SSS papers: {len(manager.get_topic_papers('Solid Solution Strengthening (SSS)'))}")
    print(f"   SSS correlations: {len(manager.get_topic_correlations('Solid Solution Strengthening (SSS)'))}")
    
    # Test 2: Search correlations for specific papers
    print("\n✅ Test 2: Search correlations")
    test_papers = ["fang2024composition", "zhou2025strategies", "wang2025hf"]
    for paper in test_papers:
        correlations = manager.search_correlations(paper)
        print(f"   {paper}: {len(correlations)} correlations found")
    
    # Test 3: Network creation
    print("\n✅ Test 3: Network creation")
    G = manager.create_network_graph("Solid Solution Strengthening (SSS)")
    print(f"   Network nodes: {G.number_of_nodes()}")
    print(f"   Network edges: {G.number_of_edges()}")
    
    # Test 4: DataFrame export
    print("\n✅ Test 4: DataFrame export")
    df = manager.export_to_dataframe("Solid Solution Strengthening (SSS)")
    print(f"   DataFrame shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Test 5: Simulate search context integration
    print("\n✅ Test 5: Search context integration simulation")
    # Simulate papers found in search results
    found_papers = ["fang2024composition", "zhou2025strategies"]
    
    # Find correlations for these papers
    all_correlations = []
    for paper in found_papers:
        correlations = manager.search_correlations(paper)
        all_correlations.extend(correlations)
    
    # Remove duplicates
    unique_correlations = {}
    for corr in all_correlations:
        key = f"{corr.source}->{corr.target}"
        if key not in unique_correlations:
            unique_correlations[key] = corr
    
    print(f"   Found {len(unique_correlations)} unique correlations for search papers")
    
    # Format correlation context (simulating what would be added to LLM prompt)
    correlation_context = "\n\n📊 **Paper Correlations Found:**\n"
    for corr in unique_correlations.values():
        correlation_context += f"• **{corr.source}** → **{corr.target}**: {corr.relationship_type} - {corr.description} (Strength: {corr.strength})\n"
    
    print(f"   Correlation context length: {len(correlation_context)} characters")
    print("   Sample correlation context:")
    print(correlation_context[:200] + "..." if len(correlation_context) > 200 else correlation_context)
    
    print("\n🎯 Integration test completed successfully!")
    print("\n📋 Summary:")
    print(f"   • {len(manager.topics)} research topics")
    print(f"   • {len(manager.get_all_papers())} total papers")
    print(f"   • {sum(len(topic.correlations) for topic in manager.topics.values())} total correlations")
    print(f"   • Network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

if __name__ == "__main__":
    test_integration() 