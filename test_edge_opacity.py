#!/usr/bin/env python3
"""
Test script to verify that the edge_opacity parameter is working correctly.
"""

import pandas as pd
import numpy as np
from d3graph import d3graph

def test_edge_opacity():
    """Test that edge_opacity parameter is correctly inherited in HTML and JS."""
    
    # Create a simple adjacency matrix
    adjmat = pd.DataFrame([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]
    ], index=['A', 'B', 'C', 'D'], columns=['A', 'B', 'C', 'D'])
    
    # Initialize d3graph
    d3 = d3graph()
    
    # Set edge properties with custom edge_opacity
    d3.set_edge_properties(edge_color='#FF0000', edge_opacity=0.5)  # Red edges with 50% opacity
    
    # Set node properties
    d3.set_node_properties(color='cluster', size='degree')
    
    # Create the graph
    d3.graph(adjmat)
    
    # Show the graph
    d3.show(filepath='test_edge_opacity.html', showfig=False)
    
    print("Test completed. Check test_edge_opacity.html to verify edge opacity is 0.5 (50% transparent).")

if __name__ == "__main__":
    test_edge_opacity() 