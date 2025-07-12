#!/usr/bin/env python3
"""
Test script to verify that the edge_color parameter is working correctly.
"""

import pandas as pd
import numpy as np
from d3graph import d3graph

def test_edge_color():
    """Test that edge_color parameter is correctly inherited in HTML and JS."""
    
    # Create a simple adjacency matrix
    adjmat = pd.DataFrame([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]
    ], index=['A', 'B', 'C', 'D'], columns=['A', 'B', 'C', 'D'])
    
    # Initialize d3graph
    d3 = d3graph()
    
    # Set edge properties with custom edge_color
    d3.set_edge_properties(edge_color='#FF0000')  # Red edges
    
    # Set node properties
    d3.set_node_properties(color='cluster', size='degree')
    
    # Create the graph
    d3.graph(adjmat)
    
    # Show the graph
    d3.show(filepath='test_edge_color.html', showfig=False)
    
    print("Test completed. Check test_edge_color.html to verify edge colors are red (#FF0000).")

if __name__ == "__main__":
    test_edge_color() 