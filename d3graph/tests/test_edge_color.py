#!/usr/bin/env python3
"""
Test script to verify that the edge_color parameter is working correctly.
"""

import pandas as pd
import numpy as np
from d3graph import d3graph, vec2adjmat

def test_edge_color():
    """Test that edge_color parameter is correctly inherited in HTML and JS."""
    
    # Initialize with default settings
    d3 = d3graph()
    
    # Load example data
    df = d3.import_example('stormofswords')
    
    # Convert df to adjmat
    adjmat = vec2adjmat(source=df['source'], target=df['target'], weight=df['weight'])
    # adjmat = np.exp(adjmat)
    # adjmat = adjmat-1
    
    # Create the network
    d3.graph(adjmat)
    
    # Set edge properties with custom edge_color
    d3.set_edge_properties(edge_color='#FF0000')  # Red edges
    
    # Set node properties
    d3.set_node_properties(color='cluster', size='degree')
    
    # Show the graph
    d3.show(showfig=False)
    
    print("Test completed. Check test_edge_color.html to verify edge colors are red (#FF0000).")

if __name__ == "__main__":
    test_edge_color() 