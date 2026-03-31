#!/usr/bin/env python3
"""
Test script to verify that the edge_opacity parameter is working correctly.
"""

import pandas as pd
import numpy as np
from d3graph import d3graph, vec2adjmat

def test_edge_opacity():
    """Test that edge_opacity parameter is correctly inherited in HTML and JS."""
    # Import library

    # Initialize with default settings
    d3 = d3graph(support=None)
    
    # Load example data
    df = d3.import_example('stormofswords')
    
    # Convert df to adjmat
    adjmat = vec2adjmat(source=df['source'], target=df['target'], weight=df['weight'])
    # adjmat = np.exp(adjmat)
    # adjmat = adjmat-1
    
    # Create the network
    d3.graph(adjmat)
    
    # Set Node properties
    d3.set_node_properties(edge_color='#000000', cmap='Blues', minmax=[5, 13], fontcolor='#808080')
    
    # Set edge properties
    d3.set_edge_properties(directed=True, marker_end='arrow')
    d3.show(showfig=False)
    
    # Set edge properties with custom edge_opacity
    d3.set_edge_properties(edge_color='#FF0000', edge_opacity=0.5)  # Red edges with 50% opacity
    
    # Set node properties
    d3.set_node_properties(color='cluster', size='degree')

    # Show the graph
    d3.show()
    
    print("Test completed. Check test_edge_opacity.html to verify edge opacity is 0.5 (50% transparent).")

if __name__ == "__main__":
    test_edge_opacity() 