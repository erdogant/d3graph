# Example 15: Edge opacity and color demonstration
print("\n=== Example 15: Edge Opacity and Color ===")
import pandas as pd
import numpy as np
from d3graph import d3graph

# Create a simple network
adjmat = pd.DataFrame([
    [0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0]
], index=['A', 'B', 'C', 'D', 'E'], columns=['A', 'B', 'C', 'D', 'E'])

# Initialize d3graph
d3 = d3graph()

# Set edge properties with custom color and opacity
d3.set_edge_properties(
    edge_color='#FF0000',      # Red edges
    edge_opacity=0.6,          # 60% opacity (40% transparent)
    edge_weight=3,             # Thicker edges
    edge_style=0               # Solid lines
)

# Set node properties
d3.set_node_properties(
    color='cluster',           # Color by cluster
    size='degree',             # Size by degree
    edge_color='#000000'       # Black node borders
)

# Create the graph
d3.graph(adjmat)

# Show the graph
d3.show(filepath='example_edge_opacity.html', showfig=False)

print("Created example_edge_opacity.html with red edges at 60% opacity") 