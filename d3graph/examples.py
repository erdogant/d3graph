# %% Libraries
import networkx as nx
import pandas as pd
import numpy as np
from d3graph import d3graph, vec2adjmat

# %%
from d3graph import d3graph

d3 = d3graph()
# Load example
adjmat = d3.import_example('small')

d3.graph(adjmat)
d3.show()

d3.set_node_properties(color=adjmat.columns.values, label=['node AA','node BB','node FF','node JJ','node MM','node CC','node ZZ'])
d3.show()



# %%


# Make some graphs
d3 = d3graph()
# Load example
adjmat, df = d3.import_example('karate')
d3.graph(adjmat)
d3.set_node_properties(color=df['label'].values, cmap='Set1')
d3.show()

d3.set_node_properties(label=df['label'].values, color=df['label'].values, cmap='Set1')
d3.show()


# %%
from d3graph import d3graph
size = [10, 20, 10, 10, 15, 10, 5]

# Example A: simple interactive network
d3 = d3graph()

# Load example
adjmat = d3.import_example('bigbang')

d3.graph(adjmat)
d3.show()

# Example B: Color nodes
d3.set_node_properties(color=adjmat.columns.values)
d3.show()

# Example C: include node size
d3.set_node_properties(color=adjmat.columns.values, size=size)
d3.show()

# Example D: include node-edge-size
d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1])
d3.show()

# Example E: include node-edge color
d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1], edge_color='#00FFFF')
d3.show()

# Example F: Change colormap
d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1], edge_color='#00FFFF', cmap='Set2')
d3.show()

# Example H: Include directed links. Arrows are set from source -> target
d3.set_edge_properties(directed=True)
d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size, edge_color='#000FFF', cmap='Set1')
d3.show()


# %%
from d3graph import d3graph, vec2adjmat
d3 = d3graph()
# Load example
adjmat = d3.import_example('small')

d3.graph(adjmat)
d3.set_node_properties(color=adjmat.columns.values)
d3.show()

# %% Extended example
from d3graph import d3graph

# Initialize
d3 = d3graph()
# Load karate example
adjmat, df = d3.import_example('karate')

label = df['label'].values
node_size = df['degree'].values

d3.graph(adjmat)
d3.set_node_properties(color=df['label'].values)
d3.show()

d3.set_node_properties(label=label, color=label, cmap='Set1')
d3.show()

d3.set_node_properties(size=node_size)
d3.show()

d3.set_node_properties(color=label, size=node_size)
d3.show()

d3.set_edge_properties(edge_distance=100)
d3.set_node_properties(color=node_size, size=node_size)
d3.show()

d3 = d3graph(charge=1000)
d3.graph(adjmat)
d3.set_node_properties(color=node_size, size=node_size)
d3.show()

d3 = d3graph(collision=1, charge=250)
d3.graph(adjmat)
d3.set_node_properties(color=label, size=node_size, edge_size=node_size, cmap='Set1')
d3.show()

d3 = d3graph(collision=1, charge=250)
d3.graph(adjmat)
d3.set_node_properties(color=label, size=node_size, edge_size=node_size, edge_color='#00FFFF', cmap='Set1')
d3.show()


# %% default example
from d3graph import d3graph

# Initialize
d3 = d3graph()

# Load karate example
adjmat, df = d3.import_example('karate')

# Initialize
d3.graph(adjmat)

# Node properties
d3.set_node_properties(label=df['label'].values, color=df['label'].values, size=df['degree'].values, edge_size=df['degree'].values, cmap='Set1')

# Plot
d3.show()


# %% Collision example
from d3graph import d3graph

# Initialize
d3 = d3graph(charge=1000)

# Load karate example
adjmat, df = d3.import_example('karate')

# Initialize
d3.graph(adjmat)

# Plot
d3.show(filepath='D://REPOS//erdogant.github.io//docs//d3graph//d3graph//karate_charge.html', figsize=(800, 600))

