# Import library
from d3graph import d3graph, vec2adjmat

# Create example network
source = ['node A','node F','node B','node B','node B','node A','node C','node Z']
target = ['node F','node B','node J','node F','node F','node M','node M','node A']
weight = [5.56, 0.5, 0.64, 0.23, 0.9, 3.28, 0.5, 0.45]
# Convert to adjacency matrix
adjmat = vec2adjmat(source, target, weight=weight)

# Initialize
d3 = d3graph()
# Proces adjmat
d3.graph(adjmat)
# Plot
d3.show()

# Make changes in node properties
d3.set_node_properties(color=adjmat.columns.values, label=['node 1','node 2','node 3','node 4','node 5','node 6','node 7'])
# Plot
d3.show(filepath='c://temp/')

# %% DATA
# https://erdogant.github.io/d3graph/pages/html/Data.html

# Import library
from d3graph import d3graph
# Initialize
d3 = d3graph()
# Load example
adjmat = d3.import_example('bigbang')

print(adjmat)

# Import library
from d3graph import d3graph, vec2adjmat

# Source node names
source = ['Penny', 'Penny', 'Amy', 'Bernadette', 'Bernadette', 'Sheldon', 'Sheldon', 'Sheldon', 'Rajesh']
# Target node names
target = ['Leonard', 'Amy', 'Bernadette', 'Rajesh', 'Howard', 'Howard', 'Leonard', 'Amy', 'Penny']
# Edge Weights
weight = [5, 3, 2, 2, 5, 2, 3, 5, 2]

# Convert the vector into a adjacency matrix
adjmat = vec2adjmat(source, target, weight=weight)

# Initialize
d3 = d3graph()
d3.graph(adjmat)
d3.show(figsize=(500, 400), filepath=r'D:\REPOS\erdogant.github.io\docs\d3blocks\d3graph_data_1.html', overwrite=False)

# %% DATA 2
# Import library
from d3graph import d3graph, vec2adjmat

# Initialize
d3 = d3graph()

# Load example
adjmat = d3.import_example('bigbang')

html = d3.graph(adjmat, html=True)

# Write to specified directory with custom filename
d3.show(figsize=(500, 400), filepath=r'D:\REPOS\erdogant.github.io\docs\d3blocks\d3graph_data_2.html', overwrite=False)

# %%
from d3graph import d3graph

# Initialization with default parameters
d3 = d3graph()
# Load karate example
adjmat, df = d3.import_example('karate')
# Process the adjacency matrix
d3.graph(adjmat)
# Node properties
d3.set_node_properties(label=df['label'].values, tooltip=df['label'].values, color='cluster')
# Plot
d3.show(figsize=(500, 600), filepath=r'D:\REPOS\erdogant.github.io\docs\d3blocks\karate_default.html', overwrite=False)

# %%
# Import library
from d3graph import d3graph
# Initialization
d3 = d3graph()
# Load karate example
adjmat, df = d3.import_example('karate')
# Process the adjacency matrix
d3.graph(adjmat)

# Set node properties
tooltip = '\nId: ' + adjmat.columns.astype(str) +'\nDegree: ' + df['degree'].astype(str) + '\nLabel: ' + df['label'].values
tooltip = tooltip.values
label = df['label'].values

# Set node properties
# d3.set_node_properties(label=label, tooltip=tooltip, color=label)
# d3.show(figsize=(500, 600), filepath=r'D:\REPOS\erdogant.github.io\docs\d3blocks\karate_label_hover.html', overwrite=False)

# If you want thinner lines
d3.set_node_properties(label=label, tooltip=tooltip, color=label, minmax=[0.1, 25])
d3.show(figsize=(500, 600), filepath=r'D:\REPOS\erdogant.github.io\docs\d3blocks\karate_label_hover.html', overwrite=True)

# %%
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

# %%