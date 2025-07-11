# -*- coding: utf-8 -*-
"""
This file contains all code examples from the documentation (.rst files), organized by topic and section.
All examples are self-contained and runnable independently.
"""

#%% --- Examples.rst ---
from d3graph import d3graph
d3 = d3graph()
adjmat = d3.import_example('bigbang')
d3.graph(adjmat, cmap='Set1', scaler='minmax')
d3.show()

# Big Bang network: Node colors
d3.set_node_properties(color=adjmat.columns.values)
d3.show()

# Big Bang network: Node fontcolors
d3.set_node_properties(color='cluster', fontcolor='node_color')
d3.show()

# Big Bang network: Node sizes
size = [10, 20, 10, 10, 15, 10, 5]
d3.set_node_properties(color=adjmat.columns.values, size=size)
d3.show()

# Big Bang network: Edge sizes
d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1])
d3.show()

# Big Bang network: Edge colors
d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1], edge_color='#00FFFF')
d3.show()

# Big Bang network: Colormap
d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1], edge_color='#00FFFF', cmap='Set2')
d3.show()

# Big Bang network: Directed arrows
d3.set_edge_properties(directed=True)
d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size, edge_color='#000FFF', cmap='Set1')
d3.show()



#%% --- Examples.rst ---
# Big Bang network: Default
from d3graph import d3graph
d3 = d3graph()
adjmat = d3.import_example('bigbang')
d3.graph(adjmat, cmap='Set1', scaler='minmax')
d3.show()

# Big Bang network: Node colors
from d3graph import d3graph
size = [10, 20, 10, 10, 15, 10, 5]
d3 = d3graph()
adjmat = d3.import_example('bigbang')
d3.graph(adjmat)
d3.set_node_properties(color=adjmat.columns.values)
d3.show()

# Big Bang network: Node fontcolors
from d3graph import d3graph
size = [10, 20, 10, 10, 15, 10, 5]
d3 = d3graph()
adjmat = d3.import_example('bigbang')
d3.graph(adjmat)
d3.set_node_properties(color='cluster', fontcolor='node_color')
d3.show()

# Big Bang network: Node sizes
from d3graph import d3graph
size = [10, 20, 10, 10, 15, 10, 5]
d3 = d3graph()
adjmat = d3.import_example('bigbang')
d3.graph(adjmat)
d3.set_node_properties(color=adjmat.columns.values, size=size)
d3.show()

# Big Bang network: Edge sizes
from d3graph import d3graph
size = [10, 20, 10, 10, 15, 10, 5]
d3 = d3graph()
adjmat = d3.import_example('bigbang')
d3.graph(adjmat)
d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1])
d3.show()

# Big Bang network: Edge colors
from d3graph import d3graph
size = [10, 20, 10, 10, 15, 10, 5]
d3 = d3graph()
adjmat = d3.import_example('bigbang')
d3.graph(adjmat)
d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1], edge_color='#00FFFF')
d3.show()

# Big Bang network: Colormap
from d3graph import d3graph
size = [10, 20, 10, 10, 15, 10, 5]
d3 = d3graph()
adjmat = d3.import_example('bigbang')
d3.graph(adjmat)
d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1], edge_color='#00FFFF', cmap='Set2')
d3.show()

# Big Bang network: Directed arrows
from d3graph import d3graph
size = [10, 20, 10, 10, 15, 10, 5]
d3 = d3graph()
adjmat = d3.import_example('bigbang')
d3.graph(adjmat)
d3.set_edge_properties(directed=True)
d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size, edge_color='#000FFF', cmap='Set1')
d3.show()

#%%  energy network: Energy 1
from d3graph import d3graph, vec2adjmat
d3 = d3graph()
df = d3.import_example('energy')
adjmat = vec2adjmat(source=df['source'], target=df['target'], weight=df['weight'])
d3.graph(adjmat)
d3.show(filepath=r'D:\REPOS\erdogant.github.io\docs\d3graph\d3graph/energy_1.html')

# energy network: Energy 2
from d3graph import d3graph, vec2adjmat
d3 = d3graph()
df = d3.import_example('energy')
adjmat = vec2adjmat(source=df['source'], target=df['target'], weight=df['weight'])
d3.graph(adjmat)
d3.set_node_properties(scaler='minmax', color=None)
d3.node_properties['Solar']['size']=30
d3.node_properties['Solar']['color']='#FF0000'
d3.node_properties['Solar']['edge_color']='#000000'
d3.node_properties['Solar']['edge_size']=5
d3.show()

#%% energy network: Energy 3
from d3graph import d3graph, vec2adjmat
d3 = d3graph()
df = d3.import_example('energy')
adjmat = vec2adjmat(source=df['source'], target=df['target'], weight=df['weight'])
d3.graph(adjmat)
d3.set_edge_properties(directed=True, marker_end='arrow')
d3.show()

#%% --- Node properties.rst ---
# Node label example
from d3graph import d3graph
d3 = d3graph()
adjmat, df = d3.import_example('karate')
d3.graph(adjmat)
d3.set_node_properties(label=df['label'].values)
d3.show()

# Tooltips example
from d3graph import d3graph
d3 = d3graph()
adjmat, df = d3.import_example('karate')
d3.graph(adjmat)
tooltip = '\nId: ' + adjmat.columns.astype(str) +'\nDegree: ' + df['degree'].astype(str) + '\nLabel: ' + df['label'].values
tooltip = tooltip.values
label = df['label'].values
d3.set_node_properties(label=label, tooltip=tooltip, color=label, size='degree')
d3.show()
d3.set_node_properties(label=label, tooltip=tooltip, color=label, size='degree', minmax=[1, 20])
d3.show()

#%% Example: Star Network (central hub with spokes)
# Demonstrates: Custom node sizes/colors, tooltips, and a central hub
from d3graph import d3graph
import numpy as np
import colourmap as cm
hub = 'Center'
spokes = [f'Leaf{i}' for i in range(1, 9)]
source = [hub] * len(spokes)
target = spokes
weight = np.random.uniform(1, 5, size=len(spokes))
size = [30] + [10]*len(spokes)
labels = [hub] + spokes

from d3graph import vec2adjmat
adjmat = vec2adjmat(source, target, weight=weight)
d3 = d3graph()
d3.graph(adjmat)
colors = cm.fromlist(labels, scheme='hex')[0]
d3.set_node_properties(size=size, color=colors, label=labels, tooltip=labels)
d3.show()

#%% Example: Ring Network with Clusters
# Demonstrates: Clustering, color maps, and circular structure
from d3graph import d3graph
import numpy as np
N = 12
nodes = [f'N{i}' for i in range(N)]
source = nodes
# Connect in a ring
target = nodes[1:] + nodes[:1]
weight = np.random.uniform(1, 3, size=N)
adjmat = vec2adjmat(source, target, weight=weight)
d3 = d3graph()
d3.graph(adjmat)
d3.set_node_properties(color='cluster', cmap='tab10', label=nodes)
d3.show()

#%% Example: Weighted Grid Network
# Demonstrates: Grid structure, edge weights, and node opacity
from d3graph import d3graph
import numpy as np
rows, cols = 4, 4
nodes = [f'G{r}_{c}' for r in range(rows) for c in range(cols)]
source, target, weight = [], [], []
for r in range(rows):
    for c in range(cols):
        idx = r * cols + c
        if c < cols - 1:
            source.append(nodes[idx])
            target.append(nodes[idx+1])
            weight.append(np.random.uniform(1, 3))
        if r < rows - 1:
            source.append(nodes[idx])
            target.append(nodes[idx+cols])
            weight.append(np.random.uniform(1, 3))
adjmat = vec2adjmat(source, target, weight=weight)
d3 = d3graph()
d3.graph(adjmat)
d3.set_node_properties(opacity='degree', size='degree', color='cluster', label=nodes)
d3.show()

