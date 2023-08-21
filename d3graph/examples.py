# %% Libraries
import networkx as nx
import pandas as pd
import numpy as np
from d3graph import d3graph, adjmat2vec, vec2adjmat

# %% opacity
from d3graph import d3graph, adjmat2vec
# intialize to load example dataset
d3 = d3graph()
# 
df = d3.import_example(data='energy')
df=vec2adjmat(source=df['source'], target=df['target'], weight=df['weight'])
# d3.graph(df, size='degree', opacity='degree', color='cluster', scaler='zscore')
d3.graph(df)

# d3.set_node_properties(opacity='centrality')
d3.show(filepath='c://temp/network.html')


# %% Edge distance

# Import
from d3graph import d3graph
# intialize to load example dataset
d3 = d3graph(support=False)
adjmat = d3.import_example('bigbang')

# Initialize with clustering colors
d3.graph(adjmat, color='cluster')

# Set all edge labels to "test"
d3.set_edge_properties(directed=True, minmax_distance=None)
d3.show(filepath=r'c:\temp\\d3graph\edge_labels_1.html')
d3.set_edge_properties(directed=True)
d3.show(filepath=r'c:\temp\\d3graph\edge_labels_3.html')
d3.set_edge_properties(directed=True, minmax_distance=[20, 200])
d3.show(filepath=r'c:\temp\\d3graph\edge_labels_3.html')
d3.set_edge_properties

d3.set_edge_properties(directed=True, label=None)
d3.show(filepath=r'c:\temp\\d3graph\edge_labels_2.html')

# Set edge labels 
d3.edge_properties

# We will first set all label properties to None and then we will adjust two of them
d3.set_edge_properties(directed=True, marker_color='#000FFF', label=None)
d3.edge_properties['Amy', 'Bernadette']['weight_scaled']=10
d3.edge_properties['Amy', 'Bernadette']['label']='amy-bern'
d3.edge_properties['Amy', 'Bernadette']['label_color']='#000FFF'
d3.edge_properties['Amy', 'Bernadette']['label_fontsize']=8
d3.edge_properties['Bernadette', 'Howard']['label']='bern-how'
d3.edge_properties['Bernadette', 'Howard']['label_fontsize']=20
d3.edge_properties['Bernadette', 'Howard']['label_color']='#000000'

d3.show(filepath=r'c:\temp\\d3graph\edge_labels_2.html')

# %% Edge link
d3 = d3graph()
adjmat, df = d3.import_example('karate')
d3.graph(adjmat, color='cluster')
d3.show(filepath=r'c:\temp\\d3graph\d3graph1.html')

# %% Edge link

# Import
from d3graph import d3graph
# intialize to load example dataset
d3 = d3graph()
adjmat = d3.import_example('bigbang')

# Initialize with clustering colors
d3.graph(adjmat, color='cluster')

# Set all edge labels to "test"
d3.set_edge_properties(directed=True, label='test')
d3.show(filepath=r'c:\temp\\d3graph\edge_labels_1.html')

# Set edge labels 
d3.edge_properties

# We will first set all label properties to None and then we will adjust two of them
d3.set_edge_properties(directed=True, marker_color='#000FFF', label=None)
d3.edge_properties['Amy', 'Bernadette']['weight_scaled']=10
d3.edge_properties['Amy', 'Bernadette']['label']='amy-bern'
d3.edge_properties['Amy', 'Bernadette']['label_color']='#000FFF'
d3.edge_properties['Amy', 'Bernadette']['label_fontsize']=8
d3.edge_properties['Bernadette', 'Howard']['label']='bern-how'
d3.edge_properties['Bernadette', 'Howard']['label_fontsize']=20
d3.edge_properties['Bernadette', 'Howard']['label_color']='#000000'

d3.show(filepath=r'c:\temp\\d3graph\edge_labels_2.html')

# %% Change color of text
import numpy as np
# Import library
from d3graph import d3graph

# Initialize with defaults
d3 = d3graph()

# Load example
adjmat = d3.import_example('bigbang')
# Color on clustering
d3.graph(adjmat)
fontsize=np.random.randint(low=6, high=40, size=adjmat.shape[0])

# Set some node properties
d3.set_node_properties(color='cluster', scaler='minmax', fontcolor='node_color')

# Set the click properties: Create green node on click with black border
d3.show(filepath=r'c:\temp\\d3graph\click_example_1.html', click={'fill': '#00FF00', 'stroke': '#000000'})

# Keep the original color but set the stroke to grey and increase both node size and stroke width
d3.show(filepath=r'c:\temp\\d3graph\click_example_2.html', click={'fill': None, 'stroke': '#F0F0F0', 'size': 2.5, 'stroke-width': 10})


d3.set_node_properties(color='cluster', scaler='minmax', fontcolor='node_color', fontsize=fontsize)
d3.show(filepath=r'c:\temp\\d3graph\d3graph.html')

# %% 
from d3graph import d3graph
adjmat = d3.import_example('bigbang')
d3.graph(adjmat, color='cluster')

hexcolors = ['#000000', '#000000', '#000000', '#000FFF', '#000FFF', '#000FFF', '#000FFF']
colors = ['cluster', '#000FFF', hexcolors]
text_colors = ['node_color', 'cluster', '#000FFF', hexcolors]

for color in colors:
    for fontcolor in text_colors:
        print('-----------------')
        print('color: %s' %(str(color)))
        print('fontcolor: %s' %(str(fontcolor)))
        d3.set_node_properties(color=color, scaler='minmax', fontcolor=fontcolor)
        d3.show(filepath=r'c:\temp\\d3graph\d3graph.html')
        input('press enter')

# %%
from d3graph import d3graph

d3 = d3graph()
# Load example
adjmat, df = d3.import_example('karate')

d3.graph(adjmat, color='cluster')
d3.show(figsize=(None, None))

d3.set_node_properties(color='cluster', scaler='minmax')
d3.show(figsize=(1500, 1200))


# %% on click
from d3graph import d3graph
d3 = d3graph()
adjmat, df = d3.import_example('karate')
d3.graph(adjmat, color='cluster')
html = d3.show(click={'fill': 'green', 'stroke': 'black', 'size': 2, 'stroke-width': 2}, filepath=r'c:\temp\d3graph\d3graph.html')
html = d3.show(click=None, filepath=r'c:\temp\d3graph\d3graph.html')
html = d3.show(filepath=r'c:\temp\d3graph\d3graph.html')

# %% notebook examples
from d3graph import d3graph
d3 = d3graph()
adjmat, df = d3.import_example('karate')
d3.graph(adjmat, color='cluster')
html = d3.show(filepath=None, notebook=False)
assert html is not None
html = d3.show(filepath=None, notebook=True)
assert html is None
html = d3.show(filepath='test.html', notebook=False)
assert html is None

# %%
from d3graph import d3graph

d3 = d3graph()
# Load example
adjmat, df = d3.import_example('karate')

d3.graph(adjmat, color='cluster')
d3.show(figsize=(None, None))

d3.set_node_properties(color='cluster', scaler='minmax')
d3.show(figsize=(1500, 1200))


# %% small example
from d3graph import d3graph

# Initialize
d3 = d3graph()
# Load example
adjmat = d3.import_example('bigbang')
# Process adjmat
d3.graph(adjmat)
d3.show(filepath='c:\\temp\\network1.html', show_slider=True)


d3.set_edge_properties(directed=True, marker_end='square', marker_color='#000000')
d3.show(filepath='c:\\temp\\network1.html')



d3.set_edge_properties(directed=True, marker_end='', marker_color='#000000')
d3.edge_properties['Penny', 'Leonard']['marker_end']='arrow'
d3.edge_properties['Sheldon', 'Howard']['marker_end']='stub'
d3.edge_properties['Sheldon', 'Leonard']['marker_end']='circle'
d3.edge_properties['Rajesh', 'Penny']['marker_end']='square'
d3.edge_properties['Penny', 'Leonard']['marker_color']='#ff0000'
d3.show(filepath='c:\\temp\\network2.html')

d3.set_node_properties(color=adjmat.columns.values, size=[10, 20, 10, 10, 15, 10, 5])

d3.node_properties['Penny']['tooltip']='test\ntest2'
d3.show(filepath='c:\\temp\\network3.html')

# %% Checks with cluster label
from d3graph import d3graph

# Make some graphs
d3 = d3graph(charge=450)
# Load example
adjmat, df = d3.import_example('karate')
d3.graph(adjmat)
d3.set_node_properties(color=df['label'].values, cmap='Set1')
d3.show()

d3.set_node_properties(color='cluster', cmap='Set1')
d3.show(filepath='c:\\temp\\network1.html')

d3.set_node_properties(label=df['label'].values, tooltip=adjmat.columns.values, color=df['label'].values, cmap='Set1')
d3.show()


# %% Convert source-target to adjmat
from d3graph import d3graph, vec2adjmat

source = ['Penny', 'Penny', 'Amy']
target = ['Leonard', 'Amy', 'Bernadette']
adjmat = vec2adjmat(source, target)
d3 = d3graph()
print(d3.config)

d3.graph(adjmat)
# d3.show(showfig=True)

d3.set_edge_properties(directed=True)
d3.show(filepath='c:\\temp\\network1.html')


# %% Issue marker edges
# http://bl.ocks.org/dustinlarimer/5888271

from d3graph import d3graph
size = [10, 20, 10, 10, 15, 10, 5]

# Initialize
d3 = d3graph()
# Load example
adjmat = d3.import_example('bigbang')
# Process adjmat
d3.graph(adjmat)
d3.set_edge_properties(directed=True)
# Show
# d3.show(filepath='c:\\temp\\network.html')

d3.set_node_properties(color=adjmat.columns.values)
d3.show(filepath='c:\\temp\\network.html')

d3.set_node_properties(color=adjmat.columns.values, size=size)
d3.show()

d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1])
d3.set_edge_properties(directed=True, marker_end=None)
d3.show()

d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1], edge_color='#00FFFF')
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
d3.set_node_properties(label=df['label'].values, color=df['label'].values, edge_size=df['degree'].values, cmap='Set1')
d3.show()

d3.set_node_properties(label=df['label'].values, color='cluster', edge_size=df['degree'].values, cmap='Set1',
                       edge_color='cluster')
d3.show()

d3.set_node_properties(label=df['label'].values, color='cluster', size=df['degree'].values,
                       edge_size=df['degree'].values, cmap='Set1', edge_color='#000000', scaler='zscore',
                       minmax=[10, 50])
d3.show()

d3.set_node_properties(label=df['label'].values, color='cluster', size=df['degree'].values, cmap='Set1',
                       edge_color='#000000', scaler=None, minmax=[10, 50])
d3.show()

d3.set_node_properties(label=df['label'].values, edge_color='cluster', edge_size=df['degree'].values, cmap='Set1')
d3.show()

d3.set_node_properties(label=df['label'].values, edge_color='cluster', edge_size=4, cmap='Set1')
d3.show()

d3.set_node_properties(label=df['label'].values, color='#000000', edge_size=4, cmap='Set1', edge_color='cluster')

d3.set_edge_properties(scaler='zscore')
# d3.set_edge_properties(scaler=None)

# Plot
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

# %% Issue edge colors
from d3graph import d3graph

size = [10, 20, 10, 10, 15, 10, 5]

# Initialize
d3 = d3graph()
# Load example
adjmat = d3.import_example('bigbang')
# Process adjmat
d3.graph(adjmat)
# Show
# d3.show()

# d3.set_node_properties(color=adjmat.columns.values)
# d3.show()

# d3.set_node_properties(color=adjmat.columns.values, size=size)
# d3.show()

# d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1])
# d3.show()

# d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1], edge_color='#00FFFF')
# d3.show()

# d3.set_edge_properties(directed=True)
# d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size, edge_color='#000FFF', cmap='Set1')
# d3.set_node_properties(color='cluster', size=size, edge_size=size, edge_color='#000FFF', cmap='Set2')
d3.set_node_properties(color='cluster', size=size, edge_size=size, edge_color='cluster', cmap='Set2')
# d3.show()

d3.edge_properties['Penny', 'Leonard']['color'] = '#ff0000'
d3.edge_properties['Bernadette', 'Howard']['color'] = '#0000ff'
d3.show()

color, cluster_label, node_names = d3.get_cluster_color()


# %% Convert source-target to adjmat
from d3graph import d3graph, vec2adjmat

source = ['Penny', 'Penny', 'Amy', 'Bernadette', 'Bernadette', 'Sheldon', 'Sheldon', 'Sheldon', 'Rajesh']
target = ['Leonard', 'Amy', 'Bernadette', 'Rajesh', 'Howard', 'Howard', 'Leonard', 'Amy', 'Penny']
adjmat = vec2adjmat(source, target)
d3 = d3graph()
print(d3.config)

d3.graph(adjmat)
# d3.show(showfig=True)

d3.set_edge_properties(directed=True)
d3.show(showfig=True)

# %% TEST EXCEPTIONS FOR WRONG COLOR
from d3graph import d3graph

d3 = d3graph()
adjmat = d3.import_example('bigbang')
d3.graph(adjmat)

# Error expected:
d3.set_node_properties(color='',
                       label=adjmat.columns.values + ' are the names',
                       tooltip=['\nFemale\nMore info', 'Female', 'Male', 'Male', 'Female', 'Male', 'Male'])

# Error expected:
d3.set_node_properties(color=[], label=adjmat.columns.values + ' are the names',
                       tooltip=['\nFemale\nMore info', 'Female', 'Male', 'Male', 'Female', 'Male', 'Male'])

# Error expected:
d3.set_node_properties(color=['#000000', '#000000', '#000000', '#000', '#000000', '#000000', '#000000'],
                       label=adjmat.columns.values + ' are the names',
                       tooltip=['\nFemale\nMore info', 'Female', 'Male', 'Male', 'Female', 'Male', 'Male'])

# Error expected:
d3.set_node_properties(color=['#000000', '#000000', '#000000', '#000000', '#000000', '#000000'],
                       label=adjmat.columns.values + ' are the names',
                       tooltip=['\nFemale\nMore info', 'Female', 'Male', 'Male', 'Female', 'Male', 'Male'])

# Error expected:
d3.set_node_properties(color=['#000000', '#000000', '#00', '#000000', '#000000', '#000000'],
                       label=adjmat.columns.values + ' are the names',
                       tooltip=['\nFemale\nMore info', 'Female', 'Male', 'Male', 'Female', 'Male', 'Male'])

# %%
from d3graph import d3graph

d3 = d3graph()
adjmat = d3.import_example('bigbang')
d3.graph(adjmat)
d3.set_node_properties(label=adjmat.columns.values + ' are the names',
                       tooltip=['\nFemale\nMore info', 'Female', 'Male', 'Male', 'Female', 'Male', 'Male'])
d3.show()

# %% Convert source-target to adjmat
from d3graph import d3graph, vec2adjmat

source = ['Penny', 'Penny', 'Amy', 'Bernadette', 'Bernadette', 'Sheldon', 'Sheldon', 'Sheldon', 'Rajesh']
target = ['Leonard', 'Amy', 'Bernadette', 'Rajesh', 'Howard', 'Howard', 'Leonard', 'Amy', 'Penny']
adjmat = vec2adjmat(source, target)
d3 = d3graph()
print(d3.config)

# adjmat.iloc[0, 0] = 2
# adjmat.iloc[0, 1] = 3
# adjmat.iloc[0, 2] = 4
# adjmat.iloc[1, 3] = 12

d3.graph(adjmat)
# d3.show(showfig=True)

d3.set_edge_properties(directed=True, minmax=[5, 30])
d3.show(showfig=True)


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
d3.set_node_properties(color=adjmat.columns.values, size=size, scaler=None)
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
d3.set_edge_properties(edge_distance=None, minmax=[5, 30], directed=True)
d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size, edge_color='#000FFF', cmap='Set1')
d3.show()

# %%
from d3graph import d3graph, vec2adjmat

d3 = d3graph()
# Load example
adjmat, _ = d3.import_example('small')

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

d3.set_node_properties(size=node_size, label=label)
d3.show()

d3.set_node_properties(color=label, size=node_size, label=label)
d3.show()

d3.set_node_properties(color=label, size=node_size, label=label, tooltip=label + ' tooltip text')
d3.show()

d3.set_node_properties(color=label, size=node_size, label=label, edge_color='cluster', edge_size=5,
                       tooltip=label + ' tooltip text')
d3.show()

d3.set_edge_properties(edge_distance=100)
d3.set_node_properties(color=node_size, size=node_size)
d3.show()

d3 = d3graph(charge=1000)
d3.graph(adjmat)
d3.set_node_properties(color=node_size, size=node_size, label=label)
d3.show()

d3 = d3graph(collision=1, charge=250)
d3.graph(adjmat)
d3.set_node_properties(color=label, size=node_size, edge_size=node_size, cmap='Set1', label=label)
d3.show()

d3 = d3graph(collision=1, charge=250)
d3.graph(adjmat)
d3.set_node_properties(color=label, size=node_size, edge_size=node_size, edge_color='#00FFFF', cmap='Set1', label=label)
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
# d3.set_node_properties(label=df['label'].values, color=df['label'].values, edge_size=df['degree'].values, cmap='Set1')
# d3.set_node_properties(label=df['label'].values, color='cluster', edge_size=df['degree'].values, cmap='Set1', edge_color='cluster')
# d3.set_node_properties(label=df['label'].values, edge_color='cluster', edge_size=df['degree'].values, cmap='Set1')
d3.set_node_properties(label=df['label'].values, color='cluster', edge_size=df['degree'].values, cmap='Set1', edge_color='#000000')

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
d3.show()


# %%
