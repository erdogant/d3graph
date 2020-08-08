# %% Libraries
import networkx as nx
import pandas as pd
import numpy as np
from d3graph import d3graph


# %%
G = nx.karate_club_graph()
adjmat = nx.adjacency_matrix(G).todense()
adjmat=pd.DataFrame(index=range(0,adjmat.shape[0]), data=adjmat, columns=range(0,adjmat.shape[0]))
adjmat.columns=adjmat.columns.astype(str)
adjmat.index=adjmat.index.astype(str)
adjmat.iloc[3,4]=5
adjmat.iloc[4,5]=6
adjmat.iloc[5,6]=7

df = pd.DataFrame(index=adjmat.index)
df['degree']=np.array([*G.degree()])[:,1]
df['other info']=np.array([*G.degree()])[:,1]
node_size=df.degree.values*2
node_color=[]

for i in range(0,len(G.nodes)):
    node_color.append(G.nodes[i]['club'])
    node_name=node_color
df['name']=node_name

# Make some graphs
out = d3graph(adjmat, df=df, node_color=node_size, node_size=node_size)

# %% Simple example
G = nx.karate_club_graph()
adjmat = nx.adjacency_matrix(G).todense()
adjmat = pd.DataFrame(index=range(0,adjmat.shape[0]), data=adjmat, columns=range(0,adjmat.shape[0]))

from d3graph import d3graph
out = d3graph(adjmat, df=df.iloc[0:34, :], node_color=adjmat.index.values, node_color_edge='#00000', node_size_edge=5)

out = d3graph(adjmat)
out = d3graph(adjmat, savepath='c://temp/')
out = d3graph(adjmat, node_color=adjmat.index.values, savepath='c://temp/')
out = d3graph(adjmat, node_color=adjmat.index.values, node_color_edge='#fffff',savepath='c://temp/')
out = d3graph(adjmat, node_color=adjmat.index.values, node_color_edge=adjmat.index.values)
out = d3graph(adjmat, node_color=adjmat.index.values, node_color_edge='#00000', node_size_edge=5)



# %% Extended example
G = nx.karate_club_graph()
adjmat = nx.adjacency_matrix(G).todense()
adjmat=pd.DataFrame(index=range(0,adjmat.shape[0]), data=adjmat, columns=range(0,adjmat.shape[0]))
adjmat.columns=adjmat.columns.astype(str)
adjmat.index=adjmat.index.astype(str)
adjmat.iloc[3,4]=5
adjmat.iloc[4,5]=6
adjmat.iloc[5,6]=7

from tabulate import tabulate
print(tabulate(adjmat.head(), tablefmt="grid", headers="keys"))


df = pd.DataFrame(index=adjmat.index)
df['degree']=np.array([*G.degree()])[:,1]
df['other info']=np.array([*G.degree()])[:,1]
node_size=df.degree.values*2
node_color=[]
for i in range(0,len(G.nodes)):
    node_color.append(G.nodes[i]['club'])
    node_name=node_color

out = d3graph(adjmat, df=df, node_size=node_size)
out = d3graph(adjmat, df=df, node_color=node_size, node_size=node_size)
out = d3graph(adjmat, df=df, node_color=node_size, node_size=node_size, edge_distance=1000)
out = d3graph(adjmat, df=df, node_color=node_size, node_size=node_size, charge=1000)
out = d3graph(adjmat, df=df, node_color=node_name, node_size=node_size, node_size_edge=node_size, cmap='Set1', collision=1, charge=250)
out = d3graph(adjmat, df=df, node_color=node_name, node_size=node_size, node_size_edge=node_size, node_color_edge='#00FFFF', cmap='Set1', collision=1, charge=250)

# %%
source = ['node A','node F','node B','node B','node B','node A','node C','node Z']
target = ['node F','node B','node J','node F','node F','node M','node M','node A']
weight = [5.56, 0.5, 0.64, 0.23, 0.9,3.28,0.5,0.45]

# Import library
from d3graph import d3graph, vec2adjmat
# Convert to adjacency matrix
adjmat = vec2adjmat(source, target, weight=weight)
print(adjmat)
# target  node A  node B  node F  node J  node M  node C  node Z
# source                                                        
# node A    0.00     0.0    5.56    0.00    3.28     0.0     0.0
# node B    0.00     0.0    1.13    0.64    0.00     0.0     0.0
# node F    0.00     0.5    0.00    0.00    0.00     0.0     0.0
# node J    0.00     0.0    0.00    0.00    0.00     0.0     0.0
# node M    0.00     0.0    0.00    0.00    0.00     0.0     0.0
# node C    0.00     0.0    0.00    0.00    0.50     0.0     0.0
# node Z    0.45     0.0    0.00    0.00    0.00     0.0     0.0

# Example A: simple interactive network
out = d3graph(adjmat)

# Example B: Color nodes
out = d3graph(adjmat, node_color=adjmat.columns.values)

# Example C: include node size
node_size = [10,20,10,10,15,10,5]
out = d3graph(adjmat, node_color=adjmat.columns.values, node_size=node_size)

# Example D: include node-edge-size
out = d3graph(adjmat, node_color=adjmat.columns.values, node_size=node_size, node_size_edge=node_size[::-1], cmap='Set2')

# Example E: include node-edge color
out = d3graph(adjmat, node_color=adjmat.columns.values, node_size=node_size, node_size_edge=node_size[::-1], node_color_edge='#00FFFF')

# Example F: Change colormap
out = d3graph(adjmat, node_color=adjmat.columns.values, node_size=node_size, node_size_edge=node_size[::-1], node_color_edge='#00FFFF', cmap='Set2')

# Example H: Include directed links. Arrows are set from source -> target
out = d3graph(adjmat, node_color=adjmat.columns.values, node_size=node_size, node_size_edge=node_size[::-1], node_color_edge='#00FFFF', cmap='Set2', directed=True)

