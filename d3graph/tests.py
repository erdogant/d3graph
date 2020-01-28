# %% Libraries
import networkx as nx
import pandas as pd
import numpy as np
from d3graph import d3graph


# %% Simple example
G = nx.karate_club_graph()
adjmat = nx.adjacency_matrix(G).todense()
adjmat=pd.DataFrame(index=range(0,adjmat.shape[0]), data=adjmat, columns=range(0,adjmat.shape[0]))

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
out = d3graph(adjmat, df=df, node_color=node_size, node_size=node_size, edge_distance=100)
out = d3graph(adjmat, df=df, node_color=node_size, node_size=node_size, charge=50)
out = d3graph(adjmat, df=df, node_color=node_name, node_size=node_size, node_size_edge=node_size, node_color_edge='#00FFFF', cmap='Set1', collision=1, charge=250)

# %%