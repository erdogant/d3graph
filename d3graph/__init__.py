from d3graph.d3graph import d3graph
from packaging import version

from d3graph.adjmat_vec import (
    vec2adjmat,
    adjmat2vec,
    )

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.1.12'

import jinja2
if version.parse(jinja2.__version__) > version.parse("2.11.3"):
    raise Exception('[d3graph] >Error: jinja2 versions > 2.11.3 gives an error! It is advised to create a new environment install d3graph or: pip install -U jinja2==2.11.3')

# module level doc-string
__doc__ = """
d3graph
=======================================================================================

Description
------------
d3graph is a python library that is build on d3js and creates interactive and stand-alone networks.
The input data is a simple adjacency matrix for which the columns and indexes are the nodes and elements>0 the edges.
The ouput is a html file that is interactive and stand alone.

Examples
--------
>>> # Load some libraries
>>> import pandas as pd
>>> import numpy as np
>>> import networkx as nx
>>> from d3graph import d3graph
>>> 
>>> # Easy Example
>>> G = nx.karate_club_graph()
>>> adjmat = nx.adjacency_matrix(G).todense()
>>> adjmat = pd.DataFrame(index=range(0,adjmat.shape[0]), data=adjmat, columns=range(0,adjmat.shape[0]))
>>> # Make the interactive graph
>>> results = d3graph(adjmat)
>>>
>>> # Example with more parameters
>>> G = nx.karate_club_graph()
>>> adjmat = nx.adjacency_matrix(G).todense()
>>> adjmat=pd.DataFrame(index=range(0,adjmat.shape[0]), data=adjmat, columns=range(0,adjmat.shape[0]))
>>> adjmat.columns=adjmat.columns.astype(str)
>>> adjmat.index=adjmat.index.astype(str)
>>> adjmat.iloc[3,4]=5
>>> adjmat.iloc[4,5]=6
>>> adjmat.iloc[5,6]=7
>>>
>>> # Create dataset
>>> df = pd.DataFrame(index=adjmat.index)
>>> df['degree']=np.array([*G.degree()])[:,1]
>>> df['other info']=np.array([*G.degree()])[:,1]
>>> node_size=df.degree.values*2
>>> node_color=[]
>>> for i in range(0,len(G.nodes)):
>>>     node_color.append(G.nodes[i]['club'])
>>>     node_name=node_color
>>>
>>> # Make some graphs
>>> out = d3graph(adjmat, df=df, node_size=node_size)
>>> out = d3graph(adjmat, df=df, node_color=node_size, node_size=node_size)
>>> out = d3graph(adjmat, df=df, node_color=node_size, node_size=node_size, edge_distance=1000)
>>> out = d3graph(adjmat, df=df, node_color=node_size, node_size=node_size, charge=1000)
>>> out = d3graph(adjmat, df=df, node_color=node_name, node_size=node_size, node_size_edge=node_size, node_color_edge='#00FFFF', cmap='Set1', collision=1, charge=250)
>>>
>>> # Example with conversion to adjacency matrix
>>> G = nx.karate_club_graph()
>>> adjmat = nx.adjacency_matrix(G).todense()
>>> adjmat = pd.DataFrame(index=range(0,adjmat.shape[0]), data=adjmat, columns=range(0,adjmat.shape[0]))
>>> import d3graph
>>> # Convert adjacency matrix to vector with source and target
>>> vec = d3graph.adjmat2vec(adjmat)
>>> # Convert vector (source and target) to adjacency matrix.
>>> adjmat1 = d3graph.vec2adjmat(vec['source'], vec['target'], vec['weight'])
>>> # Check
>>> np.all(adjmat==adjmat1.astype(int))

"""
