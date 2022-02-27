from d3graph.d3graph import d3graph
from packaging import version

from d3graph.d3graph import (
    vec2adjmat,
    adjmat2vec,
    )

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '2.0.0'

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
>>> source = ['node A', 'node F', 'node B', 'node B', 'node B', 'node A', 'node C', 'node Z']
>>> target = ['node F', 'node B', 'node J', 'node F', 'node F', 'node M', 'node M', 'node A']
>>> weight = [5.56, 0.5, 0.64, 0.23, 0.9, 3.28, 0.5, 0.45]
>>>
>>> # Convert to adjacency matrix
>>> adjmat = vec2adjmat(source, target, weight=weight)
>>> # print(adjmat)
>>>
>>> # Example A: simple interactive network
>>> d3 = d3graph()
>>> d3.graph(adjmat)
>>> d3.show()
>>>
>>> # Example B: Color nodes
>>> # d3 = d3graph()
>>> d3.graph(adjmat)
>>> # Set node properties
>>> d3.set_node_properties(color=adjmat.columns.values)
>>> d3.show()
>>>
>>> size = [10, 20, 10, 10, 15, 10, 5]
>>>
>>> # Example C: include node size
>>> d3.set_node_properties(color=adjmat.columns.values, size=size)
>>> d3.show()
>>>
>>> # Example D: include node-edge-size
>>> d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1])
>>> d3.show()
>>>
>>> # Example E: include node-edge color
>>> d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1], edge_color='#000000')
>>> d3.show()
>>>
>>> # Example F: Change colormap
>>> d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1], edge_color='#00FFFF', cmap='Set2')
>>> d3.show()
>>>
>>> # Example H: Include directed links. Arrows are set from source -> target
>>> d3.set_edge_properties(directed=True)
>>> d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1], edge_color='#00FFFF', cmap='Set2')
>>> d3.show()
>>>

"""
