from d3graph.d3graph import d3graph
from packaging import version

from d3graph.d3graph import (
    vec2adjmat,
    adjmat2vec,
    )

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '2.1.1'

import jinja2
if version.parse(jinja2.__version__) > version.parse("2.11.3"):
    print('[d3graph] >Error: jinja2 versions > 2.11.3 gives an error! It is advised to create a new environment install d3graph or: pip install -U jinja2==2.11.3')

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
>>> from d3graph import d3graph
>>>
>>> # Initialize
>>> d3 = d3graph()
>>>
>>> # Load karate example
>>> adjmat, df = d3.import_example('karate')
>>>
>>> # Initialize
>>> d3.graph(adjmat)
>>>
>>> # Node properties
>>> d3.set_node_properties(label=df['label'].values, color=df['label'].values, size=df['degree'].values, edge_size=df['degree'].values, cmap='Set1')
>>>
>>> # Edge properties
>>> d3.set_edge_properties(directed=True)
>>>
>>> # Plot
>>> d3.show()

"""
