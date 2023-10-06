from d3graph.d3graph import d3graph
from packaging import version
from datazets import get as import_example

from d3graph.d3graph import (
    vec2adjmat,
    adjmat2vec,
    make_graph,
    json_create,
    adjmat2dict,
    data_checks,
    )


__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '2.4.11'


# module level doc-string
__doc__ = """
d3graph
=======================================================================================

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

References
----------
* D3Graph: https://towardsdatascience.com/creating-beautiful-stand-alone-interactive-d3-charts-with-python-804117cb95a7
* D3Blocks: https://towardsdatascience.com/d3blocks-the-python-library-to-create-interactive-and-standalone-d3js-charts-3dda98ce97d4
* Github : https://github.com/erdogant/d3graph
* Documentation: https://erdogant.github.io/d3graph/

"""
