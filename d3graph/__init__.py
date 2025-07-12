from d3graph.d3graph import d3graph
from packaging import version
from datazets import get as import_example
import logging

from d3graph.d3graph import (
    vec2adjmat,
    adjmat2vec,
    make_graph,
    json_create,
    adjmat2dict,
    data_checks,
    check_logger,
    )

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '2.6.1'

# Setup root logger
_logger = logging.getLogger('d3graph')
_log_handler = logging.StreamHandler()
_fmt = '[{asctime}] [{name}] [{levelname}] {msg}'
_formatter = logging.Formatter(fmt=_fmt, style='{', datefmt='%d-%m-%Y %H:%M:%S')
_log_handler.setFormatter(_formatter)
_log_handler.setLevel(logging.DEBUG)
_logger.addHandler(_log_handler)
_logger.propagate = False


# module level doc-string
__doc__ = """
d3graph
=======================================================================================

d3graph is a python library that is build on d3js and creates interactive and stand-alone networks.
The input data is a simple adjacency matrix for which the columns and indexes are the nodes and elements>0 the edges.
The ouput is a html file that is interactive and stand alone.

Examples
--------
>>> from d3graph import d3graph, vec2adjmat
>>>
>>> # Initialize
>>> d3 = d3graph()
>>>
>>> # Load karate example
>>> df = d3.import_example('energy')
>>> adjmat = vec2adjmat(source=df['source'], target=df['target'], weight=df['weight'])
>>>
>>> # Initialize
>>> d3.graph(adjmat)
>>>
>>> # Node properties
>>> d3.set_node_properties(cmap='Set1')
>>>
>>> # Edge properties
>>> d3.set_edge_properties(directed=True)
>>>
>>> # Plot
>>> d3.show()

References
----------
* D3Graph: erdogant.medium.com
* D3Blocks: https://medium.com/data-science-collective/d3blocks-the-python-library-to-create-interactive-standalone-and-beautiful-d3-js-charts-ef8c65286e86
* Github : https://github.com/erdogant/d3graph
* Documentation: https://erdogant.github.io/d3graph/

"""
