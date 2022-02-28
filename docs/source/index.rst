d3graph's documentation!
========================

The ``d3graph`` library is a Python library that is built on D3 and creates a stand-alone, and interactive force-directed network graph. It allows the creation, manipulation, and study of the structure, dynamics, and functions of complex networks. The input data is an adjacency matrix for which the columns and indexes are the nodes and the elements with a value of one or larger are considered to be an edge. The output is a single HTML file that contains the interactive force-directed graph. ``d3graph`` has several features, among them a slider that can break the edges of the network based on the edge value, a double click on a node will highlight the node and its connected edges and many more options to customize the network based on the edge and node properties.

`Medium Blog: Creating beautiful stand-alone interactive D3 charts with Python <https://towardsdatascience.com/creating-beautiful-stand-alone-interactive-d3-charts-with-python-804117cb95a7>`_


Content
=======

.. toctree::
   :maxdepth: 1
   :caption: Background
   
   Abstract


.. toctree::
   :maxdepth: 1
   :caption: Installation
   
   Installation


.. toctree::
  :maxdepth: 1
  :caption: Methods

  D3
  Network properties


.. toctree::
  :maxdepth: 1
  :caption: Examples

  Examples


.. toctree::
  :maxdepth: 1
  :caption: Code Documentation
  
  Documentation
  Coding quality
  d3graph.d3graph



Quick install
-------------

.. code-block:: console

   pip install d3graph




Source code and issue tracker
------------------------------

Available on Github, `erdogant/d3graph <https://github.com/erdogant/d3graph/>`_.
Please report bugs, issues and feature extensions there.

Citing *d3graph*
----------------
Here is an example BibTeX entry:

@misc{erdogant2019d3graph,
  title={d3graph},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdogant/d3graph}}}



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
