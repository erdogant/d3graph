d3graph's documentation!
========================

The ``d3graph`` library is a Python library that is built on D3 and creates a stand-alone, and interactive force-directed network graph. It allows the creation, manipulation, and study of the structure, dynamics, and functions of complex networks. The input data is an adjacency matrix for which the columns and indexes are the nodes and the elements with a value of one or larger are considered to be an edge. The output is a single HTML file that contains the interactive force-directed graph. ``d3graph`` has several features, among them a slider that can break the edges of the network based on the edge value, a double click on a node will highlight the node and its connected edges and many more options to customize the network based on the edge and node properties.

.. tip::
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
   :caption: Data
   
   Data

.. toctree::
  :maxdepth: 1
  :caption: Methods

  D3
  Core Functionalities


.. toctree::
  :maxdepth: 1
  :caption: Examples

  Examples


.. toctree::
  :maxdepth: 1
  :caption: Code Documentation
  
  Blog
  Coding quality
  d3graph.d3graph



Quick install
-------------

.. code-block:: console

   pip install d3graph


Github
------------------------------

Please report bugs, issues and feature extensions there.
Github, `erdogant/d3graph <https://github.com/erdogant/d3graph/>`_.


Citing
----------------

.. code:: python

	@software{Taskesen_Interactive_force-directed_network_2019,
	author = {Taskesen, Erdogan},
	license = {Apache-2.0},
	month = {12},
	title = {{Interactive force-directed network creator (d3graph)}},
	url = {https://github.com/erdogant/d3graph},
	version = {0.1.12},
	year = {2019}
	}



.. raw:: html

	<div data-ea-publisher="erdogantgithubio" data-ea-type="text" id="ad"></div>

.. raw:: html

	<div data-ea-publisher="erdogantgithubio" data-ea-type="text" id="ad"></div>
	<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

