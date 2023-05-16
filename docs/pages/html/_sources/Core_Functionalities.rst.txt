Core Functionalities
######################

In order to create, manipulate, and study the structure, dynamics, and functions of complex networks, it is usefull to understand the various functions of ``d3graph``. Here I will describe the core functionalities that can help to customize your network. In the following examples I will be using the **karate** network.


Import
************************************
Importing the ``d3graph`` library is the first step after the pip installation.

.. code:: python
	
	# Import library
	from d3graph import d3graph


Initalization 
************************************

The initialization is directly performed after importing the ``d3graph`` library. During the initialization, the following parameters can be set:

.. code-block::

	* collision	: 0.5		: Response of the network. Higher means that more collisions are prevented.
	* charge	: 250		: Edge length of the network. Towards zero becomes a dense network.
	* slider	: [None, None]	: Slider to break the network. The default is based on the edge weights.
	* verbose	: 20		: Print progress to screen, 60: None, 40: Error, 30: Warn, 20: Info, 10: Debug

**A run with default initialization.**

.. code:: python
	
	# Initialization with default parameters
	d3 = d3graph()
	# Load karate example
	adjmat, df = d3.import_example('karate')
	# Process the adjacency matrix
	d3.graph(adjmat)
	# Node properties
	d3.set_node_properties(label=df['label'].values, tooltip=df['label'].values, color='cluster')
	# Plot
	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/karate_default.html" height="700px" width="850px", frameBorder="0"></iframe>


**The collision parameter**

This network will go wild because it tries to prevent collisions from happening. At some point, the network will stop trying. You can reset it by breaking the network with the silder.

.. code:: python

	# Import library
	from d3graph import d3graph
	# Initialization to make the network be more nervous when nodes are close together.
	d3 = d3graph(collision=3)
	# Load karate example
	adjmat, df = d3.import_example('karate')
	# Process the adjacency matrix
	d3.graph(adjmat)
	# Plot
	d3.show()


.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/karate_collision.html" height="700px" width="850px", frameBorder="0"></iframe>


**The charge parameter.**

This network is much wider than the previous ones. This is certainly helpfull if you have a dense network and need to expand it for visualization purposes.

.. code:: python

	# Import library
	from d3graph import d3graph
	# Initialization to make network edges reltively longer.
	d3 = d3graph(charge=1000)
	# Load karate example
	adjmat, df = d3.import_example('karate')
	# Process the adjacency matrix
	d3.graph(adjmat)
	# Plot
	d3.show()


.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/karate_charge.html" height="700px" width="850px", frameBorder="0"></iframe>


Processing
************************************

The graph function :func:`d3graph.d3graph.d3graph.graph` processes the adjacency matrix to create a network with default *node properties* and *edge properties*. The nodes are the column and index names, and a connect edge for vertices with value larger than 0. The strenght of edges are based on the vertices values. The input for ``d3graph`` is the adjacency matrix.

Show
************************************

The show function :func:`d3graph.d3graph.d3graph.show` has several tasks.
	
	* 1. Creating networkx *graph G* based on the node properties and edge properties.
	* 2. Embedding of the data.
	* 3. Writes the final HTML file to disk.
	* 4. Opens the webbroswer with the network graph.




Hide Slider
************************************

The slider can be hidden from the output HTML by setting the ``show_slider=False`` parameter.

.. code:: python

	from d3graph import d3graph

	# Initialize
	d3 = d3graph()
	# Load example
	adjmat, df = d3.import_example('karate')
	# Process adjmat
	d3.graph(adjmat)
	d3.show(show_slider=False)



.. include:: add_bottom.add