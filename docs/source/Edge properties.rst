Edge properties
################

The **edge properties** can be customized using four options. After creating the ``d3.graph()``, the edges are based on the strength of the vertices.

General
*************************

Edge network properties can also be changed for the edges:

.. note::
	* 1. weight
	* 2. edge_distance
	* 3. edge_distance_minmax
	* 4. color
	* 5. directed
	* 6. marker


.. code:: python

	# Import library
	from d3graph import d3graph
	# Initialization
	d3 = d3graph()
	# Load karate example
	adjmat = d3.import_example('bigbang')
	# Process the adjacency matrix
	d3.graph(adjmat)

	# Examine the node properties
	print(d3.edge_properties)
	# ('Sheldon', 'Amy'): {'weight': 5.0, 'weight_scaled': 20.0, 'color': '#000000'},
	# ('Sheldon', 'Howard'): {'weight': 2.0, 'weight_scaled': 1.0, 'color': '#000000'},
	# ('Sheldon', 'Leonard'): {'weight': 3.0,'weight_scaled': 7.3333, 'color': '#000000'}}
	# ...
  
	# Set to directed edges
	d3.set_edge_properties(directed=True)

	# Customize the properties of one specific edge
	d3.edge_properties[('Sheldon', 'Howard')]['weight']=10
	d3.edge_properties[('Penny', 'Leonard')]['color']='#ff0000'
	
	# Plot
	d3.show()


.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/edge_properties_1.html" height="400px" width="750px", frameBorder="0"></iframe>



Markers
*************************

The **start** and **end** of the edges can be set for the following markers:
	* arrow
	* circle
	* square
	* stub
	* None or ''

The default ``marker_end`` is set to **arrow** whereas the ``marker_start`` is set to **None**.
Each marker can be customized using the ``edge_properties``.

.. code:: python

	# Import library
	from d3graph import d3graph
	# Initialization
	d3 = d3graph()
	# Load karate example
	adjmat = d3.import_example('bigbang')
	# Process the adjacency matrix
	d3.graph(adjmat)
	# Set some node properties
	d3.set_node_properties(color=adjmat.columns.values, size=[10, 20, 10, 10, 15, 10, 5])

  	# Edge properties
	print(d3.edge_properties)
	# {('Amy', 'Bernadette'): {'weight': 2.0, 'weight_scaled': 2.0, 'color': '#808080', 'marker_start': '', 'marker_end': 'arrow', ...
	
	# Set all marker-end to square and keep marker_start to be None or ''
	d3.set_edge_properties(directed=True, marker_end='square', marker_start='')
	d3.show()

	# Make some customized changes in the marker-end by removing all markers and set one for penny-leonard.
	d3.set_edge_properties(directed=True, marker_end='', label='weight')

	# Set markers for individual edges
	d3.edge_properties['Penny', 'Leonard']['marker_end']='arrow'
	d3.edge_properties['Sheldon', 'Howard']['marker_end']='stub'
	d3.edge_properties['Sheldon', 'Leonard']['marker_end']='circle'
	d3.edge_properties['Rajesh', 'Penny']['marker_end']='square'
	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/edge_properties_2.html" height="400px" width="750px", frameBorder="0"></iframe>

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/edge_properties_3.html" height="400px" width="750px", frameBorder="0"></iframe>



Scaling edges
************************************

There are two manners to scale the edges; scaling using the **minmax** or scaling using the **z-score**.
The default option is the z-score because the results tends to better in most use-cases.
Let's see the differences between the different methods.

.. code:: python

	# Import library
	from d3graph import d3graph
	# Initialization
	d3 = d3graph()
	# Load karate example
	adjmat = d3.import_example('bigbang')
	# Process the adjacency matrix
	d3.graph(adjmat)

  	# Set to no scaler (default)
	d3.set_edge_properties(directed=True)
	d3.show()

	# Set to minmax scaler
	d3.set_edge_properties(directed=True, minmax=[1, 20], scaler='minmax')
	d3.show()

	# Set to zscore scaler (default)
	d3.set_edge_properties(directed=True, minmax=[1, 20], scaler='zscore')
	d3.show()


.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/edge_properties_4.html" height="400px" width="750px", frameBorder="0"></iframe>

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/edge_properties_5.html" height="400px" width="750px", frameBorder="0"></iframe>

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/edge_properties_6.html" height="400px" width="750px", frameBorder="0"></iframe>



Edge Labels
************************************

The labels of the edges can be set per edge in a similar manner as for the other edge properties.

.. code:: python

	# Import
	from d3graph import d3graph
	# intialize to load example dataset
	d3 = d3graph()
	adjmat = d3.import_example('bigbang')

	# Initialize with clustering colors
	d3.graph(adjmat, color='cluster')


Set all edge labels to "test".

.. code:: python

    # Adding specifc labels to the edges
	d3.set_edge_properties(directed=True, label='test')
	
	# Adding the weights can be as following:
    d3.set_edge_properties(directed=True, label='weight')

	print(d3.edge_properties)

	# {('Amy', 'Bernadette'): {'weight': 2.0,
	#   'weight_scaled': 2.0,
	#   'color': '#808080',
	#   'marker_start': '',
	#   'marker_end': 'arrow',
	#   'marker_color': '#808080',
	#   'label': 'test',
	#   'label_color': '#808080',
	#   'label_fontsize': 8},
	#  ('Bernadette', 'Howard'): {'weight': 5.0,
	#   'weight_scaled': 5.0,
	#   'color': '#808080',
	#   'marker_start': '',
	#   'marker_end': 'arrow',
	#   'marker_color': '#808080',
	#   'label': 'test',
	#   'label_color': '#808080',
	#   'label_fontsize': 8},

	d3.show()


.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/edge_labels_1.html" height="400px" width="750px", frameBorder="0"></iframe>


We will first set all label properties to None and then we will adjust two of them.

.. code:: python

	# Change the label properties for the first edge
	d3.set_edge_properties(directed=True, marker_color='#000FFF', label=None)
	d3.edge_properties['Amy', 'Bernadette']['weight_scaled']=10
	d3.edge_properties['Amy', 'Bernadette']['label']='amy-bern'
	d3.edge_properties['Amy', 'Bernadette']['label_color']='#000FFF'
	d3.edge_properties['Amy', 'Bernadette']['label_fontsize']=8

	# Change the label properties for the second edge
	d3.edge_properties['Bernadette', 'Howard']['label']='bern-how'
	d3.edge_properties['Bernadette', 'Howard']['label_fontsize']=20
	d3.edge_properties['Bernadette', 'Howard']['label_color']='#000000'

	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/edge_labels_2.html" height="400px" width="750px", frameBorder="0"></iframe>


Set distance
*************************

.. code:: python
	
	# Set edge properties with a edge distance
	d3.set_edge_properties(edge_distance=100)

	# Plot
	d3.show()



.. include:: add_bottom.add