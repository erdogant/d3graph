.. _code_directive:

-------------------------------------

Node properties
'''''''''''''''''

There are various possabilities to customize the node properties. After creating the ``d3.graph``, it will generate default node properties which can be customized.

	* 1. label
	* 2. color
	* 3. size
	* 4. edge_color
	* 5. edge_size


Create example network
-----------------------

.. code:: python
	
	# Import library
	from d3graph import d3graph, vec2adjmat
	
	# Create example network
	source = ['node A', 'node F', 'node B', 'node B', 'node B', 'node A', 'node C', 'node Z']
	target = ['node F', 'node B', 'node J', 'node F', 'node F', 'node M', 'node M', 'node A']
	weight = [5.56, 0.5, 0.64, 0.23, 0.9, 3.28, 0.5, 0.45]

	# Convert to adjacency matrix
	adjmat = vec2adjmat(source, target, weight=weight)

	# Create network with all default node properties
	d3 = d3graph()
	d3.graph(adjmat)
	d3.show()


Customize the properties of one specific node
-----------------------------------------------

.. code:: python

	# Examine the node properties
	print(d3.node_properties)

	# Customize the properties of one specific node
	d3.node_properties['node_A']['label']='CUSTOMIZED NODE'
	d3.node_properties['node_A']['color']='#FF00FF'
	d3.node_properties['node_A']['size']=20
	d3.node_properties['node_A']['edge_size']=5
	d3.node_properties['node_A']['edge_color']='#000000'

	print(d3.node_properties['node_A'])
	# {'label': 'CUSTOMIZED NODE', 'color': '#FF00FF', 'size': 20, 'edge_size': 5, 'edge_color': '#000000'}

	# Plot
	d3.show()


.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/example_1.html" height="1000px" width="2000px", frameBorder="0"></iframe>


Customize all nodes
-----------------------------------------------

.. code:: python

	# The order of the node names is as following:
	print(d3.node_properties)
	# dict_keys(['node_A', 'node_B', 'node_F', 'node_J', 'node_M', 'node_C', 'node_Z'])

	# Customize node colors based on node names
	# Size is set
	# edge size is same as the size
	# edge color is the same for all nodes: '#000FFF'
	# cmap is set to 'Set1'

	# Set size for the nodes
	size = [10, 20, 10, 10, 15, 10, 5]
	
	# Set node properties
	d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size, edge_color='#000FFF', cmap='Set1')

	# Plot
	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/example_2.html" height="1000px" width="2000px", frameBorder="0"></iframe>


Edge properties
'''''''''''''''''

There are various possabilities to customize the edge properties. After creating the ``d3.graph``, it will generate default node properties which can be customized.

	* 1. weight
	* 2. weight_scaled
	* 3. color

Network properties can also be changed for the edges:

	* 1. edge_distance
	* 2. edge_distance_minmax
	* 3. directed


Create example network
-----------------------

.. code:: python
	
	# Import library
	from d3graph import d3graph, vec2adjmat
	
	# Create example network
	source = ['node A', 'node F', 'node B', 'node B', 'node B', 'node A', 'node C', 'node Z']
	target = ['node F', 'node B', 'node J', 'node F', 'node F', 'node M', 'node M', 'node A']
	weight = [5.56, 0.5, 0.64, 0.23, 0.9, 3.28, 0.5, 0.45]

	# Convert to adjacency matrix
	adjmat = vec2adjmat(source, target, weight=weight)

	# Create network with all default node properties
	d3 = d3graph()
	d3.graph(adjmat)
	d3.show()


Customize the properties of one specific edge
-----------------------------------------------

.. code:: python

	# Examine the node properties
	print(d3.edge_properties)

	# Customize the properties of one specific node
	d3.edge_properties[('node_A', 'node_F')]['weight']=10
	d3.edge_properties[('node_A', 'node_F')]['weight_scaled']=25
	d3.edge_properties[('node_A', 'node_F')]['color']='#000000'

	print(d3.edge_properties[('node_A', 'node_F')])
	# {'weight': 10, 'weight_scaled': 25, 'color': '#000000'}

	# Plot
	d3.show()



Customize all edges
-----------------------------------------------

.. code:: python
	
	# Set edge properties with a edge distance
	d3.set_edge_properties(edge_distance=100)

	# Plot
	d3.show()

