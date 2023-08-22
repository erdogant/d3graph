Node properties
######################

There are various possabilities to customize the network using the node properties function :func:`d3graph.d3graph.d3graph.set_node_properties`. Intially, all default node properties are created which can than be customized. The underneath properties can be changed for each node. I will use the **karate** network to demonstrate the working.

.. note::
	* 1. Node label
	* 1. Node tooltip
	* 3. Node color
	* 4. Node size
	* 5. Node opacity
	* 6. Node edge color
	* 7. Node fontcolor
	* 8. Node fontsize
	* 9. Node edge size


Node label
************************************

Lets change the **node labels** from the *karate* example into something more meaningfull.

.. code:: python

	# Import library
	from d3graph import d3graph
	# Initialization
	d3 = d3graph()
	# Load karate example
	adjmat, df = d3.import_example('karate')
	# Process the adjacency matrix
	d3.graph(adjmat)

	# Set node properties
	d3.set_node_properties(label=df['label'].values)

	# Plot
	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/karate_label.html" height="700px" width="850px", frameBorder="0"></iframe>


Tooltips
**************************************************

Getting more information when hovering over a node can be easily done using the ``tooltip`` parameter.

.. code:: python

	# Import library
	from d3graph import d3graph
	# Initialization
	d3 = d3graph()
	# Load karate example
	adjmat, df = d3.import_example('karate')
	# Process the adjacency matrix
	d3.graph(adjmat)

	# Set node properties
	tooltip = '\nId: ' + adjmat.columns.astype(str) +'\nDegree: ' + df['degree'].astype(str) + '\nLabel: ' + df['label'].values
	tooltip = tooltip.values
	label = df['label'].values

	# Set node properties
	d3.set_node_properties(label=label, tooltip=tooltip, color=label, size='degree')
	d3.show()

	# If you want thinner lines
	d3.set_node_properties(label=label, tooltip=tooltip, color=label, size='degree', minmax=[0.1, 15])
	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/karate_label_hover.html" height="700px" width="850px", frameBorder="0"></iframe>


Node color
**************************************************

Lets change the **node colors** from the *karate* example using the label information. We do not need to re-initialize the whole graph but we can simply update the node properties.

.. code:: python

	# Set node properties
	d3.set_node_properties(label=df['label'].values, color=df['label'].values)

	# Plot
	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/karate_label_color.html" height="700px" width="850px", frameBorder="0"></iframe>


Node color on clustering
**************************************************

We can also change the node color on the clustering.

.. code:: python

	# Set node properties
	d3.set_node_properties(label=df['label'].values, color='cluster')

	# Plot
	d3.show()


Node fontcolor
**************************************************

Lets change the **node font colors** and ajust it according to the node color.

.. code:: python

	# Set node properties
	d3.set_node_properties(label=df['label'].values, color='cluster', fontcolor='node_color')

	# Plot
	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/karate_node_text.html" height="700px" width="850px", frameBorder="0"></iframe>



Node fontsize
**************************************************

Change the **node fontsize** and ajust it according to the node color.

.. code:: python

	d3 = d3graph()
	adjmat = d3.import_example('bigbang')

	fontsize=np.random.randint(low=6, high=40, size=adjmat.shape[0])
	d3.set_node_properties(color='cluster', scaler='minmax', fontcolor='node_color', fontsize=fontsize)

	# Plot
	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/bigbang_nodefontsize.html" height="700px" width="850px", frameBorder="0"></iframe>



Node edge color on clustering
**************************************************

We can also change the node color on the clustering.

.. code:: python

	# Set node properties
	d3.set_node_properties(label=df['label'].values, edge_color='cluster')

	# Plot
	d3.show()



Node size
**************************************************

Lets change the **node size** from the *karate* example using the degree of the network. We do not need to re-initialize the whole graph but we can simply update the node properties.

.. code:: python

	# Set node properties
	d3.set_node_properties(label=df['label'].values, color=df['label'].values, size=df['degree'].values)
	# Plot
	d3.show()

	# Set node properties
	d3.set_node_properties(label=df['label'].values, color=df['label'].values, size='degree')
	# Plot
	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/karate_label_color_size.html" height="700px" width="850px", frameBorder="0"></iframe>


Node opacity
**************************************************

We can change the **node opacity** using the degree of the network. We do not need to re-initialize the whole graph but we can simply update the node properties.

.. code:: python

	# Set node properties
	d3.set_node_properties(opacity='degree', size='degree', color='cluster')

	# Plot
	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/karate_label_color_size.html" height="700px" width="850px", frameBorder="0"></iframe>


Node edge size
**************************************************

Lets change the **node edge size** from the *karate* example using the degree of the network. We do not need to re-initialize the whole graph but we can simply update the node properties.

.. code:: python

	# Set node properties
	d3.set_node_properties(label=df['label'].values, color=df['label'].values, size=df['degree'].values, edge_size=df['degree'].values)

	# Plot
	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/karate_label_color_size_edge_size.html" height="700px" width="850px", frameBorder="0"></iframe>


Node edge color
**************************************************

Lets change the **node edge color** from the *karate* example using a specified color. We do not need to re-initialize the whole graph but we can simply update the node properties.

.. code:: python

	# Set node properties
	d3.set_node_properties(label=df['label'].values, color=df['label'].values, size=df['degree'].values, edge_size=df['degree'].values, edge_color='#FFF000')

	# Plot
	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/karate_label_color_size_edge_size_edge_color.html" height="700px" width="850px", frameBorder="0"></iframe>



Customize the properties of one specific node
**************************************************

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
	print(d3.node_properties)
	# {'Amy': {'label': 'Amy', 'color': '#000080', 'size': 10, 'edge_size': 0.1, 'edge_color': '#000000'},
	# 'Bernadette': {'label': 'Bernadette', 'color': '#000080', 'size': 10, 'edge_size': 0.1, 'edge_color': '#000000'}, 
	# 'Howard': {'label': 'Howard', 'color': '#000080', 'size': 10, 'edge_size': 0.1, 'edge_color': '#000000'},
	# 'Leonard': {'label': 'Leonard', 'color': '#000080', 'size': 10, 'edge_size': 0.1, 'edge_color': '#000000'},
	# 'Penny': {'label': 'Penny', 'color': '#000080', 'size': 10, 'edge_size': 0.1, 'edge_color': '#000000'},
	# 'Rajesh': {'label': 'Rajesh', 'color': '#000080', 'size': 10, 'edge_size': 0.1, 'edge_color': '#000000'},
	# 'Sheldon': {'label': 'Sheldon', 'color': '#000080', 'size': 10, 'edge_size': 0.1, 'edge_color': '#000000'}}

	# Customize the properties of one specific node
	d3.node_properties['Penny']['label']='Penny Hofstadter'
	d3.node_properties['Penny']['color']='#ffc0cb' # Pink
	d3.node_properties['Penny']['size']=20
	d3.node_properties['Penny']['edge_size']=5
	d3.node_properties['Penny']['edge_color']='#0000ff' # Blue

	# Customize a specific edge property
	d3.edge_properties['Penny', 'Leonard']['color']='#FF0000' # red
	
	# Print
	print(d3.node_properties['Penny'])
	# {'label': 'Penny Hofstadter', 'color': '#ffc0cb', 'size': 20, 'edge_size': 5, 'edge_color': '#000000'}

	# Plot
	d3.show()


.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/bigbang_Penny.html" height="700px" width="850px", frameBorder="0"></iframe>



.. include:: add_bottom.add