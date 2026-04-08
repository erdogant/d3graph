Big Bang network
####################

Default
************************************

.. code:: python

	from d3graph import d3graph
	size = [10, 20, 10, 10, 15, 10, 5]

	# Initialize
	d3 = d3graph()
	# Load example
	adjmat = d3.import_example('bigbang')
	# Process adjmat
	d3.graph(adjmat)
	# Show
	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/bigbang_default.html" height="800px" width="850px", frameBorder="0"></iframe>


Node colors
************************************

.. code:: python

	d3.set_node_properties(color=adjmat.columns.values)
	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/bigbang_color.html" height="800px" width="850px", frameBorder="0"></iframe>


Node fontcolors
************************************

.. code:: python

	d3.set_node_properties(color='cluster', fontcolor='node_color')
	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/bigbang_text_color.html" height="800px" width="850px", frameBorder="0"></iframe>



Node sizes
************************************

.. code:: python

	d3.set_node_properties(color=adjmat.columns.values, size=size)
	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/bigbang_color_size.html" height="800px" width="850px", frameBorder="0"></iframe>

Edge sizes
************************************

.. code:: python

	d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1])
	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/bigbang_color_size_edge.html" height="800px" width="850px", frameBorder="0"></iframe>


Edge colors
************************************

.. code:: python

	d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1], edge_color='#00FFFF')
	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/bigbang_color_size_edge_color.html" height="800px" width="850px", frameBorder="0"></iframe>


Colormap
************************************

.. code:: python

	d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1], edge_color='#00FFFF', cmap='Set2')
	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/bigbang_cmap.html" height="800px" width="850px", frameBorder="0"></iframe>


Directed arrows
************************************

.. code:: python

	d3.set_edge_properties(directed=True)
	d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size, edge_color='#000FFF', cmap='Set1')
	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/bigbang_directed.html" height="800px" width="850px", frameBorder="0"></iframe>


Node text inside
************************************

.. code:: python

	from d3graph import d3graph
	size = [10, 20, 10, 10, 15, 10, 5]

	# Initialize
	d3 = d3graph()
	# Load example
	adjmat = d3.import_example('bigbang')
	# Process adjmat
	d3.graph(adjmat)
	# Show
	d3.set_node_properties(fontcolor='#000000')
	d3.show(node_text_inside=True)

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/bigbang_node_text_inside.html" height="800px" width="850px", frameBorder="0"></iframe>


Change nodes types
************************************

.. code:: python

    from d3graph import d3graph, import_example
    
    d3 = d3graph()
    adjmat, df = import_example('karate')
    d3.graph(adjmat)
    
    # Mix shapes per node
    markers = ['circle', 'star', 'diamond', 'square', 'pentagon', 'hexagon', 'rectangle', 'triangle-down', 'triangle'] * 8
    d3.set_node_properties(marker=markers[:len(adjmat.columns)], color=df['label'].values, cmap='Set1')
    d3.show(filepa)

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/node_shapes.html" height="800px" width="850px", frameBorder="0"></iframe>


.. code:: python
    
    from d3graph import d3graph, import_example
    
    d3 = d3graph()
    adjmat, df = import_example('karate')
    d3.graph(adjmat)
    
    # Set shape for node
    d3.set_node_properties(marker='hexagon', color=df['label'].values, cmap='Set1')
    d3.show()


.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/node_shape.html" height="800px" width="850px", frameBorder="0"></iframe>





Karate Club network
####################

.. code:: python

	from d3graph import d3graph, vec2adjmat

	# Initialize
	d3 = d3graph()
	# Load energy example
	df = d3.import_example('energy')
	adjmat = vec2adjmat(source=df['source'], target=df['target'], weight=df['weight'])
	
	# Process adjmat
	d3.graph(adjmat)
	d3.show(filepath=r'D:\REPOS\erdogant.github.io\docs\d3graph\d3graph/energy_1.html')


.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/energy_1.html" height="800px" width="850px", frameBorder="0"></iframe>



.. code:: python

	from d3graph import d3graph, vec2adjmat

	# Initialize
	d3 = d3graph()
	# Load energy example
	df = d3.import_example('energy')
	adjmat = vec2adjmat(source=df['source'], target=df['target'], weight=df['weight'])
	
	# Process adjmat
	d3.graph(adjmat)

    # Change node properties
    d3.set_node_properties(scaler='minmax', color=None)
    d3.node_properties['Solar']['size']=30
    d3.node_properties['Solar']['color']='#FF0000'
    d3.node_properties['Solar']['edge_color']='#000000'
    d3.node_properties['Solar']['edge_size']=5

    # Show
	d3.show()
    

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/energy_2.html" height="800px" width="850px", frameBorder="0"></iframe>



.. code:: python

    d3.set_edge_properties(directed=True, marker_end='arrow')
	d3.show()

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/energy_3.html" height="800px" width="850px", frameBorder="0"></iframe>


.. include:: add_bottom.add