On Click Actions
'''''''''''''''''''''' 

Changing the click actions is easy using the click dictionary.


.. code:: python
	
	# Import library
	from d3graph import d3graph

	# Initialize with defaults to load example
	d3 = d3graph()
	adjmat = d3.import_example('bigbang')
	
	# Initialize with data
	d3.graph(adjmat)

	# Set node properties
	d3.set_node_properties(color='cluster', scaler='minmax', fontcolor='node_color')




Set the click properties that changes the color to green on click and with a black border.

.. code:: python

	d3.show(click={'fill': '#00FF00', 'stroke': '#000000'})


.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/click_example_1.html" height="700px" width="850px", frameBorder="0"></iframe>



Keep original node color on click but set the stroke to grey and increase both node size and stroke width.


.. code:: python

	d3.show(click={'fill': None, 'stroke': '#F0F0F0', 'size': 2.5, 'stroke-width': 10})


.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/click_example_2.html" height="700px" width="850px", frameBorder="0"></iframe>



.. include:: add_bottom.add