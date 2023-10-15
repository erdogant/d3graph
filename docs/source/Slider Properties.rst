Slider Properties
'''''''''''''''''''''' 

The function of the slider is to break the network on its weights with an interactive slider.


Set Slider Threshold
************************************
The slider can be set in a particular state on start-up with the `set_slider` parameter.

.. code:: python
	
	# Import library
	from d3graph import d3graph

	# Initialize
	d3 = d3graph()
	# Load example data set
	adjmat = d3.import_example('bigbang')
	
	# Create the graph with default setting
	d3.graph(adjmat)

    # Show the chat and break the network on threshold value 3.
    d3.show(set_slider=3, filepath=r'D:\REPOS\erdogant.github.io\docs\d3graph\d3graph\set_slider.html', figsize=(600, 400))


.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/set_set_slider.html" height="700px" width="850px", frameBorder="0"></iframe>



Remove Slider
************************************
The slider can be removed from the html.

.. code:: python
	
	# Import library
	from d3graph import d3graph

	# Initialize
	d3 = d3graph()
	# Load example data set
	adjmat = d3.import_example('bigbang')
	
	# Create the graph with default setting
	d3.graph(adjmat)

    # Show the chat and break the network on threshold value 3.
    d3.show(show_slider=False, filepath=r'D:\REPOS\erdogant.github.io\docs\d3graph\d3graph\show_slider.html', figsize=(600, 400))


.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/d3graph/show_slider.html" height="700px" width="850px", frameBorder="0"></iframe>




.. include:: add_bottom.add