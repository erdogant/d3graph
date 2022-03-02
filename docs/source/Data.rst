.. _code_directive:

-------------------------------------

Input
'''''''''''''''

It is important to get the data in the right shape for the javascript file. In ``d3graph``, the input is an adjacency matrix for which the columns and indexes are the nodes and the elements with a values larger than 0 are an edge. The data is converted in a *json-file* and is embedded in the final HTML file. By embedding the data directly in the HTML file will integrate all scripts and data into a single file which can be very practical. Note that depending on the amount of data, it can result in a heavy HTML file. 

In its simplest form, the input for ``d3.graph()`` is an adjacency matrix in the form of a ``pd.DataFrame()``. The index names are the *source* nodes and the column names the *target* nodes. An edge is created when the vertice between the source and target is larger than 0. This value also represents the strength of the edge.


Adjacency matrix
---------------------------------

Example of a simple adjacency matrix with 4 nodes. The *True* booleans represent the value 1 and the *False* the value 0.

.. table::
  
  +-----------+--------+-----------+--------+-----------+
  |           | Node 1 | Node 2    | Node 3 | Node 4    |
  +===========+========+===========+========+===========+
  | Node 1    | False  | True      | True   | False     |
  +-----------+--------+-----------+--------+-----------+
  | Node 2    | False  | False     | False  | True      |
  +-----------+--------+-----------+--------+-----------+
  | Node 3    | False  | False     | False  | True      |
  +-----------+--------+-----------+--------+-----------+
  | Node 4    | False  | False     | False  | False     |
  +-----------+--------+-----------+--------+-----------+

The following edges will be created:

	* (Node 1, Node 2)
	* (Node 1, Node 3)
	* (Node 2, Node 4)
	* (Node 3, Node 4)


Let's load the *bigbang* network example and examine the adjacency matrix. Here we can clearly see that the source nodes are the index names and and target nodes ar the column names.

.. code:: python
	
	# Import library
	from d3graph import d3graph
	# Initialize
	d3 = d3graph()
	# Load example
	adjmat = d3.import_example('bigbang')

	print(adjmat)
	# target      Amy  Bernadette  Howard  Leonard  Penny  Rajesh  Sheldon
	# source                                                              
	# Amy         0.0         2.0     0.0      0.0    0.0     0.0      0.0
	# Bernadette  0.0         0.0     5.0      0.0    0.0     2.0      0.0
	# Howard      0.0         0.0     0.0      0.0    0.0     0.0      0.0
	# Leonard     0.0         0.0     0.0      0.0    0.0     0.0      0.0
	# Penny       3.0         0.0     0.0      5.0    0.0     0.0      0.0
	# Rajesh      0.0         0.0     0.0      0.0    2.0     0.0      0.0
	# Sheldon     5.0         0.0     2.0      3.0    0.0     0.0      0.0




Create adjacency matrix
---------------------------------

A manner to create a network is by specifying the *source* to *target* nodes with its weights.
The function ``vec2adjmat`` helps doing this: :func:`d3graph.d3graph.d3graph.vec2adjmat`. 
In the following example we will create the *bigbang* network from scratch:

.. code:: python
	
	# Import library
	from d3graph import d3graph, vec2adjmat
	
	# Source node names
	source = ['Penny', 'Penny', 'Amy', 'Bernadette', 'Bernadette', 'Sheldon', 'Sheldon', 'Sheldon', 'Rajesh']
	# Target node names
	target = ['Leonard', 'Amy', 'Bernadette', 'Rajesh', 'Howard', 'Howard', 'Leonard', 'Amy', 'Penny']
	# Edge Weights
	weight = [5, 3, 2, 2, 5, 2, 3, 5, 2]

	# Convert the vector into a adjacency matrix
	adjmat = vec2adjmat(source, target, weight=weight)

	# Initialize
	d3 = d3graph()
	d3.graph(adjmat)
	d3.show()



Output
'''''''''''''''

The output is a single HTML file that contains all scripts and data that forms the interactive force-directed graph. 
If no output directory is specfied, the *systems temporary directory* will be used.
The following example will write the final HTML to a custom directory with a custome file name.

.. code:: python
	
	# Import library
	from d3graph import d3graph, vec2adjmat

	# Initialize
	d3 = d3graph()

	# Load example
	adjmat = d3.import_example('bigbang')

	d3.graph(adjmat)

	# Write to specified directory with custom filename
	d3.show(filepath='c://temp/d3graph_bigbang.html')


.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>

