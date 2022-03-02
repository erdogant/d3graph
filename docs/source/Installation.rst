.. _code_directive:

-------------------------------------

Quickstart
''''''''''

A quick example how to learn a model on a given dataset.


.. code:: python

	# Import library
	from d3graph import d3graph, vec2adjmat

	# Create example network
	source = ['node A','node F','node B','node B','node B','node A','node C','node Z']
	target = ['node F','node B','node J','node F','node F','node M','node M','node A']
	weight = [5.56, 0.5, 0.64, 0.23, 0.9, 3.28, 0.5, 0.45]
	# Convert to adjacency matrix
	adjmat = vec2adjmat(source, target, weight=weight)

	# Initialize
	d3 = d3graph()
	# Proces adjmat
	d3.graph(adjmat)
	# Plot
	d3.show()

	# Make changes in node properties
	d3.set_node_properties(color=adjmat.columns.values, label=['node 1','node 2','node 3','node 4','node 5','node 6','node 7'])
	# Plot
	d3.show(filepath='c://temp/')


Installation
''''''''''''

Create environment
------------------


If desired, install ``d3graph`` from an isolated Python environment using conda:

.. code-block:: python

    conda create -n env_d3graph python=3.8
    conda activate env_d3graph


Install via ``pip``:

.. code-block:: console

    pip install d3graph


Install directly from ``github``:

.. code-block:: console

    pip install git+https://github.com/erdogant/d3graph


Uninstalling
''''''''''''

If you want to remove your ``d3graph`` installation with your environment, it can be as following:

.. code-block:: console

   # List all the active environments. d3graph should be listed.
   conda env list

   # Remove the d3graph environment
   conda env remove --name d3graph

   # List all the active environments. d3graph should be absent.
   conda env list


.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>

