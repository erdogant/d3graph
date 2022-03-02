.. _code_directive:

-------------------------------------

Abstract
''''''''

Background
	Visualizing your data can be the key to success in projects because it can reveal hidden insights in the data, and improve understanding. The best way to convince people is by letting them see and interact with their data. Despite many visualization packages being available in Python, it is not always straightforward to make beautiful stand-alone and interactive charts that can also work outside your own machine. The key advantage of D3 is that it works with web standards so you don’t need any other technology than a browser to make use of D3. Importantly, interactive charts can help to not just tell the reader something but let the reader see, engage, and ask questions. 

Aim
	Creating a a python package that simplifies the task of creating interactive and stand-alone networks in d3 javascript using python. 

Results
	The ``d3graph`` library is a Python library that is built on D3 and creates a stand-alone, and interactive force-directed network graph. The input data is an adjacency matrix for which the columns and indexes are the nodes and the elements with a value of one or larger are considered to be an edge. The output is a single HTML file that contains the interactive force-directed graph. ``d3graph`` has several features, among them a slider that can break the edges of the network based on the edge value, a double click on a node will highlight the node and its connected edges and many more options to customize the network based on the edge and node properties.

	For this package I was inspired by d3 javascript examples but there was no python package that could create such interactive networks. Therefore I decided to create a package that automatically creates d3js javascript and html code based on a input adjacency matrix in python! This library does not require you any additional installation, downloads or setting paths to your systems environments. You just need python and this library. All other is taken care off. Huray!

    
Schematic overview
'''''''''''''''''''

The schematic overview of our approach is as following:

.. _schematic_overview:

.. figure:: ../figs/schematic_overview.jpg


.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>

