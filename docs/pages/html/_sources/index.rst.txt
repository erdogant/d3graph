D3Graph
=======

|python| |pypi| |docs| |LOC| |downloads_month| |downloads_total| |license| |forks| |open issues| |project status| |medium| |colab| |donate|

.. note::
	`D3Graph: Creating beautiful stand-alone interactive D3 charts with Python. <https://towardsdatascience.com/creating-beautiful-stand-alone-interactive-d3-charts-with-python-804117cb95a7>`_

.. note::
	`D3Blocks: The Python Library to Create Interactive and Standalone D3js Charts. <https://towardsdatascience.com/d3blocks-the-python-library-to-create-interactive-and-standalone-d3js-charts-3dda98ce97d4>`_


.. raw:: html

   <iframe src="https://erdogant.github.io\docs\d3blocks\d3graph_example3.html" height="775px" width="775px", frameBorder="0"></iframe>


The ``d3graph`` library is a Python library that is built on D3 and creates a stand-alone, and interactive force-directed network graph. It allows the creation, manipulation, and study of the structure, dynamics, and functions of complex networks. The input data is an adjacency matrix for which the columns and indexes are the nodes and the elements with a value of one or larger are considered to be an edge. The output is a single HTML file that contains the interactive force-directed graph. ``d3graph`` has several features, among them a slider that can break the edges of the network based on the edge value, a double click on a node will highlight the node and its connected edges and many more options to customize the network based on the edge and node properties.


-----------------------------------

.. note::
	**Your ❤️ is important to keep maintaining this package.** You can `support <https://erdogant.github.io/d3graph/pages/html/Documentation.html>`_ in various ways, have a look at the `sponser page <https://erdogant.github.io/d3graph/pages/html/Documentation.html>`_.
	Report bugs, issues and feature extensions at `github <https://github.com/erdogant/d3graph/>`_ page.

	.. code-block:: console

	   pip install d3graph

-----------------------------------


Content
=======

.. toctree::
   :maxdepth: 1
   :caption: Background
   
   Abstract
   D3


.. toctree::
   :maxdepth: 1
   :caption: Installation
   
   Installation


.. toctree::
   :maxdepth: 1
   :caption: Data
   
   Data

.. toctree::
  :maxdepth: 1
  :caption: Methods

  Core_Functionalities
  Node properties
  Edge properties
  On Click Actions


.. toctree::
  :maxdepth: 1
  :caption: Examples

  Examples


.. toctree::
  :maxdepth: 1
  :caption: Documentation
  
  Documentation
  Coding quality
  d3graph.d3graph



Indices and tables
====================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



.. |python| image:: https://img.shields.io/pypi/pyversions/d3graph.svg
    :alt: |Python
    :target: https://erdogant.github.io/d3graph/

.. |pypi| image:: https://img.shields.io/pypi/v/d3graph.svg
    :alt: |Python Version
    :target: https://pypi.org/project/d3graph/

.. |docs| image:: https://img.shields.io/badge/Sphinx-Docs-blue.svg
    :alt: Sphinx documentation
    :target: https://erdogant.github.io/d3graph/

.. |LOC| image:: https://sloc.xyz/github/erdogant/d3graph/?category=code
    :alt: lines of code
    :target: https://github.com/erdogant/d3graph

.. |downloads_month| image:: https://static.pepy.tech/personalized-badge/d3graph?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month
    :alt: Downloads per month
    :target: https://pepy.tech/project/d3graph

.. |downloads_total| image:: https://static.pepy.tech/personalized-badge/d3graph?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads
    :alt: Downloads in total
    :target: https://pepy.tech/project/d3graph

.. |license| image:: https://img.shields.io/badge/license-BSD3-green.svg
    :alt: License
    :target: https://github.com/erdogant/d3graph/blob/master/LICENSE

.. |forks| image:: https://img.shields.io/github/forks/erdogant/d3graph.svg
    :alt: Github Forks
    :target: https://github.com/erdogant/d3graph/network

.. |open issues| image:: https://img.shields.io/github/issues/erdogant/d3graph.svg
    :alt: Open Issues
    :target: https://github.com/erdogant/d3graph/issues

.. |project status| image:: http://www.repostatus.org/badges/latest/active.svg
    :alt: Project Status
    :target: http://www.repostatus.org/#active

.. |medium| image:: https://img.shields.io/badge/Medium-Blog-green.svg
    :alt: Medium Blog
    :target: https://erdogant.github.io/d3graph/pages/html/Documentation.html#medium-blog

.. |donate| image:: https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors
    :alt: donate
    :target: https://erdogant.github.io/d3graph/pages/html/Documentation.html#

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Colab example
    :target: https://erdogant.github.io/d3graph/pages/html/Documentation.html#colab-notebook



.. include:: add_bottom.add