# Interactive force-directed network creator (d3graph)

[![Python](https://img.shields.io/pypi/pyversions/d3graph)](https://img.shields.io/pypi/pyversions/d3graph)
[![PyPI Version](https://img.shields.io/pypi/v/d3graph)](https://pypi.org/project/d3graph/)
[![License](https://img.shields.io/badge/license-BSD3-green.svg)](https://github.com/erdogant/d3graph/blob/master/LICENSE)
[![Github Forks](https://img.shields.io/github/forks/erdogant/d3graph.svg)](https://github.com/erdogant/d3graph/network)
[![GitHub Open Issues](https://img.shields.io/github/issues/erdogant/d3graph.svg)](https://github.com/erdogant/d3graph/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![DOI](https://zenodo.org/badge/228166657.svg)](https://zenodo.org/badge/latestdoi/228166657)
[![Downloads](https://pepy.tech/badge/d3graph)](https://pepy.tech/project/d3graph)
[![Downloads](https://pepy.tech/badge/d3graph/month)](https://pepy.tech/project/d3graph/month)
[![Sphinx](https://img.shields.io/badge/Sphinx-Docs-Green)](https://erdogant.github.io/d3graph/)
[![Medium](https://img.shields.io/badge/Medium-Blog-green)](https://erdogant.github.io/d3graph/pages/html/Documentation.html#medium-blog)
<!---[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)-->
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->


``d3graph`` is a python package that simplifies the task of creating **interactive** and **stand-alone** networks in **d3 javascript** using **python**.
For this package I was inspired by d3 javascript examples but there was no python package that could create such interactive networks. Here it is; a library that automatically creates D3 javascript and HTML code based on an input adjacency matrix in python! This library does not require you any additional installation, downloads or setting paths to your systems environments. You just need python and this library. All other is taken care off. Huray!

This library will create an interactive and stand-alone network that is build on d3 javascript. ``d3graph`` only requirs an adjacency matrix in the form of an pandas dataframe. Each column and index name represents a node whereas values >0 in the matrix represents an edge. Node links are build from rows to columns. Building the edges from row to columns only matters in directed cases. The network nodes and edges can be adjusted in weight, color etc, based on user defined paramters. 



# 
**⭐️ Star this repo if you like it ⭐️**
#

### Blogs

Read the blog [Creating beautiful stand-alone interactive D3 charts with Python](https://erdogant.github.io/d3graph/pages/html/Documentation.html#medium-blog) to get a structured overview and usage of ``d3graph``.



# 

### [Documentation pages](https://erdogant.github.io/d3graph/)

On the [documentation pages](https://erdogant.github.io/d3graph/) you can find detailed information about the working of the ``d3graph`` with many examples. 

# 

### Installation

##### Install from PyPI

```bash
pip install d3graph
```

##### Import package

```python
from d3graph import d3graph
```

# 

### Examples

Click on the following image to load the interactive **Titanic** network that is created with ``d3graph``. Note that the relations are determined using [HNet. Click here to go to the page with code to make the network.](https://erdogant.github.io/hnet/pages/html/Use%20Cases.html#titanic-dataset)

<p align="left">
  <a href="https://erdogant.github.io/docs/d3graph/titanic_example/index.html">
     <img src="https://github.com/erdogant/d3graph/blob/master/docs/titanic_example/d3graph.png" width="600"/>
  </a>
</p>


##### [Example: Changing node properties](https://erdogant.github.io/d3graph/pages/html/Core%20Functionalities.html#node-label)

<p align="left">
  <a href="https://erdogant.github.io/d3graph/pages/html/Core%20Functionalities.html#node-label">
     <img src="https://github.com/erdogant/d3graph/blob/master/docs/figs/d3graph_node_properties.png" width="900"/>
  </a>
</p>

#

##### [Example: Convert source-target list to adjacency matrix](https://erdogant.github.io/d3graph/pages/html/Data.html#create-adjacency-matrix)

#

##### [Example: Breaking of networks using slider](https://erdogant.github.io/d3graph/pages/html/Examples.html)


<p align="left">
  <a href="https://erdogant.github.io/d3graph/pages/html/Examples.html">
    <img src="https://github.com/erdogant/d3graph/blob/master/docs/figs/d3graph example.png" width="600"/>
  </a>
</p>

<hr>

### Contribute
* All kinds of contributions are welcome!

### Citation
Please cite d3graph in your publications if this is useful for your research. See column right for citation information.

### Maintainer
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
* Contributions are welcome.
* If you wish to buy me a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a> for this work, it is very appreciated :)
