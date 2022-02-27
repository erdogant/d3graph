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
[![Medium](https://img.shields.io/badge/Medium-Blog-green)](https://towardsdatascience.com/creating-beautiful-stand-alone-interactive-d3-charts-with-python-804117cb95a7)
<!---[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)-->
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->

### I am working on an improved version of d3graph with lower code complexity,  removal of redundant features and documenation pages. It will also be much much easier to change any of the edge and node properties. Be patient. Its coming soon. For those that can not wait. A working version is readily present in github.

``d3graph`` is a python package that simplifies the task of creating **interactive** and **stand-alone** networks in **d3 javascript** using **python**.
For this package I was inspired by d3 javascript examples but there was no python package that could create such interactive networks. Therefore I decided to create a package that automatically creates d3js javascript and html code based on a input adjacency matrix in python! This library does not require you any additional installation, downloads or setting paths to your systems environments. You just need python and this library. All other is taken care off. Huray!

# 
**Star this repo if you like it! ⭐️**
#

### Blog/Documentation

* [**Medium blog: Creating beautiful stand-alone interactive D3 charts with Python**](https://towardsdatascience.com/creating-beautiful-stand-alone-interactive-d3-charts-with-python-804117cb95a7)
* [**d3graph documentation pages (Sphinx)**](https://erdogant.github.io/d3graph/)


<p align="center">
  <a href="https://erdogant.github.io/docs/d3graph/titanic_example/index.html">
     <img src="https://github.com/erdogant/d3graph/blob/master/docs/titanic_example/d3graph.png" width="600"/>
  </a>
</p>

This package provides functionality to create a interactive and stand-alone network that is build on d3 javascript. D3graph only requirs an adjacency matrix in the form of an pandas dataframe. Each column and index name represents a node whereas values >0 in the matrix represents an edge. Node links are build from rows to columns. Building the edges from row to columns only matters in directed cases. The network nodes and edges can be adjusted in weight, color etc, based on user defined paramters. 

* <a href="https://erdogant.github.io/docs/d3graph/titanic_example/index.html">d3graph example</a> 

## Installation
d3graph is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. **Note**: d3graph requires networkx to be v2 or higher.
It is distributed under the Apache 2.0 license. There are two ways to install d3graph:


* Install d3graph from PyPI (recommended):
```
pip install d3graph
```

* Install d3graph from the GitHub source:
```bash
git clone https://github.com/erdogant/d3graph.git
cd d3graph
pip install -U .
```  

## Quick Start
In order to create an interactive and stand-alone d3graph, following workflow can be used:

* Import d3graph method
```python
from d3graph import d3graph
```

* Create simple example dataset for which the input matrix should look this:


```python
G = nx.karate_club_graph()
adjmat = nx.adjacency_matrix(G).todense()
adjmat = pd.DataFrame(index=range(0,adjmat.shape[0]), data=adjmat, columns=range(0,adjmat.shape[0]))
adjmat.iloc[3,4]=5
adjmat.iloc[4,5]=6


# Import library
from d3graph import d3graph
# Initialize
d3 = d3graph()
# Process adjacency matrix
d3.graph(adjmat)
# Show plot
d3.show()

```

The output looks as below:
<p align="center">
  <img src="https://github.com/erdogant/d3graph/blob/master/docs/figs/d3graph_1.png" width="300" />
  <img src="https://github.com/erdogant/d3graph/blob/master/docs/figs/d3graph_2.png" width="300" />
  <img src="https://github.com/erdogant/d3graph/blob/master/docs/figs/d3graph_3.png" width="300" />
  <img src="https://github.com/erdogant/d3graph/blob/master/docs/figs/d3graph_4.png" width="300" />
</p>


### Simple example with various settings

```python
source = ['node A', 'node F', 'node B', 'node B', 'node B', 'node A', 'node C', 'node Z']
target = ['node F', 'node B', 'node J', 'node F', 'node F', 'node M', 'node M', 'node A']
weight = [5.56, 0.5, 0.64, 0.23, 0.9, 3.28, 0.5, 0.45]

# Convert to adjacency matrix
adjmat = vec2adjmat(source, target, weight=weight)

print(adjmat)
# target  node A  node B  node F  node J  node M  node C  node Z
# source                                                        
# node A    0.00     0.0    5.56    0.00    3.28     0.0     0.0
# node B    0.00     0.0    1.13    0.64    0.00     0.0     0.0
# node F    0.00     0.5    0.00    0.00    0.00     0.0     0.0
# node J    0.00     0.0    0.00    0.00    0.00     0.0     0.0
# node M    0.00     0.0    0.00    0.00    0.00     0.0     0.0
# node C    0.00     0.0    0.00    0.00    0.50     0.0     0.0
# node Z    0.45     0.0    0.00    0.00    0.00     0.0     0.0


# Example A: simple interactive network
d3 = d3graph()
d3.graph(adjmat)
d3.show()

# Example B: Color nodes
# d3 = d3graph()
d3.graph(adjmat)
# Set node properties
d3.set_node_properties(color=adjmat.columns.values)
d3.show()

size = [10, 20, 10, 10, 15, 10, 5]

# Example C: include node size
d3.set_node_properties(color=adjmat.columns.values, size=size)
d3.show()

# Example D: include node-edge-size
d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1])
d3.show()

# Example E: include node-edge color
d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1], edge_color='#000000')
d3.show()

# Example F: Change colormap
d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1], edge_color='#00FFFF', cmap='Set2')
d3.show()

# Example H: Include directed links. Arrows are set from source -> target
d3.set_edge_properties(directed=True)
d3.set_node_properties(color=adjmat.columns.values, size=size, edge_size=size[::-1], edge_color='#00FFFF', cmap='Set2')
d3.show()


```

### Example with various settings

```python

import networkx as nx
import pandas as pd
from d3graph import d3graph

G = nx.karate_club_graph()
adjmat = nx.adjacency_matrix(G).todense()
adjmat=pd.DataFrame(index=range(0,adjmat.shape[0]), data=adjmat, columns=range(0,adjmat.shape[0]))
adjmat.columns=adjmat.columns.astype(str)
adjmat.index=adjmat.index.astype(str)
adjmat.iloc[3,4]=5
adjmat.iloc[4,5]=6
adjmat.iloc[5,6]=7

from tabulate import tabulate
print(tabulate(adjmat.head(), tablefmt="grid", headers="keys"))

df = pd.DataFrame(index=adjmat.index)
df['degree']=np.array([*G.degree()])[:,1]
df['other info']=np.array([*G.degree()])[:,1]
node_size=df.degree.values*2
node_color=[]
for i in range(0,len(G.nodes)):
    node_color.append(G.nodes[i]['club'])
    node_name=node_color

# Make some graphs
d3 = d3graph()

d3.graph(adjmat)
d3.set_node_properties(color=node_color, cmap='Set1')
d3.show()

d3.set_node_properties(label=node_name, color=node_color, cmap='Set1')
d3.show()

d3.set_node_properties(adjmat, size=node_size)
d3.show()

d3.set_node_properties(color=node_size, size=node_size)
d3.show()

d3.set_edge_properties(edge_distance=100)
d3.set_node_properties(color=node_size, size=node_size)
d3.show()

d3 = d3graph(charge=1000)
d3.graph(adjmat)
d3.set_node_properties(color=node_size, size=node_size)
d3.show()

d3 = d3graph(collision=1, charge=250)
d3.graph(adjmat)
d3.set_node_properties(color=node_name, size=node_size, edge_size=node_size, cmap='Set1')
d3.show()

d3 = d3graph(collision=1, charge=250)
d3.graph(adjmat)
d3.set_node_properties(color=node_name, size=node_size, edge_size=node_size, edge_color='#00FFFF', cmap='Set1')
d3.show()

```


### Contribute
* Thanks to Oliver Verver who helped to fix some bugs in d3js (https://github.com/oliver3).
* All kinds of contributions are welcome!

### Citation
Please cite d3graph in your publications if this is useful for your research. See column right for citation information.

### Maintainer
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
* Contributions are welcome.
* If you wish to buy me a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a> for this work, it is very appreciated :)
