# Interactive force-directed network creator (d3graph)
[![Python](https://img.shields.io/pypi/pyversions/d3graph)](https://img.shields.io/pypi/pyversions/d3graph)
[![PyPI Version](https://img.shields.io/pypi/v/d3graph)](https://pypi.org/project/d3graph/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/erdogant/d3graph/blob/master/LICENSE)
[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)
[![Github Forks](https://img.shields.io/github/forks/erdogant/d3graph.svg)](https://github.com/erdogant/d3graph/network)
[![GitHub Open Issues](https://img.shields.io/github/issues/erdogant/d3graph.svg)](https://github.com/erdogant/d3graph/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Downloads](https://pepy.tech/badge/d3graph)](https://pepy.tech/project/d3graph)
[![Downloads](https://pepy.tech/badge/d3graph/month)](https://pepy.tech/project/d3graph/month)
[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)
[![DOI](https://zenodo.org/badge/228166657.svg)](https://zenodo.org/badge/latestdoi/228166657)
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->

    Star it if you like it!

* ``d3graph`` is a python package that simplifies the task of creating **interactive** and **stand-alone** networks in **d3 javascript** using **python**.
For this package I was inspired by d3 javascript examples but there was no python package that could create such interactive networks. Therefore I decided to create a package that automatically creates d3js javascript and html code based on a input adjacency matrix in python! This library does not require you any additional installation, downloads or setting paths to your systems environments. You just need python and this library. All other is taken care off. Huray!


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

+----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+
|    |   0 |   1 |   2 |   3 |   4 |   5 |   6 |   7 |   8 |   9 |   10 |   11 |   12 |   13 |   14 |   15 |   16 |   17 |   18 |   19 |   20 |   21 |   22 |   23 |   24 |   25 |   26 |   27 |   28 |   29 |   30 |   31 |   32 |   33 |
+====+=====+=====+=====+=====+=====+=====+=====+=====+=====+=====+======+======+======+======+======+======+======+======+======+======+======+======+======+======+======+======+======+======+======+======+======+======+======+======+
|  0 |   0 |   1 |   1 |   1 |   1 |   1 |   1 |   1 |   1 |   0 |    1 |    1 |    1 |    1 |    0 |    0 |    0 |    1 |    0 |    1 |    0 |    1 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    1 |    0 |    0 |
+----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+
|  1 |   1 |   0 |   1 |   1 |   0 |   0 |   0 |   1 |   0 |   0 |    0 |    0 |    0 |    1 |    0 |    0 |    0 |    1 |    0 |    1 |    0 |    1 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    1 |    0 |    0 |    0 |
+----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+
...
+----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+
| 32 |   1 |   1 |   1 |   0 |   5 |   0 |   0 |   1 |   0 |   0 |    0 |    0 |    1 |    1 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |
+----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+
| 33 |   1 |   0 |   0 |   0 |   0 |   6 |   1 |   0 |   0 |   0 |    1 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |
+----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+------+


```

```python
G = nx.karate_club_graph()
adjmat = nx.adjacency_matrix(G).todense()
adjmat = pd.DataFrame(index=range(0,adjmat.shape[0]), data=adjmat, columns=range(0,adjmat.shape[0]))
adjmat.iloc[3,4]=5
adjmat.iloc[4,5]=6
```

* Make d3graph
```python
G_d3   = d3graph(adjmat)
```

The output looks as below:
<p align="center">
  <img src="https://github.com/erdogant/d3graph/blob/master/docs/figs/d3graph_1.png" width="300" />
  <img src="https://github.com/erdogant/d3graph/blob/master/docs/figs/d3graph_2.png" width="300" />
  <img src="https://github.com/erdogant/d3graph/blob/master/docs/figs/d3graph_3.png" width="300" />
  <img src="https://github.com/erdogant/d3graph/blob/master/docs/figs/d3graph_4.png" width="300" />
</p>


```python

source = ['node A','node F','node B','node B','node B','node A','node C','node Z']
target = ['node F','node B','node J','node F','node F','node M','node M','node A']
weight = [5.56, 0.5, 0.64, 0.23, 0.9,3.28,0.5,0.45]

# Import library
from d3graph import d3graph, vec2adjmat

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
out = d3graph(adjmat)

# Example B: Color nodes
out = d3graph(adjmat, node_color=adjmat.columns.values)

# Example C: include node size
node_size = [10,20,10,10,15,10,5]
out = d3graph(adjmat, node_color=adjmat.columns.values, node_size=node_size)

# Example D: include node-edge-size
out = d3graph(adjmat, node_color=adjmat.columns.values, node_size=node_size, node_size_edge=node_size[::-1], cmap='Set2')

# Example E: include node-edge color
out = d3graph(adjmat, node_color=adjmat.columns.values, node_size=node_size, node_size_edge=node_size[::-1], node_color_edge='#00FFFF')

# Example F: Change colormap
out = d3graph(adjmat, node_color=adjmat.columns.values, node_size=node_size, node_size_edge=node_size[::-1], node_color_edge='#00FFFF', cmap='Set2')

# Example H: Include directed links. Arrows are set from source -> target
out = d3graph(adjmat, node_color=adjmat.columns.values, node_size=node_size, node_size_edge=node_size[::-1], node_color_edge='#00FFFF', cmap='Set2', directed=True)

```

### Example: including Dataframe with additional node information

```python

G = nx.karate_club_graph()
adjmat = nx.adjacency_matrix(G).todense()
adjmat=pd.DataFrame(index=range(0,adjmat.shape[0]), data=adjmat, columns=range(0,adjmat.shape[0]))
adjmat.columns=adjmat.columns.astype(str)
adjmat.index=adjmat.index.astype(str)

# Make the dataframe
df = pd.DataFrame(index=adjmat.index)

# Add some columns. Note that columns that start with: node_ are removed from the information.

df['degree']=np.array([*G.degree()])[:,1]
df['other info']=np.array([*G.degree()])[:,1]

node_color = []
for i in range(0,len(G.nodes)):
    node_color.append(G.nodes[i]['club'])
    node_name=node_color
df['name']=node_name

node_size = df.degree.values*2

# Make some graphs
out = d3graph(adjmat, df=df, node_color=node_size, node_size=node_size)
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
