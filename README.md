# Interactive network creator (d3graph)
[![Build Status](https://travis-ci.org/erdoganta/d3graph.svg?branch=master)](https://travis-ci.org/erdoganta/d3graph)
[![PyPI Version](https://img.shields.io/pypi/v/d3graph)](https://pypi.org/project/d3graph/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/erdoganta/d3graph/blob/master/LICENSE)

Creation of an interactive network in d3 javascript from an adjacency matrix.

d3graph is a python package that simplifies the task of creating **interactive** and **stand-alone** networks in **d3 javascript**.
For this package I was inspired by various examples shown on the internet, such as www.alanzucconi.com/2015/11/01/interactive-graphs-in-the-browser/ or https://andrewmellor.co.uk/blog/articles/2014/12/14/d3-networks/ or http://bl.ocks.org/d3noob/5141278. But all of these examples are purely based on javascript without any easy python package to generate the networks. Therefore I decided to create a package that automatically creates d3js javascript and html code based on a input adjacency matrix in python! Huray!

<p align="center">
  <img src="docs/titanic_example/d3graph.png" width="600" />
</p>

This package provides functionality to create a interactive and stand-alone network that is build on d3 javascript. D3graph only requirs an adjacency matrix in the form of an pandas dataframe. Each column and index name represents a node whereas values >0 in the matrix represents an edge. Node links are build from rows to columns. Building the edges from row to columns only matters in directed cases. The network nodes and edges can be adjusted in weight, color etc, based on user defined paramters. 

## Contents
- [Installation](#%EF%B8%8F-installation)
- [Quick Start](#-quick-start)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

## Installation
d3graph is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. **Note**: d3graph requires networkx to be v2 or higher.
It is distributed under the Apache 2.0 license. There are two ways to install d3graph:


* Install d3graph from PyPI (recommended):
```
pip install d3graph
```

* Install d3graph from the GitHub source:
```bash
git clone https://github.com/erdoganta/d3graph.git
cd d3graph
pip install "networkx>=2"
python setup.py install
```  

## Quick Start

In order to create an interactive and stand-alone d3graph, following workflow can be used:

- Import d3graph method

```python
from d3graph import d3graph
```

- Create simple example dataset

```python
G = nx.karate_club_graph()
adjmat = nx.adjacency_matrix(G).todense()
adjmat = pd.DataFrame(index=range(0,adjmat.shape[0]), data=adjmat, columns=range(0,adjmat.shape[0]))
adjmat.iloc[3,4]=5
adjmat.iloc[4,5]=6
```
- Make d3graph

```python
G_d3   = d3graph(adjmat)
```

The output looks as below:

<p align="center">
  <img src="docs/figs/d3graph_1.png" width="300" />
  <img src="docs/figs/d3graph_2.png" width="300" />
  <img src="docs/figs/d3graph_3.png" width="300" />
  <img src="docs/figs/d3graph_4.png" width="300" />
</p>


## Contribute
We welcome all kinds of contributions.
See the [Contribution](CONTRIBUTING.md) guide for more details.

## Citation
Please cite d3graph in your publications if this is useful for your research. Here is an example BibTeX entry:
```BibTeX
@misc{erdoganta2019d3graph,
  title={d3graph},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdoganta/d3graph}},
}
```

## Maintainers
* Erdogan Taskesen, github: [erdoganta](https://github.com/erdoganta)

## © Copyright
See [LICENSE](LICENSE) for details.
