# Interactive network creator (d3graph)
Creation of an interactive network in d3 javascript from an adjacency matrix

[![Build Status](https://travis-ci.org/erdoganta/d3graph.svg?branch=master)](https://travis-ci.org/erdoganta/d3graph)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen)](https://erdoganta.github.io/d3graph/)
[![codecov](https://codecov.io/gh/erdoganta/d3graph/branch/master/graph/badge.svg)](https://codecov.io/gh/erdoganta/d3graph)
[![PyPI Version](https://img.shields.io/pypi/v/imagededup)](https://pypi.org/project/d3graph/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/erdoganta/d3graph/blob/master/LICENSE)

d3graph is a python package that simplifies the task of creating **interactive** networks in **d3 javascript**.

<p align="center">
  <img src="readme_figures/d3graph.png" width="600" />
</p>

This package provides functionality to create a interactive and stand-alone network that is build on d3 javascript. The required input data is an adjacency matrix in the form of an pandas dataframe. Each column and index name represents a node whereas values >0 in the matrix represents a edge between the vertices. Node links are build in a directed manner from row to column. The edges, and nodes are adjusted according to the input parameters. 

Detailed documentation for the package can be found at: [https://erdoganta.github.io/d3graph/](https://erdoganta.github.io/d3graph/)

d3graph is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
It is distributed under the Apache 2.0 license.

## Contents
- [Installation](#%EF%B8%8F-installation)
- [Quick Start](#-quick-start)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

## Installation
There are two ways to install d3graph:

* Install d3graph from PyPI (recommended):

```
pip install d3graph
```

> ⚠️ **Note**: d3graph requires networkx to be v2 or higher. 

* Install d3graph from the GitHub source:

```bash
git clone https://github.com/erdoganta/d3graph.git
cd imagededup
pip install "cython>=0.29"
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
G      = nx.karate_club_graph()
adjmat = nx.adjacency_matrix(G).todense()
adjmat = pd.DataFrame(index=range(0,adjmat.shape[0]), data=adjmat, columns=range(0,adjmat.shape[0]))
adjmat.iloc[3,4]=5
adjmat.iloc[4,5]=6
adjmat.iloc[5,6]=7
```
- Make d3graph

```python
G_d3   = d3graph(adjmat)
```

The output looks as below:

<p align="center">
  <img src="readme_figures/plot_d3graph_1.png" width="600" />
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
