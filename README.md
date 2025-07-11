# d3graph: Interactive force-directed networks

[![Python](https://img.shields.io/pypi/pyversions/d3graph)](https://img.shields.io/pypi/pyversions/d3graph)
[![Pypi](https://img.shields.io/pypi/v/d3graph)](https://pypi.org/project/d3graph/)
[![Docs](https://img.shields.io/badge/Sphinx-Docs-Green)](https://erdogant.github.io/d3graph/)
[![LOC](https://sloc.xyz/github/erdogant/d3graph/?category=code)](https://github.com/erdogant/d3graph/)
[![Downloads](https://static.pepy.tech/personalized-badge/d3graph?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month)](https://pepy.tech/project/d3graph)
[![Downloads](https://static.pepy.tech/personalized-badge/d3graph?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/d3graph)
[![License](https://img.shields.io/badge/license-BSD3-green.svg)](https://github.com/erdogant/d3graph/blob/master/LICENSE)
[![Forks](https://img.shields.io/github/forks/erdogant/d3graph.svg)](https://github.com/erdogant/d3graph/network)
[![Issues](https://img.shields.io/github/issues/erdogant/d3graph.svg)](https://github.com/erdogant/d3graph/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![DOI](https://zenodo.org/badge/228166657.svg)](https://zenodo.org/badge/latestdoi/228166657)
[![Medium](https://img.shields.io/badge/Medium-Blog-black)](https://towardsdatascience.com/creating-beautiful-stand-alone-interactive-d3-charts-with-python-804117cb95a7)
[![Donate](https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors)](https://erdogant.github.io/d3graph/pages/html/Documentation.html#)
<!---[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)-->
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->


### 

<div>

<a href="https://erdogant.github.io/d3graph/"><img src="https://github.com/erdogant/d3graph/blob/master/docs/figs/logo.png" width="175" align="left" /></a>
``d3graph`` is a python package that simplifies the task of creating **interactive** and **stand-alone** networks in **d3 javascript** using **python**.
For this package, I was inspired by d3 JavaScript, but there was no Python package that could create such interactive networks. Here it is: a library that automatically creates D3 JavaScript and HTML code based on an input adjacency matrix in Python! This library does not require any additional installation, downloads, or setting paths to your system's environments. You just need Python and this library. All other is taken care of. Huray! Navigate to [API documentations](https://erdogant.github.io/d3graph/) for more detailed information. **⭐️ Star it if you like it ⭐️**
</div>

---

### Key Pipelines

| Feature | Description |
|--------|-------------|
| [**Graph**](https://erdogant.github.io/d3graph/pages/html/Core_Functionalities.html) | Create the network Graph. |
| [**set_node_properties**](https://erdogant.github.io/d3graph/pages/html/Node%20properties.html) | Set the node properties for the network graph |
| [**set_edge_properties**](https://erdogant.github.io/d3graph/pages/html/Edge%20properties.html) | Set the edge properties for the network graph |

---

### Resources and Links
- **Blog Posts:** [Medium](https://erdogant.medium.com/)
- **Documentation:** [Website](https://erdogant.github.io/d3graph)
- **Bug Reports and Feature Requests:** [GitHub Issues](https://github.com/erdogant/d3graph/issues)

---

### Installation

##### Install d3graph from PyPI
```bash
pip install d3graph
```
##### Install d3graph from GitHub source
```bash
pip install git+https://github.com/erdogant/d3graph
```
##### Load library
```python
# Import library
from d3graph import d3graph
```
---

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
