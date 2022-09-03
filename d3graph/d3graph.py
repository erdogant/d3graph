"""Make interactive network in D3 javascript."""
# ---------------------------------
# Name        : d3graph.py
# Licence     : See licences
# ---------------------------------

import logging
import os
import time
import webbrowser
from json import dumps
from pathlib import Path
from sys import platform
from tempfile import TemporaryDirectory
from typing import List, Union, Tuple
from unicodedata import normalize

import colourmap as cm
import networkx as nx
import numpy as np
import pandas as pd
from ismember import ismember
from jinja2 import Environment, PackageLoader
from packaging import version
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger('')
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
console = logging.StreamHandler()
# formatter = logging.Formatter('[%(asctime)s] [XXX]> %(levelname)s> %(message)s', datefmt='%H:%M:%S')
formatter = logging.Formatter('[d3graph] %(levelname)s> %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
logger = logging.getLogger()


# %%
class d3graph:
    """Make interactive network in D3 javascript."""

    def __init__(self, collision: float = 0.5, charge: int = 250, slider: List[int] = None, verbose: int = 20) -> None:
        """Initialize d3graph.

        Description
        -----------
        d3graph is a Python library that is built on d3js and creates interactive and standalone networks. The input
        data is a simple adjacency matrix for which the columns and indexes are the nodes and elements>0 the edges.
        The output is an HTML file that is interactive and standalone.

        Parameters
        ----------
        collision : float, (default: 0.5)
            Response of the network. Higher means that more collisions are prevented.
        charge : int, (default: 250)
            Edge length of the network. Towards zero becomes a dense network. Higher make edges longer.
        slider : list [min: int, max: int]:, (default: [None, None])
            Slider is automatically set to the range of the edge weights.
        verbose : int, (default: 20)
            Print progress to screen.
            60: None, 40: Error, 30: Warn, 20: Info, 10: Debug

        Returns
        -------
        None.

        """
        if slider is None:
            slider = [None, None]
        # Cleaning
        self._clean()
        # Some library compatibility checks
        library_compatibility_checks()
        # Set the logger
        set_logger(verbose=verbose)
        # Setup configurations
        self.config = {}
        self.config['network_collision'] = collision
        self.config['network_charge'] = charge * -1
        self.config['slider'] = slider
        # Set paths
        self.config['curpath'] = os.path.dirname(os.path.abspath(__file__))
        self.config['d3_library'] = os.path.abspath(os.path.join(self.config['curpath'], 'd3js/d3.v3.js'))
        self.config['d3_script'] = os.path.abspath(os.path.join(self.config['curpath'], 'd3js/d3graphscript.js'))
        self.config['css'] = os.path.abspath(os.path.join(self.config['curpath'], 'd3js/style.css'))

    def _clean(self, clean_config: bool = True) -> None:
        """Clean previous results to ensure correct working."""
        for attr in ('adjmat', 'edge_properties', 'G', 'node_properties'):
            self.__dict__.pop(attr, None)
        if clean_config and hasattr(self, 'config'): del self.config

    def show(self, figsize: Tuple[int, int] = (1500, 800), title: str = 'd3graph', filepath: str = 'd3graph.html',
             showfig: bool = True, overwrite: bool = True) -> None:
        """Build and show the graph.

        Parameters
        ----------
        figsize : tuple, (default: (1500, 800))
            Size of the figure in the browser, [height, width].
        title : String, (default: None)
            Title of the figure.
        filepath : String, (Default: user temp directory)
            File path to save the output
        showfig : bool, (default: True)
            Open the window to show the network.
        overwrite : bool, (default: True)
            Overwrite the existing html file.

        Returns
        -------
        None.

        """
        # Some checks
        if not hasattr(self, 'edge_properties') or not hasattr(self, 'node_properties'):
            logger.warning('No graph detected. <return> Hint: "d3.graph(df)"')
            return None

        self.config['figsize'] = figsize
        self.config['network_title'] = title
        # self.config['path'] = '' if filepath is None else filepath
        self.config['showfig'] = showfig
        self.config['filepath'] = self.set_path(filepath)

        # Create dataframe from co-occurrence matrix
        self.G = make_graph(self.node_properties, self.edge_properties)
        # Make slider
        self.setup_slider()
        # Create json
        json_data = json_create(self.G)
        # Create html with json file embedded
        self.write_html(json_data, overwrite=overwrite)
        # Open the webbrowser
        if self.config['showfig']:
            self._showfig(self.config['filepath'])
            # webbrowser.open(os.path.abspath(self.config['filepath']), new=2)
        # Return
        return self.G

    @staticmethod
    def _showfig(filepath: str, sleep: float = 0.5) -> None:
        # Sleeping is required to prevent overlapping windows
        time.sleep(sleep)
        file_location = os.path.abspath(filepath)
        if platform == "darwin":  # check if on OSX
            file_location = "file:///" + file_location
        webbrowser.open(file_location, new=2)

    def set_edge_properties(self, edge_distance: int = None, minmax: List[float] = None, directed: bool = False, scaler: str = 'zscore', marker_start=None, marker_end='arrow', marker_color='#808080') -> dict:
        """Edge properties.

        Parameters
        ----------
        edge_distance : Int (default: 30)
            Distance of nodes on the edges.
            * 0: Weighted approach using edge weights in the adjacency matrix. Weights are normalized between the minmax
            * 80: Constant edge distance
        minmax : tuple(int,int), (default: [0.5, 15.0])
            Weights are normalized between minimum and maximum
            * [0.5, 15]
        scaler : str, (default: 'zscore')
            Scale the edge-width in the range of a minimum and maximum [0.5, 15] using the following scaler:
            'zscore' : Scale values to Z-scores.
            'minmax' : The sklearn scaler will shrink the distribution.
        directed : Bool, (default: False)
            True: Edges are shown with an marker (e.g. arrow)
            False: Edges do not show markers.
        marker_start : (list of) str, (default: 'arrow')
            The start of the edge can be one of the following markers:
            'arrow','square','circle','stub',None or ''
        marker_end : (list of) str, (default: 'arrow')
            The end of the edge can be one of the following markers:
            'arrow','square','circle','stub',None or ''

        Returns
        -------
        edge_properties: dict
            key: (source, target)
                'weight': weight of the edge
                'weight_scaled': scaled weight of the edge
                'color': color of the edge

        """
        if minmax is None:
            minmax = [0.5, 15.0]
        self.config['directed'] = directed
        self.config['edge_distance'] = 30 if edge_distance is None else edge_distance
        self.config['minmax'] = minmax
        self.config['edge_scaler'] = scaler
        self.config['marker_start'] = marker_start
        self.config['marker_end'] = marker_end
        self.config['marker_color'] = marker_color

        # Set the edge properties
        # Set the edge properties
        self.edge_properties = adjmat2dict(self.adjmat,
                                           min_weight=0,
                                           minmax=self.config['minmax'],
                                           scaler=self.config['edge_scaler'],
                                           marker_start=self.config['marker_start'],
                                           marker_end=self.config['marker_end'],
                                           marker_color=self.config['marker_color'])

		logger.debug('Number of edges: %.0d', len(self.edge_properties.keys()))

    def set_node_properties(self, label: List[str] = None, hover: List[str] = None, color: Union[str, List[str]] = '#000080', size=10, edge_color='#000000', edge_size=1, cmap='Set1', scaler='zscore', minmax = None):
        """Node properties.

        Parameters
        ----------
        label : list of names (default: None)
            The text that is shown on the Node.
            If not specified, the label text will be inherited from the adjacency matrix column-names.
            * ['label 1','label 2','label 3', ...]
        tooltip : list of names (default: None)
            The text that is shown when hovering over the Node.
            If not specified, the text will inherit from the label.
            * ['hover 1','hover 2','hover 3', ...]
        color : list of strings (default: '#000080')
            Color of the node.
            * 'cluster' : Colours are based on the community distance clusters.
            * None: All nodes will have the same color (auto generated).
            * ['#000000']: All nodes will have the same hex color.
            * ['#377eb8','#ffffff','#000000',...]: Hex colors are directly used.
            * ['A']: All nodes will have hte same color. Color is generated on CMAP and the unique labels.
            * ['A','A','B',...]:  Colors are generated using cmap and the unique labels accordingly colored.
        size : array of integers (default: 5)
            Size of the nodes.
            * 10: all nodes sizes are set to 10
            * [10, 5, 3, 1, ...]: Specify node sizes
        edge_color : list of strings (default: '#000080')
            Edge color of the node.
            * 'cluster' : Colours are based on the community distance clusters.
            * ['#377eb8','#ffffff','#000000',...]: Hex colors are directly used.
            * ['A']: All nodes will have hte same color. Color is generated on CMAP and the unique labels.
            * ['A','A','B',...]:  Colors are generated using cmap and the unique labels recordingly colored.
        edge_size : array of integers (default: 1)
            Size of the node edge. Note that node edge sizes are automatically scaled between [0.1 - 4].
            * 1: All nodes will be set on this size,
            * [2,5,1,...]  Specify per node the edge size.
        cmap : String, (default: 'Set1')
            All colors can be reversed with '_r', e.g. 'binary' to 'binary_r'
            'Set1',  'Set2', 'rainbow', 'bwr', 'binary', 'seismic', 'Blues', 'Reds', 'Pastel1', 'Paired'
        scaler : str, (default: 'zscore')
            Scale the node size in the range of a minimum and maximum [0.5, 15] using the following scaler:
            'zscore' : Scale values to Z-scores.
            'minmax' : The sklearn scaler will shrink the distribution.
            None : No scaler is used.
        minmax : tuple, (default: [10, 50])
            Scale the node size in the range of a minimum and maximum [5, 50] using the following scaler:
            'zscore' : Scale values to Z-scores.
            'minmax' : The sklearn scaler will shrink the distribution.
            None : No scaler is used.

        Returns
        -------
        node_properties: dict
            key: node_name
                'label': Label of the node
                'color': color of the node
                'size': size of the node
                'edge_size': edge_size of the node
                'edge_color': edge_color of the node

        """
        if minmax is None:
            minmax = [10, 50]
        node_names = self.adjmat.columns.astype(str)
        nodecount = self.adjmat.shape[0]
        cluster_label = np.zeros_like(node_names).astype(int)
        if isinstance(color, str) and len(color) != 7: raise ValueError(
            'Input parameter [color] has wrong format. Must be like color="#000000"')
        if isinstance(color, list) and len(color) == 0: raise ValueError(
            'Input parameter [color] has wrong format and length. Must be like: color=["#000000", "...", "#000000"]')
        if isinstance(color, list) and (not np.all(list(map(lambda x: len(x) == 7, color)))): raise ValueError(
            '[color] contains incorrect length of hex-color! Hex must be of length 7: ["#000000", "#000000", etc]')
        if isinstance(color, list) and len(color) != nodecount:
            raise ValueError(f'Input parameter [color] has wrong length. Must be of length: {str(nodecount)}')

        self.config['cmap'] = 'Paired' if cmap is None else cmap
        self.config['node_scaler'] = scaler

        # Set node label
        if isinstance(label, list):
            label = np.array(label).astype(str)
        elif 'numpy' in str(type(label)):
            pass
        elif isinstance(label, str):
            label = np.array([label] * nodecount)
        elif self.adjmat is not None:
            label = self.adjmat.columns.astype(str)
        else:
            label = np.array([''] * nodecount)
        if len(label) != nodecount: raise ValueError("[label] must be of same length as the number of nodes")

        # tooltip text
        if isinstance(tooltip, list):
            tooltip = np.array(tooltip).astype(str)
        elif 'numpy' in str(type(tooltip)):
            pass
        elif isinstance(tooltip, str):
            tooltip = np.array([tooltip] * nodecount)
        elif label is not None:
            tooltip = label
        else:
            hover = np.array([''] * nodecount)
        if len(hover) != nodecount: raise ValueError("[Hover text] must be of same length as the number of nodes")

        # Set node color
        if isinstance(color, list) and len(color) == nodecount:
            color = np.array(color)
        elif 'numpy' in str(type(color)):
            color = _get_hexcolor(color, cmap=self.config['cmap'])
        elif isinstance(color, str) and color == 'cluster':
            color, cluster_label, _ = self.get_cluster_color(node_names=node_names)
        elif isinstance(color, str):
            color = np.array([color] * nodecount)
        elif color is None:
            color = np.array(['#000080'] * nodecount)
        else:
            assert 'Node color not possible'
        if len(color) != nodecount: raise ValueError("[color] must be of same length as the number of nodes")

        # Set node color edge
        if isinstance(edge_color, list):
            edge_color = np.array(edge_color)
        elif 'numpy' in str(type(edge_color)):
            pass
        elif isinstance(edge_color, str) and edge_color == 'cluster':
            # Only cluster if this is not done previously. Otherwise, the random generator can create slightly
            # different clustering results, and thus colors.
            if len(np.unique(cluster_label)) == 1:
                edge_color, cluster_label, _ = self.get_cluster_color(node_names=node_names)
            else:
                edge_color = color
        elif isinstance(edge_color, str):
            edge_color = np.array([edge_color] * nodecount)
        elif isinstance(edge_color, type(None)):
            edge_color = np.array(['#000000'] * nodecount)
        else:
            assert 'Node color edge not possible'

        # Check correctness of hex colors
        edge_color = _get_hexcolor(edge_color, cmap=self.config['cmap'])
        # Check length edge color with node count. This should match.
        if len(edge_color) != nodecount: raise ValueError("[edge_color] must be of same length as the number of nodes")

        # Set node size
        if isinstance(size, list):
            size = np.array(size)
        elif 'numpy' in str(type(size)):
            pass
        elif isinstance(size, type(None)):
            # Set all nodes same default size
            size = np.ones(nodecount, dtype=int) * 5
        elif isinstance(size, (int, float)):
            size = np.ones(nodecount, dtype=int) * size
        else:
            raise ValueError(logger.error("Node size not possible"))
        # Scale the sizes
        size = _normalize_size(size.reshape(-1, 1), minmax[0], minmax[1], scaler=self.config['node_scaler'])
        if len(size) != nodecount: raise ValueError("Node size must be of same length as the number of nodes")

        # Set node edge size
        if isinstance(edge_size, list):
            edge_size = np.array(edge_size)
        elif 'numpy' in str(type(edge_size)):
            pass
        elif isinstance(edge_size, type(None)):
            # Set all nodes same default size
            edge_size = np.ones(nodecount, dtype=int) * 1
        elif isinstance(edge_size, (int, float)):
            edge_size = np.ones(nodecount, dtype=int) * edge_size
        else:
            raise ValueError(logger.error("Node edge size not possible"))

        # Scale the edge-sizes
        edge_size = _normalize_size(edge_size.reshape(-1, 1), 0.5, 4, scaler=self.config['node_scaler'])
        if len(edge_size) != nodecount: raise ValueError("[edge_size] must be of same length as the number of nodes")

        # Store in dict
        self.node_properties = {}
        for i, node in enumerate(node_names):
            self.node_properties[node] = {'name'         : node,
                                          'label'        : label[i],
                                          'hover'        : hover[i],
                                          'color'        : color[i].astype(str),
                                          'size'         : size[i],
                                          'edge_size'    : edge_size[i],
                                          'edge_color'   : edge_color[i],
                                          'cluster_label': cluster_label[i]}

        logger.info('Number of unique nodes: %.0d', len(self.node_properties.keys()))

    # compute clusters
    def get_cluster_color(self, node_names: list = None) -> tuple:
        """Clustering of graph labels.

        Parameters
        ----------
        df : Pandas DataFrame with columns
            'source'
            'target'
            'weight'

        Returns
        -------
        dict cluster_label.

        """
        import networkx as nx
        from community import community_louvain
        if node_names is None: node_names = [*self.node_properties.keys()]

        df = adjmat2vec(self.adjmat.copy())
        G = nx.from_pandas_edgelist(df, edge_attr=True, create_using=nx.MultiGraph)
        # Partition
        G = G.to_undirected()
        cluster_labels = community_louvain.best_partition(G)
        # Extract clusterlabels
        y = list(map(lambda x: cluster_labels.get(x), cluster_labels.keys()))
        hex_colors, _ = cm.fromlist(y, cmap=self.config['cmap'], scheme='hex')
        labx = {key: {'name': key, 'color': hex_colors[i], 'cluster_label': cluster_labels.get(key)} for i, key in
                enumerate(cluster_labels.keys())}

        # return
        color = np.array(list(map(lambda x: labx.get(x)['color'], node_names)))
        cluster_label = np.array(list(map(lambda x: labx.get(x)['cluster_label'], node_names)))
        return color, cluster_label, node_names

    def setup_slider(self) -> None:
        """Minimum maximum range of the slider.

        Returns
        -------
        tuple: [min, max].

        """
        tmplist = [*self.G.edges.values()]
        # edge_weight = list(map(lambda x: x['weight_scaled'], tmplist))
        edge_weight = list(map(lambda x: x['weight'], tmplist))

        # if self.config['slider'] == [None, None]:
        max_slider = np.ceil(np.max(edge_weight))
        # max_slider = np.ceil(np.sort(edge_weight)[-2])
        if len(np.unique(edge_weight)) > 1:
            min_slider = np.maximum(np.floor(np.min(edge_weight)) - 1, 0)
        else:
            min_slider = 0

        # Store the slider range
        self.config['slider'] = [int(min_slider), int(max_slider)]
        logger.info('Slider range is set to [%g, %g]' % (self.config['slider'][0], self.config['slider'][1]))

    def graph(self, adjmat) -> None:
        """Process the adjacency matrix and set all properties to default.

        Description
        -----------
        This function processes the adjacency matrix. The nodes are the column and index names.
        A connect edge is seen in case vertices have values larger than 0. The strenght of the edge is based on the vertices values.

        Parameters
        ----------
        adjmat : pd.DataFrame()
            Adjacency matrix (symmetric). Values > 0 are edges.

        Examples
        --------
        >>> from d3graph import d3graph
        >>>
        >>> # Initialize
        >>> d3 = d3graph()
        >>>
        >>> # Load karate example
        >>> adjmat, df = d3.import_example('karate')
        >>>
        >>> # Initialize
        >>> d3.graph(adjmat)
        >>>
        >>> # Node properties
        >>> d3.set_node_properties(label=df['label'].values, color=df['label'].values, size=df['degree'].values, edge_size=df['degree'].values, cmap='Set1')
        >>>
        >>> # Edge properties
        >>> d3.set_edge_properties(directed=True)
        >>>
        >>> # Plot
        >>> d3.show()

        Returns
        -------
        None

        """
        # Clean readily fitted models to ensure correct results
        self._clean(clean_config=False)
        # Checks
        self.adjmat = data_checks(adjmat.copy())
        # Set default edge properties
        self.set_edge_properties()
        # Set default node properties
        self.set_node_properties()

    def write_html(self, json_data, overwrite: bool = True) -> None:
        """Write html.

        Parameters
        ----------
        json_data : json file

        Returns
        -------
        None.

        """
        content = {'json_data'    : json_data,
                   'title'        : self.config['network_title'],
                   'width'        : self.config['figsize'][0],
                   'height'       : self.config['figsize'][1],
                   'charge'       : self.config['network_charge'],
                   'edge_distance': self.config['edge_distance'],
                   'min_slider'   : self.config['slider'][0],
                   'max_slider'   : self.config['slider'][1],
                   'directed'     : self.config['directed'],
                   'collision'    : self.config['network_collision']}

        jinja_env = Environment(loader=PackageLoader(package_name=__name__, package_path='d3js'))
        index_template = jinja_env.get_template('index.html.j2')
        index_file = Path(self.config['filepath'])
        logger.info(f'Write to path: [{index_file.absolute()}]')
        # index_file.write_text(index_template.render(content))
        if os.path.isfile(index_file) and overwrite:
            logger.info(f'File already exists and will be overwritten: [{index_file}]')
            os.remove(index_file)
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(index_template.render(content))

    def set_path(self, filepath='d3graph.html') -> str:
        """Set the file path.

        Parameters
        ----------
        filepath : str
            filename and or full pathname.
            * 'd3graph.html'
            * 'c://temp/'
            * 'c://temp/d3graph.html'

        Returns
        -------
        filepath : str
            Path to graph.

        """
        dirname, filename = os.path.split(filepath)

        if filename in (None, ''):
            filename = 'd3graph.html'

        if dirname in (None, ''):
            dirname = TemporaryDirectory().name

        os.makedirs(dirname, exist_ok=True)
        filepath = os.path.abspath(os.path.join(dirname, filename))
        logger.debug(f'filepath is set to [{filepath}]')
        return filepath

    def import_example(self, network='small'):
        """Import example.

        Parameters
        ----------
        network : str, optional
            Import example adjacency matrix. The default is 'small'.

        Returns
        -------
        adjmat : pd.DataFrame()

        """
        if network == 'small':
            source = ['node A', 'node F', 'node B', 'node B', 'node B', 'node A', 'node C', 'node Z']
            target = ['node F', 'node B', 'node J', 'node F', 'node F', 'node M', 'node M', 'node A']
            weight = [5.56, 0.5, 0.64, 0.23, 0.9, 3.28, 0.5, 0.45]
            adjmat = vec2adjmat(source, target, weight=weight)
            return adjmat
        elif network == 'bigbang':
            source = ['Penny', 'Penny', 'Amy', 'Bernadette', 'Bernadette', 'Sheldon', 'Sheldon', 'Sheldon', 'Rajesh']
            target = ['Leonard', 'Amy', 'Bernadette', 'Rajesh', 'Howard', 'Howard', 'Leonard', 'Amy', 'Penny']
            weight = [5, 3, 2, 2, 5, 2, 3, 5, 2]
            adjmat = vec2adjmat(source, target, weight=weight)
            return adjmat
        elif network == 'karate':
            import scipy
            if version.parse(scipy.__version__) < version.parse('1.8.0'):
                raise ImportError(
                    '[d3graph] >Error: This release requires scipy version >= 1.8.0. Try: pip install -U scipy>=1.8.0')

            G = nx.karate_club_graph()
            adjmat = nx.adjacency_matrix(G).todense()
            adjmat = pd.DataFrame(index=range(adjmat.shape[0]), data=adjmat, columns=range(adjmat.shape[0]))
            adjmat.columns = adjmat.columns.astype(str)
            adjmat.index = adjmat.index.astype(str)
            adjmat.iloc[3, 4] = 5
            adjmat.iloc[4, 5] = 6
            adjmat.iloc[5, 6] = 7

            df = pd.DataFrame(index=adjmat.index)
            df['degree'] = np.array([*G.degree()])[:, 1]
            df['label'] = [G.nodes[i]['club'] for i in range(len(G.nodes))]

            return adjmat, df


# %%
def set_logger(verbose: int = 20) -> None:
    """Set the logger for verbosity messages."""
    logger.setLevel(verbose)


# %% Write network in json file
def json_create(G: nx.Graph) -> str:
    """Create json from Graph.

    Parameters
    ----------
    G : Networkx object
        Graph G

    Returns
    -------
    json dumps

    """
    # Ensure indexing of nodes is correct with the edges
    node_ui = np.array([*G.nodes()])
    node_id = np.arange(0, len(node_ui)).astype(str)
    edges = [*G.edges()]
    source = []
    target = []
    for edge in edges:
        source.append(node_id[edge[0] == node_ui][0])
        target.append(node_id[edge[1] == node_ui][0])

    links = pd.DataFrame([*G.edges.values()]).T.to_dict()
    links_new = []
    for i in range(len(links)):
        links[i]['edge_width'] = links[i].pop('weight_scaled')
        links[i]['edge_weight'] = links[i]['weight']
        links[i]['source'] = int(source[i])
        links[i]['target'] = int(target[i])
        links[i]['source_label'] = edges[i][0]
        links[i]['target_label'] = edges[i][1]
        links[i]['marker_start'] = links[i]['marker_start']
        links[i]['marker_end'] = links[i]['marker_end']
        links[i]['marker_color'] = links[i]['marker_color']
        links_new.append(links[i])
    nodes = pd.DataFrame([*G.nodes.values()]).T.to_dict()
    nodes_new = [None] * len(nodes)
    for i, node in enumerate(nodes):
        nodes[i]['node_name'] = nodes[i].pop('label')
        # nodes[i]['node_label'] = nodes[i].pop('label')
        nodes[i]['node_tooltip'] = nodes[i].pop('tooltip')
        nodes[i]['node_color'] = nodes[i].pop('color')
        nodes[i]['node_size'] = nodes[i].pop('size')
        nodes[i]['node_size_edge'] = nodes[i].pop('edge_size')
        nodes[i]['node_color_edge'] = nodes[i].pop('edge_color')
        # Combine all information into new list
        nodes_new[i] = nodes[i]
    data = {'links': links_new, 'nodes': nodes_new}
    return dumps(data, separators=(',', ':'))


# %%  Convert adjacency matrix to vector
def adjmat2dict(adjmat: pd.DataFrame, min_weight: float = 0.0, minmax=None, scaler: str = 'zscore', marker_start=None, marker_end='arrow', marker_color='#808080') -> dict:
    """Convert adjacency matrix into vector with source and target.

    Parameters
    ----------
    adjmat : pd.DataFrame()
        Adjacency matrix.
    min_weight : float
        edges are returned with a minimum weight.
    minmax : tuple(int,int), (default: [0.5, 15])
        Weights are normalized between minimum and maximum
        * [0.5, 15]
    scaler : str

    Returns
    -------
    edge_properties: dict
        key: (source, target)
            'weight': weight of the edge.
            'weight_scaled': scaled weight of the edge.
            'color': color of the edge.
            'marker_start': '', 'circle', 'square', 'arrow', 'stub'
            'marker_end': '', 'circle', 'square', 'arrow', 'stub'
            'marker_color': hex color of the marker.

    """
    # Convert adjacency matrix into vector
    if minmax is None:
        minmax = [0.5, 15]
    df = adjmat.stack().reset_index()
    # Set columns
    df.columns = ['source', 'target', 'weight']
    # Remove self loops and no-connected edges
    Iloc = df['source'] != df['target']
    # Keep only edges with a minimum edge strength
    if min_weight:
        logger.info("Keep only edges with weight>%g" % min_weight)
        Iloc2 = df['weight'] > min_weight
        Iloc = Iloc & Iloc2
    df = df.loc[Iloc, :]
    df.reset_index(drop=True, inplace=True)

    # Scale the weights for visualization purposes
    if len(np.unique(df['weight'].values.reshape(-1, 1))) > 2:
        df['weight_scaled'] = _normalize_size(df['weight'].values.reshape(-1, 1), minmax[0], minmax[1], scaler=scaler)
    else:
        df['weight_scaled'] = np.ones(df.shape[0]) * 1

    # Set marker start-end
    if marker_start is None: marker_start=''
    if marker_end is None: marker_end=''
    df['marker_start']=marker_start
    df['marker_end']=marker_end
    df['marker_color']=marker_color

    # Creation dictionary
    source_target = list(zip(df['source'], df['target']))
    # Return
    return {edge: {'weight': df['weight'].iloc[i], 'weight_scaled': df['weight_scaled'].iloc[i], 'color': '#808080', 'marker_start': df['marker_start'].iloc[i], 'marker_end': df['marker_end'].iloc[i], 'marker_color': df['marker_color'].iloc[i]} for
            i, edge in enumerate(source_target)}

# %% Convert dict with edges to graph (G) (also works with lower versions of networkx)
def edges2G(edge_properties: dict, G: nx.Graph = None) -> nx.Graph:
    """Convert edges to Graph.

    Parameters
    ----------
    edge_properties : dictionary
        Dictionary containing edge properties
    G : Networkx object
        Graph G.

    Returns
    -------
    Graph G

    """
    # Create new graph G
    G = nx.DiGraph() if G is None else G
    edges = [*edge_properties]
    # Create edges in graph
    for edge in edges:
        #G.add_edge(edge[0], edge[1], weight_scaled=np.abs(edge_properties[edge]['weight_scaled']),
        #           weight=np.abs(edge_properties[edge]['weight']), color=edge_properties[edge]['color'])
        G.add_edge(edge[0], edge[1], marker_color=edge_properties[edge]['marker_color'], marker_start=edge_properties[edge]['marker_start'], marker_end=edge_properties[edge]['marker_end'], weight_scaled=np.abs(edge_properties[edge]['weight_scaled']), weight=np.abs(edge_properties[edge]['weight']), color=edge_properties[edge]['color'])

	# Return
    return G


# %% Convert dict with nodes to graph (G)
def nodes2G(node_properties: dict, G: nx.Graph = None) -> nx.Graph:
    """Convert nodes to Graph.

    Parameters
    ----------
    node_properties : dictionary
        Dictionary containing node properties
    G : Networkx object
        Graph G.

    Returns
    -------
    Graph G

    """
    # Create new graph G
    if G is None:
        G = nx.DiGraph()
    # Get node properties
    node_properties = pd.DataFrame(node_properties).T
    # Store node properties in Graph
    if not node_properties.empty:
        getnodes = np.array([*G.nodes])
        for col in node_properties.columns:
            for i in range(node_properties.shape[0]):
                idx = node_properties.index.values[i]
                if np.any(np.isin(getnodes, node_properties.index.values[i])):
                    G.nodes[idx][col] = str(node_properties[col][idx])
                else:
                    logger.warning(f'Node [{node_properties.index.values[i]}] not found')
    return G


# %% Convert adjacency matrix to graph
def make_graph(node_properties: dict, edge_properties: dict) -> dict:
    """Make graph from node and edge properties.

    Parameters
    ----------
    node_properties : dictionary
        Dictionary containing node properties
    edge_properties : dictionary
        Dictionary containing edge properties

    Returns
    -------
    dict containing Graph G and dataframe.

    """
    # Create new Graph, add edges, and nodes
    G = nx.DiGraph()
    G = edges2G(edge_properties, G=G)
    G = nodes2G(node_properties, G=G)

    return G


# %% Normalize in good d3 range
def _normalize_size(getsizes, minscale: Union[int, float] = 0.5, maxscale: Union[int, float] = 4,
                    scaler: str = 'zscore'):
    # Instead of Min-Max scaling, that shrinks any distribution in the [0, 1] interval, scaling the variables to
    # Z-scores is better. Min-Max Scaling is too sensitive to outlier observations and generates unseen problems,
    # out-of-scale datapoints.
    if scaler == 'zscore' and len(np.unique(getsizes)) > 3:
        getsizes = (getsizes.flatten() - np.mean(getsizes)) / np.std(getsizes)
        getsizes = getsizes + (minscale - np.min(getsizes))
    elif scaler == 'minmax':
        getsizes = MinMaxScaler(feature_range=(minscale, maxscale)).fit_transform(getsizes).flatten()
    else:
        getsizes = getsizes.ravel()
    # Max digits is 4
    getsizes = np.array(list(map(lambda x: round(x, 4), getsizes)))

    return getsizes


# %% Convert to hex color
def _get_hexcolor(label, cmap: str = 'Paired'):
    label = label.astype(str)
    if label[0][0] != '#':
        label = label.astype(dtype='U7')
        uinode = np.unique(label)
        tmpcolors = cm.rgb2hex(cm.fromlist(uinode, cmap=cmap, method='seaborn')[0])
        IA, IB = ismember(label, uinode)
        label[IA] = tmpcolors[IB]

    return label


# %% Do checks
def library_compatibility_checks() -> None:
    """Library compatibility checks.

    Returns
    -------
    None.

    """
    if not version.parse(nx.__version__) >= version.parse('2.5'):
        logger.error('Networkx version should be >= 2.5')
        logger.info('Hint: pip install -U networkx')


# %% Do checks
def data_checks(adjmat: pd.DataFrame) -> pd.DataFrame:
    """Check input Adjacency matrix.

    Parameters
    ----------
    adjmat : pd.DataFrame()
        Adjacency matrix (symmetric). Values > 0 are edges.

    Returns
    -------
    adjmat : pd.DataFrame()

    """
    if 'numpy' in str(type(adjmat)):
        logger.info('Converting numpy matrix into Pandas DataFrame')
        adjmat = pd.DataFrame(index=range(adjmat.shape[0]), data=adjmat, columns=range(adjmat.shape[0]))

    # Set the column and index names as str type
    adjmat.index = adjmat.index.astype(str)
    adjmat.columns = adjmat.columns.astype(str)
    # Column names and index should have the same order.
    if not np.all(adjmat.columns == adjmat.index.values):
        raise ValueError(logger.error('adjmat columns and index must have the same identifiers'))
    # Remove special characters from column names
    adjmat = remove_special_chars(adjmat)

    return adjmat


# %% Remove special characters from column names
def remove_special_chars(adjmat):
    """Remove special characters.

    Parameters
    ----------
    adjmat : pd.DataFrame()
        Adjacency matrix (symmetric). Values > 0 are edges.

    Returns
    -------
    adjmat : pd.DataFrame()

    """
    logger.debug('Removing special chars and replacing with "_"')
    adjmat.columns = list(
        map(lambda x: normalize('NFD', x).encode('ascii', 'ignore').decode("utf-8").replace(' ', '_'),
            adjmat.columns.values.astype(str)))
    adjmat.index = list(
        map(lambda x: normalize('NFD', x).encode('ascii', 'ignore').decode("utf-8").replace(' ', '_'),
            adjmat.index.values.astype(str)))
    return adjmat


# %%  Convert adjacency matrix to vector
def vec2adjmat(source: list, target: list, weight: List[int] = None, symmetric: bool = True, aggfunc='sum') -> pd.DataFrame:
    """Convert source and target into adjacency matrix.

    Parameters
    ----------
    source : list
        The source node.
    target : list
        The target node.
    weight : list of int
        The Weights between the source-target values
    symmetric : bool, optional
        Make the adjacency matrix symmetric with the same number of rows as columns. The default is True.
    aggfunc : str, optional
        Aggregate function in case multiple values exists for the same relationship.
        'sum' (default)

    Returns
    -------
    pd.DataFrame
        adjacency matrix.

    Examples
    --------
    >>> source = ['Cloudy', 'Cloudy', 'Sprinkler', 'Rain']
    >>> target = ['Sprinkler', 'Rain', 'Wet_Grass', 'Wet_Grass']
    >>> vec2adjmat(source, target)

    >>> weight = [1, 2, 1, 3]
    >>> vec2adjmat(source, target, weight=weight)

    """
    if len(source) != len(target): raise ValueError('[d3graph] >Source and Target should have equal elements.')
    if weight is None: weight = [1] * len(source)

    df = pd.DataFrame(np.c_[source, target], columns=['source', 'target'])
    # Make adjacency matrix
    adjmat = pd.crosstab(df['source'], df['target'], values=weight, aggfunc=aggfunc).fillna(0)
    # Get all unique nodes
    nodes = np.unique(list(adjmat.columns.values) + list(adjmat.index.values))
    # nodes = np.unique(np.c_[adjmat.columns.values, adjmat.index.values].flatten())

    # Make the adjacency matrix symmetric
    if symmetric:
        # Add missing columns
        node_columns = np.setdiff1d(nodes, adjmat.columns.values)
        for node in node_columns:
            adjmat[node] = 0

        # Add missing rows
        node_rows = np.setdiff1d(nodes, adjmat.index.values)
        adjmat = adjmat.T
        for node in node_rows:
            adjmat[node] = 0
        adjmat = adjmat.T

        # Sort to make ordering of columns and rows similar
        logger.debug('Order columns and rows.')
        _, IB = ismember(adjmat.columns.values, adjmat.index.values)
        adjmat = adjmat.iloc[IB, :]
        adjmat.index.name = 'source'
        adjmat.columns.name = 'target'

    # Force columns to be string type
    adjmat.columns = adjmat.columns.astype(str)
    return adjmat



# %%  Convert adjacency matrix to vector
def adjmat2vec(adjmat, min_weight: float = 1.0) -> pd.DataFrame:
    """Convert adjacency matrix into vector with source and target.

    Parameters
    ----------
    adjmat : pd.DataFrame()
        Adjacency matrix.

    min_weight : float
        edges are returned with a minimum weight.

    Returns
    -------
    pd.DataFrame()
        nodes that are connected based on source and target

    Examples
    --------
    >>> source = ['Cloudy', 'Cloudy', 'Sprinkler', 'Rain']
    >>> target = ['Sprinkler', 'Rain', 'Wet_Grass', 'Wet_Grass']
    >>> adjmat = vec2adjmat(source, target, weight=[1, 2, 1, 3])
    >>> vector = adjmat2vec(adjmat)

    """
    # Convert adjacency matrix into vector
    adjmat = adjmat.stack().reset_index()
    # Set columns
    adjmat.columns = ['source', 'target', 'weight']
    # Remove self loops and no-connected edges
    Iloc1 = adjmat['source'] != adjmat['target']
    Iloc2 = adjmat['weight'] >= min_weight
    Iloc = Iloc1 & Iloc2
    # Take only connected nodes
    adjmat = adjmat.loc[Iloc, :]
    adjmat.reset_index(drop=True, inplace=True)
    return adjmat
