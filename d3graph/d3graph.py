# ---------------------------------
# Name        : d3graph.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ---------------------------------

# Custom packages
from ismember import ismember

# Popular
import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import json
from jinja2 import Environment, PackageLoader
import unicodedata
import logging
import time

# Internal
# from shutil import copyfile
from packaging import version
import webbrowser
import os
from pathlib import Path

import sys
# sys.stdin.reconfigure(encoding='utf-8')

logger = logging.getLogger('')
for handler in logger.handlers[:]: #get rid of existing old handlers
    logger.removeHandler(handler)
console = logging.StreamHandler()
# formatter = logging.Formatter('[%(asctime)s] [XXX]> %(levelname)s> %(message)s', datefmt='%H:%M:%S')
formatter = logging.Formatter('[d3graph] >%(levelname)s> %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
logger = logging.getLogger()

# %%
class d3graph():
    """Make interactive network in D3 javascript."""

    def __init__(self, collision=0.5, charge=250, slider=[None, None], verbose=20):
        """Initialize d3graph.

        Description
        -----------
        Initialize d3graph network with its behavior properties.

        Parameters
        ----------
        collision : float, (default: 0.5)
            Nodes wants to prevent a collision. The higher the number, the more collisions are prevented.
        charge : int, (default: 250)
            Scaling/Response of the network. Towards zero becomes a more nervous network.
        slider : typle [min: int, max: int]:, (default: [0, 10])
            set slider in range to break the network on edge-strength.
        cmap : TYPE, optional
            DESCRIPTION. The default is 'Paired'.
        verbose : int, (default: 20)
            Print progress to screen.
            60: None, 40: Error, 30: Warn, 20: Info, 10: Debug

        Returns
        -------
        None.

        """
        # Some library compatibily checks
        library_compatibility_checks()
        # Set the logger
        set_logger(verbose=verbose)
        # Setup configurations
        self.config = {}
        self.config['network_collision'] = collision
        self.config['network_charge'] = charge * -1
        self.config['slider'] = slider
        # Set path locations
        self.config['curpath'] = os.path.dirname(os.path.abspath(__file__))
        self.config['d3_library'] = os.path.abspath(os.path.join(self.config['curpath'], 'd3js/d3.v3.js'))
        self.config['d3_script'] = os.path.abspath(os.path.join(self.config['curpath'], 'd3js/d3graphscript.js'))
        self.config['css'] = os.path.abspath(os.path.join(self.config['curpath'], 'd3js/style.css'))

    def clean(self):
        """Clean previous results to ensure correct working."""
        if hasattr(self, 'adjmat'): del self.adjmat
        if hasattr(self, 'node_properties'): del self.node_properties

    def show(self, width=1500, height=800, title='d3graph', filepath=None, savename='index', showfig=True):
        """Build and show the graph.

        Parameters
        ----------
        width : int, (default: 1500)
            Width of the window.
        height : int, (default: 800)
            height of the window.
        title : String, (default: None)
            Title of the figure.
        filepath : String, (Default: user temp directory)
            Directory path to save the output, such as 'c://temp/'
        savename : string, (default: 'index')
            Name of the html file.
        showfig : bool, (default: True)
            Open the window to show the network.

        Returns
        -------
        None.

        """
        time.sleep(0.5)
        self.config['network_width'] = width
        self.config['network_height'] = height
        self.config['network_title'] = title
        self.config['path'] = '' if filepath is None else filepath
        self.config['showfig'] = showfig

        # Create dataframe from co-occurence matrix
        self.G = make_graph(self.node_properties, self.edge_properties)
        # Make slider
        self.setup_slider()
        # filepath
        self.config['filepath'] = set_path(filepath, self.config['path'], savename)
        # Create json
        json_data = json_create(self.G)
        # Create html with json file embedded
        self.write_html(json_data)
        # Open the webbrowser
        if self.config['showfig']:
            webbrowser.open(os.path.abspath(self.config['path'] + 'index.html'), new=2)
        # Return
        return self.G

    def set_edge_properties(self, edge_distance=None, edge_distance_minmax=[None, None], directed=False):
        """Edge properties.

        Parameters
        ----------
        edge_distance : Int (default: 30)
            Distance of nodes on the edges.
            * 0: Weighted approach using edge weights in the adjadancy matrix. Weights are normalized between the edge_distance_minmax
            * 80: Constant edge distance
        edge_distance_minmax : tuple(int,int), (default: [None,None].)
            Min and max Distance of nodes on the edges. e.g., [10, 100] Weights are normalized between minimum and maximum (default)
        directed : Bool, (default: False)
            True: Directed edges (with arrow), False: Undirected edges (without arrow)

        Returns
        -------
        None

        """
        self.config['directed'] = directed
        self.config['edge_distance'] = 30 if edge_distance is None else edge_distance
        if edge_distance_minmax[0] is None: edge_distance_minmax[0]=1
        if edge_distance_minmax[1] is None: edge_distance_minmax[1]=20
        self.config['edge_distance_minmax'] = edge_distance_minmax
        # edges to graph (G) (also works with lower versions of networkx)
        self.edge_properties = adjmat2dict(self.adjmat, min_weight=0, edge_distance_minmax=self.config['edge_distance_minmax'])

        # Create color column
        # df['color']=None
        # df.loc[df['weight']>0, 'color']='#FF0000'
        # df.loc[df['weight']<0, 'color']='#0000FF'
        # df.loc[df['weight']==0, 'color']='#000000'


    def set_node_properties(self, label=None, color='#000080', size=10, edge_color='#000000', edge_size=1, cmap='Paired', return_properties=False):
        """Node properties.

        Parameters
        ----------
        color : list of strings (default : '#000080')
            Color of the node.
            * None: All nodes will be have the same color (auto generated).
            * ['#000000']: All nodes will have the same hex color.
            * ['#377eb8','#ffffff','#000000',...]: Hex colors are directly used.
            * ['A']: All nodes will have hte same color. Color is generated on CMAP and the unique labels.
            * ['A','A','B',...]:  Colors are generated using cmap and the unique labels recordingly colored.
        edge_color : list of strings (default : '#000080')
            See color.
        size : array of integers (default=5)
            Size of the node edge., e.g.,  None 10: All nodes will be set on this size, [2,5,1,...]  Specify per node the size
        edge_size : array of integers (default : 1)
            Size of the node edge. Note that node edge sizes are automatically scaled between [0.1 - 4].
            * 10: All nodes will be set on this size,
            * [2,5,1,...]  Specify per node the edge size.
        cmap : String, (default: 'Set1')
            All colors can be reversed with '_r', e.g. 'binary' to 'binary_r'
            'Set1',  'Set2', 'rainbow', 'bwr', 'binary', 'seismic', 'Blues', 'Reds', 'Pastel1', 'Paired'

        Returns
        -------
        None.

        """
        self.config['cmap'] = 'Paired' if cmap is None else cmap
        nodecount = self.adjmat.shape[0]

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
        if not (len(label)==nodecount): raise Exception(logger.warning("Node label must be of same length as the number of nodes"))

        # Set node color
        if isinstance(color, list):
            color = np.array(color)
        elif 'numpy' in str(type(color)):
            pass
        elif isinstance(color, str):
            color = np.array([color] * nodecount)
        elif color is None:
            color = np.array(['#000080'] * nodecount)
        else:
            assert 'Node color not possible'
        color = _get_hexcolor(color, cmap=self.config['cmap'])
        if not (len(color)==nodecount): raise Exception(logger.warning("Node color must be of same length as the number of nodes"))

        # Set node color edge
        if isinstance(edge_color, list):
            edge_color = np.array(edge_color)
        elif 'numpy' in str(type(edge_color)):
            pass
        elif isinstance(edge_color, str):
            edge_color = np.array([edge_color] * nodecount)
        elif isinstance(edge_color, type(None)):
            edge_color = np.array(['#000000'] * nodecount)
        else:
            assert 'Node color edge not possible'
        edge_color = _get_hexcolor(edge_color, cmap=self.config['cmap'])
        if not (len(edge_color)==nodecount): raise Exception(logger.warning("Node color edge must be of same length as the number of nodes"))

        # Set node size
        if isinstance(size, list):
            size = np.array(size)
        elif 'numpy' in str(type(size)):
            pass
        elif isinstance(size, type(None)):
            # Set all nodes same default size
            size = np.ones(nodecount, dtype=int) * 5
        elif isinstance(size, int) or isinstance(size, float):
            size = np.ones(nodecount, dtype=int) * size
        else:
            raise Exception(logger.error("Node size not possible"))
        if not (len(size)==nodecount): raise Exception(logger.warning("Node size must be of same length as the number of nodes"))

        # Set node edge size
        if isinstance(edge_size, list):
            edge_size = np.array(edge_size)
        elif 'numpy' in str(type(edge_size)):
            pass
        elif isinstance(edge_size, type(None)):
            # Set all nodes same default size
            edge_size = np.ones(nodecount, dtype=int) * 1
        elif isinstance(edge_size, int) or isinstance(edge_size, float):
            edge_size = np.ones(nodecount, dtype=int) * edge_size
        else:
            raise Exception(logger.error("Node edge size not possible"))

        # Scale the sizes and get hex colors
        edge_size = _normalize_size(edge_size.reshape(-1, 1), 0.1, 4)
        if not (len(edge_size)==nodecount): raise Exception(logger.warning("Node size edge must be of same length as the number of nodes"))

        # Store in dict
        node_names = self.adjmat.columns.astype(str)
        self.node_properties = {}
        for i, node in enumerate(node_names):
            self.node_properties[node] = {
                'label': label[i],
                'color': color[i].astype(str),
                'size': size[i],
                'edge_size': edge_size[i],
                'edge_color': edge_color[i]}

        # Store in dataframe
        # if not hasattr(self, 'node_properties') or (self.node_properties is None): self.node_properties=pd.DataFrame()
        # self.node_properties['node_name'] = label
        # self.node_properties['node_color'] = color.astype(str)
        # self.node_properties['node_size'] = size.astype(str)
        # self.node_properties['node_size_edge'] = edge_size.astype(str)
        # self.node_properties['node_color_edge'] = edge_color
        # Make strings of the identifiers
        # self.node_properties.index = self.node_properties.index.astype(str)
        # Return properties
        if return_properties:
            return self.node_properties

    def setup_slider(self):
        """Mininum maximum range of the slider.

        Returns
        -------
        tuple: [min, max].

        """
        tmplist = [*self.G.edges.values()]
        edge_weight = list(map(lambda x: x['weight_scaled'], tmplist))

        if self.config['slider'] == [None, None]:
            max_slider = np.ceil(np.max(edge_weight))
            if len(np.unique(edge_weight))>1:
                min_slider = np.maximum(np.floor(np.min(edge_weight)) - 1, 0)
            else:
                min_slider = 0
        else:
            assert len(self.config['slider'])==2, 'Slider must be of type [int, int]'
            min_slider = self.config['slider'][0]
            max_slider = self.config['slider'][1]
        # Store the slider range
        logger.info('Slider range is set to [%g, %g]' %(min_slider, max_slider))
        self.config['slider'] = [min_slider, max_slider]

    def graph(self, adjmat, df=None):
        """Make interactive network in d3js.

        Description
        -----------
        d3graph is a python library that is build on d3js and creates interactive and stand-alone networks.
        The input data is a simple adjacency matrix for which the columns and indexes are the nodes and elements>0 the edges.
        The ouput is a html file that is interactive and stand alone.

        Parameters
        ----------
        adjmat : pd.DataFrame()
            Adjacency matrix (symmetric). Values > 0 are edges.
        df : pd.DataFrame, (default: None)
            index: Samples in the same order as the adjmat. column: columns represents an additional feature of the node

        Examples
        --------
        >>> # Load some libraries
        >>> import pandas as pd
        >>> import numpy as np
        >>> import networkx as nx
        >>> from d3graph import d3graph
        >>>
        >>> # Easy Example
        >>> G = nx.karate_club_graph()
        >>> adjmat = nx.adjacency_matrix(G).todense()
        >>> adjmat = pd.DataFrame(index=range(0,adjmat.shape[0]), data=adjmat, columns=range(0,adjmat.shape[0]))
        >>> # Make the interactive graph
        >>> results = d3graph(adjmat)
        >>>
        >>> # Example with more parameters
        >>> G = nx.karate_club_graph()
        >>> adjmat = nx.adjacency_matrix(G).todense()
        >>> adjmat=pd.DataFrame(index=range(0,adjmat.shape[0]), data=adjmat, columns=range(0,adjmat.shape[0]))
        >>> adjmat.columns=adjmat.columns.astype(str)
        >>> adjmat.index=adjmat.index.astype(str)
        >>> adjmat.iloc[3,4]=5
        >>> adjmat.iloc[4,5]=6
        >>> adjmat.iloc[5,6]=7
        >>>
        >>> # Create dataset
        >>> df = pd.DataFrame(index=adjmat.index)
        >>> df['degree']=np.array([*G.degree()])[:,1]
        >>> df['other info']=np.array([*G.degree()])[:,1]
        >>> weight=df.degree.values*2
        >>> node_color=[]
        >>> for i in range(0,len(G.nodes)):
        >>>     node_color.append(G.nodes[i]['club'])
        >>>     label=node_color
        >>>
        >>> # Make some graphs
        >>> out = d3graph(adjmat, df=df, weight=weight)
        >>> out = d3graph(adjmat, df=df, node_color=weight, weight=weight)
        >>> out = d3graph(adjmat, df=df, node_color=weight, weight=weight, edge_distance=1000)
        >>> out = d3graph(adjmat, df=df, node_color=weight, weight=weight, charge=1000)
        >>> out = d3graph(adjmat, df=df, node_color=label, weight=weight, edge_size=weight, edge_color='#00FFFF', cmap='Set1', collision=1, charge=250)
        >>>
        >>> # Example with conversion to adjacency matrix
        >>> G = nx.karate_club_graph()
        >>> adjmat = nx.adjacency_matrix(G).todense()
        >>> adjmat = pd.DataFrame(index=range(0,adjmat.shape[0]), data=adjmat, columns=range(0,adjmat.shape[0]))
        >>> # Import library
        >>> import d3graph
        >>> # Convert adjacency matrix to vector with source and target
        >>> vec = d3graph.adjmat2vec(adjmat)
        >>> # Convert vector (source and target) to adjacency matrix.
        >>> adjmat1 = d3graph.vec2adjmat(vec['source'], vec['target'], vec['weight'])
        >>> # Check
        >>> np.all(adjmat==adjmat1.astype(int))

        Returns
        -------
        dict with graph and filepath.

        """
        # Clean readily fitted models to ensure correct results
        self.clean()
        # Checks
        self.adjmat = data_checks(adjmat.copy())
        # Set edge properties
        self.set_edge_properties()
        # Set node properties
        self.set_node_properties()

    def write_html(self, json_data):
        """Write html.

        Parameters
        ----------
        json_data : json file

        Returns
        -------
        None.

        """
        content = {
            'json_data': json_data,
            'title': self.config['network_title'],
            'width': self.config['network_width'],
            'height': self.config['network_height'],
            'charge': self.config['network_charge'],
            'edge_distance': self.config['edge_distance'],
            'min_slider': self.config['slider'][0],
            'max_slider': self.config['slider'][1],
            'directed': self.config['directed'],
            'collision': self.config['network_collision']
        }

        jinja_env = Environment(loader=PackageLoader(package_name=__name__, package_path='d3js'))
        index_template = jinja_env.get_template('index.html.j2')
        index_file = Path(self.config['filepath'])
        logger.info('Write to path: [%s]' % index_file.absolute())
        # index_file.write_text(index_template.render(content))
        with open(index_file, "w", encoding="utf-8") as f:
            f.write(index_template.render(content))


# %%
def set_path(filepath, path, savename):
    if filepath is None:
        filepath = savename + '.html'
    else:
        filepath = filepath + savename + '.html'

    if (path is not None) and (not os.path.isdir(path)) and (path != ''):
        logger.info('Create directory: [%s]' %(path))
        os.mkdir(path)

    logger.info("filepath is set to [%s]" %(filepath))
    return filepath


# %%
def set_logger(verbose=20):
    """Set the logger for verbosity messages."""
    logger.setLevel(verbose)


# %% Write network in json file
def json_create(G):
    # Make sure indexing of nodes is correct with the edges
    node_ui = np.array([*G.nodes()])
    node_id = np.arange(0, len(node_ui)).astype(str)
    edges = [*G.edges()]
    source = []
    target = []
    for edge in edges:
        source.append(node_id[edge[0]==node_ui][0])
        target.append(node_id[edge[1]==node_ui][0])

    data = {}
    links = pd.DataFrame([*G.edges.values()]).T.to_dict()
    links_new = []
    for i in range(0, len(links)):
    # for i in node_id:
        links[i]['edge_width'] = links[i].pop('weight_scaled')
        links[i]['edge_weight'] = links[i]['weight']
        links[i]['source'] = int(source[i])
        links[i]['target'] = int(target[i])
        links[i]['source_label'] = edges[i][0]
        links[i]['target_label'] = edges[i][1]
        links_new.append(links[i])
    data['links']=links_new

    nodes = pd.DataFrame([*G.nodes.values()]).T.to_dict()
    # nodeid = [*G.nodes]
    nodes_new = [None] * len(nodes)
    for i in range(0, len(nodes)):
    # for i in node_id:
        nodes[i]['node_name'] = nodes[i].pop('label')
        nodes[i]['node_color'] = nodes[i].pop('color')
        nodes[i]['node_size'] = nodes[i].pop('size')
        nodes[i]['node_size_edge'] = nodes[i].pop('edge_size')
        nodes[i]['node_color_edge'] = nodes[i].pop('edge_color')
        nodes_new[i] = nodes[i]
    data['nodes'] = nodes_new

    return json.dumps(data, separators=(',', ':'))


# %%  Convert adjacency matrix to vector
def adjmat2dict(adjmat, min_weight=0, edge_distance_minmax=[1, 20]):
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
    >>> import bnlearn as bn

    """
    # Convert adjacency matrix into vector
    df = adjmat.stack().reset_index()
    # Set columns
    df.columns = ['source', 'target', 'weight']
    # Remove self loops and no-connected edges
    Iloc = df['source']!=df['target']
    # Keep only edges with a minimum edge strength
    if min_weight is not None:
        logger.info("Keep only edges with weight>%g" %(min_weight))
        Iloc2 = df['weight']>min_weight
        Iloc = Iloc & Iloc2
    df = df.loc[Iloc, :]
    df.reset_index(drop=True, inplace=True)

    # Scale the weights for visualization purposes
    if len(np.unique(df['weight'].values.reshape(-1, 1)))>2:
        df['weight_scaled'] = _normalize_size(df['weight'].values.reshape(-1, 1), edge_distance_minmax[0], edge_distance_minmax[1])
    else:
        df['weight_scaled'] = np.ones(df.shape[0]) * 5

    # Creation dictionary
    source_target = list(zip(df['source'], df['target']))
    dict_edges = {}
    for i, edge in enumerate(source_target):
        dict_edges[edge] = {'weight': df['weight'].iloc[i], 'weight_scaled': df['weight_scaled'].iloc[i], 'color': '#000000'}

    # Return
    return(dict_edges)


# %% Convert dict with edges to graph (G) (also works with lower versions of networkx)
def edges2G(edge_properties, G=None):
    # Create new graph G
    if G is None:
        G = nx.Graph()
    edges = [*edge_properties]
    # Create edges in graph
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight_scaled=np.abs(edge_properties[edge]['weight_scaled']), weight=np.abs(edge_properties[edge]['weight']), color=edge_properties[edge]['color'])
    # Return
    return(G)

# %% Convert adjacency matrix to graph
def make_graph(node_properties, edge_properties):
    """Make graph from node and edge properties.

    Parameters
    ----------
    node_properties : dictionary
        Dictionary containing node properties
    edge_properties : dictionary
        Dictionary containing edge properties
    edge_distance_min : int, (default: None)
        Scale the weights with a minimum value.
    edge_distance_max : int, (default: None)
        Scale the weights with a maximum value.

    Returns
    -------
    dict containing Graph G and dataframe.

    """
    # Convert adjmat
    # node_names = adjmat.index.values
    # adjmat.reset_index(drop=True, inplace=True)
    # adjmat.index = adjmat.index.values.astype(str)
    # adjmat.columns = np.arange(0, adjmat.shape[1]).astype(str)
    # adjmat = adjmat.stack().reset_index()
    # adjmat.columns = ['source', 'target', 'weight']
    adjmat = pd.DataFrame(edge_properties).T.reset_index(level=[0, 1])
    adjmat.rename(columns={"level_0": "source", "level_1": "target"}, inplace=True)
    node_names = [*node_properties.keys()]

    # Width of edge, scale between [1-10] if there is are more then 2 diferent weights

    adjmat['edge_weight'] = adjmat['weight_scaled']

    # Keep only edges with weight
    # edge_weight_original = adjmat['weight'].values.reshape(-1, 1).flatten()
    # adjmat = adjmat.loc[edge_weight_original > 0, :].reset_index(drop=True)

    # Remove self-loops
    # Iloc = adjmat['source'] != adjmat['target']
    # adjmat = adjmat.loc[Iloc, :].reset_index(drop=True)

    # Include source-target label
    source_label = np.repeat('', adjmat.shape[0]).astype('O')
    target_label = np.repeat('', adjmat.shape[0]).astype('O')
    for i in range(0, len(node_names)):
        source_label[adjmat['source']==node_names[i]] = node_names[i]
    for i in range(0, len(node_names)):
        target_label[adjmat['target']==node_names[i]] = node_names[i]
    adjmat['source_label'] = source_label
    adjmat['target_label'] = target_label

    # Make sure indexing of nodes is correct with the edges
    uilabels = np.unique(np.append(adjmat['source'], adjmat['target']))
    tmplabels=adjmat[['source', 'target']]
    adjmat['source']=None
    adjmat['target']=None
    for i in range(0, len(uilabels)):
        I1 = tmplabels['source']==uilabels[i]
        I2 = tmplabels['target']==uilabels[i]
        adjmat.loc[I1, 'source'] = str(i)
        adjmat.loc[I2, 'target'] = str(i)
    
    # Create new Graph
    G = nx.Graph()
    # Add edges to graph
    G = edges2G(edge_properties, G=G)
    
    # try:
    #     G = nx.from_pandas_edgelist(adjmat, 'source', 'target', ['weight', 'edge_weight', 'weight_scaled', 'source_label', 'target_label', 'source', 'target'])
    # except:
    #     G = nx.from_pandas_dataframe(adjmat, 'source', 'target', edge_attr=['weight', 'edge_weight', 'weight_scaled', 'source_label', 'target_label', 'source', 'target'])

    # Add node information
    # A = pd.concat([pd.DataFrame(adjmat[['target', 'target_label']].values), pd.DataFrame(adjmat[['source', 'source_label']].values)], axis=0)
    # A = A.groupby([0, 1]).size().reset_index(name='Freq')
    # IA, IB = ismember(node_properties['node_names'], A[1])
    # node_properties = node_properties.loc[IA, :]
    # node_properties.index = A[0].loc[IB].values.astype(str)
    
    node_properties = pd.DataFrame(node_properties).T
    # node_properties.rename(columns={"index": "node_name"}, inplace=True)

    if not node_properties.empty:
        getnodes = np.array([*G.nodes])
        for col in node_properties.columns:
            for i in range(0, node_properties.shape[0]):
                idx = node_properties.index.values[i]
                if np.any(np.isin(getnodes, node_properties.index.values[i])):
                    G.nodes[idx][col] = str(node_properties[col][idx])
                else:
                    logger.warning("Node not found")

    return G


# %% Normalize in good d3 range
def _normalize_size(getsizes, minscale=0.1, maxscale=4):
    getsizes = MinMaxScaler(feature_range=(minscale, maxscale)).fit_transform(getsizes).flatten()
    getsizes = np.array(list(map(lambda x: round(x, 4), getsizes)))
    return(getsizes)


# %% Convert to hex color
def _get_hexcolor(label, cmap='Paired'):
    label = label.astype(str)
    if label[0][0]!='#':
        label = label.astype(dtype='U7')
        uinode = np.unique(label)
        tmpcolors = np.array(sns.color_palette(cmap, len(uinode)).as_hex())
        IA, IB = ismember(label, uinode)
        label[IA] = tmpcolors[IB]

    return(label)


# %% Do checks
def library_compatibility_checks():
    """Library compatibiliy checks.

    Returns
    -------
    None.

    """
    if not version.parse(nx.__version__) >= version.parse("2.5"):
        logger.error('Networkx version should be >= 2.5')
        logger.info('Hint: pip install -U networkx')


# %% Do checks
def data_checks(adjmat):
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
        adjmat = pd.DataFrame(index=range(0, adjmat.shape[0]), data=adjmat, columns=range(0, adjmat.shape[0]))
    # Set the column and index names as str type
    adjmat.index=adjmat.index.astype(str)
    adjmat.columns=adjmat.columns.astype(str)
    # Column names and index should have the same order.
    if not np.all(adjmat.columns==adjmat.index.values):
        raise Exception(logger.error('adjmat columns and index must have the same identifiers'))
    # Remove special characters from column names
    adjmat = remove_special_chars(adjmat)

    # Return
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
    adjmat.columns = list(map(lambda x: unicodedata.normalize('NFD', x).encode('ascii', 'ignore').decode("utf-8").replace(' ', '_'), adjmat.columns.values.astype(str)))
    adjmat.index = list(map(lambda x: unicodedata.normalize('NFD', x).encode('ascii', 'ignore').decode("utf-8").replace(' ', '_'), adjmat.index.values.astype(str)))
    return adjmat


# %%  Convert adjacency matrix to vector
def vec2adjmat(source, target, weight=None, symmetric=True):
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

    Returns
    -------
    pd.DataFrame
        adjacency matrix.

    Examples
    --------
    >>> source=['Cloudy','Cloudy','Sprinkler','Rain']
    >>> target=['Sprinkler','Rain','Wet_Grass','Wet_Grass']
    >>> vec2adjmat(source, target)
    >>>
    >>> weight=[1,2,1,3]
    >>> vec2adjmat(source, target, weight=weight)

    """
    if len(source)!=len(target): raise Exception('[d3graph] >Source and Target should have equal elements.')
    if weight is None: weight = [1] * len(source)

    df = pd.DataFrame(np.c_[source, target], columns=['source', 'target'])
    # Make adjacency matrix
    adjmat = pd.crosstab(df['source'], df['target'], values=weight, aggfunc='sum').fillna(0)
    # Get all unique nodes
    nodes = np.unique(list(adjmat.columns.values) + list(adjmat.index.values))
    # nodes = np.unique(np.c_[adjmat.columns.values, adjmat.index.values].flatten())

    # Make the adjacency matrix symmetric
    if symmetric:
        # Add missing columns
        node_columns = np.setdiff1d(nodes, adjmat.columns.values)
        for node in node_columns:
            adjmat[node]=0

        # Add missing rows
        node_rows = np.setdiff1d(nodes, adjmat.index.values)
        adjmat=adjmat.T
        for node in node_rows:
            adjmat[node]=0
        adjmat=adjmat.T

        # Sort to make ordering of columns and rows similar
        logger.debug('Order columns and rows.')
        IA, IB = ismember(adjmat.columns.values, adjmat.index.values)
        adjmat = adjmat.iloc[IB, :]
        adjmat.index.name='source'
        adjmat.columns.name='target'

    return(adjmat)


# %%  Convert adjacency matrix to vector
def adjmat2vec(adjmat, min_weight=1):
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
    >>> source=['Cloudy','Cloudy','Sprinkler','Rain']
    >>> target=['Sprinkler','Rain','Wet_Grass','Wet_Grass']
    >>> adjmat = vec2adjmat(source, target)
    >>> vector = adjmat2vec(adjmat)

    """
    # Convert adjacency matrix into vector
    adjmat = adjmat.stack().reset_index()
    # Set columns
    adjmat.columns = ['source', 'target', 'weight']
    # Remove self loops and no-connected edges
    Iloc1 = adjmat['source']!=adjmat['target']
    Iloc2 = adjmat['weight']>=min_weight
    Iloc = Iloc1 & Iloc2
    # Take only connected nodes
    adjmat = adjmat.loc[Iloc,:]
    adjmat.reset_index(drop=True, inplace=True)
    return(adjmat)

# %% Main
if __name__ == '__main__':
    d3graph(sys.argv[1:])
