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

# Internal
# from shutil import copyfile
from packaging import version
import webbrowser
import os
from pathlib import Path

import sys
# sys.stdin.reconfigure(encoding='utf-8')

# %% Main
def d3graph(adjmat, df=None, node_name=None, node_color='#000080', node_color_edge='#000000', node_size=10, node_size_edge=1, width=1500, height=800, collision=0.5, charge=250, edge_width=None, edge_distance=None, directed=False, edge_distance_minmax=[None,None], title='d3graph', slider=None, savepath=None, savename='index', cmap='Paired', showfig=True, verbose=3):
    """Make the interactive network in d3js.

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
    node_name : array, (default: None)
        Label name for the nodes, e.g., ['1','2','3',...]
    node_color : list of strings (default : '#000080')
        Color of the node.
        * None: Do something style: All nodes will be have the same color (auto generated).
        * ['A']: All nodes will have hte same color. Color is generated on CMAP and the unique labels.
        * ['#000000']: All nodes will have the same hex color.
        * ['A','A','B',...]:  Colors are generated using cmap and the unique labels recordingly colored.
        * ['#377eb8','#ffffff','#000000',...]: Hex colors are directly used.
    node_color_edge : list of strings (default : '#000080')
        See node_color.
    node_size : array of integers (default=5)
        Size of the node edge., e.g.,  None 10: All nodes will be set on this size, [2,5,1,...]  Specify per node the size
    node_size_edge : array of integers (default : 1)
        Size of the node edge. Note that node edge sizes are automatically scaled between [0.1 - 4].
        * 10: All nodes will be set on this size,
        * [2,5,1,...]  Specify per node the edge size.
    edge_width : float (default: None)
        width of the edges.
        * None: The values in the data are rescaled between [1-10] and used for the width.
        * 1: all values have the same width
    edge_distance : Int (default: 30)
        Distance of nodes on the edges.
        * 0: Weighted approach using edge weights in the adjadancy matrix. Weights are normalized between the edge_distance_minmax
        * 80: Constant edge distance

    NETWORK BEHAVIOR SETTINGS
    --------------------------
    collision : float, (default: 0.5)
        Nodes wants to prevent a collision. The higher the number, the more collisions are prevented.
    charge : int, (default: 250)
        Scaling/Response of the network. Towards zero becomes a more nervous network.
    directed : Bool, (default: False)
        True: Directed edges (with arrow), False: Undirected edges (without arrow)
    edge_distance_minmax : tuple(int,int), (default: [None,None].)
        Min and max Distance of nodes on the edges. e.g., [10, 100] Weights are normalized between minimum and maximum (default)

    FIGURE SETTINGS
    ---------------
    width : int, (default: 1500)
        Width of the window.
    height : int, (default: 800)
        height of the window.
    title : String, (default: None)
        Title of the figure.
    cmap : String, (default: 'Set1')
        All colors can be reversed with '_r', e.g. 'binary' to 'binary_r'
        * 'Set1'
        * 'Set2'
        * 'rainbow'
        * 'bwr'
        * 'binary'
        * 'seismic'
        * 'Blues'
        * 'Reds'
        * 'Pastel1'
        * 'Paired'
        * 'Set1'
    slider : typle [int, int]:, (default: [0,10])
        Slider to break the network on edge-strength, e.g. set slider in range 0 to 10
    savepath : String, (Default: user temp directory)
        Directory path to save the output, such as 'c://temp/'
    savename : string, (default: 'index')
        Name of the html file.
    showfig : bool, (default: True)
        Open the window to show the network.
    verbose : int [0-5], (default: 3)
        Verbosity to print the working-status. The higher the number, the more information. 0: nothing, 1: Error, 2: Warning, 3: general, 4: debug, 5: trace.

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
    >>> node_size=df.degree.values*2
    >>> node_color=[]
    >>> for i in range(0,len(G.nodes)):
    >>>     node_color.append(G.nodes[i]['club'])
    >>>     node_name=node_color
    >>>
    >>> # Make some graphs
    >>> out = d3graph(adjmat, df=df, node_size=node_size)
    >>> out = d3graph(adjmat, df=df, node_color=node_size, node_size=node_size)
    >>> out = d3graph(adjmat, df=df, node_color=node_size, node_size=node_size, edge_distance=1000)
    >>> out = d3graph(adjmat, df=df, node_color=node_size, node_size=node_size, charge=1000)
    >>> out = d3graph(adjmat, df=df, node_color=node_name, node_size=node_size, node_size_edge=node_size, node_color_edge='#00FFFF', cmap='Set1', collision=1, charge=250)
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
    dict with graph and savepath.

    """
    # Checks
    adjmat = _do_checks(adjmat.copy())
    # Remove special chars
    adjmat = _remove_special_chars(adjmat)
    # Parameters
    config = _set_configurations(width, height, collision, charge, edge_distance_minmax, edge_distance, edge_width, directed, title, slider, savepath, savename, cmap, showfig, verbose)
    # Make node configurations
    df = _node_config(df, node_name, node_color, node_size, node_size_edge, node_color_edge, adjmat=adjmat, cmap=config['cmap'])
    # Create dataframe from co-occurence matrix
    G, df = adjmat2G(adjmat, df, config['edge_distance_minmax'][0], config['edge_distance_minmax'][1], config['edge_width'], verbose=config['verbose'])
    # Make slider
    [min_slider, max_slider] = _setup_slider(G, slider=slider)
    config['min_slider'] = min_slider
    config['max_slider'] = max_slider
    # Check path
    _set_source_dir(config)
    # Create json
    json_data = _create_json(G)
    # Create html with json file embedded
    _write_index_html(config, json_data)
    # Final
    out = {'G': G, 'path':config['path'] + 'index.html'}
    # Open the webbrowser
    if config['showfig']: webbrowser.open(os.path.abspath(out['path']), new=2)
    # Return
    return(out)


# %% Setup slider
def _setup_slider(G, slider=None):
    tmplist = [*G.edges.values()]
    edge_weight = []
    for i in range(0, len(tmplist)):
        edge_weight = np.append(edge_weight, tmplist[i]['edge_weight'])

    if slider is None:
        max_slider = np.ceil(np.max(edge_weight))
        if len(np.unique(edge_weight))>1:
            min_slider = np.maximum(np.floor(np.min(edge_weight)) - 1, 0)
        else:
            min_slider = 0
    else:
        assert len(slider)==2, 'Slider must be of type [int,int]'
        min_slider = slider[0]
        max_slider = slider[1]

    return(min_slider, max_slider)


# %% Source directory
def _set_source_dir(config):
    if config['path'] is None:
        config['path']=''

    # Check path
    if not os.path.isdir(config['path']) and config['path']!='':
        if config['verbose']>=3: print('[d3graph] >Creating directory: %s' %(config['path']))
        os.mkdir(config['path'])


# %% Write network in json file
def _create_json(G):
    data=dict()
    links=pd.DataFrame([*G.edges.values()]).T.to_dict()
    links_new=[]
    for i in range(0, len(links)):
        links[i]['target'] = int(links[i]['target'])
        links[i]['source'] = int(links[i]['source'])
        links_new.append(links[i])
    data['links']=links_new

    nodes = pd.DataFrame([*G.nodes.values()]).T.to_dict()
    nodeid = [*G.nodes]
    nodes_new = [None] * len(nodes)
    for i in range(0, len(nodes)):
        nodes_new[int(nodeid[i])] = nodes[i]
    data['nodes'] = nodes_new

    return json.dumps(data, separators=(',', ':'))


# %% Write the index.html
def _write_index_html(config, json_data):
    content = {
        'json_data': json_data,
        'title': config['network_title'],
        'width': config['network_width'],
        'height': config['network_height'],
        'charge': config['network_charge'],
        'edge_distance': config['edge_distance'],
        'min_slider': config['min_slider'],
        'max_slider': config['max_slider'],
        'directed': config['directed'],
        'collision': config['network_collision']
    }

    jinja_env = Environment(loader=PackageLoader(package_name=__name__, package_path='d3js') )
    index_template = jinja_env.get_template('index.html.j2')
    index_file = Path(config['savepath'])
    if config['verbose']>=3: print('[d3graph] >Writing %s' % index_file.absolute())
    # index_file.write_text(index_template.render(content))
    with open(index_file, "w", encoding="utf-8") as f:
        f.write(index_template.render(content))


# %%
def _node_config(df, node_name, node_color, node_size, node_size_edge, node_color_edge, cmap='Paired', adjmat=None):
    # getcolors=sns.color_palette(cmap, nodecount).as_hex()
    nodecount = adjmat.shape[0]

    # Set node label
    if isinstance(node_name, list):
        node_name = np.array(node_name)
    elif 'numpy' in str(type(node_name)):
        pass
    elif isinstance(node_name, str):
        node_name = np.array([node_name] * nodecount)
    # elif isinstance(adjmat, type(None)) is False:
    elif adjmat is not None:
        node_name = adjmat.columns.astype(str)
    else:
        node_name = np.array([''] * nodecount)

    # Set node color
    if isinstance(node_color, list):
        node_color = np.array(node_color)
    elif 'numpy' in str(type(node_color)):
        pass
    elif isinstance(node_color, str):
        node_color = np.array([node_color] * nodecount)
    elif isinstance(node_color, type(None)):
        node_color = np.array(['#000080'] * nodecount)
    else:
        assert 'Node color not possible'

    # Set node color edge
    if isinstance(node_color_edge, list):
        node_color_edge = np.array(node_color_edge)
    elif 'numpy' in str(type(node_color_edge)):
        pass
    elif isinstance(node_color_edge, str):
        node_color_edge = np.array([node_color_edge] * nodecount)
    elif isinstance(node_color_edge, type(None)):
        node_color_edge = np.array(['#000000'] * nodecount)
    else:
        assert 'Node color edge not possible'

    # Set node size
    if isinstance(node_size, list):
        node_size = np.array(node_size)
    elif 'numpy' in str(type(node_size)):
        pass
    elif isinstance(node_size, type(None)):
        # Set all nodes same default size
        node_size = np.ones(nodecount, dtype=int) * 5
    elif isinstance(node_size, int) or isinstance(node_size, float):
        node_size = np.ones(nodecount, dtype=int) * node_size
    else:
        assert 'Node size not possible'

    # Set node edge size
    if isinstance(node_size_edge, list):
        node_size_edge = np.array(node_size_edge)
    elif 'numpy' in str(type(node_size_edge)):
        pass
    elif isinstance(node_size_edge, type(None)):
        # Set all nodes same default size
        node_size_edge = np.ones(nodecount, dtype=int) * 1
    elif isinstance(node_size_edge, int) or isinstance(node_size_edge, float):
        node_size_edge = np.ones(nodecount, dtype=int) * node_size_edge
    else:
        assert 'Node edge size not possible'

    # Scale the weights and get hex colors
    node_size_edge = _normalize_size(node_size_edge.reshape(-1, 1), 0.1, 4)
    node_color = _get_hexcolor(node_color, cmap=cmap)
    node_color_edge = _get_hexcolor(node_color_edge, cmap=cmap)

    # Sanity checks
    assert len(node_size)==nodecount, 'Node size must be of same length as the number of nodes'
    assert len(node_size_edge)==nodecount, 'Node size edge must be of same length as the number of nodes'
    assert len(node_color)==nodecount, 'Node color must be of same length as the number of nodes'
    assert len(node_color_edge)==nodecount, 'Node color edge must be of same length as the number of nodes'
    assert len(node_name)==nodecount, 'Node label must be of same length as the number of nodes'

    # Store in dataframe
    if isinstance(df, type(None)): df=pd.DataFrame()
    df['node_name']=node_name
    df['node_color']=node_color.astype(str)
    df['node_size']=node_size.astype(str)
    df['node_size_edge']=node_size_edge.astype(str)
    df['node_color_edge']=node_color_edge
    # Make strings of the identifiers
    df.index=df.index.astype(str)
    # df.index=df.index.str.replace(' ', '_', regex=True)
    # df.columns=df.columns.str.replace(' ', '_', regex=True)

    return(df)


# %% Convert adjacency matrix to graph
def adjmat2G(adjmat, df, edge_distance_min=None, edge_distance_max=None, edge_width=None, verbose=3):
    """Convert adjacency matrix to graph.

    Parameters
    ----------
    adjmat : dataframe
        Dataframe for which the nodes are the columns and index and edges are the values in the array.
    df : dataframe
        Dataframe that is connected to adjmat.
    edge_distance_min : int, (default: None)
        Scale the weights with a minimum value.
    edge_distance_max : int, (default: None)
        Scale the weights with a maximum value.
    edge_width : int, (default: None)
        Width of edge, scale between [1-10] if there is are more then 2 diferent weights. The default is None.

    Returns
    -------
    dict containing Graph G and dataframe.

    """
    # Convert adjmat
    node_names = adjmat.index.values
    adjmat.reset_index(drop=True, inplace=True)
    adjmat.index = adjmat.index.values.astype(str)
    adjmat.columns = np.arange(0, adjmat.shape[1]).astype(str)
    adjmat = adjmat.stack().reset_index()
    adjmat.columns = ['source', 'target', 'weight']

    # Width of edge, scale between [1-10] if there is are more then 2 diferent weights
    if isinstance(edge_width, type(None)):
        if len(np.unique(adjmat['weight'].values.reshape(-1, 1)))>2:
            adjmat['edge_width'] = _normalize_size(adjmat['weight'].values.reshape(-1, 1), 1, 20)
        else:
            edge_width=1
    if not isinstance(edge_width, type(None)):
        adjmat['edge_width'] = np.ones(adjmat.shape[0]) * edge_width

    # Scale the weights towards an edge weight
    if not isinstance(edge_distance_min, type(None)):
        adjmat['edge_weight'] = _normalize_size(adjmat['weight'].values.reshape(-1, 1), edge_distance_min, edge_distance_max)
    else:
        adjmat['edge_weight'] = adjmat['weight'].values.reshape(-1, 1)

    # Keep only edges with weight
    edge_weight_original = adjmat['weight'].values.reshape(-1, 1).flatten()
    adjmat = adjmat.loc[edge_weight_original > 0, :].reset_index(drop=True)

    # Remove self-loops
    Iloc = adjmat['source'] != adjmat['target']
    adjmat = adjmat.loc[Iloc, :].reset_index(drop=True)

    # Include source-target label
    source_label = np.repeat('', adjmat.shape[0]).astype('O')
    target_label = np.repeat('', adjmat.shape[0]).astype('O')
    for i in range(0, len(node_names)):
        source_label[adjmat['source']==str(i)] = node_names[i]
    for i in range(0, len(node_names)):
        target_label[adjmat['target']==str(i)] = node_names[i]
    adjmat['source_label'] = source_label
    adjmat['target_label'] = target_label

    # Make sure indexing of nodes is correct with the edges.
    uilabels = np.unique(np.append(adjmat['source'], adjmat['target']))
    tmplabels=adjmat[['source', 'target']]
    adjmat['source']=None
    adjmat['target']=None
    for i in range(0, len(uilabels)):
        I1 = tmplabels['source']==uilabels[i]
        I2 = tmplabels['target']==uilabels[i]
        adjmat.loc[I1, 'source'] = str(i)
        adjmat.loc[I2, 'target'] = str(i)

    G = nx.Graph()
    # G.add_nodes_from(np.unique(uilabels))
    try:
        G = nx.from_pandas_edgelist(adjmat, 'source', 'target', ['weight', 'edge_weight','edge_width','source_label','target_label','source', 'target'])
    except:
        # Version of networkx<2
        G = nx.from_pandas_dataframe(adjmat, 'source', 'target', edge_attr=['weight', 'edge_weight','edge_width','source_label','target_label','source', 'target'])

    # Add node information
    A = pd.concat([pd.DataFrame(adjmat[['target', 'target_label']].values),pd.DataFrame(adjmat[['source', 'source_label']].values)], axis=0)
    A = A.groupby([0, 1]).size().reset_index(name='Freq')
    [IA,IB] = ismember(df['node_name'], A[1])
    df = df.loc[IA,:]
    df.index = A[0].loc[IB].values.astype(str)

    if not df.empty:
        getnodes = np.array([*G.nodes])
        for col in df.columns:
            for i in range(0, df.shape[0]):
                idx = df.index.values[i]
                if np.any(np.isin(getnodes, df.index.values[i])):
                    G.nodes[idx][col] = str(df[col][idx])
                else:
                    if verbose>=2: print('[d3graph] >WARNING: Not found')

    return(G, df)


# %% Normalize in good d3 range
def _normalize_size(getsizes, minscale=0.1, maxscale=4):
    getsizes = MinMaxScaler(feature_range=(minscale,maxscale)).fit_transform(getsizes).flatten()
    return(getsizes)


# %% Convert to hex color
def _get_hexcolor(label, cmap='Paired'):
    label = label.astype(str)
    if label[0][0]!='#':
        label = label.astype(dtype='U7')
        uinode = np.unique(label)
        tmpcolors = np.array(sns.color_palette(cmap, len(uinode)).as_hex())
        [IA,IB] = ismember(label,uinode)
        label[IA] = tmpcolors[IB]

    return(label)


# %%
def _set_configurations(width, height, collision, charge, edge_distance_minmax, edge_distance, edge_width, directed, title, slider, savepath, savename, cmap, showfig, verbose):
    # curpath=os.path.realpath(sys.argv[0])
    curpath = os.path.dirname(os.path.abspath(__file__))

    config = dict()
    config['verbose'] = verbose
    config['path'] = savepath
    config['network_width'] = width
    config['network_height'] = height
    config['network_title'] = title
    config['network_charge'] = charge * -1
    config['network_collision'] = collision
    config['edge_distance'] = edge_distance
    config['edge_width'] = edge_width
    config['edge_distance_minmax'] = edge_distance_minmax
    config['directed'] = directed
    config['showfig'] = showfig
    config['d3_library'] = os.path.abspath(os.path.join(curpath,'d3js/d3.v3.js'))
    config['d3_script'] = os.path.abspath(os.path.join(curpath,'d3js/d3graphscript.js'))
    config['css'] = os.path.abspath(os.path.join(curpath,'d3js/style.css'))
    config['cmap'] = cmap

    if isinstance(savepath, type(None)):
        config['savepath'] = savename + '.html'
    else:
        config['savepath'] = savepath + savename + '.html'

    # Get color schemes
    if config['cmap'] is None:
        config['cmap']='Paired'

    if (not isinstance(savepath, type(None))) and (not os.path.isdir(config['path'])):
        if verbose>=2: print('[d3graph] >Creating directory [%s]' %(config['path']))
        os.mkdir(config['path'])

    if config['edge_distance'] is None:
        config['edge_distance'] = 30

    return(config)


# %% Do checks
def _do_checks(adjmat):
    if not version.parse(nx.__version__) >= version.parse("2.5"):
        print('[d3graph] >ERROR: networkx version should be >= 2.5.\nTry to: pip install -U networkx')
    if 'numpy' in str(type(adjmat)):
        adjmat = pd.DataFrame(index=range(0,adjmat.shape[0]), data=adjmat, columns=range(0,adjmat.shape[0]))

    adjmat.index=adjmat.index.astype(str)
    adjmat.columns=adjmat.columns.astype(str)
    assert np.all(adjmat.columns==adjmat.index.values), 'adjmat columns and index must have the same identifiers'
    return(adjmat)


# %% Remov special characters from column names
def _remove_special_chars(adjmat, verbose=3):
    if verbose>=3: print('[d3graph] >Removing special chars and replacing with "_"')
    adjmat.columns = list(map(lambda x: unicodedata.normalize('NFD', x).encode('ascii', 'ignore').decode("utf-8").replace(' ','_'), adjmat.columns.values.astype(str)))
    adjmat.index = list(map(lambda x: unicodedata.normalize('NFD', x).encode('ascii', 'ignore').decode("utf-8").replace(' ','_'), adjmat.index.values.astype(str)))
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
    if weight is None: weight = [1]*len(source)
    
    df = pd.DataFrame(np.c_[source, target], columns=['source','target'])
    # Make adjacency matrix
    adjmat = pd.crosstab(df['source'], df['target'], values=weight, aggfunc='sum').fillna(0)
    # Get all unique nodes
    nodes = np.unique(list(adjmat.columns.values)+list(adjmat.index.values))
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
        [IA, IB] = ismember(adjmat.columns.values, adjmat.index.values)
        adjmat = adjmat.iloc[IB,:]
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
