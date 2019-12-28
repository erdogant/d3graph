""" This function creates an interactive network using D3 javascript and html.
	from d3graph import d3graph

	out = d3graph(data, <optional>)

 INPUT:
   adjmat:          [Pandas DataFrame] Adjacency matrix (symmetric)
                    rows    = samples (nodes)
                    colums  = samples (nodes)
                    values > 0 are edges
 OPTIONAL

   df:              [Pandas DataFrame]
                    empty: (default)
                    index:   Samples in the same order as the adjmat
                    column:  columns represents an additional feature of the node

   node_name:       [list][numpy array] Label name for the nodes 
                    None: (default)
                    ['1','2','3',...]

   node_color :     [list][numpy array] Color of the node.
                    None: (default)                      Do something style: All nodes will be have the same color (auto generated)
                    ['A']                                Lazy  style:        All nodes will have hte same color. Color is generated on CMAP and the unique labels
                    ['#000000']                          Lazy+ style:        All nodes will have the same hex color.
                    ['A','A','B',...]                    Pro   style:        Colors are generated using cmap and the unique labels recordingly colored
                    ['#377eb8','#ffffff','#000000',...]  Pro+  style:        Hex colors are directly used

   node_color_edge: [list][numpy array] Color of the node edge.
                    None: (default)                      Do something style: All edges will be have the same color (auto generated)
                    ['A']                                Lazy  style:        All edges will have hte same color. Color is generated on CMAP and the unique labels
                    ['#000000']                          Lazy+ style:        All edges will have the same hex color.
                    ['A','A','B',...]                    Pro   style:        Colors are generated using cmap and the unique labels recordingly colored
                    ['#377eb8','#ffffff','#000000',...]  Pro+  style:        Hex colors are directly used

   node_size:       [list][numpy array][integer] Size of the node.
                    None: (default=5)
                    10:           All nodes will be set on this size
                    [2,5,1,...]  Specify per node the size
                   
   node_size_edge:  [list][numpy array][integer] Size of the node edge. Note that node edge sizes are automatically scaled between [0.1 - 4]
                    None: (default=1)
                    10:           All nodes will be set on this size
                    [2,5,1,...]  Specify per node the edge size

   edge_width:      float:  width of the edges. 
                    None: The values in the data are rescaled between [1-10] and used for the width.
                    1: all values have the same width

   edge_distance:   Integer:  Distance of nodes on the edges.
                    None (default)
                    0     Weighted approach using edge weights in the adjadancy matrix. Weights are normalized between the edge_distance_minmax
                    80    Constant edge distance

   edge_distance_minmax: [int, int]:  Min and max Distance of nodes on the edges.
                    [None, None] (Default)
                    [10, 100]    Weights are normalized between minimum and maximum (default)

   
    NETWORK SETTINGS
   
   charge:          Integer:  Scaling/Response of the network. Towards zero becomes a more nervous network
                    250 (default)

   collision:       Float: Nodes wants to prevent a collision. The higher the number, the more collisions are prevented
                    1 (default)

   cmap:           String: Colormap  https://matplotlib.org/examples/color/colormaps_reference.html
                   'Paired'       (default)     
                   'Set2'       
                   'rainbow'
                   'bwr'        Blue-white-red
                   'binary' or 'binary_r'
                   'seismic'    Blue-white-red 
                   'Blues'      white-to-blue
                   'Reds'       white-to-red
                   'Pastel1'    Discrete colors
                   'Paired'     Discrete colors
                   'Set1'       Discrete colors

   
   slider:         [int, int]: Slider to break the network on edge-strength
                   None:   (default) Slider is set on minimum and maximum edge size in the adjmatrix
                   [0,10]:  e.g. set slider in range 0 to 10

   height:         Integer:  Height of figure
                   500: (default)

   width:          Integer:  Width of figure
                   1200: (default)

   title:          String:  Title of hte .html page
                   'd3graph': (default)
                   
   path:           [String]: Directory path to save the output
                   None: (default) Current working directory
                   'c://temp/magweg/'

   savename:       [String]: Name of the html file
                   'index': (default)
                   
   verbose:        Boolean [True,False]
                   False: No (default)
                   True: Yes

 OUTPUT
	output

 DESCRIPTION
   Use Networkx and D3 to produce interactive network diagrams to display multiple layers of data. 

 EXAMPLE
   #=======================================================================
   %reset -f
   %matplotlib auto
   import networkx as nx
   import pandas as pd
   import numpy as np
   from d3graph import d3graph

   #=======================================================================
   # Simple example
   G = nx.karate_club_graph()
   adjmat = nx.adjacency_matrix(G).todense()
   adjmat=pd.DataFrame(index=range(0,adjmat.shape[0]), data=adjmat, columns=range(0,adjmat.shape[0]))

   A = d3graph(adjmat, path='c://temp/magweg/')
   A = d3graph(adjmat, node_color=adjmat.index.values, path='c://temp/magweg/')
   A = d3graph(adjmat, node_color=adjmat.index.values, node_color_edge='#fffff',path='c://temp/magweg/')
   A = d3graph(adjmat, node_color=adjmat.index.values, node_color_edge=adjmat.index.values, path='c://temp/magweg/')
   A = d3graph(adjmat, node_color=adjmat.index.values, node_color_edge='#00000', node_size_edge=5, path='c://temp/magweg/')

   #=======================================================================
   # Extended example
   G = nx.karate_club_graph()
   adjmat = nx.adjacency_matrix(G).todense()
   adjmat=pd.DataFrame(index=range(0,adjmat.shape[0]), data=adjmat, columns=range(0,adjmat.shape[0]))
   adjmat.columns=adjmat.columns.astype(str)
   adjmat.index=adjmat.index.astype(str)
   adjmat.iloc[3,4]=5
   adjmat.iloc[4,5]=6
   adjmat.iloc[5,6]=7

   df=pd.DataFrame(index=adjmat.index)
   df['degree']=np.array([*G.degree()])[:,1]
   df['other info']=np.array([*G.degree()])[:,1]
   node_size=df.degree.values*2
   node_color=[]
   for i in range(0,len(G.nodes)):
       node_color.append(G.nodes[i]['club'])
   node_name=node_color

   A = d3graph(adjmat, df=df, path='c://temp/magweg/',node_size=node_size)
   A = d3graph(adjmat, df=df, node_color=node_size, node_size=node_size, node_name=node_name, path='c://temp/magweg/', directed=False)
   A = d3graph(adjmat, df=df, node_color=node_size, node_size=node_size, node_name=node_name, path='c://temp/magweg/', edge_distance=100, savename='index_ld')
   A = d3graph(adjmat, df=df, node_color=node_size, node_size=node_size, node_name=node_name, path='c://temp/magweg/', charge=50, savename='index_charge')
   A = d3graph(adjmat, df=df, node_color=node_name, node_size=node_size, node_name=node_name, node_size_edge=node_size, node_color_edge='#00FFFF', path='c://temp/magweg/', cmap='Set1', collision=1, charge=250)

   #=======================================================================
   G = nx.karate_club_graph()
   A = d3graph.G2d3(G.copy(), path='c://temp/magweg/')
   #=======================================================================

 SEE ALSO
   http://www.coppelia.io/2014/07/an-a-to-z-of-extra-features-for-the-d3-force-layout/
   https://www.alanzucconi.com/2015/11/01/interactive-graphs-in-the-browser/
   https://andrewmellor.co.uk/blog/articles/2014/12/14/d3-networks/
   http://bl.ocks.org/d3noob/5141278
"""

#--------------------------------------------------------------------------
# Name        : d3graph.py
# Version     : 0.1
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : Dec. 2019
#--------------------------------------------------------------------------

#%% Libraries
# Basics
import os
import sys
from shutil import copyfile
import webbrowser
# Popular
import pandas as pd
import numpy as np
import seaborn as sns
import json
import networkx as nx
from sklearn.preprocessing import MinMaxScaler 
# Custom
from d3graph.ismember import ismember

#%% Main
def d3graph(adjmat, df=None, node_name=None, node_color='#000080', node_size=10, node_size_edge=1, node_color_edge='#000000', width=1500, height=800, collision=0.5, charge=250, edge_width=None, edge_distance=None, directed=False, edge_distance_minmax=[None,None], title='d3graph', slider=None, path=None, savename='index', cmap='Paired', showfig=True, verbose=3):
    # Checks
    adjmat=do_checks(adjmat)
	# Parameters
    config = set_configurations(width, height, collision, charge, edge_distance_minmax, edge_distance, edge_width, directed, title, slider, path, savename, cmap, showfig, verbose)
    # Make node configurations
    df = node_config(df, node_name, node_color, node_size, node_size_edge, node_color_edge, adjmat=adjmat, cmap=config['cmap'])
    # Create dataframe from co-occurence matrix
    [G, df] = adjmat2G(adjmat, df, config['edge_distance_minmax'][0], config['edge_distance_minmax'][1], config['edge_width'])
    # Make slider
    [min_slider, max_slider]=setup_slider(G, slider=slider)
    config['min_slider']=min_slider
    config['max_slider']=max_slider
    # Check path
    set_source_dir(config)
    # Write to json
    jsonpath = write_json(G, config)
    # Create html with json file embedded
    write_index_html(config, jsonpath)
    # Set user-defined parameters in javascript
    write_jscript(df, config)
    # Cleaning
    os.remove(jsonpath)
    # Final
    out = dict()
    out['G']=G
    out['path']=config['path']+'index.html'
    # Open the webbrowser
    if config['showfig']: webbrowser.open(os.path.abspath(out['path']), new=2)
    # Return
    return(out)

#%% Setup slider
def setup_slider(G, slider=None):
    tmplist=[*G.edges.values()]
    edge_weight=[]
    for i in range(0,len(tmplist)):
        edge_weight=np.append(edge_weight, tmplist[i]['edge_weight'])

    if slider==None:
        max_slider = np.ceil(np.max(edge_weight))            
        if len(np.unique(edge_weight))>1:
            min_slider = np.maximum(np.floor(np.min(edge_weight))-1, 0)
        else:
            min_slider = 0
    else:
        assert len(slider)==2, 'Slider must be of type [int,int]'
        min_slider = slider[0]
        max_slider = slider[1]

    return(min_slider, max_slider)
    
#%% Source directory
def set_source_dir(config):
    if config['path']==None:
        config['path']=''

    # Check path
    if not os.path.isdir(config['path']) and config['path']!='':
        if config['verbose']>=3: print('[D3GRAPH] Creating directory: %s' %(config['path']))
        os.mkdir(config['path'])

    # Copy files to destination directory
    copyfile(config['d3_script'], config['path']+os.path.basename(config['d3_script']))
    copyfile(config['css'], config['path']+os.path.basename(config['css']))
    copyfile(config['d3_library'], config['path']+os.path.basename(config['d3_library']))
    
#%% Write network in json file
def write_json(G, config):
    jsonpath = config['path']+'d3graph.json'
#    data = json_graph.node_link_data(G)
#    data.pop('directed')
#    data.pop('multigraph')
#    data.pop('graph')
    
#    A=pd.DataFrame(data['links'])
    data=dict()
    links=pd.DataFrame([*G.edges.values()]).T.to_dict()
    links_new=[]
    for i in range(0, len(links)):
        links[i]['target'] = int(links[i]['target'])
        links[i]['source'] = int(links[i]['source'])
        links_new.append(links[i])
    data['links']=links_new

    nodes=pd.DataFrame([*G.nodes.values()]).T.to_dict()
    nodeid=[*G.nodes]
    nodes_new=[None]*len(nodes)
    for i in range(0,len(nodes)):
        nodes_new[int(nodeid[i])] = nodes[i]
    data['nodes']=nodes_new
    

    with open(jsonpath, 'w') as f:
        json.dump(data, f, indent=4)
    
    return(jsonpath)
    
#%% Write the index.html
def write_index_html(config, jsonpath):
    
    f=open(config['savepath'],'w+')
    f.write('<!DOCTYPE html>\n')
    f.write('<html lang="en" >\n')
    f.write('<head>\n')
    f.write('<meta charset="UTF-8">\n')
    
    config['network_title']
    
    f.write(('<title>'+config['network_title']+'</title>\n'))
    f.write('<script type="text/javascript" src="'+os.path.basename(config['d3_library'])+'"></script>\n')
    # f.write('<script type="text/javascript" src="//d3js.org/d3.v3.js"></script>\n')
    f.write('<link rel="stylesheet" href="style.css">\n')
    f.write('</head>\n')
    f.write('<body>\n')
    f.write('<script type="application/json" id="d3graph"> {\n')

    # Write graph
    fs = open(jsonpath,'r')
    lines = fs.readlines()[1:]
    for line in lines:
        f.write(line)
    fs.close()

    # Write last part html
    f.write('\n')
    f.write('</script>\n')
    
    if (config['max_slider']>config['min_slider']):
        f.write('<form>\n')
        # f.write('<h3> Link threshold 0 <input type="range" id="thersholdSlider" name="points" value=0 min="0" max="10" onchange="threshold(this.value)"> 10 </h3>\n')
        slidertxt='<h3> Link threshold 0 <input type="range" id="thersholdSlider" name="points" value='+str(config['min_slider'])+' min="'+str(config['min_slider'])+'" max="'+str(config['max_slider'])+'" onchange="threshold(this.value)"> '+str(config['max_slider'])+' </h3>\n'
        f.write(slidertxt)
        f.write('</form>\n')
    
    f.write('<script  src="d3graphscript.js"></script>\n')
    f.write('</body>\n')
    f.write('</html>\n')

    f.close()

#%% Write javascript with the configurations
def write_jscript(df, config):
    jspathscript=config['path']+os.path.basename(config['d3_script'])

    # Read in the file
    with open(jspathscript, 'r') as file :
        d3graphscript = file.read()
    
    # Create hover-over textbox for the nodes
    idx=np.where(df.columns.str.contains('node_')==False)[0]
    idx=np.append(idx, np.where(df.columns=='node_name')[0])
    HOVEROVER=''
    DIRECTED_LINKS='//'
    DIRECTED_LINKS_RESTART='//'
    
    if config['directed']:
        DIRECTED_LINKS=''
        DIRECTED_LINKS_RESTART=''
            
    if len(idx)>0:
        str_middle=''
        for i in range(0, len(idx)):
            str_middle = str_middle + ('+ "\\n" + "' + df.columns[idx[i]] + ': " + d.' + df.columns[idx[i]] + ' ')
    
        str_start='function(d) { return "Node: " + d.id '
        str_end=';}'
        HOVEROVER = str_start+str_middle+str_end

    # Replace the target string
    d3graphscript = d3graphscript.replace('$CHARGE$',    str(config['network_charge']))
    if config['edge_distance']=='.linkDistance(function(d) {return d.edge_weight;})':
        d3graphscript = d3graphscript.replace('$LINKDIST$',  str(config['edge_distance']))
    else:
        d3graphscript = d3graphscript.replace('$LINKDIST$', '')

    d3graphscript = d3graphscript.replace('$WIDTH$',     str(config['network_width']))
    d3graphscript = d3graphscript.replace('$HEIGHT$',    str(config['network_height']))
    d3graphscript = d3graphscript.replace('$COLLISION$', str(config['network_collision']))
    d3graphscript = d3graphscript.replace('$HOVEROVER$', HOVEROVER)
    d3graphscript = d3graphscript.replace('$DIRECTED_LINKS$', DIRECTED_LINKS)
    d3graphscript = d3graphscript.replace('$DIRECTED_LINKS_RESTART$', DIRECTED_LINKS_RESTART)
    
    # Write the file out again
    with open(jspathscript, 'w') as file:
        file.write(d3graphscript)

#%%
def node_config(df, node_name, node_color, node_size, node_size_edge, node_color_edge, cmap='Paired', adjmat=None):

#    getcolors=sns.color_palette(cmap, nodecount).as_hex()
    nodecount=adjmat.shape[0]

    # Set node label
    if isinstance(node_name, list):
        node_name = np.array(node_name)
    elif 'numpy' in str(type(node_name)):
        pass
    elif isinstance(node_name, str):
        node_name=np.array([node_name]*nodecount)
    elif isinstance(adjmat, type(None))==False:
        node_name=adjmat.columns.astype(str)
    else:
        node_name=np.array(['']*nodecount)

    # Set node color
    if isinstance(node_color, list):
        node_color = np.array(node_color)
    elif 'numpy' in str(type(node_color)):
        pass
    elif isinstance(node_color, str):
        node_color=np.array([node_color]*nodecount)
    elif isinstance(node_color, type(None)):
        node_color=np.array(['#000080']*nodecount)
    else:
        assert 'Node color not possible'

    # Set node color edge
    if isinstance(node_color_edge, list):
        node_color_edge = np.array(node_color_edge)
    elif 'numpy' in str(type(node_color_edge)):
        pass
    elif isinstance(node_color_edge, str):
        node_color_edge=np.array([node_color_edge]*nodecount)
    elif isinstance(node_color_edge, type(None)):
        node_color_edge=np.array(['#000000']*nodecount)
    else:
        assert 'Node color edge not possible'

    # Set node size
    if isinstance(node_size, list):
        node_size = np.array(node_size)
    elif 'numpy' in str(type(node_size)):
        pass
    elif isinstance(node_size, type(None)):
        # Set all nodes same default size
        node_size=np.ones(nodecount, dtype=int)*5
    elif isinstance(node_size, int) or isinstance(node_size, float):
        node_size=np.ones(nodecount, dtype=int)*node_size
    else:
        assert 'Node size not possible'

    # Set node edge size
    if isinstance(node_size_edge, list):
        node_size_edge = np.array(node_size_edge)
    elif 'numpy' in str(type(node_size_edge)):
        pass
    elif isinstance(node_size_edge, type(None)):
        # Set all nodes same default size
        node_size_edge=np.ones(nodecount, dtype=int)*1
    elif isinstance(node_size_edge, int) or isinstance(node_size_edge, float):
        node_size_edge=np.ones(nodecount, dtype=int)*node_size_edge
    else:
        assert 'Node edge size not possible'

    # Scale the weights and get hex colors
    node_size_edge  = normalize_size(node_size_edge.reshape(-1, 1), 0.1, 4)
    node_color      = get_hexcolor(node_color, cmap=cmap)
    node_color_edge = get_hexcolor(node_color_edge, cmap=cmap)

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
    df.columns=df.columns.str.replace(' ','_')
    df.columns=df.columns.str.replace('.','_')
    
    # Return
    return(df)

#%% Convert adjacency matrix to graph
def adjmat2G(adjmat, df, edge_distance_min=None, edge_distance_max=None, edge_width=None):
    # Convert adjmat
    node_names     = adjmat.index.values
    adjmat.reset_index(drop=True, inplace=True)
    adjmat.index   = adjmat.index.values.astype(str)
    adjmat.columns = np.arange(0,adjmat.shape[1]).astype(str)
    adjmat         = adjmat.stack().reset_index()
    adjmat.columns = ['source', 'target', 'weight']

    # Width of edge, scale between [1-10] if there is are more then 2 diferent weights
    if isinstance(edge_width, type(None)):
        if len(np.unique(adjmat['weight'].values.reshape(-1,1)))>2:
            adjmat['edge_width'] = normalize_size(adjmat['weight'].values.reshape(-1,1), 1, 20)
        else:
            edge_width=1
    if not isinstance(edge_width, type(None)):
        adjmat['edge_width'] = np.ones(adjmat.shape[0])*edge_width
        
    # Scale the weights towards an edge weight
    if not isinstance(edge_distance_min, type(None)):
        adjmat['edge_weight'] = normalize_size(adjmat['weight'].values.reshape(-1,1), edge_distance_min, edge_distance_max)
    else:
        adjmat['edge_weight'] = adjmat['weight'].values.reshape(-1,1)

    # Keep only edges with weight
    edge_weight_original = adjmat['weight'].values.reshape(-1,1).flatten()
    adjmat=adjmat.loc[edge_weight_original>0,:].reset_index(drop=True)
    
    # Remove self-loops
    I=adjmat['source']!=adjmat['target']
    adjmat=adjmat.loc[I,:].reset_index(drop=True)
    
    # Include source-target label
    source_label=np.repeat('',adjmat.shape[0]).astype('O')
    target_label=np.repeat('',adjmat.shape[0]).astype('O')
    for i in range(0,len(node_names)):
        source_label[adjmat['source']==str(i)]= node_names[i]
    for i in range(0,len(node_names)):
        target_label[adjmat['target']==str(i)]= node_names[i]
    adjmat['source_label']=source_label
    adjmat['target_label']=target_label

    # Make sure indexing of nodes is correct with the edges    
    uilabels=np.unique(np.append(adjmat['source'], adjmat['target']))
    tmplabels=adjmat[['source','target']]
    adjmat['source']=None
    adjmat['target']=None
    for i in range(0,len(uilabels)):
        I1=tmplabels['source']==uilabels[i]
        I2=tmplabels['target']==uilabels[i]
        adjmat['source'].loc[I1]=str(i)
        adjmat['target'].loc[I2]=str(i)
    
#    adjmat['source']=ismember(tmplabels['source'],uilabels)[1].astype(str)
#    adjmat['target']=ismember(tmplabels['target'],uilabels)[1].astype(str)
    G = nx.Graph()
#    G.add_nodes_from(np.unique(uilabels))
    try:
        G = nx.from_pandas_edgelist(adjmat, 'source', 'target', ['weight', 'edge_weight','edge_width','source_label','target_label','source', 'target'])
    except:
        # Version of networkx<2
        G = nx.from_pandas_dataframe(adjmat, 'source', 'target', edge_attr=['weight', 'edge_weight','edge_width','source_label','target_label','source', 'target'])

    # Add node information
    A=pd.concat([pd.DataFrame(adjmat[['target', 'target_label']].values),pd.DataFrame(adjmat[['source', 'source_label']].values)], axis=0)
    A=A.groupby([0, 1]).size().reset_index(name='Freq')
    [IA,IB]=ismember(df['node_name'], A[1])
    df=df.loc[IA,:]
    df.index=A[0].loc[IB].values.astype(str)
    
    if not df.empty:
        getnodes=np.array([*G.nodes])
        for col in df.columns:
            for i in range(0, df.shape[0]):
                idx=df.index.values[i]
                if np.any(np.isin(getnodes, df.index.values[i])):
                    G.nodes[idx][col] = str(df[col][idx])
                else:
                    print('not found')
    
#    mapping = dict(zip(np.arange(0,len(getnodes)), getnodes))
#    G = nx.relabel_nodes(G, mapping)
    #    A=pd.DataFrame([*G.nodes.values()])
     
    return(G, df)

#%% Normalize in good d3 range
def normalize_size(getsizes, minscale=0.1, maxscale=4):
    getsizes = MinMaxScaler(feature_range=(minscale,maxscale)).fit_transform(getsizes).flatten()
    return(getsizes)
    
#%% Convert to hex color
def get_hexcolor(label, cmap='Paired'):
    label = label.astype(str)
    if label[0][0]!='#':
        label=label.astype(dtype='U7')
        uinode=np.unique(label)
        tmpcolors=np.array(sns.color_palette(cmap, len(uinode)).as_hex())
        [IA,IB]=ismember(label,uinode)
        label[IA]=tmpcolors[IB]
    
    return(label)

#%%
def set_configurations(width, height, collision, charge, edge_distance_minmax, edge_distance, edge_width, directed, title, slider, path, savename, cmap, showfig, verbose):
    #curpath=os.path.realpath(sys.argv[0])
    curpath = os.path.dirname(os.path.abspath( __file__ ))

    config = dict()
    config['verbose']              = verbose
    config['path']                 = path
    config['network_width']        = width
    config['network_height']       = height
    config['network_title']        = title
    config['network_charge']       = charge*-1
    config['network_collision']    = collision
    config['edge_distance']        = edge_distance #80
    config['edge_width']           = edge_width
    config['edge_distance_minmax'] = edge_distance_minmax
    config['directed']             = directed
    config['showfig']              = showfig
    config['d3_library']           = os.path.abspath(os.path.join(curpath,'d3js/d3.v3.js'))
    config['d3_script']            = os.path.abspath(os.path.join(curpath,'d3js/d3graphscript.js'))
    config['css']                  = os.path.abspath(os.path.join(curpath,'d3js/style.css'))

    config['cmap']                 = cmap

    if isinstance(path, type(None)): 
        config['savepath'] = savename+'.html'
    else:
        config['savepath'] = path+savename+'.html'

    # Get color schemes
    if config['cmap']==None:
        config['cmap']='Paired'
        
    if (not isinstance(path, type(None))) and (not os.path.isdir(config['path'])):
        if verbose>=2: print('[NETWORK D3] Creating directory [%s]' %(config['path']))
        os.mkdir(config['path'])

    if not isinstance(config['edge_distance'], type(None)):
        config['edge_distance'] = '.linkDistance(function(d) {return d.edge_weight;})' # Distance of nodes depends on the weight of the edges
    
    return(config)

#%% Do checks
def do_checks(adjmat):
    assert float(nx.__version__)>2, 'This function requires networkx to be v2 or higher. Try to: pip install --upgrade networkx'
    if 'numpy' in str(type(adjmat)):
        adjmat = pd.DataFrame(index=range(0,adjmat.shape[0]), data=adjmat, columns=range(0,adjmat.shape[0]))

    adjmat.index=adjmat.index.astype(str)
    adjmat.columns=adjmat.columns.astype(str)
#    if not df.empty: assert df.shape[0]==adjmat.shape[0], 'df must be of same length as the nodes in adjmat'
#    if not df.empty: assert np.all(df.index.values==adjmat.index.values), 'adjmat and df must have the same identifiers'
    assert np.all(adjmat.columns==adjmat.index.values), 'adjmat columns and index must have the same identifiers'
    return(adjmat)

#%%
#def G2d3(G, width=1800, height=500, collision=0.5, charge=250, edge_distance_minmax=[1,100], title='d3graph', slider=None, path=None, savename='index', cmap='Paired', verbose=log.INFO):
#    
#    config = set_configurations(width, height, collision, charge, edge_distance_minmax, title, slider, path, savename, cmap, verbose)
#    config['edge_distance'] = '.linkDistance(function(d) {return d.edge_weight;})' # Distance of nodes depends on the weight of the edges
#    
#    # Inlucde information in Graph
#    [df_nodes, df_edges] = network.G2df(G, edge_distance_minmax=edge_distance_minmax)
#    # And back to graph
#    G1 = network.df2G(df_nodes, df_edges)
#
#    # Slider
#    [min_slider, max_slider]=setup_slider(G1, slider=slider)
#    config['min_slider']=min_slider
#    config['max_slider']=max_slider
#
    # A=pd.DataFrame([*G.node.values()])
    # A=pd.DataFrame([*G.edges.values()])
    # edges = G.edges()
    # edge_weights = np.array([G[u][v]['source'] for u,v in edges])

#    # Check path
#    set_source_dir(config)
#    # Write to json
#    jsonpath = write_json(G1, config)
#    # Create html with json file embedded
#    write_index_html(config, jsonpath)
#    # Set user-defined parameters in javascript
#    write_jscript(df_nodes, config)
#    # Cleaning
#    os.remove(jsonpath)
#    
#    import NETWORKS.network as network
#    node_name=pd.DataFrame([*G.node.values()])['node_name']
#    node_size=pd.DataFrame([*G.node.values()])['node_size'].values
#    node_color=pd.DataFrame([*G.node.values()])['node_color'].values
#    network.plot(G, node_color=node_color, node_size=node_size.astype(float), node_size_scale=[400,800], alpha=0.8, cmap='Paired', width=20, height=15, pos=None, filename=None, title=None, methodtype='default', verbose=3)

#    out=dict()
#    out['G']=G1
#    out['path']=config['path']+'index.html'
#    webbrowser.open(out['path'], new=2)
#    
#    return(out)

#%% Main
# if __name__ == '__main__':
    # main(sys.argv[1:])
