# ---------------------------------
# Name        : adjmatvec.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ---------------------------------

import pandas as pd
import numpy as np
from ismember import ismember


#%%  Convert adjacency matrix to vector
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
    if len(source)!=len(target): raise Exception('[hnet] >Source and Target should have equal elements.')
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
