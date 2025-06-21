# ---------------------------------
# Name        : adjmatvec.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ---------------------------------

import numpy as np
import pandas as pd
from ismember import ismember
import logging

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='[{asctime}] [{name}] [{levelname}] {msg}', style='{', datefmt='%d-%m-%Y %H:%M:%S')

#%%  Convert adjacency matrix to vector
def vec2adjmat(source, target, weight=None, symmetric: bool = True, aggfunc='sum', logger=None) -> pd.DataFrame:
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
    if logger is not None: logger.info('Converting source-target into adjacency matrix..')


    df = pd.DataFrame(np.c_[source, target], columns=['source', 'target'])
    # Make adjacency matrix
    adjmat = pd.crosstab(df['source'], df['target'], values=weight, aggfunc=aggfunc).fillna(0)
    # Get all unique nodes
    nodes = np.unique(list(adjmat.columns.values) + list(adjmat.index.values))
    # nodes = np.unique(np.c_[adjmat.columns.values, adjmat.index.values].flatten())

    # Make the adjacency matrix symmetric
    if symmetric:
        logger.info('Making the matrix symmetric..')
        # Add missing columns
        # node_columns = np.setdiff1d(nodes, adjmat.columns.values)
        # for node in node_columns:
        #     adjmat[node] = 0

        # # Add missing rows
        # node_rows = np.setdiff1d(nodes, adjmat.index.values)
        # adjmat = adjmat.T
        # for node in node_rows:
        #     adjmat[node] = 0

        # Add missing columns
        IA, _ = ismember(nodes, adjmat.columns.values)
        node_columns = nodes[~IA]
        if len(node_columns) > 0:
            df_new_columns = pd.DataFrame(0, index=adjmat.index, columns=node_columns)
            adjmat = pd.concat([adjmat, df_new_columns], axis=1)

        # # Add missing rows
        IA, _ = ismember(nodes, adjmat.index.values)
        node_rows = nodes[~IA]
        # node_rows = np.setdiff1d(nodes, adjmat.index.values)
        if len(node_rows) > 0:
            df_new_rows = pd.DataFrame(0, index=node_rows, columns=adjmat.columns)
            adjmat = pd.concat([adjmat, df_new_rows], axis=0)

        # adjmat = adjmat.T

        # Sort to make ordering of columns and rows similar
        if logger is not None: logger.debug('Order columns and rows.')
        _, IB = ismember(adjmat.columns.values, adjmat.index.values)
        adjmat = adjmat.iloc[IB, :]
        adjmat.index.name = 'source'
        adjmat.columns.name = 'target'

    # Force columns to be string type
    adjmat.columns = adjmat.columns.astype(str)
    return adjmat


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
