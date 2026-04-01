from copy import deepcopy
import numpy as np
import pandas as pd
from d3graph import d3graph, adjmat2vec, vec2adjmat, import_example
from pathlib import Path

def test_filepath():
    df = import_example('stormofswords')
    adjmat = vec2adjmat(source=df['source'], target=df['target'], weight=df['weight'])
    output_path = Path.cwd() / 'd3graph.html'
    d3 = d3graph()
    d3.graph(adjmat)
    d3.set_path(output_path)
    d3.show(showfig=False)
    assert d3.config['filepath']==output_path
    
    output_path = Path.cwd() / 'd3graph1.html'
    d3 = d3graph()
    d3.graph(adjmat)
    d3.show(showfig=False, filepath=output_path)
    assert d3.config['filepath']==output_path
    

    
def test_instantiate_d3graph_no_args() -> None:
    """Test instantiation works with defaults"""
    d3 = d3graph()
    assert isinstance(d3, type(d3graph()))


def test_clean(d3, helpers) -> None:
    """Test _clean method deletes the attributes in clean_fields"""
    clean_fields: tuple = ('adjmat', 'config', 'edge_properties', 'G', 'node_properties')

    # Set attrs to dummy value (i.e., 0) and assert they exist in the object
    original_attrs = {field: 0 for field in clean_fields}

# Convert the array to a DataFrame for comparison
def test_adjmat2vec_and_back() -> None:
    edges  = np.array([('Cloudy', 'Sprinkler'), ('Cloudy', 'Rain'), ('Sprinkler', 'Wet_Grass'), ('Rain', 'Wet_Grass')])
    weight = [1,2,3,4]
    df_edges = pd.DataFrame(edges, columns=['source', 'target'])
    df_edges['weight']=weight

    df_edges_c = adjmat2vec(vec2adjmat(source=df_edges['source'], target=df_edges['target'], weight=df_edges['weight']))
    adjmat = vec2adjmat(source=df_edges_c['source'], target=df_edges_c['target'], weight=df_edges_c['weight'])
    df_edges_c = adjmat2vec(adjmat)

    df_edges_c = df_edges_c.sort_values(by='weight')
    df_edges_c['isin']=0

    for index, edge in df_edges.iterrows():
        Iloc = np.sum(edge==df_edges_c[df_edges.columns], axis=1)==len(df_edges.columns)
        df_edges_c.loc[Iloc, 'isin'] = 1

    assert df_edges_c['isin'].sum()==df_edges.shape[0]

