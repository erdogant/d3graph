from copy import deepcopy

from d3graph import d3graph


def test_instantiate_d3graph_no_args() -> None:
    """Test instantiation works with defaults"""
    d3 = d3graph()
    assert isinstance(d3, type(d3graph()))


def test_clean(d3, helpers) -> None:
    """Test _clean method deletes the attributes in clean_fields"""
    clean_fields: tuple = ('adjmat', 'config', 'edge_properties', 'G', 'node_properties')

    # Set attrs to dummy value (i.e., 0) and assert they exist in the object
    original_attrs = {field: 0 for field in clean_fields}
    
