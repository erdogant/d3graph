from d3graph import d3graph
from copy import deepcopy


def test_instantiate_d3graph_no_args() -> None:
    """Test instantiation works with defaults"""
    d3 = d3graph()
    assert isinstance(d3, type(d3graph()))


def test_clean(d3, helpers) -> None:
    """Test _clean method deletes the attributes in clean_fields"""
    clean_fields: tuple = ('adjmat', 'edge_properties', 'G', 'node_properties', 'config')

    # Set attrs and assert they exist in the object
    original_attrs = {'adjmat': 0, 'edge_properties': 0, 'G': 0, 'node_properties': 0, 'config': 0}
    d3_og = helpers.setattrs(obj=d3, **original_attrs)

    # Make a copy of the object and apply _clean()
    d3_new = deepcopy(d3_og)
    d3_new._clean()

    assert len([attr for attr in vars(d3_og) if attr in clean_fields]) == len(clean_fields)
    assert all(isinstance(i, int) for i in map(vars(d3_og).get, clean_fields))
    assert not [attr for attr in vars(d3_new) if attr in clean_fields]
    assert not any(hasattr(d3_new, attr) for attr in vars(d3_new) if attr in clean_fields)
