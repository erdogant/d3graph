from typing import Tuple

import pandas as pd
import pytest

from d3graph import d3graph


@pytest.fixture(scope="session")
def d3() -> d3graph:
    """An instance of d3graph"""
    d3 = d3graph()
    return d3


@pytest.fixture(scope="session")
def load_default_karate_example(d3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    adjmat, df = d3.import_example('karate')
    return adjmat, df


class Helpers:
    """Util functions"""

    @staticmethod
    def setattrs(obj: d3graph, **kwargs: dict) -> d3graph:
        for k, v in kwargs.items():
            setattr(obj, k, v)
        return obj


@pytest.fixture(scope="session")
def helpers() -> Helpers:
    return Helpers
