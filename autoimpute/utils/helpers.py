"""Module of helper functions used throughout utils"""

import pandas as pd

def _sq_output(data, cols, square=False):
    """Return array or dataframe, depending on parameters passed"""
    data = pd.DataFrame(data, columns=cols)
    if square:
        data.index = data.columns
    return data

def _index_output(data, index):
    """Return array or dataframe, with index set"""
    data = pd.DataFrame(data, index=index)
    return data
