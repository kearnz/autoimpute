"""Module of helper functions used throughout utils"""

import pandas as pd

def _sq_output(data, square=False):
    """Return array or dataframe, depending on parameters passed"""
    if square:
        data.index = data.columns
    return data

def _index_output(data, index):
    """Return array or dataframe, with index set"""
    data = pd.DataFrame(data)
    if len(index) == data.shape[0]:
        data.index = index
        return data
    else:
        raise ValueError("Length of index must equal rows in DataFrame")
