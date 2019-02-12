"""Module of helper functions used throughout utils"""

import pandas as pd

def _sq_output(data, cols, square=False):
    """Return dataframe, where index = columns if sq matrix"""
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=cols)
    if square:
        data.index = data.columns
    return data

def _index_output(data, index):
    """Return dataframe with index set based on index passed"""
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, index=index)
    return data
