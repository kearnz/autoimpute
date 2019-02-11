"""Module of helper functions used throughout utils"""

import warnings
import numpy as np
import pandas as pd

ACCEPTED_TYPES = (list, tuple, np.ndarray, pd.core.indexes.base.Index)

def _cols_decider(data, cols):
    """Maintain columns of DataFrame if data is in fact a DataFrame"""
    if isinstance(data, pd.DataFrame):
        if not cols is None:
            warnings.warn("WARNING: cols overriden when data is DataFrame")
        return data, data.columns
    else:
        return cols

def _cols_type(cols):
    """Allowed types to set indices and columns of dataframes"""
    if isinstance(cols, ACCEPTED_TYPES):
        return list(cols)
    else:
        raise TypeError("Cols must be list, tuple, array, or pd column index")

def _cols_output(data, cols=None, square=False):
    """Return array or dataframe, depending on parameters passed"""
    if cols is None:
        return data
    else:
        cols = _cols_type(cols)
        if len(cols) == data.shape[1]:
            if square:
                return pd.DataFrame(data, columns=cols, index=cols)
            else:
                return pd.DataFrame(data, columns=cols)
        else:
            raise ValueError("Length of cols must equal data shape columns")

def _index_output(data, index=None):
    """Return array or dataframe, with index set"""
    if index is None:
        return data
    else:
        index = _cols_type(index)
        df = pd.DataFrame(data)
        if len(index) == df.shape[0]:
            df.index = index
            return df
        else:
            raise ValueError("Length of index must equal rows in DataFrame")
