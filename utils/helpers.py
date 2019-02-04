"""Module of helper functions used throughout utils"""

import numpy as np
import pandas as pd

def _pattern_output(data, cols=None, square=False):
    """Return array or dataframe, depending on parameters passed"""
    if cols is None:
        return data
    elif isinstance(cols, (list, tuple)):
        if len(cols) == data.shape[1]:
            if square:
                return pd.DataFrame(data, columns=cols, index=cols)
            else:
                return pd.DataFrame(data, columns=cols)
        else:
            raise ValueError("Length of cols must equal data shape columns")
    else:
        raise TypeError("Optional cols must be list or tuple")

def _is_null(data):
    if data.dtype in (np.dtype('float32'), np.dtype('float64')):
        r = np.isnan(data)
    else:
        r = pd.isnull(data)
    return r
