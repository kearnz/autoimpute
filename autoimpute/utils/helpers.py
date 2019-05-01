"""Helper functions used throughout other methods in automipute.utils."""

import warnings
import pandas as pd

def _sq_output(data, cols, square=False):
    """Private method to turn unlabeled data into a DataFrame."""
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=cols)
    if square:
        data.index = data.columns
    return data

def _index_output(data, index):
    """Private method to transform data to DataFrame and set the index."""
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, index=index)
    return data

def _nan_col_dropper(data):
    """Private method to drop columns w/ all missing values from DataFrame."""
    cb = set(data.columns.tolist())
    data.dropna(axis=1, how='all', inplace=True)
    ca = set(data.columns.tolist())
    cdiff = cb.difference(ca)
    if cdiff:
        wrn = f"{cdiff} dropped from DataFrame because all rows missing."
        warnings.warn(wrn)
    return data, cdiff
