"""Helper functions that are used throughout other methods in automipute.

This module contains helper functions which are intended for private use
within other functions. The methods are used both within the utils lib
and anywhere needed throughout the package, such as imputations and visuals.
The methods generally perform small checks, if/else handling, and property
setting. They abstract away functionality needed by many methods, for example,
in the patterns.py file. They are generally used for mutation or manipulation
as apposed to data validation, although at times these methods throw errors as
well (if they cannot determine what to output).

Methods:
    _sq_output(data, cols, square=False)
    _index_output(data, index)
    _nan_col_dropper(data)
"""

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
