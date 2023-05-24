"""Helper functions used throughout other methods in automipute.utils."""

import warnings
import numpy as np
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

def _one_hot_encode(X, used_columns=None):
    """Private method to handle one hot encoding for categoricals."""
    cats = X.select_dtypes(include=(object,)).columns.size
    if cats > 0:
        X_temp = pd.get_dummies(X, drop_first=True)
        if used_columns is None:
            used_columns = X_temp.columns
        if len(X_temp.columns) != len(used_columns):
            one_hot = pd.get_dummies(X)
            # if wasn't in `used_columns`, then it's the first category
            to_drop = set(one_hot.columns).difference(used_columns)
            one_hot.drop(to_drop, axis=1, inplace=True)
            # if wasn't in `one_hot`, there were no instances of this category
            to_add = set(used_columns).difference(one_hot.columns)
            X_temp = one_hot.assign(**{col:0 for col in to_add})
            X_temp = X_temp.reindex(columns=used_columns, copy=False)
        X = X_temp
    return X
