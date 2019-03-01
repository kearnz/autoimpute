"""Helper functions that are used throughout other methods in automipute.

This module contains helper functions which are intended for private use
within other functions. The methods are used both within the utils lib
and anywhere needed throughout the package, such as imputations and visuals.
The methods generally perform small checks, if/else handling, and property
setting. They abstract away functionality needed by many methods, for example,
in the patterns.py file.

Methods:
    _sq_output(data, cols, square=False)
    _index_output(data, index)
    _nan_col_dropper(data)
"""

import warnings
import pandas as pd

def _sq_output(data, cols, square=False):
    """Turn unlabeled data into a pandas DataFrame.

    This method transforms unlabeled data, such as a numpy array,
    into a pandas DataFrame. It requires the user to pass data as well as
    labels for each column within that data. These labels become the
    columns of the returned DataFrame. If the data is a square matrix,
    then the method allows the user to set the index of the DataFrame
    equal to its column labels.

    Args:
        data (iterator): Numpy array, list, tuple, etc. that contains
            data to be turned into a pandas DataFrame.
        cols (iterator): Labels for the columns of the new DataFrame.
        square (boolean, optional): Whether or not the data is square.
            Defaults to False.

    Returns:
        pd.DataFrame
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=cols)
    if square:
        data.index = data.columns
    return data

def _index_output(data, index):
    """Transform data to pandas DataFrame and set the index.

    Args:
        data (iterator): data to transform to DataFrame.
        index (iterator): labels to set as te index.

    Returns:
        pd.DataFrame
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, index=index)
    return data

def _nan_col_dropper(data):
    """Drop columns with missing all rows missing from a DataFrame.

    This method drops columns from a DataFrame if all values in the
    column are missing. Fully incomplete columns can not be imputed,
    nor are they useful to impute other columns. Therefore, they are
    useless in imputation analysis and removed from a DataFrame. If
    columns are removed, a warning is issued with the names of each
    column removed in the process.

    Args:
        data (pd.DataFrame): DataFrame with potentially NaN columns.

    Returns:
        pd.DataFrame
    """
    cb = set(data.columns.tolist())
    data.dropna(axis=1, how='all', inplace=True)
    ca = set(data.columns.tolist())
    cdiff = cb.difference(ca)
    if cdiff:
        wrn = f"{cdiff} dropped from DataFrame because all rows were missing."
        warnings.warn(wrn)
    return data, cdiff
