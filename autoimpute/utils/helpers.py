"""Helper functions that are used throughout other methods in utils dir.

This module contains helper functions which are intended for private use
within other functions. The methods are used mainly within the utils lib,
although they may appear anywhere needed throughout the package. The
methods perform small checks, if/else handling, and property setting.
They abstract away functionality needed by many methods, for example,
in the patterns.py file.

Methods:
    _sq_output(data, cols, square=False)
    _index_output(data, index)
"""

import pandas as pd

def _sq_output(data, cols, square=False):
    """Turn unlabeled data into a pandas DataFrame

    This method transforms unlabeled data, such as a numpy array,
    into a pandas DataFrame. It requires the user to pass data as well as
    labels for each column within that data. These labels become the
    columns of the returned DataFrame. If the data is square,
    then the method allows the user to set the index of the DataFrame
    equal to its column labels.

    Args:
        data (iterator): Numpy array, list, tuple, etc. that contains
            data to be turned into a pandas DataFrame.
        cols (iterator): Labels for the columns of the data
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
        data (iterator): data to transform to DataFrame
        index (iterator): labels to set as te index

    Returns:
        pd.DataFrame
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, index=index)
    return data
