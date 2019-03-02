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
    _mode_output(series, mode, strategy)
"""

import warnings
import numpy as np
import pandas as pd

def _sq_output(data, cols, square=False):
    """Turn unlabeled data into a pandas DataFrame.

    This method transforms unlabeled data, such as a numpy array, into a
    pandas DataFrame. It requires the user to pass data as well as labels
    for each column within that data. These labels become the columns of the
    returned DataFrame. If the data is a square matrix, then the method allows
    the user to set the index of the DataFrame equal to its column labels.

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

    This method drops columns from a DataFrame if all values in the column are
    missing. Fully incomplete columns cannot be imputed, nor are they useful to
    impute other columns. Therefore, they are useless in imputation and removed
    from a DataFrame. If columns are removed, a warning is issued with the
    names of each column removed in the process.

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

def _mode_output(series, mode, strategy):
    """Determine number of modes and which to use for imputation.

    A dataset can have multiple modes if more than one distinct value ties for
    the greatest frequency in the series. This method determines which mode to
    use during imputation. If only one mode exists, use that mode. If more than
    one mode exists, default to the first mode (as is done in scipy). If the
    user specifies `random` as the strategy, then randomly sample from the
    modes and impute the random sample.

    Args:
        series (pd.Series): the series to impute with the mode.
        mode (pd.Series): the mode method from pandas always returns a Series.
        strategy (string): strategy to employ. Default `None` in Imputer class.

    Returns:
        None: imputes values in place.

    Raises:
        ValueError: if strategy is not `None` or `random`.
    """
    num_modes = len(mode)
    if num_modes == 1:
        return series.fillna(mode[0], inplace=True)
    else:
        if strategy is None:
            return series.fillna(mode[0], inplace=True)
        elif strategy == "random":
            ind = series[series.isnull()].index
            fills = np.random.choice(mode, len(ind))
            series.loc[ind] = fills
        else:
            err = f"{strategy} not accepted for mode imputation"
            raise ValueError(err)
