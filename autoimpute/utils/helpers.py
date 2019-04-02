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
    _get_observed(method, predictors, series, verbose)
    _neighbors(x, n, df, choose)
    _pymc3_logger(verbose)
"""

import warnings
import logging
import numpy as np
import pandas as pd
from autoimpute.imputations.deletion import listwise_delete
from autoimpute.imputations.errors import _not_num_matrix

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

def _get_observed(method, predictors, series, verbose):
    """Helper method to test datasets and get observed data."""
    _not_num_matrix(method, predictors)
    conc = pd.concat([predictors, series], axis=1)
    if verbose:
        null_pred = pd.isnull(predictors)
        null_ser = pd.isnull(series)
        for each in null_pred:
            sum_null_pred = null_pred[each].sum()
            print(f"Missing values in predictor {each}: {sum_null_pred}")
        sum_null_ser = null_ser.sum()
        print(f"Missing values in response {series.name}: {sum_null_ser}")

    # perform listwise delete on predictors and series
    # resulting data serves as the `observed` data for fit modeling
    predictors = listwise_delete(conc, verbose=verbose)
    series = predictors.pop(series.name)
    return predictors, series

def _neighbors(x, n, df, choose):
    al = len(df.index)
    if n > al:
        err = "# neighbors greater than # predictions. Reduce neighbor count."
        raise ValueError(err)
    indexarr = np.argpartition(abs(df["y_pred"] - x), n)[:n]
    neighbs = df.loc[indexarr, "y"].values
    return choose(neighbs)

def _pymc3_logger(verbose):
    """Private method to handle pymc3 logging."""
    progress = 1
    if not verbose:
        progress = 0
        logger = logging.getLogger('pymc3')
        logger.setLevel(logging.ERROR)
    return progress
