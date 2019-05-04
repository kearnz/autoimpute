"""Private helper methods for the imputations folder."""

import logging
import numpy as np
import pandas as pd
from autoimpute.imputations.errors import _not_num_matrix
from autoimpute.imputations.deletion import listwise_delete

def _get_observed(method, predictors, series, verbose):
    """Private method to test datasets and get observed data."""
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

def _local_residuals(x, n, df, choose):
    al = len(df.index)
    if n > al:
        err = "# neighbors greater than # predictions. Reduce neighbor count."
        raise ValueError(err)
    indexarr = np.argpartition(abs(df["y_pred"] - x), n)[:n]
    neighbs = df.loc[indexarr, "y"].values
    distances = df.loc[indexarr, "y_pred"].values - x
    resids = neighbs + distances
    return choose(resids)

def _pymc3_logger(verbose):
    """Private method to handle pymc3 logging."""
    progress = 1
    if not verbose:
        progress = 0
        logger = logging.getLogger('pymc3')
        logger.setLevel(logging.ERROR)
    return progress
