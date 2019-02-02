"""Module to assess patterns in missing data, numerically or graphically"""

import numpy as np
import pandas as pd
from .checks import check_dimensions

@check_dimensions
def md_pairs(data):
    """
    calculates pairwise missing data statistics
    rr:  response-response pairs
    rm:  response-missing pairs
    mr:  missing-response pairs
    mm:  missing-missing pairs
    returns a square matrix, where n = number of columns
    """
    int_ln = lambda arr: np.logical_not(arr)*1
    r = int_ln(np.isnan(data))
    rr = np.matmul(r.T, r)
    mm = np.matmul(int_ln(r).T, int_ln(r))
    mr = np.matmul(int_ln(r).T, r)
    rm = np.matmul(r.T, int_ln(r))
    return dict(rr=rr, rm=rm, mr=mr, mm=mm)

@check_dimensions
def inbound_stat(data, cols=None):
    """
    Calculates proportion of usable cases. From Van Buuren:
    'The proportion of usable cases Ijk equals 1 if variable Yk
    is observed in all records where Yj is missing.
    The statistic can be used to quickly select potential predictors
    Yk for imputing Yj based on the missing data pattern.'
    High values are preferred.
    """
    pairs = md_pairs(data)
    with np.errstate(divide="ignore", invalid="ignore"):
        inbound = pairs["mr"]/(pairs["mr"]+pairs["mm"])
    if cols is None:
        return inbound
    elif isinstance(cols, (list, tuple)):
        if len(cols) == inbound.shape[1]:
            return pd.DataFrame(inbound, columns=cols, index=cols)
        else:
            raise Exception('length of cols must equal inbound shape')
    else:
        raise TypeError("optional cols must be list or tuple")

@check_dimensions
def outbound_stat(data, cols=None):
    """
    Calculates the outbound statistic. From Van Buuren:
    'The outbound statistic Ojk measures how observed data in variable
    Yj connect to missing data in the rest of the data.
    The quantity Ojk equals 1 if variable Yj is observed in all records where
    Yk is missing. The statistic can be used to evaluate whether Yj
    is a potential predictor for imputing Yk.'
    High values are preferred
    """
    pairs = md_pairs(data)
    with np.errstate(divide="ignore", invalid="ignore"):
        outbound = pairs["rm"]/(pairs["rm"]+pairs["rr"])
    if cols is None:
        return outbound
    elif isinstance(cols, (list, tuple)):
        if len(cols) == outbound.shape[1]:
            return pd.DataFrame(outbound, columns=cols, index=cols)
        else:
            raise Exception('length of cols must equal outbound shape')
    else:
        raise TypeError("optional cols must be list or tuple")
