"""Module to assess patterns in missing data, numerically or graphically"""

import numpy as np
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
def inbound_stat(data):
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
    return inbound

@check_dimensions
def outbound_stat(data):
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
    return outbound
