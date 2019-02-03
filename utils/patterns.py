"""Module to assess patterns in missing data, numerically or graphically"""

import numpy as np
import pandas as pd
from .checks import check_dimensions
from .helpers import _pattern_output

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

def _inbound(pairs):
    """Helper to get inbound from pairs"""
    return pairs["mr"]/(pairs["mr"]+pairs["mm"])

def _outbound(pairs):
    """Helper to get outbound from pairs"""
    return pairs["rm"]/(pairs["rm"]+pairs["rr"])

def _influx(pairs):
    """Helper to get influx from pairs"""
    num = np.nansum(pairs["mr"], axis=1)
    denom = np.nansum(pairs["mr"]+pairs["rr"], axis=1)
    return num/denom

def _outflux(pairs):
    """Helper to get outflux from pairs"""
    num = np.nansum(pairs["rm"], axis=1)
    denom = np.nansum(pairs["rm"]+pairs["mm"], axis=1)
    return num/denom

def get_stat_for(func, data):
    """
    Generic method to get a missing data statistic from data
    Can be used directly in tandem with helper methods, but this is discouraged
    Instead, use specific methods below (inbound, outbound, etc)
    These methods utilize this function to compute specific stats
    """
    pairs = md_pairs(data)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = func(pairs)
    return stat

def inbound(data, cols=None):
    """
    Calculates proportion of usable cases. From Van Buuren:
    'The proportion of usable cases Ijk equals 1 if variable Yk
    is observed in all records where Yj is missing.
    The statistic can be used to quickly select potential predictors
    Yk for imputing Yj based on the missing data pattern.'
    High values are preferred.
    """
    inbound_coeff = get_stat_for(_inbound, data)
    inbound_ = _pattern_output(inbound_coeff, cols, True)
    return inbound_

def outbound(data, cols=None):
    """
    Calculates the outbound statistic. From Van Buuren:
    'The outbound statistic Ojk measures how observed data in variable
    Yj connect to missing data in the rest of the data.
    The quantity Ojk equals 1 if variable Yj is observed in all records where
    Yk is missing. The statistic can be used to evaluate whether Yj
    is a potential predictor for imputing Yk.'
    High values are preferred
    """
    outbound_coeff = get_stat_for(_outbound, data)
    outbound_ = _pattern_output(outbound_coeff, cols, True)
    return outbound_

def influx(data, cols=None):
    """
    Calculates the influx coefficient Ij. From Van Buuren:
    'The coefficient is equal to the number of variable pairs (Yj,Yk)
    with Yj missing and Yk observed,
    divided by the total number of observed data cells.
    The value of Ij depends on the proportion of missing data of the variable.
    Influx of a completely observed variable is equal to 0,
    whereas for completely missing variables we have Ij=1.
    For two variables with the same proportion of missing data,
    the variable with higher influx is better connected to the observed data,
    and might thus be easier to impute.
    """
    influx_coeff = get_stat_for(_influx, data)
    influx_ = _pattern_output(influx_coeff, cols, True)
    return influx_

def outflux(data, cols=None):
    """
    Calculates the outflux coefficient Oj. From Van Buuren:
    'Oj is the number of variable pairs with Yj observed and Yk missing,
    divided by the total number of incomplete data cells.
    Outflux indicates potential usefulness of Yj for imputing other variables.
    Outflux depends on the proportion of missing data of the variable.
    Outflux of a completely observed variable is equal to 1,
    whereas outflux of a completely missing variable is equal to 0.
    For two variables having the same proportion of missing data,
    the variable with higher outflux is better connected to the missing data,
    and thus potentially more useful for imputing other variables.
    """
    outflux_coeff = get_stat_for(_outflux, data)
    outflux_ = _pattern_output(outflux_coeff, cols, True)
    return outflux_

def flux(data, cols):
    """
    Port of Van Buuren's flux method in R. Calculates:
    - pobs: Proportion observed
    - ainb: Average inbound statistic
    - aout: Average outbound statistic
    - influx: Influx coefficient (Ij)
    - outflux: Outflux coefficient (Oj)
    """
    row_mean = lambda row: np.nansum(row)/(len(row) - 1)
    pairs = md_pairs(data)
    with np.errstate(divide="ignore", invalid="ignore"):
        pobs = np.mean(~np.isnan(data), axis=0)
        ainb = np.apply_along_axis(row_mean, 1, _inbound(pairs))
        aout = np.apply_along_axis(row_mean, 1, _outbound(pairs))
        inf = _influx(pairs)
        outf = _outflux(pairs)
        res = dict(pobs=pobs, influx=inf, outflux=outf, ainb=ainb, aout=aout)
    return pd.DataFrame(res, index=cols)
