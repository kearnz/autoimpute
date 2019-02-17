"""Module to numerically assess patterns in missing data"""

import numpy as np
import pandas as pd
from autoimpute.utils.checks import check_data_structure, check_missingness
from autoimpute.utils.helpers import _sq_output, _index_output

@check_data_structure
def md_locations(data, both=True):
    """
    Produces locations where values are missing in dataset
    - Normally, fully complete or fully empty throws error
    - But this method simply shows missingness locations
    - So standard for mixed complete-missing not necessary
    - In missing locations, 1 = missing, 0 = not missing
    Returns original dataframe concatenated with missingness dataframe
    """
    md_df = pd.isnull(data)*1
    if both:
        md_df = pd.concat([data, md_df], axis=1)
    return md_df

@check_data_structure
def md_pairs(data):
    """
    Calculates pairwise missing data statistics
    - rr: response-response pairs
    - rm: response-missing pairs
    - mr: missing-response pairs
    - mm: missing-missing pairs
    Returns a square matrix for each, where n = number of columns
    """
    int_ln = lambda arr: np.logical_not(arr)*1
    r = int_ln(pd.isnull(data.values))
    rr = np.matmul(r.T, r)
    mm = np.matmul(int_ln(r).T, int_ln(r))
    mr = np.matmul(int_ln(r).T, r)
    rm = np.matmul(r.T, int_ln(r))
    pairs = dict(rr=rr, rm=rm, mr=mr, mm=mm)
    pairs = {k: _sq_output(v, data.columns, True)
             for k, v in pairs.items()}
    return pairs

@check_data_structure
def md_pattern(data):
    """
    Calculates row-wise missing data statistics, where
    - 0 is missing, 1 is not missing
    - num rows is num different row patterns
    - 'nmis' is number of missing values in a row pattern
    - 'count' is number of total rows with row pattern
    """
    cols = data.columns.tolist()
    r = pd.isnull(data.values)
    nmis = np.sum(r, axis=0)
    r = r[:, np.argsort(nmis)]
    num_string = lambda row: "".join(str(e) for e in row)
    pat = np.apply_along_axis(num_string, 1, r*1)
    sort_r = r[np.argsort(pat), :]*1
    sort_r_df = _sq_output(sort_r, cols, False)
    sort_r_df = sort_r_df.groupby(cols).size().reset_index()
    sort_r_df.columns = cols + ["count"]
    sort_r_df["nmis"] = sort_r_df[cols].sum(axis=1)
    sort_r_df[cols] = sort_r_df[cols].apply(np.logical_not)*1
    return sort_r_df[["count"] + cols + ["nmis"]]

@check_missingness
def feature_cov(data):
    """
    Calculates the covariance between features in a dataframe
    - Note that this method DROPS NA VALUES to compute cov
    - Checks to ensure dataset not fully missing, or else no cov possible
    Returns dataframe with correlation between each feature
    """
    return data.cov()

@check_missingness
def feature_corr(data, method="pearson"):
    """
    Calculates the correlation between features in a dataframe
    - Note that this method DROPS NA VALUES to compute corr
    - Default method is pearson
    - If dataset encodes discrete features, proper method is spearman
    - Checks to ensure dataset not fully missing, or else no corr possible
    Returns dataframe with correlation between each numerical feature
    """
    accepted_methods = ("pearson", "kendall", "spearman")
    if method not in accepted_methods:
        raise ValueError(f"Correlation method must be in {accepted_methods}")
    return data.corr(method=method)

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
    Instead, use specific methods below (inbound, outbound, etc.)
    These special methods utilize this function to compute specific stats
    """
    pairs = md_pairs(data)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = func(pairs)
    return stat

def inbound(data):
    """
    Calculates proportion of usable cases (Ijk)
    - Ijk = 1 if variable Yk observed in all records where Yj missing
    - Used to quickly select potential predictors Yk for imputing Yj
    - High values are preferred
    """
    inbound_coeff = get_stat_for(_inbound, data)
    inbound_ = _sq_output(inbound_coeff, data.columns, True)
    return inbound_

def outbound(data):
    """
    Calculates the outbound statistic (Ojk)
    - Ojk measures how observed data Yj connect to rest of missing data
    - Ojk = 1 if Yj observed in all records where Yk is missing
    - Used to evaluate whether Yj is a potential predictor for imputing Yk
    - High values are preferred
    """
    outbound_coeff = get_stat_for(_outbound, data)
    outbound_ = _sq_output(outbound_coeff, data.columns, True)
    return outbound_

def influx(data):
    """
    Calculates the influx coefficient (Ij)
    - Ij = # pairs (Yj,Yk) w/ Yj missing & Yk observed / # observed data cells
    - Value depends on the proportion of missing data of the variable
    - Influx of a completely observed variable is equal to 0
    - Influx for completely missing variables is equal to 1
    - For two variables with the same proportion of missing data:
        - Var with higher influx is better connected to the observed data
        - Var with higher influx might thus be easier to impute
    """
    influx_coeff = get_stat_for(_influx, data)
    influx_coeff = influx_coeff.reshape(1, len(influx_coeff))
    influx_ = _sq_output(influx_coeff, data.columns, False)
    influx_.index = ["Influx"]
    return influx_

def outflux(data):
    """
    Calculates the outflux coefficient (Oj)
    - Oj = # pairs w/ Yj observed and Yk missing / # incomplete data cells
    - Value depends on the proportion of missing data of the variable
    - Outflux of a completely observed variable is equal to 1
    - Outflux of a completely missing variable is equal to 0.
    - For two variables having the same proportion of missing data:
        - Var with higher outflux is better connected to the missing data
        - Var with higher outflux more useful for imputing other variables
    """
    outflux_coeff = get_stat_for(_outflux, data)
    outflux_coeff = outflux_coeff.reshape(1, len(outflux_coeff))
    outflux_ = _sq_output(outflux_coeff, data.columns, False)
    outflux_.index = ["Outflux"]
    return outflux_

@check_data_structure
def proportions(data):
    """
    Calculates the proportions of the data missing and observed
    - poms: Proportion of missing size
    - pobs: Proportion of observed size
    """
    poms = np.mean(pd.isnull(data), axis=0)
    pobs = np.mean(np.logical_not(pd.isnull(data)), axis=0)
    proportions_dict = dict(poms=poms, pobs=pobs)
    proportions_ = _index_output(proportions_dict, data.columns)
    return proportions_

def flux(data):
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
        pobs = proportions(data)["pobs"]
        ainb = np.apply_along_axis(row_mean, 1, _inbound(pairs))
        aout = np.apply_along_axis(row_mean, 1, _outbound(pairs))
        inf = _influx(pairs)
        outf = _outflux(pairs)
        res = dict(pobs=pobs, influx=inf, outflux=outf, ainb=ainb, aout=aout)
    flux_ = _index_output(res, data.columns)
    return flux_
