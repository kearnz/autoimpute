"""Methods to numerically assess patterns in missing data.

This module is a collection of methods to explore missing data and its
patterns. The module's methods are heavily influenced by those found in
section 4.1 of Flexible Imputation of Missing Data (Van Buuren).

Methods:
    md_locations(data, both=False)
    md_pairs(data)
    md_pattern(data)
    feature_corr(data)
    feature_cov(data)
    inbound(data)
    outbound(data)
    influx(data)
    outflux(data)
    proportions(data)
    flux(data)

Todo:
    * Add futher functionality to assess missing data patterns
        - Examples include those in missingno package
        - Other R packages have EDA not contained in VB 4.1
    * Add examples of each method in respective function docstrings
"""

import numpy as np
import pandas as pd
from autoimpute.utils.checks import check_data_structure, check_missingness
from autoimpute.utils.helpers import _sq_output, _index_output

@check_data_structure
def md_locations(data, both=False):
    """Produces locations where values are missing in a DataFrame.

    Takes in a DataFrame and identifies locations where data is complete or
    missing. Normally, fully complete or fully empty throws error, but
    this method simply shows missingness locations, so standard for mixed
    complete-missing not necessary. Method marks 1 = missing, 0 = not missing.

    Args:
        data (pd.DataFrame): data to locate missing and complete
        both (boolean, optional): return data along with missingness indicator.
            Defaults to False, so just missingness indicator returned.

    Returns:
        pd.DataFrame: missingness indicator DataFrame.

    Raises:
        TypeError: if data is not a DataFrame. Error raised through decorator.
    """
    md_df = pd.isnull(data)*1
    if both:
        md_df = pd.concat([data, md_df], axis=1)
    return md_df

@check_data_structure
def md_pairs(data):
    """Calculates pairwise missing data statistics

    This method mimics the behavior of MICE md.pairs
    - rr: response-response pairs
    - rm: response-missing pairs
    - mr: missing-response pairs
    - mm: missing-missing pairs
    Returns a square matrix for each, where n = number of columns

    Args:
        data (pd.DataFrame): data to calculate pairwise stats.

    Returns:
        dict: keys are pair types, values are DataFrames w/ pair stats.

    Raises:
        TypeError: if data is not a DataFrame. Error raised through decorator.
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
    """Calculates row-wise missing data statistics in input data.

    Method is a port of md.pattern method from VB 4.1
    - 0 is missing, 1 is not missing
    - num rows is num different row patterns
    - 'nmis' is number of missing values in a row pattern
    - 'count' is number of total rows with row pattern

    Args:
        data (pd.DataFrame): data to calculate missing data pattern.

    Returns:
        pd.DataFrame: DataFrame with missing data pattern and two
            additional cols w/ row-wise stats: count and nmis
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
    """Calculates the covariance between features in a DataFrame

    Method to calculate covariance:
    - Note that this method DROPS NA VALUES to compute cov
    - Checks to ensure dataset not fully missing, or else no cov possible

    Args:
        data (pd.DataFrame): data to calculate covariance matrix.

    Returns:
        pd.DataFrame: DataFrame with covariance between each feature.

    Raises:
        TypeError: If data not pd.DataFrame
        ValueError: If DataFrame values all missing and none complete.
    """
    return data.cov()

@check_missingness
def feature_corr(data, method="pearson"):
    """Calculates the correlation between features in a DataFrame.

    Method to calculate correlation:
    - Note that this method DROPS NA VALUES to compute corr
    - Default method is pearson
    - If dataset encodes discrete features, proper method is spearman
    - Checks to ensure dataset not fully missing, or else no corr possible

    Args:
        data (pd.DataFrame): data to calculate correlation matrix.
        method (string, optional): corr method to use. Default is pearson,
            but spearman should be used if any features categorically encoded.

    Returns:
        pd.DataFrame: DataFrame with correlation between each feature.

    Raises:
        TypeError: If data not pd.DataFrame.
        ValueError: If DataFrame values all missing and none complete.
        ValueError: If method for correlation not an accepted method.
    """
    accepted_methods = ("pearson", "kendall", "spearman")
    if method not in accepted_methods:
        raise ValueError(f"Correlation method must be in {accepted_methods}")
    return data.corr(method=method)

def _inbound(pairs):
    """Helper to get inbound from pairs. Intended for private use.

    Args:
        pairs (dict): Pairs generated from md_pairs function

    Returns:
        np.ndarray: Pairwise stat for inbound
    """
    return pairs["mr"]/(pairs["mr"]+pairs["mm"])

def _outbound(pairs):
    """Helper to get outbound from pairs. Intended for private use.

    Args:
        pairs (dict): Pairs generated from md_pairs function

    Returns:
        np.ndarray: Pairwise stat for outbound
    """
    return pairs["rm"]/(pairs["rm"]+pairs["rr"])

def _influx(pairs):
    """Helper to get influx from pairs. Intended for private use.

    Args:
        pairs (dict): Pairs generated from md_pairs function

    Returns:
        np.ndarray: Pairwise stat for influx
    """
    num = np.nansum(pairs["mr"], axis=1)
    denom = np.nansum(pairs["mr"]+pairs["rr"], axis=1)
    return num/denom

def _outflux(pairs):
    """Helper to get outflux from pairs. Intended for private use.

    Args:
        pairs (dict): Pairs generated from md_pairs function

    Returns:
        np.ndarray: Pairwise stat for outflux
    """
    num = np.nansum(pairs["rm"], axis=1)
    denom = np.nansum(pairs["rm"]+pairs["mm"], axis=1)
    return num/denom

def get_stat_for(func, data):
    """Generic method to get a missing data statistic from data.

    This method can be used directly in tandem with helper methods,
    but this behavior is discouraged. Instead, use specific methods below
    (inbound, outbound, etc.). These special methods utilize this function
    to compute specific stats.

    Args:
        func (function): Function that calculates a statistic
        data (pd.DataFrame): data on which to run the function

    Returns:
        np.ndarray: Output from statistic chosen.
    """
    pairs = md_pairs(data)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = func(pairs)
    return stat

def inbound(data):
    """Calculates proportion of usable cases (Ijk) from Van Buuren 4.1

    Method ported from VB, called "inbound statistic".
    - Ijk = 1 if variable Yk observed in all records where Yj missing
    - Used to quickly select potential predictors Yk for imputing Yj
    - High values are preferred

    Args:
        data (pd.DataFrame): Data to calculate inbound stat.

    Returns:
        pd.DataFrame: inbound statistic b/w each feature and other features.
            Inbound b/w a feature and itself is 0.
    """
    inbound_coeff = get_stat_for(_inbound, data)
    inbound_ = _sq_output(inbound_coeff, data.columns, True)
    return inbound_

def outbound(data):
    """Calculates the outbound statistic (Ojk) from Van Buuren 4.1

    Method ported from VB, called "outbound statistic".
    - Ojk measures how observed data Yj connect to rest of missing data
    - Ojk = 1 if Yj observed in all records where Yk is missing
    - Used to evaluate whether Yj is a potential predictor for imputing Yk
    - High values are preferred

    Args:
        data (pd.DataFrame): Data to calculate outbound stat.

    Returns:
        pd.DataFrame: outbound statistic b/w each feature and other features.
            Outbound b/w a feature and itself is 0.
    """
    outbound_coeff = get_stat_for(_outbound, data)
    outbound_ = _sq_output(outbound_coeff, data.columns, True)
    return outbound_

def influx(data):
    """Calculates the influx coefficient (Ij) from Van Buuren 4.1

    Method ported from VB, called "influx coefficient"
    - Ij = # pairs (Yj,Yk) w/ Yj missing & Yk observed / # observed data cells
    - Value depends on the proportion of missing data of the variable
    - Influx of a completely observed variable is equal to 0
    - Influx for completely missing variables is equal to 1
    - For two variables with the same proportion of missing data:
        - Var with higher influx is better connected to the observed data
        - Var with higher influx might thus be easier to impute

    Args:
        data (pd.DataFrame): Data to calculate influx coefficient.

    Returns:
        pd.DataFrame: influx coefficient for each column.
    """
    influx_coeff = get_stat_for(_influx, data)
    influx_coeff = influx_coeff.reshape(1, len(influx_coeff))
    influx_ = _sq_output(influx_coeff, data.columns, False)
    influx_.index = ["Influx"]
    return influx_

def outflux(data):
    """Calculates the outflux coefficient (Oj) from Van Buuren 4.1

    Method ported from VB, called "outflux coefficient"
    - Oj = # pairs w/ Yj observed and Yk missing / # incomplete data cells
    - Value depends on the proportion of missing data of the variable
    - Outflux of a completely observed variable is equal to 1
    - Outflux of a completely missing variable is equal to 0.
    - For two variables having the same proportion of missing data:
        - Var with higher outflux is better connected to the missing data
        - Var with higher outflux more useful for imputing other variables

    Args:
        data (pd.DataFrame): Data to calculate outflux coefficient.

    Returns:
        pd.DataFrame: outflux coefficient for each column.
    """
    outflux_coeff = get_stat_for(_outflux, data)
    outflux_coeff = outflux_coeff.reshape(1, len(outflux_coeff))
    outflux_ = _sq_output(outflux_coeff, data.columns, False)
    outflux_.index = ["Outflux"]
    return outflux_

@check_data_structure
def proportions(data):
    """Calculates the proportions of the data missing and observed.

    Method calculates two arrays:
    - poms: Proportion of missing size
    - pobs: Proportion of observed size

    Args:
        data (pd.DataFrame): Data to calculate proportions.

    Returns:
        pd.DataFrame: two columns, one for poms and one for pobs. The
            sum of each row should equal 1. Index = original data cols.

    Raises:
        TypeError: if data not DataFrame. Error raised through decorator.
    """
    poms = np.mean(pd.isnull(data), axis=0)
    pobs = np.mean(np.logical_not(pd.isnull(data)), axis=0)
    proportions_dict = dict(poms=poms, pobs=pobs)
    proportions_ = _index_output(proportions_dict, data.columns)
    return proportions_

def flux(data):
    """Caclulates inbound, influx, outbound, outflux, pobs, for data

    Port of Van Buuren's flux method in R. Calculates:
    - pobs: Proportion observed
    - ainb: Average inbound statistic
    - aout: Average outbound statistic
    - influx: Influx coefficient (Ij)
    - outflux: Outflux coefficient (Oj)

    Args:
        data (pd.DataFrame): Data to calculate proportions.

    Returns:
        pd.DataFrame: one column for each summary statistic. Index = data cols.
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
