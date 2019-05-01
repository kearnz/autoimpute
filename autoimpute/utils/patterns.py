"""Methods to numerically assess patterns in missing data.

This module is a collection of methods to explore missing data and its
patterns. The module's methods are heavily influenced by those found in
section 4.1 of Flexible Imputation of Missing Data (Van Buuren). Their main
purpose is to identify trends and patterns in missing data that can help
inform what type of imputation method may apply or what cautions to take
when performing imputations in general.
"""

import numpy as np
import pandas as pd
from autoimpute.utils import check_data_structure, check_missingness
from autoimpute.utils.helpers import _sq_output, _index_output

@check_data_structure
def md_locations(data, both=False):
    """Produces locations where values are missing in a DataFrame.

    Takes in a DataFrame and identifies locations where data is complete or
    missing. Normally, fully complete issues warning, and fully incomplete
    throws error, but this method simply shows missingness locations,
    so the general standard for mixed complete-missing pattern not necessary.
    Method marks 1 = missing, 0 = not missing.

    Args:
        data (pd.DataFrame): DataFrame to find missing & complete observations.
        both (boolean, optional): return data along with missingness indicator.
            Defaults to False, so just missingness indicator returned.

    Returns:
        pd.DataFrame: missingness indicator DataFrame OR
        pd.DataFrame: missingness indicator DataFrame concatenated column-wise
            with original DataFame.

    Raises:
        TypeError: if data is not a DataFrame. Error raised through decorator.
    """
    md_df = pd.isnull(data)*1
    if both:
        md_df = pd.concat([data, md_df], axis=1)
    return md_df

@check_data_structure
def md_pairs(data):
    """Calculates pairwise missing data statistics.

    This method mimics the behavior of MICE md.pairs.
    - rr: response-response pairs
    - rm: response-missing pairs
    - mr: missing-response pairs
    - mm: missing-missing pairs
    Returns a square matrix for each, where n = number of columns.

    Args:
        data (pd.DataFrame): DataFrame to calculate pairwise stats.

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

    Method is a port of md.pattern method from VB 4.1. The number of rows
    indicates the number of different row patterns of missingness. The 'nmis'
    column is the number of missing values in a given row pattern. The
    'count' is number of total rows with a given row pattern.
    In this method, 0 = missing, 1 = missing.

    Args:
        data (pd.DataFrame): DataFrame to calculate missing data pattern.

    Returns:
        pd.DataFrame: DataFrame with missing data pattern and two
            additional columns w/ row-wise stats: `count` and `nmis`.
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
def nullility_cov(data):
    """Calculates the nullility covariance between features in a DataFrame.

    Leverages pandas method to calculate covariance of nullility. Note that
    this method drops NA values to compute covariance. It also employs
    `check_missingness` decorator to ensure DataFrame not fully missing. If
    a DataFrame is fully observed, nothing is returned, as there is no
    nullility.

    Args:
        data (pd.DataFrame): DataFrame to calculate nullility covariance.

    Returns:
        pd.DataFrame: DataFrame with nullility covariance b/w each feature.

    Raises:
        TypeError: If data not pd.DataFrame. Raised through decorator.
        ValueError: If DataFrame values all missing and none complete.
            Also raised through decorator.
    """
    data_cov = data.isnull().cov()
    return data_cov.dropna(axis=0, how="all").dropna(axis=1, how="all")

@check_missingness
def nullility_corr(data, method="pearson"):
    """Calculates the nullility correlation between features in a DataFrame.

    Leverages pandas method to calculate correlation of nullility. Note that
    this method drops NA values to compute correlation. It also employs
    `check_missingness` decorator to ensure DataFrame not fully missing. If
    a DataFrame is fully observed, nothing is returned, as there is no
    nullility.

    Args:
        data (pd.DataFrame): DataFrame to calculate nullility correlation.
        method (string, optional): correlation method to use. Default pearson,
            but spearman should be used with categorical or ordinal encoding.

    Returns:
        pd.DataFrame: DataFrame with nullility correlation b/w each feature.

    Raises:
        TypeError: If data not pd.DataFrame. Raised through decorator.
        ValueError: If DataFrame values all missing and none complete.
            Also raised through decorator.
        ValueError: If method for correlation not an accepted method.
    """
    accepted_methods = ("pearson", "kendall", "spearman")
    if method not in accepted_methods:
        err = f"Correlation method must be in {accepted_methods}"
        raise ValueError(err)
    data_corr = data.isnull().corr(method=method)
    return data_corr.dropna(axis=0, how="all").dropna(axis=1, how="all")

def _inbound(pairs):
    """Private method to get inbound from pairs."""
    return pairs["mr"]/(pairs["mr"]+pairs["mm"])

def _outbound(pairs):
    """Private method to get outbound from pairs."""
    return pairs["rm"]/(pairs["rm"]+pairs["rr"])

def _influx(pairs):
    """Private method to get influx from pairs."""
    num = np.nansum(pairs["mr"], axis=1)
    denom = np.nansum(pairs["mr"]+pairs["rr"], axis=1)
    return num/denom

def _outflux(pairs):
    """Private method to get outflux from pairs."""
    num = np.nansum(pairs["rm"], axis=1)
    denom = np.nansum(pairs["rm"]+pairs["mm"], axis=1)
    return num/denom

def get_stat_for(func, data):
    """Generic method to get a missing data statistic from data.

    This method can be used directly with helper methods, but this behavior
    is discouraged. Instead, use specific public methods below. These special
    methods utilize this function internally to compute summary statistics.

    Args:
        func (function): Function that calculates a statistic.
        data (pd.DataFrame): DataFrame on which to run the function.

    Returns:
        np.ndarray: Output from statistic chosen.
    """
    pairs = md_pairs(data)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = func(pairs)
    return stat

def inbound(data):
    """Calculates proportion of usable cases (Ijk) from Van Buuren 4.1.

    Method ported from VB, called "inbound statistic", Ijk.
    Ijk = 1 if variable Yk observed in all records where Yj missing.
    Used to quickly select potential predictors Yk for imputing Yj.
    High values are preferred.

    Args:
        data (pd.DataFrame): DataFrame to calculate inbound statistic.

    Returns:
        pd.DataFrame: inbound statistic between each of the features.
            Inbound between a feature and itself is 0.
    """
    inbound_coeff = get_stat_for(_inbound, data)
    inbound_ = _sq_output(inbound_coeff, data.columns, True)
    return inbound_

def outbound(data):
    """Calculates the outbound statistic (Ojk) from Van Buuren 4.1.

    Method ported from VB, called "outbound statistic", Ojk.
    Ojk measures how observed data Yj connect to rest of missing data.
    Ojk = 1 if Yj observed in all records where Yk is missing.
    Used to evaluate whether Yj is a potential predictor for imputing Yk.
    High values are preferred.

    Args:
        data (pd.DataFrame): DataFrame to calculate outbound statistic.

    Returns:
        pd.DataFrame: outbound statistic between each of the features.
            Outbound between a feature and itself is 0.
    """
    outbound_coeff = get_stat_for(_outbound, data)
    outbound_ = _sq_output(outbound_coeff, data.columns, True)
    return outbound_

def influx(data):
    """Calculates the influx coefficient (Ij) from Van Buuren 4.1.

    Method ported from VB, called "influx coefficient", Ij.
    Ij = # pairs (Yj,Yk) w/ Yj missing & Yk observed / # observed data cells.
    Value depends on the proportion of missing data of the variable.
    Influx of a completely observed variable is equal to 0.
    Influx for completely missing variables is equal to 1.
    For two variables with the same proportion of missing data:
    - Variable with higher influx is better connected to the observed data.
    - Variable with higher influx might thus be easier to impute.

    Args:
        data (pd.DataFrame): DataFrame to calculate influx coefficient.

    Returns:
        pd.DataFrame: influx coefficient for each column.
    """
    influx_coeff = get_stat_for(_influx, data)
    influx_coeff = influx_coeff.reshape(1, len(influx_coeff))
    influx_ = _sq_output(influx_coeff, data.columns, False)
    influx_.index = ["Influx"]
    return influx_

def outflux(data):
    """Calculates the outflux coefficient (Oj) from Van Buuren 4.1.

    Method ported from VB, called "outflux coefficient", Oj.
    Oj = # pairs w/ Yj observed and Yk missing / # incomplete data cells.
    Value depends on the proportion of missing data of the variable.
    Outflux of a completely observed variable is equal to 1.
    Outflux of a completely missing variable is equal to 0.
    For two variables having the same proportion of missing data:
    - Variable with higher outflux is better connected to the missing data.
    - Variable with higher outflux more useful for imputing other variables.

    Args:
        data (pd.DataFrame): DataFrame to calculate outflux coefficient.

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
    """Calculates the proportions of the data missing and data observed.

    Method calculates two arrays:
    - `poms`: Proportion of missing data.
    - `pobs`: Proportion of observed data.

    Args:
        data (pd.DataFrame): DataFrame to calculate proportions.

    Returns:
        pd.DataFrame: two columns, one for `poms` and one for `pobs`.
            The sum of each row should equal 1. Index = original data cols.

    Raises:
        TypeError: if data not DataFrame. Error raised through decorator.
    """
    poms = np.mean(pd.isnull(data), axis=0)
    pobs = np.mean(np.logical_not(pd.isnull(data)), axis=0)
    proportions_dict = dict(poms=poms, pobs=pobs)
    proportions_ = _index_output(proportions_dict, data.columns)
    return proportions_

def flux(data):
    """Caclulates inbound, influx, outbound, outflux, pobs, for DataFrame.

    Port of Van Buuren's flux method in R. Calculates:
    - `pobs`: Proportion observed (column from the `proportions` method).
    - `ainb`: Average inbound statistic.
    - aout: Average outbound statistic.
    - influx: Influx coefficient, Ij (from the `influx` method).
    - outflux: Outflux coefficient, Oj (from the `outflux` method).

    Args:
        data (pd.DataFrame): DataFrame to calculate relevant statistics.

    Returns:
        pd.DataFrame: one column for each summary statistic.
            Columns of DataFrame equal the name of the summary statistics.
            Indices of DataFrame equal the original DataFrame columns.
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
