"""Private imputation methods used by different Imputer Classes."""

import numpy as np
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from scipy.stats import norm
from autoimpute.imputations.errors import _not_num_series, _not_cat_series
# pylint:disable=unused-argument

# FIT IMPUTATION
# --------------
# Methods below represent fits for associated fit methods above.

def _fit_none(series):
    """Private method for no imputation (leave series as is)."""
    method = "none"
    return None, method

def _fit_mean(series):
    """Private method to fit data for mean imputation."""
    method = "mean"
    _not_num_series(method, series)
    return series.mean(), method

def _fit_median(series):
    """Private method to fit data for median imputation."""
    method = "median"
    _not_num_series(method, series)
    return series.median(), method

def _fit_mode(series):
    """Private method to fit data for mode imputation."""
    method = "mode"
    return series.mode(), method

def _fit_random(series):
    """Private method to fit data for random imputation."""
    method = "random"
    return series[~series.isnull()].unique(), method

def _fit_norm(series):
    """Private method to fit data for normal distribution imputation."""
    method = "norm"
    _not_num_series(method, series)
    mu = series.mean()
    sd = series.std()
    return (mu, sd), method

def _fit_categorical(series):
    """Private method to fit data for categorical distribution imputation."""
    method = "categorical"
    _not_cat_series(method, series)
    proportions = series.value_counts() / np.sum(~series.isnull())
    return proportions, method

def _fit_single_default(series):
    """Private method to fit data for single, cs default imputation."""
    if is_numeric_dtype(series):
        return _fit_mean(series)
    elif is_string_dtype(series):
        return _fit_mode(series)
    else:
        return _fit_none(series)

# TRANSFORM IMPUTATION
# --------------------
# Methods below represent transformations for associated fit methods above.

def _imp_central(X, col_name, fill_val):
    """Private method used generally for mean and median imputation."""
    X[col_name].fillna(fill_val, inplace=True)

def _imp_mode(X, col_name, mode, strategy):
    """Private method used for mode imputation."""
    num_modes = len(mode)
    if num_modes == 1:
        return X[col_name].fillna(mode[0], inplace=True)
    else:
        if strategy is None:
            return X[col_name].fillna(mode[0], inplace=True)
        elif strategy == "random":
            ind = X[col_name][X[col_name].isnull()].index
            fills = np.random.choice(mode, len(ind))
            X.loc[ind, col_name] = fills
        else:
            err = f"{strategy} not accepted for mode imputation"
            raise ValueError(err)

def _imp_random(X, col_name, fill_val, imp_ind):
    """Private method for random value imputation."""
    fills = np.random.choice(fill_val, len(imp_ind))
    X.loc[imp_ind, col_name] = fills

def _imp_norm(X, col_name, fill_val, imp_ind):
    """Private method for imputation using random draws from norm dist."""
    mu, std = fill_val
    fills = norm.rvs(loc=mu, scale=std, size=len(imp_ind))
    X.loc[imp_ind, col_name] = fills

def _imp_categorical(X, col_name, fill_val, imp_ind):
    """Private method for imputation using random draws from cat pmf."""
    cats = fill_val.index
    proportions = fill_val.tolist()
    fills = np.random.choice(cats, size=len(imp_ind), p=proportions)
    X.loc[imp_ind, col_name] = fills
