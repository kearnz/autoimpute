"""Private imputation methods used by Time-Based Imputer Classes."""

import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from autoimpute.imputations.errors import _not_num_err
from autoimpute.imputations.single_methods import _fit_none, _fit_mode

# FIT IMPUTATION
# --------------
# Methods below represent fits for associated fit methods above.

def _fit_linear(series):
    """Private method to fit data for linear interpolation."""
    method = "linear"
    _not_num_err(method, series)
    return None, method

def _fit_time(series):
    """Private method to fit data for time-weighted interpolation."""
    method = "time"
    _not_num_err(method, series)
    return None, method

def _fit_locf(series):
    """Private method to fit data for last obs carried forward imputation."""
    method = "locf"
    _not_num_err(method, series)
    # return mean incase needed for first observation
    return series.mean(), method

def _fit_nocb(series):
    """Private method to fit data for next obs carried backward imputation."""
    method = "nocb"
    _not_num_err(method, series)
    # return mean incase needed for last observation
    return series.mean(), method

def _fit_ts_default(series):
    """Private method to fit data for single, ts default imputation."""
    if is_numeric_dtype(series):
        return _fit_linear(series)
    elif is_string_dtype(series):
        return _fit_mode(series)
    else:
        return _fit_none(series)

# TRANSFORM IMPUTATION
# --------------------
# Methods below represent transformations for associated fit methods above.

def _imp_interp(X, col_name, method):
    """Private method to wrap interpolation methods for imputation."""
    X[col_name].interpolate(method=method,
                            limit=None,
                            limit_direction="both",
                            inplace=True)

def _imp_locf(X, col_name, fill_val):
    """Private method for last obs carried forward imputation."""
    first = X.index[0]
    if pd.isnull(X.loc[first, col_name]):
        X.loc[first, col_name] = fill_val
    X[col_name].fillna(method="ffill", inplace=True)

def _imp_nocb(X, col_name, fill_val):
    """Private method for next obs carried backward imputation."""
    last = X.index[-1]
    if pd.isnull(X.loc[last, col_name]):
        X.loc[last, col_name] = fill_val
    X[col_name].fillna(method="bfill", inplace=True)
