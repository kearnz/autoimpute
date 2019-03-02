"""Imputation methods used by different Imputer Classes."""

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
# pylint:disable=unused-argument

def _none(series):
    """don't impute a series. leave as is"""
    method = "none"
    return None, method

def _mean(series):
    """helper mean"""
    method = "mean"
    if not is_numeric_dtype(series):
        typ = series.dtype
        err = f"{method} not appropriate for Series with type {typ}"
        raise TypeError(err)
    return series.mean(), method

def _median(series):
    """helper median"""
    method = "median"
    if not is_numeric_dtype(series):
        typ = series.dtype
        err = f"{method} not appropriate for Series with type {typ}"
        raise TypeError(err)
    return series.median(), method

def _mode(series):
    """helper mode"""
    method = "mode"
    return series.mode(), method

def _single_default(series):
    """helper function for default"""
    if is_numeric_dtype(series):
        return _mean(series)
    elif is_string_dtype(series):
        return _mode(series)
    else:
        return _none(series)

def _random(series):
    """return random values to select from"""
    method = "random"
    return series[~series.isnull()].unique(), method

def _linear(series):
    """helper method for linear interpolation"""
    method = "linear"
    if not is_numeric_dtype(series):
        typ = series.dtype
        err = f"{method} not appropriate for Series with type {typ}"
        raise TypeError(err)
    return None, method
