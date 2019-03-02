"""Imputation methods used by different Imputer Classes."""

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

def _mean(series):
    """helper mean"""
    return series.mean(), "mean"

def _median(series):
    """helper median"""
    return series.median(), "median"

def _mode(series):
    """helper mode"""
    return series.mode(), "mode"

def _default(series):
    """helper function for default"""
    if is_numeric_dtype(series):
        return _mean(series)
    if is_string_dtype(series):
        return _mode(series)

def _random(series):
    """return random values to select from"""
    return series[~series.isnull()].unique(), "random"
