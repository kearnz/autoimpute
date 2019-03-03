"""Imputation methods used by different Imputer Classes."""

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
# pylint:disable=unused-argument

def _err_method(m, s):
    if not is_numeric_dtype(s):
        typ = s.dtype
        err = f"{m} not appropriate for Series with type {typ}"
        raise TypeError(err)

def _none(series):
    """don't impute a series. leave as is"""
    method = "none"
    return None, method

def _mean(series):
    """helper mean"""
    method = "mean"
    _err_method(method, series)
    return series.mean(), method

def _median(series):
    """helper median"""
    method = "median"
    _err_method(method, series)
    return series.median(), method

def _mode(series):
    """helper mode"""
    method = "mode"
    return series.mode(), method

def _random(series):
    """return random values to select from"""
    method = "random"
    return series[~series.isnull()].unique(), method

def _interp(series, method):
    """Interpolation wrapper"""
    series.interpolate(method=method,
                       limit=None,
                       limit_direction="both",
                       inplace=True)


def _linear(series):
    """helper method for linear interpolation"""
    method = "linear"
    _err_method(method, series)
    return None, method

def _time(series):
    """helper method for time interpolation"""
    method = "time"
    _err_method(method, series)
    return None, method

def _single_default(series):
    """helper function for default"""
    if is_numeric_dtype(series):
        return _mean(series)
    elif is_string_dtype(series):
        return _mode(series)
    else:
        return _none(series)

def _ts_default(series):
    """helper function for default"""
    if is_numeric_dtype(series):
        return _linear(series)
    elif is_string_dtype(series):
        return _mode(series)
    else:
        return _none(series)
