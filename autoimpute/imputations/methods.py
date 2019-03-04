"""Imputation methods used by different Imputer Classes."""

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
# pylint:disable=unused-argument

def _not_num_err(m, s):
    """error hanlding for all the private methods in this module."""
    if not is_numeric_dtype(s):
        typ = s.dtype
        err = f"{m} not appropriate for Series with type {typ}"
        raise TypeError(err)

def _none(series):
    """Private method for no imputation (leave series as is)."""
    method = "none"
    return None, method

def _mean(series):
    """Private method for mean imputation."""
    method = "mean"
    _not_num_err(method, series)
    return series.mean(), method

def _median(series):
    """Private method for median imputation."""
    method = "median"
    _not_num_err(method, series)
    return series.median(), method

def _mode(series):
    """Private method for mode imputation."""
    method = "mode"
    return series.mode(), method

def _random(series):
    """Private method for random imputation."""
    method = "random"
    return series[~series.isnull()].unique(), method

def _interp(series, method):
    """Private method to wrap interpolation methods."""
    series.interpolate(method=method,
                       limit=None,
                       limit_direction="both",
                       inplace=True)


def _linear(series):
    """Private method for linear interpolation."""
    method = "linear"
    _not_num_err(method, series)
    return None, method

def _time(series):
    """Private method for time-weighted interpolation."""
    method = "time"
    _not_num_err(method, series)
    return None, method

def _norm(series):
    """Private method for normal distribution imputation."""
    method = "norm"
    _not_num_err(method, series)
    mu = series.mean()
    sd = series.std()
    return (mu, sd), method

def _single_default(series):
    """Private method for single, cs default imputation."""
    if is_numeric_dtype(series):
        return _mean(series)
    elif is_string_dtype(series):
        return _mode(series)
    else:
        return _none(series)

def _ts_default(series):
    """Private method for single, ts default imputation."""
    if is_numeric_dtype(series):
        return _linear(series)
    elif is_string_dtype(series):
        return _mode(series)
    else:
        return _none(series)
