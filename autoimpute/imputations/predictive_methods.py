"""Private imputation methods used by different Imputer Classes."""

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from autoimpute.imputations.single_methods import _fit_none
# pylint:disable=unused-argument

# FIT IMPUTATION
# --------------
# Methods below represent fits for associated fit methods above.

def _fit_linear_reg(series):
    """Private method to fit data for linear regression imputation."""
    method = "linear"
    return None, method

def _fit_binary_logistic(series):
    """Private method to fit data for binary logistic imputation."""
    method = "binary logistic"
    return None, method

def _fit_multi_logistic(series):
    """Private method to fit data for multinomial logistic imputation."""
    method = "multinomial logistic"
    return None, method

def _predictive_default(series):
    """Private method to fit data for default predictive imputation."""
    method = "default"
    if is_numeric_dtype(series):
        return _fit_linear_reg(series)
    if is_string_dtype(series):
        ser_unique = series.unique()
        ser_len = len(ser_unique)
        if ser_len == 1:
            return series.unique()[0], method
        if ser_len == 2:
            return _fit_binary_logistic(series)
        if ser_len > 2:
            return _fit_multi_logistic(series)
    else:
        return _fit_none(series)
    return None, method
