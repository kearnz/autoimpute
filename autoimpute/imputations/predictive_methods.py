"""Private imputation methods used by different Imputer Classes."""

import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from autoimpute.imputations.deletion import listwise_delete
from autoimpute.imputations.single_methods import _fit_none
from autoimpute.imputations.errors import _not_num_series, _not_num_matrix

# pylint:disable=unused-argument
# pylint:disable=inconsistent-return-statements

# FIT IMPUTATION
# --------------
# Methods below represent fits for associated fit methods above.

def _get_observed(method, predictors, series, verbose):
    """Helper method to test datasets and get observed data."""
    _not_num_series(method, series)
    _not_num_matrix(method, predictors)
    conc = pd.concat([predictors, series], axis=1)
    predictors = listwise_delete(conc, verbose=verbose)
    series = predictors.pop(series.name)
    return predictors, series

def _fit_linear_reg(predictors, series, verbose, **kwargs):
    """Private method to fit data for linear regression imputation."""
    method = "linear"
    predictors, series = _get_observed(method, predictors, series, verbose)
    return None, method

def _fit_binary_logistic(predictors, series, verbose, **kwargs):
    """Private method to fit data for binary logistic imputation."""
    method = "binary logistic"
    predictors, series = _get_observed(method, predictors, series, verbose)
    return None, method

def _fit_multi_logistic(predictors, series, verbose, **kwargs):
    """Private method to fit data for multinomial logistic imputation."""
    method = "multinomial logistic"
    predictors, series = _get_observed(method, predictors, series, verbose)
    return None, method

def _predictive_default(predictors, series, verbose, **kwargs):
    """Private method to fit data for default predictive imputation."""
    method = "default"
    if is_numeric_dtype(series):
        return _fit_linear_reg(predictors, series, verbose, **kwargs)
    elif is_string_dtype(series):
        ser_unique = series.unique()
        ser_len = len(ser_unique)
        if ser_len == 1:
            return series.unique()[0], method
        if ser_len == 2:
            return _fit_binary_logistic(predictors, series, verbose, **kwargs)
        if ser_len > 2:
            return _fit_multi_logistic(predictors, series, verbose, **kwargs)
    else:
        return _fit_none(series)
