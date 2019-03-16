"""Private imputation methods used by different Imputer Classes."""

import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LinearRegression, LogisticRegression
from autoimpute.imputations import single_methods
from autoimpute.imputations.deletion import listwise_delete
from autoimpute.imputations.errors import _not_num_series, _not_num_matrix
sm = single_methods
# pylint:disable=unused-argument
# pylint:disable=inconsistent-return-statements
# pylint:disable=protected-access

# FIT IMPUTATION
# --------------
# Methods below represent fits for associated fit methods above.

def _get_observed(method, predictors, series, verbose):
    """Helper method to test datasets and get observed data."""
    _not_num_matrix(method, predictors)
    conc = pd.concat([predictors, series], axis=1)
    predictors = listwise_delete(conc, verbose=verbose)
    series = predictors.pop(series.name)
    return predictors, series

def _fit_least_squares_reg(predictors, series, verbose):
    """Private method to fit data for linear regression imputation."""
    method = "least squares"
    _not_num_series(method, series)
    X, y = _get_observed(method, predictors, series, verbose)
    lm = LinearRegression()
    lm.fit(X, y)
    return lm, method

def _fit_binary_logistic_reg(predictors, series, verbose):
    """Private method to fit data for binary logistic imputation."""
    method = "binary logistic"
    X, y = _get_observed(method, predictors, series, verbose)
    glm = LogisticRegression(solver="liblinear")
    y = y.astype("category").cat
    glm.fit(X, y.codes)
    return (glm, y.categories), method

def _fit_multi_logistic_reg(predictors, series, verbose):
    """Private method to fit data for multinomial logistic imputation."""
    method = "multinomial logistic"
    X, y = _get_observed(method, predictors, series, verbose)
    glm = LogisticRegression(solver="saga", multi_class="multinomial")
    y = y.astype("category").cat
    glm.fit(X, y.codes)
    return (glm, y.categories), method

def _predictive_default(predictors, series, verbose):
    """Private method to fit data for default predictive imputation."""
    method = "default"
    if is_numeric_dtype(series):
        return _fit_least_squares_reg(predictors, series, verbose)
    elif is_string_dtype(series):
        ser_unique = series.dropna().unique()
        ser_len = len(ser_unique)
        if ser_len == 1:
            return series.unique()[0], method
        if ser_len == 2:
            return _fit_binary_logistic_reg(predictors, series, verbose)
        if ser_len > 2:
            return _fit_multi_logistic_reg(predictors, series, verbose)
    else:
        return sm._fit_none(series)

# TRANSFORM IMPUTATION
# --------------------
# Methods below represent transformations for associated fit methods above.

def _imp_least_squares_reg(X, col_name, x, lm, imp_ind):
    """Private method to perform linear regression imputation."""
    fills = lm.predict(x)
    X.loc[imp_ind, col_name] = fills

def _imp_logistic_reg(X, col_name, x, lm, imp_ind):
    """Private method to perform linear regression imputation."""
    model, labels = lm
    fills = model.predict(x)
    label_dict = {i:j for i, j in enumerate(labels.values)}
    X.loc[imp_ind, col_name] = fills
    X[col_name].replace(label_dict, inplace=True)
