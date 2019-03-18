"""Private imputation methods used by different Imputer Classes."""

import warnings
import logging
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
import pymc3 as pm
from autoimpute.imputations import single_methods
from autoimpute.imputations.deletion import listwise_delete
from autoimpute.imputations.errors import _not_num_series, _not_num_matrix
sm = single_methods
# pylint:disable=unused-argument
# pylint:disable=inconsistent-return-statements
# pylint:disable=protected-access
# pylint:disable=unused-variable
# pylint:disable=no-member

# FIT IMPUTATION
# --------------
# Methods below represent fits for associated fit methods above.

def _get_observed(method, predictors, series, verbose):
    """Helper method to test datasets and get observed data."""
    _not_num_matrix(method, predictors)
    conc = pd.concat([predictors, series], axis=1)
    if verbose:
        null_pred = pd.isnull(predictors)
        null_ser = pd.isnull(series)
        for each in null_pred:
            sum_null_pred = null_pred[each].sum()
            print(f"Missing values in predictor {each}: {sum_null_pred}")
        sum_null_ser = null_ser.sum()
        print(f"Missing values in predictor {series.name}: {sum_null_ser}")
    predictors = listwise_delete(conc, verbose=verbose)
    series = predictors.pop(series.name)
    return predictors, series

def _pymc3_logger(verbose):
    """Private method to handle pymc3 logging."""
    progress = 1
    if not verbose:
        progress = 0
        logger = logging.getLogger('pymc3')
        logger.setLevel(logging.ERROR)
    return progress

def _fit_least_squares_reg(predictors, series, verbose):
    """Private method to fit data for linear regression imputation."""
    method = "least squares"
    _not_num_series(method, series)
    X, y = _get_observed(method, predictors, series, verbose)
    lm = LinearRegression()
    lm.fit(X, y)
    return lm, method

def _fit_stochastic_reg(predictors, series, verbose):
    """Private method to fit data for stochastic regression imputation."""
    method = "stochastic"
    _not_num_series(method, series)
    X, y = _get_observed(method, predictors, series, verbose)
    lm = LinearRegression()
    lm.fit(X, y)
    preds = lm.predict(X)
    mse = mean_squared_error(y, preds)
    return (lm, mse), method

def _fit_bayes_least_squares_reg(predictors, series, verbose):
    """Private method to fit data for bayesian regression imputation."""
    method = "bayesian least squares"
    _not_num_series(method, series)
    X, y = _get_observed(method, predictors, series, verbose)
    # initialize model for bayesian linear regression
    with pm.Model() as fit_model:
        alpha = pm.Normal("alpha", 0, sd=10)
        beta = pm.Normal("beta", 0, sd=10, shape=len(X.columns))
        sigma = pm.HalfCauchy("σ", 1)
        mu = alpha+beta.dot(X.T)
        score = pm.Normal("score", mu, sd=sigma, observed=y)
    return fit_model, method

def _fit_binary_logistic_reg(predictors, series, verbose):
    """Private method to fit data for binary logistic imputation."""
    method = "binary logistic"
    X, y = _get_observed(method, predictors, series, verbose)
    glm = LogisticRegression(solver="liblinear")
    y = y.astype("category").cat
    y_cat_l = len(y.codes.unique())
    if y_cat_l != 2:
        err = "This method requires 2 categories. Use multinomial instead."
        raise ValueError(err)
    glm.fit(X, y.codes)
    return (glm, y.categories), method

def _fit_multi_logistic_reg(predictors, series, verbose):
    """Private method to fit data for multinomial logistic imputation."""
    method = "multinomial logistic"
    X, y = _get_observed(method, predictors, series, verbose)
    glm = LogisticRegression(solver="saga", multi_class="multinomial")
    y = y.astype("category").cat
    y_cat_l = len(y.codes.unique())
    if y_cat_l == 2:
        w = "Multiple categories expected. Consider binary instead if c = 2."
        warnings.warn(w)
    glm.fit(X, y.codes)
    return (glm, y.categories), method

def _fit_bayes_binary_logistic_reg(predictors, series, verbose):
    """Private method to fit data for binary bayesian logistic imputation."""
    method = "bayesian binary logistic"
    X, y = _get_observed(method, predictors, series, verbose)
    y = y.astype("category").cat
    y_cat_l = len(y.codes.unique())
    if y_cat_l != 2:
        err = "Only two categories supported. Multinomial bayes coming soon."
        raise ValueError(err)
    # initialize model for bayesian linear regression
    with pm.Model() as fit_model:
        mu = pm.Normal("mu", 0, sd=10)
        beta = pm.Normal("beta", 0, sd=10, shape=len(X.columns))
        p = pm.invlogit(mu + beta.dot(X.T))
        score = pm.Bernoulli("score", p, observed=y.codes)
    return (fit_model, y.categories), method

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

def _imp_least_squares_reg(X, col_name, x, lm, imp_ix):
    """Private method to perform linear regression imputation."""
    fills = lm.predict(x)
    X.loc[imp_ix, col_name] = fills

def _imp_stochastic_reg(X, col_name, x, lm, imp_ix):
    """Private method to perform stochastic regression imputation."""
    model, mse = lm
    preds = model.predict(x)
    mse_dist = norm.rvs(loc=0, scale=mse, size=len(preds))
    fills = preds + mse_dist
    X.loc[imp_ix, col_name] = fills

def _imp_bayes_least_squares_reg(X, col_name, x, lm, imp_ix, fv, verbose):
    """Private method to perform bayesian regression imputation."""
    progress = _pymc3_logger(verbose)
    with lm:
        mu_pred = pm.Deterministic("mu_pred", lm["alpha"]+lm["beta"].dot(x.T))
        tr = pm.sample(1000, tune=1000, progress_bar=progress)
    if not fv or fv == "mean":
        fills = tr["mu_pred"].mean(0)
    elif fv == "random":
        fills = np.apply_along_axis(np.random.choice, 0, tr["mu_pred"])
    else:
        err = f"{fv} not accepted reducer. Choose `mean` or `random`."
        raise ValueError(err)
    X.loc[imp_ix, col_name] = fills
    return tr

def _imp_logistic_reg(X, col_name, x, lm, imp_ix):
    """Private method to perform linear regression imputation."""
    model, labels = lm
    fills = model.predict(x)
    label_dict = {i:j for i, j in enumerate(labels.values)}
    X.loc[imp_ix, col_name] = fills
    X[col_name].replace(label_dict, inplace=True)

def _imp_bayes_logistic_reg(X, col_name, x, lm, imp_ix,
                            fv, verbose, thresh=0.5):
    """Private method to perform bayesian logistic imputation."""
    progress = _pymc3_logger(verbose)
    model, labels = lm
    with model:
        p_pred = pm.Deterministic(
            "p_pred", pm.invlogit(model["mu"] + model["beta"].dot(x.T))
        )
        tr = pm.sample(1000, tune=1000, progress_bar=progress)
    if not fv or fv == "mean":
        fills = tr["p_pred"].mean(0)
    elif fv == "random":
        fills = np.apply_along_axis(np.random.choice, 0, tr["p_pred"])
    else:
        err = f"{fv} not accepted reducer. Choose `mean` or `random`."
        raise ValueError(err)
    fill_thresh = np.vectorize(lambda f: 1 if f > thresh else 0)
    fills = fill_thresh(fills)
    label_dict = {i:j for i, j in enumerate(labels.values)}
    X.loc[imp_ix, col_name] = fills
    X[col_name].replace(label_dict, inplace=True)
    return tr
