"""Private imputation methods used by different Imputer Classes."""

import warnings
import logging
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from scipy.stats import norm, multivariate_normal
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
# pylint:disable=too-many-locals
# pylint:disable=too-many-arguments

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
        print(f"Missing values in response {series.name}: {sum_null_ser}")
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

def _fit_bayes_least_squares_reg(predictors, series, verbose,
                                 am=0, asd=10, bm=0, bsd=10, sig=1):
    """Private method to fit data for bayesian regression imputation."""
    method = "bayesian least squares"
    _not_num_series(method, series)
    X, y = _get_observed(method, predictors, series, verbose)
    # initialize model for bayesian linear regression
    with pm.Model() as fit_model:
        alpha = pm.Normal("alpha", am, sd=asd)
        beta = pm.Normal("beta", bm, sd=bsd, shape=len(X.columns))
        sigma = pm.HalfCauchy("σ", sig)
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

def _fit_bayes_binary_logistic_reg(predictors, series, verbose,
                                   am=0, asd=10, bm=0, bsd=10):
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
        alpha = pm.Normal("alpha", am, asd)
        beta = pm.Normal("beta", bm, bsd, shape=len(X.columns))
        p = pm.invlogit(alpha + beta.dot(X.T))
        score = pm.Bernoulli("score", p, observed=y.codes)
    return (fit_model, y.categories), method

def _fit_pmm_reg(predictors, series, verbose,
                 am=0, asd=10, bm=0, bsd=10, sig=1):
    """Private method to fit data for predictive mean matching imputation."""
    method = "pmm"
    _not_num_series(method, series)
    X, y = _get_observed(method, predictors, series, verbose)
    # get predictions for the observed, which will be used for "closest" vals
    lm = LinearRegression()
    y_pred = lm.fit(X, y).predict(X)
    y_df = pd.DataFrame({"y":y, "y_pred":y_pred})
    lm_bayes, _ = _fit_bayes_least_squares_reg(
        predictors, series, False, am, asd, bm, bsd, sig
        )
    return (lm_bayes, lm, y_df), method

def _predictive_default(predictors, series, verbose):
    """Private method to fit data for default predictive imputation."""
    method = "default"
    if is_numeric_dtype(series):
        return _fit_pmm_reg(predictors, series, verbose)
    elif is_string_dtype(series):
        ser_unique = series.dropna().unique()
        ser_len = len(ser_unique)
        if ser_len <= 2:
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

def _imp_bayes_least_squares_reg(X, col_name, x, lm, imp_ix, fv, verbose,
                                 sample=1000, tune=1000, init="auto"):
    """Private method to perform bayesian regression imputation."""
    progress = _pymc3_logger(verbose)
    with lm:
        mu_pred = pm.Deterministic("mu_pred", lm["alpha"]+lm["beta"].dot(x.T))
        tr = pm.sample(
            sample=sample, tune=tune, init=init, progress_bar=progress
        )
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

def _imp_bayes_logistic_reg(X, col_name, x, lm, imp_ix, fv, verbose,
                            sample=1000, tune=1000, init="auto", thresh=0.5):
    """Private method to perform bayesian logistic imputation."""
    progress = _pymc3_logger(verbose)
    model, labels = lm
    with model:
        p_pred = pm.Deterministic(
            "p_pred", pm.invlogit(model["alpha"] + model["beta"].dot(x.T))
        )
        tr = pm.sample(
            sample=sample, tune=tune, init=init, progress_bar=progress
        )
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

def _neighbors(x, n, df, choose):
    al = len(df.index)
    if n > al:
        err = "# neighbors greater than # predictions. Reduce neighbor count."
        raise ValueError(err)
    indexarr = np.argpartition(abs(df["y_pred"] - x), n)[:n]
    neighbs = df.loc[indexarr, "y"].values
    return choose(neighbs)

def _imp_pmm_reg(X, col_name, x, lm, imp_ix, fv, verbose, n=5,
                 sample=1000, tune=1000, init="auto"):
    """Private method to perform predictive mean matching imputation."""
    progress = _pymc3_logger(verbose)
    model, _, df = lm
    df = df.reset_index(drop=True)
    # generate posterior for alpha, beta
    with model:
        tr = pm.sample(
            sample=sample, tune=tune, init=init, progress_bar=progress
            )

    # sample random alpha, get mean and covariance of beta coefficients
    # beta assumed multivariate normal by linear reg rules
    # sample beta w/ cov structure to create realistic variability
    alpha_bayes = np.random.choice(tr["alpha"])
    beta_means = tr["beta"].mean(0)
    beta_cov = np.cov(tr["beta"].T)
    beta_bayes = np.array(multivariate_normal(beta_means, beta_cov).rvs())

    # predictions for missing y, using bayes alpha + coeff samples
    # use these preds for nearest neighbor search from reg results
    y_pred_bayes = alpha_bayes + beta_bayes.dot(x.T)
    if not fv or fv == "mean":
        fills = [_neighbors(x, n, df, np.mean) for x in y_pred_bayes]
    elif fv == "random":
        fills = [_neighbors(x, n, df, np.random.choice) for x in y_pred_bayes]
    else:
        err = f"{fv} not accepted reducer. Choose `mean` or `random`."
        raise ValueError(err)

    # finally, impute and return trace for bayes
    X.loc[imp_ix, col_name] = fills
    stats = {"tr": tr, "y_pred": y_pred_bayes,
             "betas": beta_bayes, "alpha": alpha_bayes}
    return stats
