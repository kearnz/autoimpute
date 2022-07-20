"""This module implements predictive mean matching via the PMMImputer.

This module contains the PMMImputer, which implements predictive mean matching
to impute missing values. Predictive mean matching is a semi-supervised,
hot-deck technique to impute missing values. Dataframe imputers utilize this
class when its strategy is requested. Use SingleImputer or MultipleImputer
with strategy = `pmm` to broadcast the strategy across all the columns in a
dataframe, or specify this strategy for a given column.
"""

import numpy as np
import pymc as pm
from pandas import DataFrame
from scipy.stats import multivariate_normal
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
from autoimpute.imputations.helpers import _neighbors
from autoimpute.imputations.errors import _not_num_series
from .base import ISeriesImputer
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=no-member
# pylint:disable=unused-variable

class PMMImputer(ISeriesImputer):
    """Impute missing values using predictive mean matching.

    The PMMIMputer produces predictions using a combination of bayesian
    approach to least squares and least squares itself. For each missing value
    PMM finds the `n` closest neighbors from a least squares regression
    prediction set, and samples from the corresponding true values for y of
    each of those `n` predictions. The imputation is the resulting sample.
    To implement bayesian least squares, the imputer utlilizes the pymc
    library. The imputer can be used directly, but such behavior is
    discouraged. PmmImputer does not have the flexibility / robustness of
    dataframe imputers, nor is its behavior identical. Preferred use is
    MultipleImputer(strategy="pmm").
    """
    # class variables
    strategy = methods.PMM

    def __init__(self, **kwargs):
        """Create an instance of the PMMImputer class.

        The class requires multiple arguments necessary to create priors for
        a bayesian linear regression equation and least squares itself.
        Therefore, PMM arguments include all of those seen in bayesian least
        squares and least squares itself. New parameters include `neighbors`,
        or the number of neighbors that PMM uses to sample observed.

        Args:
            **kwargs: default keyword arguments for lm & bayesian analysis.
                Note - kwargs popped for default arguments defined below.
                Next set of kwargs popped and sent to linear regression.
                Rest of kwargs passed as params to sampling (see pymc).
            am (float, Optional): mean of alpha prior. Default 0.
            asd (float, Optional): std. deviation of alpha prior. Default 10.
            bm (float, Optional): mean of beta priors. Default 0.
            bsd (float, Optional): std. deviation of beta priors. Default 10.
            sig (float, Optional): parameter of sigma prior. Default 1.
            sample (int, Optional): number of posterior samples per chain.
                Default = 1000. More samples, longer to run, but better
                approximation of the posterior & chance of convergence.
            tune (int, Optional): parameter for tuning. Draws done in addition
                to sample. Default = 1000.
            init (str, Optional): MCMC algo to use for posterior sampling.
                Default = 'auto'. See pymc docs for more info on choices.
            fill_value (str, Optional): How to draw from the posterior to
                create imputations. Default is "random". 'random' and 'mean'
                supported for explicit options.
            neighbors (int, Optional): number of neighbors. Default is 5.
                Value should be greater than 0 and less than # observed,
                although anything greater than 10-20 generally too high
                unless dataset is massive.
            fit_intercept (bool, Optional): sklearn LinearRegression param.
            normalize (bool, Optional): sklearn LinearRegression param.
            copy_x (bool, Optional): sklearn LinearRegression param.
            n_jobs (int, Optional): sklearn LinearRegression param.
        """
        self.am = kwargs.pop("am", None)
        self.asd = kwargs.pop("asd", 10)
        self.bm = kwargs.pop("bm", None)
        self.bsd = kwargs.pop("bsd", 10)
        self.sig = kwargs.pop("sig", 1)
        self.sample = kwargs.pop("sample", 1000)
        self.tune = kwargs.pop("tune", 1000)
        self.init = kwargs.pop("init", "auto")
        self.fill_value = kwargs.pop("fill_value", "random")
        self.neighbors = kwargs.pop("neighbors", 5)
        self.fit_intercept = kwargs.pop("fit_intercept", True)
        self.copy_x = kwargs.pop("copy_x", True)
        self.n_jobs = kwargs.pop("n_jobs", None)
        self.lm = LinearRegression(
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_x,
            n_jobs=self.n_jobs
        )
        self.sample_kwargs = kwargs

    def fit(self, X, y):
        """Fit the Imputer to the dataset by fitting bayesian and LS model.

        Args:
            X (pd.Dataframe): dataset to fit the imputer.
            y (pd.Series): response, which is eventually imputed.

        Returns:
            self. Instance of the class.
        """
        _not_num_series(self.strategy, y)
        nc = len(X.columns)

        # get predictions for the data, which will be used for "closest" vals
        y_pred = self.lm.fit(X, y).predict(X)
        y_df = DataFrame({"y": y, "y_pred": y_pred})

        # calculate bayes and use appropriate means for alpha and beta priors
        # here we specify the point estimates from the linear regression as the
        # means for the priors. This will greatly speed up posterior sampling
        # and help ensure that convergence occurs
        if self.am is None:
            self.am = self.lm.intercept_
        if self.bm is None:
            self.bm = self.lm.coef_

        # initialize model for bayesian linear reg. Default vals for priors
        # assume data is scaled and centered. Convergence can struggle or fail
        # if not the case and proper values for the priors are not specified
        # separately, also assumes each beta is normal and "independent"
        # while betas likely not independent, this is technically a rule of OLS
        with pm.Model() as fit_model:
            alpha = pm.Normal("alpha", self.am, self.asd)
            beta = pm.Normal("beta", self.bm, self.bsd, shape=nc)
            sigma = pm.HalfCauchy("σ", self.sig)
            mu = alpha+beta.dot(X.T)
            score = pm.Normal("score", mu, sigma, observed=y)
        params = {"model": fit_model, "y_obs": y_df}
        self.statistics_ = {"param": params, "strategy": self.strategy}
        return self

    def impute(self, X):
        """Generate imputations using predictions from the fit bayesian model.

        The transform method returns the values for imputation. Missing values
        in a given dataset are replaced with the random selection from the PMM
        process. Again, PMM imputes actually observed values, and the observed
        values are selected by finding the closest least squares predictions
        to a given prediction from the bayesian model.

        Args:
            X (pd.DataFrame): predictors to determine imputed values.

        Returns:
            np.array: imputed dataset.
        """
        # check if fitted then predict with least squares
        check_is_fitted(self, "statistics_")
        model = self.statistics_["param"]["model"]
        df = self.statistics_["param"]["y_obs"]
        df = df.reset_index(drop=True)

        # generate posterior distribution for alpha, beta coefficients
        with model:
            tr = pm.sample(
                self.sample,
                tune=self.tune,
                init=self.init,
                **self.sample_kwargs
            )
        self.trace_ = tr

        # support for pymc - handling InferenceData obj instead of MultiTrace
        # we have to compress chains ourselves w/ InferenceData obj (xarray)
        post = tr.posterior
        alpha_, beta_ = post.alpha.values, post.beta.values
        chain, draws, beta_dim = beta_.shape
        beta_ = beta_.reshape(chain*draws, beta_dim)

        # sample random alpha from alpha posterior distribution
        alpha_bayes = np.random.choice(alpha_.ravel())

        # get the mean and covariance of the multivariate betas
        # betas assumed multivariate normal by linear reg rules
        # sample beta w/ cov structure to create realistic variability
        beta_means, beta_cov = beta_.mean(0), np.cov(beta_.T)
        beta_bayes = np.array(multivariate_normal(beta_means, beta_cov).rvs())

        # predictions for missing y, using bayes alpha + coeff samples
        # use these preds for nearest neighbor search from reg results
        # neighbors are nearest from prediction model fit on observed
        # imputed values are actual y vals corresponding to nearest neighbors
        # therefore, this is a form of "hot-deck" imputation
        y_pred_bayes = alpha_bayes + beta_bayes.dot(X.T)
        n_ = self.neighbors
        if X.columns.size == 1:
            y_pred_bayes = y_pred_bayes[0]
        if self.fill_value == "mean":
            imp = [_neighbors(x, n_, df, np.mean) for x in y_pred_bayes]
        elif self.fill_value == "random":
            choice = np.random.choice
            imp = [_neighbors(x, n_, df, choice) for x in y_pred_bayes]
        else:
            err = f"{self.fill_value} must be `mean` or `random`."
            raise ValueError(err)

        # finally, set last class values and return imputations
        self.y_pred = y_pred_bayes
        self.alphas = alpha_bayes
        self.betas = beta_bayes
        return imp

    def fit_impute(self, X, y):
        """Fit impute method to generate imputations where y is missing.

        Args:
            X (pd.Dataframe): predictors in the dataset.
            y (pd.Series): response w/ missing values to impute.

        Returns:
            np.array: imputed dataset.
        """
        # transform occurs with records from X where y is missing
        miss_y_ix = y[y.isnull()].index
        return self.fit(X, y).impute(X.loc[miss_y_ix])
