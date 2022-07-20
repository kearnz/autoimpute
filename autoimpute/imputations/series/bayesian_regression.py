"""This module implements bayesian techniques to impute missing data.

This module contains BayesLeastSquaresImputer and BayesBinaryLogisticImputer.
Both imputers are the bayesian equivalent of their frequentist counterparts
(LeastSquaresImputer and BinaryLogisticImputer). Dataframe imputers utilize
the classes in this module when each's respective strategy is requested.
Use SingleImputer or MultipleImputer with strategy = `bayesian least squares`
or `bayesian binary logistic` to broadcast the strategies across all the
columns in a dataframe, or specify either strategy for a given column.
"""

import numpy as np
import pymc as pm
from pandas import Series
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
from autoimpute.imputations.errors import _not_num_series
from .base import ISeriesImputer
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=too-many-arguments
# pylint:disable=unused-variable
# pylint:disable=no-member
# pylint:disable=too-many-instance-attributes
# pylint:disable=unsubscriptable-object

class BayesianLeastSquaresImputer(ISeriesImputer):
    """Impute missing values using bayesian least squares regression.

    The BayesianLeastSquaresImputer produces predictions using the bayesian
    approach to least squares. Prior distributions are fit for the model
    parameters of interest (alpha, beta, epsilon). Imputations for missing
    values are samples from posterior predictive distribution of each missing
    point. To implement bayesian least squares, the imputer utlilizes the
    pymc library. The imputer can be used directly, but such behavior is
    discouraged. BayesianLeastSquaresImputer does not have the flexibility /
    robustness of dataframe imputers, nor is its behavior identical. Preferred
    use is MultipleImputer(strategy="bayesian least squares").
    """
    # class variables
    strategy = methods.BAYESIAN_LS

    def __init__(self, **kwargs):
        """Create an instance of the BayesianLeastSquaresImputer class.

        The class requires multiple arguments necessary to create priors for
        a bayesian linear regression equation. The regression is:
        alpha + beta * X + epsilson. Because paramaters are treated as
        random variables, we must specify their distributions, including
        the parameters of those distributions. In thie init method we also
        include arguments used to sample the posterior distributions.

        Args:
            **kwargs: default keyword arguments used for bayesian analysis.
                Note - kwargs popped for default arguments defined below.
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
                create imputations. Default is None. 'random' and 'mean'
                supported for explicit options.
        """
        self.am = kwargs.pop("am", 0)
        self.asd = kwargs.pop("asd", 10)
        self.bm = kwargs.pop("bm", 0)
        self.bsd = kwargs.pop("bsd", 10)
        self.sig = kwargs.pop("sig", 1)
        self.sample = kwargs.pop("sample", 1000)
        self.tune = kwargs.pop("tune", 1000)
        self.init = kwargs.pop("init", "auto")
        self.fill_value = kwargs.pop("fill_value", None)
        self.sample_kwargs = kwargs

    def fit(self, X, y):
        """Fit the Imputer to the dataset by fitting bayesian model.

        Args:
            X (pd.Dataframe): dataset to fit the imputer.
            y (pd.Series): response, which is eventually imputed.

        Returns:
            self. Instance of the class.
        """
        _not_num_series(self.strategy, y)
        nc = len(X.columns)

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
        self.statistics_ = {"param": fit_model, "strategy": self.strategy}
        return self

    def impute(self, X, k=None):
        """Generate imputations using predictions from the fit bayesian model.

        The transform method returns the values for imputation. Missing values
        in a given dataset are replaced with the samples from the posterior
        predictive distribution of each missing data point.

        Args:
            X (pd.DataFrame): predictors to determine imputed values.
            k (integer): optional, pass if and only if receiving from MICE
        Returns:
            np.array: imputed dataset.
        """
        # check if fitted then predict with least squares
        check_is_fitted(self, "statistics_")
        model = self.statistics_["param"]

        # add a Deterministic node for each missing value
        # sampling then pulls from the posterior predictive distribution
        # each missing data point. I.e. distribution for EACH missing
        base_name = "mu_pred"
        if k is not None:
            base_name = f"{base_name}_{k}"
        with model:
            mu_pred = pm.Deterministic(
                base_name, model["alpha"]+model["beta"].dot(X.T)
            )
            tr = pm.sample(
                self.sample,
                tune=self.tune,
                init=self.init,
                **self.sample_kwargs
            )
        self.trace_ = tr

        # support for pymc - handling InferenceData obj instead of MultiTrace
        # we have to compress chains ourselves w/ InferenceData obj (xarray)
        post = tr.posterior[base_name].values
        chain, draws, dim = post.shape
        post = post.reshape(chain*draws, dim)

        # decide how to impute. Use mean of posterior predictive or random draw
        # not supported yet, but eventually consider using the MAP
        if not self.fill_value or self.fill_value == "mean":
            imp = post.mean(0)
        elif self.fill_value == "random":
            imp = np.apply_along_axis(np.random.choice, 0, post)
        else:
            err = f"{self.fill_value} must be 'mean' or 'random'."
            raise ValueError(err)
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


class BayesianBinaryLogisticImputer(ISeriesImputer):
    """Impute missing values using bayesian binary losgistic regression.

    The BayesianBinaryLogisticImputer produces predictions using the bayesian
    approach to logistic regression. Prior distributions are fit for the model
    parameters of interest (alpha, beta, epsilon). Imputations for missing
    values are samples from the posterior predictive distribution of each
    missing point. To implement bayesian logistic regression, the imputer uses
    the pymc library. The imputer can be used directly, but such behavior is
    discouraged. BayesianBinaryLogisticImputer does not have the flexibility /
    robustness of dataframe imputers, nor is its behavior identical.
    Preferred use is MultipleImputer(strategy="bayesian binary logistic").
    """
    # class variables
    strategy = methods.BAYESIAN_BINARY_LOGISTIC

    def __init__(self, **kwargs):
        """Create an instance of the BayesianBinaryLogisticImputer class.

        The class requires multiple arguments necessary to create priors for
        a bayesian logistic regression equation. The parameters are the same
        as linear regression, but the regression equation is transformed using
        pymc's invlogit method. Because paramaters are treated as random
        variables, we must specify their distributions, including
        the parameters of those distributions. In the init method we also
        include arguments used to sample the posterior distributions.

        Args:
            **kwargs: default keyword arguments used for bayesian analysis.
                Note - kwargs popped for default arguments defined below.
                Rest of kwargs passed as params to sampling (see pymc).
            am (float, Optional): mean of alpha prior. Default 0.
            asd (float, Optional): std. deviation of alpha prior. Default 10.
            bm (float, Optional): mean of beta priors. Default 0.
            bsd (float, Optional): std. deviation of beta priors. Default 10.
            thresh (float, Optional): threshold for class membership.
                Default 0.5. Max = 1, min = 0. Tune threshhold depending on
                class imbalance. Same as with logistic regression equation.
            sample (int, Optional): number of posterior samples per chain.
                Default = 1000. More samples, longer to run, but better
                approximation of the posterior & chance of convergence.
            tune (int, Optional): parameter for tuning. Draws done in addition
                to sample. Default = 1000.
            init (str, Optional): MCMC algo to use for posterior sampling.
                Default = 'auto'. See pymc docs for more info on choices.
            fill_value (str, Optional): How to draw from the posterior to
                create imputations. Default is None. 'random' and 'mean'
                supported for explicit options.
        """
        self.am = kwargs.pop("am", 0)
        self.asd = kwargs.pop("asd", 10)
        self.bm = kwargs.pop("bm", 0)
        self.bsd = kwargs.pop("bsd", 10)
        self.thresh = kwargs.pop("thresh", 0.5)
        self.sample = kwargs.pop("sample", 1000)
        self.tune = kwargs.pop("tune", 1000)
        self.init = kwargs.pop("init", "auto")
        self.fill_value = kwargs.pop("fill_value", None)
        self.sample_kwargs = kwargs

    def fit(self, X, y):
        """Fit the Imputer to the dataset by fitting bayesian model.

        Args:
            X (pd.Dataframe): dataset to fit the imputer.
            y (pd.Series): response, which is eventually imputed.

        Returns:
            self. Instance of the class.
        """
        y = y.astype("category").cat
        y_cat_l = len(y.codes.unique())

        # bayesian logistic regression. Mutliple categories not supported yet
        if y_cat_l != 2:
            err = "Only two categories supported. Multinomial coming soon."
            raise ValueError(err)
        nc = len(X.columns)

        # initialize model for bayesian logistic reg. Default vals for priors
        # assume data is scaled and centered. Convergence can struggle or fail
        # if not the case and proper values for the priors are not specified
        # separately, also assumes each beta is normal and "independent"
        # while betas likely not independent, this is technically a rule of OLS
        with pm.Model() as fit_model:
            alpha = pm.Normal("alpha", self.am, self.asd)
            beta = pm.Normal("beta", self.bm, self.bsd, shape=nc)
            p = pm.invlogit(alpha + beta.dot(X.T))
            score = pm.Bernoulli("score", p, observed=y.codes)

        params = {"model": fit_model, "labels": y.categories}
        self.statistics_ = {"param": params, "strategy": self.strategy}
        return self

    def impute(self, X, k=None):
        """Generate imputations using predictions from the fit bayesian model.

        The impute method returns the values for imputation. Missing values
        in a given dataset are replaced with the samples from the posterior
        predictive distribution of each missing data point.

        Args:
            X (pd.DataFrame): predictors to determine imputed values.
            k (integer): optional, pass if and only if receiving from MICE
        Returns:
            np.array: imputated dataset.
        """
        # check if fitted then predict with least squares
        check_is_fitted(self, "statistics_")
        model = self.statistics_["param"]["model"]
        labels = self.statistics_["param"]["labels"]

        # add a Deterministic node for each missing value
        # sampling then pulls from the posterior predictive distribution
        # each missing data point. I.e. distribution for EACH missing
        base_name = "p_pred"
        if k is not None:
            base_name = f"{base_name}_{k}"
        with model:
            p_pred = pm.Deterministic(
                base_name, pm.invlogit(model["alpha"] + model["beta"].dot(X.T))
            )
            tr = pm.sample(
                self.sample,
                tune=self.tune,
                init=self.init,
                **self.sample_kwargs
            )
        self.trace_ = tr

        # support for pymc - handling InferenceData obj instead of MultiTrace
        # we have to compress chains ourselves w/ InferenceData obj (xarray)
        post = tr.posterior[base_name].values
        chain, draws, dim = post.shape
        post = post.reshape(chain*draws, dim)

        # decide how to impute. Use mean of posterior predictive or random draw
        # not supported yet, but eventually consider using the MAP
        if not self.fill_value or self.fill_value == "mean":
            imp = post.mean(0)
        elif self.fill_value == "random":
            imp = np.apply_along_axis(np.random.choice, 0, post)
        else:
            err = f"{self.fill_value} must be 'mean' or 'random'."
            raise ValueError(err)

        # convert probabilities to class membership
        # then map class membership to corresponding label
        fill_thresh = np.vectorize(lambda f: 1 if f > self.thresh else 0)
        preds = fill_thresh(imp)
        label_dict = {i:j for i, j in enumerate(labels.values)}
        imp = Series(preds).replace(label_dict, inplace=False)
        return imp.values

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
