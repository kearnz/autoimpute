"""This module implements norm imputation via the NormImputer.

The NormImputer constructs a normal distribution with mean and variance
determined from observed values. Missing values are imputed with random draws
from the resulting normal distribution. Right now, this imputer supports
imputation on Series only. Use SingleImputer(strategy="norm") to broadcast
the imputation strategy across multiple columns of a DataFrame.
"""

import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=unnecessary-pass

class NormImputer(BaseEstimator, TransformerMixin):
    """Techniques to impute with draw from a dataset's normal distribution.

    More complex autoimpute Imputers delegate work to the NormImputer if
    norm is a specified strategy. That being said, NormImputer is a
    stand-alone class and valid sklearn transformer. It can be used directly,
    but such behavior is discouraged, because this imputer
    supports Series only. NormImputer does not have the flexibility or
    robustness of more complex imputers, nor is its behavior identical.
    Instead, use SingleImputer(strategy="norm").
    """
    # class variables
    strategy = methods.NORM

    def __init__(self):
        """Create an instance of the NormImputer class."""
        pass

    def fit(self, X):
        """Fit the Imputer to the dataset and calculate mean and variance.

        Args:
            X (pd.Series): Dataset to fit the imputer

        Returns:
            self. Instance of the class.
        """

        # get the moments for the normal distribution of feature X
        moments = (X.mean(), X.var()/(len(X.index)-1))
        self.statistics_ = {"param": moments, "strategy": self.strategy}
        return self

    def transform(self, X):
        """Perform imputations using the statistics generated from fit.

        The transform method handles the actual imputation. Transform
        constructs a normal distribution for each feature using the mean
        and variance from fit. It then imputes missing values with a
        random draw from the respective distribution

        Args:
            X (pd.Series): Dataset to fit the imputer

        Returns:
            pd.Series -- imputed dataset
        """
        # check if fitted and identify location of missingness
        check_is_fitted(self, "statistics_")
        ind = X[X.isnull()].index

        # create normal distribution and sample from it
        imp_mean, imp_var = self.statistics_["param"]
        samples = norm(imp_mean, imp_var).rvs(size=len(ind))
        fills = pd.Series(samples, index=ind)

        # fill missing values in X with draws from normal
        X.fillna(fills, inplace=True)
        return X
