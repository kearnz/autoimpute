"""This module implements categorical imputation via the CategoricalImputer.

The CategoricalImputer determines the proportions of discrete features within
observed data. It then samples this distribution to impute missing values.
Right now, this imputer supports imputation on Series only. Use
SingleImputer(strategy="categorical") to broadcast the imputation strategy
across multiple columns of a DataFrame.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=unnecessary-pass

class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Techniques to impute w/ draw from dataset's categorical distribution.

    More complex autoimpute Imputers delegate work to the CategoricalImputer
    if categorical is a specified strategy. That being said,
    CategoricalImputer is a stand-alone class and valid sklearn transformer.
    It can be used directly, but such behavior is discouraged, because this
    imputer supports Series only. CategoricalImputer does not have the
    flexibility or robustness of more complex imputers, nor is its behavior
    identical. Instead, use SingleImputer(strategy="categorical").
    """
    # class variables
    strategy = methods.CATEGORICAL

    def __init__(self):
        """Create an instance of the CategoricalImputer class."""
        pass

    def fit(self, X):
        """Fit the Imputer to the dataset and calculate proportions.

        Args:
            X (pd.Series): Dataset to fit the imputer

        Returns:
            self. Instance of the class.
        """

        # get proportions of discrete observed values to sample from
        proportions = X.value_counts() / np.sum(~X.isnull())
        self.statistics_ = {"param": proportions, "strategy": self.strategy}
        return self

    def transform(self, X):
        """Perform imputations using the statistics generated from fit.

        The transform method handles the actual imputation. Transform
        constructs a categoricalal distribution for each feature using the mean
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

        # get observed weighted by count of total and sample
        imp = self.statistics_["param"]
        cats = imp.index
        proportions = imp.tolist()
        samples = np.random.choice(cats, size=len(ind), p=proportions)
        imp = pd.Series(samples, index=ind)

        # fill missing values in X with samples from distribution
        X.fillna(imp, inplace=True)
        return X
