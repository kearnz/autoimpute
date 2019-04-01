"""This module implements categorical imputation via the CategoricalImputer.

The CategoricalImputer determines the proportions of discrete features within
observed data. It then samples this distribution to impute missing values.
Right now, this imputer supports imputation on Series only. Use
SingleImputer(strategy="categorical") to broadcast the imputation strategy
across multiple columns of a DataFrame.
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
from autoimpute.imputations.errors import _not_cat_series
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=unnecessary-pass

class CategoricalImputer(BaseEstimator):
    """Impute missing data w/ draw from dataset's categorical distribution.

    The categorical imputer computes the proportion of observed values for
    each category within a discrete dataset. The imputer then samples the
    distribution to impute missing values with a respective random draw. The
    imputer can be used directly, but such behavior is discouraged because
    the imputer supports Series only. CategoricalImputer does not have the
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
            X (pd.Series): Dataset to fit the imputer.

        Returns:
            self. Instance of the class.
        """
        _not_cat_series(self.strategy, X)
        # get proportions of discrete observed values to sample from
        proportions = X.value_counts() / np.sum(~X.isnull())
        self.statistics_ = {"param": proportions, "strategy": self.strategy}
        return self

    def impute(self, X):
        """Perform imputations using the statistics generated from fit.

        The impute method handles the actual imputation. Transform
        constructs a categorical distribution for each feature using the
        proportions of observed values from fit. It then imputes missing
        values with a random draw from the respective distribution.

        Args:
            X (pd.Series): Dataset to impute missing data from fit.

        Returns:
            np.array -- imputed dataset.
        """
        # check if fitted and identify location of missingness
        check_is_fitted(self, "statistics_")
        _not_cat_series(self.strategy, X)
        ind = X[X.isnull()].index

        # get observed weighted by count of total and sample
        param = self.statistics_["param"]
        cats = param.index
        proportions = param.tolist()
        imp = np.random.choice(cats, size=len(ind), p=proportions)
        return imp

    def fit_impute(self, X):
        """Convenience method to perform fit and imputation in one go."""
        return self.fit(X).impute(X)
