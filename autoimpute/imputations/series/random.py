"""This module implements random imputation via the RandomImputer.

The RandomImputer imputes missing data using a random draw with replacement
from the observed data. Dataframe imputers utilize this class when its
strategy is requested. Use SingleImputer or MultipleImputer with
strategy = `random` to broadcast the strategy across all the columns in a
dataframe, or specify this strategy for a given column.
"""

import numpy as np
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
from .base import ISeriesImputer
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=unnecessary-pass

class RandomImputer(ISeriesImputer):
    """Impute missing data using random draws from observed data.

    The RandomImputer samples with replacement from observed data. The imputer
    can be used directly, but such behavior is discouraged. RandomImputer does
    not have the flexibility / robustness of dataframe imputers, nor is its
    behavior identical. Preferred use is MultipleImputer(strategy="random").
    """
    # class variables
    strategy = methods.RANDOM

    def __init__(self):
        """Create an instance of the RandomImputer class."""
        pass

    def fit(self, X, y=None):
        """Fit the Imputer to the dataset and get unique observed to sample.

        Args:
            X (pd.Series): Dataset to fit the imputer.
            y (None): ignored, None to meet requirements of base class

        Returns:
            self. Instance of the class.
        """

        # determine set of observed values to sample from
        random = list(set(X[~X.isnull()]))
        self.statistics_ = {"param": random, "strategy": self.strategy}
        return self

    def impute(self, X):
        """Perform imputations using the statistics generated from fit.

        The transform method handles the actual imputation. Each missing value
        in a given dataset is replaced with a random draw from unique set of
        observed values determined during the fit stage.

        Args:
            X (pd.Series): Dataset to impute missing data from fit.

        Returns:
            np.array -- imputed dataset
        """
        # check if fitted and identify location of missingness
        check_is_fitted(self, "statistics_")
        ind = X[X.isnull()].index

        # get the observed values and sample from them
        param = self.statistics_["param"]
        imp = np.random.choice(param, len(ind))
        return imp

    def fit_impute(self, X, y=None):
        """Convenience method to perform fit and imputation in one go."""
        return self.fit(X, y).impute(X)
