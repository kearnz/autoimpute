"""This module implements categorical imputation via the CategoricalImputer.

The CategoricalImputer determines the proportions of discrete features within
observed data. It then samples this distribution to impute missing values.
Dataframe imputers utilize this class when its strategy is requested. Use
SingleImputer or MultipleImputer with strategy = `categorical` to broadcast
the strategy across all the columns in a dataframe, or specify this strategy
for a given column.
"""

import numpy as np
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
from autoimpute.imputations.errors import _not_cat_series
from .base import ISeriesImputer
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=unnecessary-pass

class CategoricalImputer(ISeriesImputer):
    """Impute missing data w/ draw from dataset's categorical distribution.

    The categorical imputer computes the proportion of observed values for
    each category within a discrete dataset. The imputer then samples the
    distribution to impute missing values with a respective random draw. The
    imputer can be used directly, but such behavior is discouraged.
    CategoricalImputer does not have the flexibility / robustness of dataframe
    imputers, nor is its behavior identical. Preferred use is
    MultipleImputer(strategy="categorical").
    """
    # class variables
    strategy = methods.CATEGORICAL

    def __init__(self):
        """Create an instance of the CategoricalImputer class."""
        pass

    def fit(self, X, y=None):
        """Fit the Imputer to the dataset and calculate proportions.

        Args:
            X (pd.Series): Dataset to fit the imputer.
            y (None): ignored, None to meet requirements of base class

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

    def fit_impute(self, X, y=None):
        """Convenience method to perform fit and imputation in one go."""
        return self.fit(X, y).impute(X)
