"""This module implements mean imputation via the MeanImputer.

The MeanImputer imputes missing data with the mean of observed data.
Right now, this imputer supports imputation on Series only.
Use SingleImputer(strategy="mean") to broadcast the imputation strategy across
multiple columns of a DataFrame.
"""

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
from autoimpute.imputations.errors import _not_num_series
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=unnecessary-pass

class MeanImputer(BaseEstimator):
    """Impute missing values with the mean of the observed data.

    This imputer imputes missing values with the mean of observed data.
    The imputer can be used directly, but such behavior is discouraged because
    the imputer supports Series only. MeanImputer does not have the
    flexibility or robustness of more complex imputers, nor is its behavior
    identical. Instead, use SingleImputer(strategy="mean").
    """
    # class variables
    strategy = methods.MEAN

    def __init__(self):
        """Create an instance of the MeanImputer class."""
        pass

    def fit(self, X):
        """Fit the Imputer to the dataset and calculate the mean.

        Args:
            X (pd.Series): Dataset to fit the imputer.

        Returns:
            self. Instance of the class.
        """
        _not_num_series(self.strategy, X)
        mu = X.mean()
        self.statistics_ = {"param": mu, "strategy": self.strategy}
        return self

    def impute(self, X):
        """Perform imputations using the statistics generated from fit.

        The impute method handles the actual imputation. Missing values
        in a given dataset are replaced with the respective mean from fit.

        Args:
            X (pd.Series): Dataset to impute missing data from fit.

        Returns:
            float -- imputed dataset.
        """
        # check if fitted then impute with mean
        check_is_fitted(self, "statistics_")
        _not_num_series(self.strategy, X)
        imp = self.statistics_["param"]
        return imp

    def fit_impute(self, X):
        """Convenience method to perform fit and imputation in one go."""
        return self.fit(X).impute(X)
