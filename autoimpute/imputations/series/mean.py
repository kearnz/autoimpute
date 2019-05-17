"""This module implements mean imputation via the MeanImputer.

The MeanImputer imputes missing data with the mean of observed data.
Dataframe imputers utilize this class when its strategy is requested. Use
SingleImputer or MultipleImputer with strategy = `mean` to broadcast the
strategy across all the columns in a dataframe, or specify this strategy
for a given column.
"""

from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
from autoimpute.imputations.errors import _not_num_series
from .base import ISeriesImputer
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=unnecessary-pass

class MeanImputer(ISeriesImputer):
    """Impute missing values with the mean of the observed data.

    This imputer imputes missing values with the mean of observed data.
    The imputer can be used directly, but such behavior is discouraged.
    MeanImputer does not have the flexibility / robustness of dataframe
    imputers, nor is its behavior identical. Preferred use is
    MultipleImputer(strategy="mean").
    """
    # class variables
    strategy = methods.MEAN

    def __init__(self):
        """Create an instance of the MeanImputer class."""
        pass

    def fit(self, X, y):
        """Fit the Imputer to the dataset and calculate the mean.

        Args:
            X (pd.Series): Dataset to fit the imputer.
            y (None): ignored, None to meet requirements of base class

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

    def fit_impute(self, X, y=None):
        """Convenience method to perform fit and imputation in one go."""
        return self.fit(X, y).impute(X)
