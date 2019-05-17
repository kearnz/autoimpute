"""This module implements median imputation via the MedianImputer.

The MedianImputer imputes missing data with the median of observed data.
Dataframe imputers utilize this class when its strategy is requested. Use
SingleImputer or MultipleImputer with strategy = `median` to broadcast the
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

class MedianImputer(ISeriesImputer):
    """Impute missing values with the median of the observed data.

    This imputer imputes missing values with the median of observed data.
    The imputer can be used directly, but such behavior is discouraged.
    MedianImputer does not have the flexibility / robustness of dataframe
    imputers, nor is its behavior identical. Preferred use is
    MultipleImputer(strategy="median").
    """
    # class variables
    strategy = methods.MEDIAN

    def __init__(self):
        """Create an instance of the MedianImputer class."""
        pass

    def fit(self, X, y=None):
        """Fit the Imputer to the dataset and calculate the median.

        Args:
            X (pd.Series): Dataset to fit the imputer.
            y (None): ignored, None to meet requirements of base class

        Returns:
            self. Instance of the class.
        """
        _not_num_series(self.strategy, X)
        median = X.median()
        self.statistics_ = {"param": median, "strategy": self.strategy}
        return self

    def impute(self, X):
        """Perform imputations using the statistics generated from fit.

        The impute method handles the actual imputation. Missing values
        in a given dataset are replaced with the respective median from fit.

        Args:
            X (pd.Series): Dataset to impute missing data from fit.

        Returns:
            float -- imputed dataset.
        """
        # check is fitted then impute with median
        check_is_fitted(self, "statistics_")
        _not_num_series(self.strategy, X)
        imp = self.statistics_["param"]
        return imp

    def fit_impute(self, X, y=None):
        """Convenience method to perform fit and imputation in one go."""
        return self.fit(X, y).impute(X)
