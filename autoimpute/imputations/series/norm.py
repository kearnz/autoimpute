"""This module implements norm imputation via the NormImputer.

The NormImputer imputes missing data with random draws from a construted
normal distribution. Dataframe imputers utilize this class when its strategy
is requested. Use SingleImputer or MultipleImputer with strategy = `norm` to
broadcast the strategy across all the columns in a dataframe, or specify this
strategy for a given column.
"""

from scipy.stats import norm
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
from autoimpute.imputations.errors import _not_num_series
from .base import ISeriesImputer
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=unnecessary-pass

class NormImputer(ISeriesImputer):
    """Impute missing data with draws from normal distribution.

    The NormImputer constructs a normal distribution using the sample mean and
    variance of the observed data. The imputer then randomly samples from this
    distribution to impute missing data. The imputer can be used directly, but
    such behavior is discouraged. NormImputer does not have the flexibility /
    robustness of dataframe imputers, nor is its behavior identical.
    Preferred use is MultipleImputer(strategy="norm").
    """
    # class variables
    strategy = methods.NORM

    def __init__(self):
        """Create an instance of the NormImputer class."""
        pass

    def fit(self, X, y=None):
        """Fit Imputer to dataset and calculate mean and sample variance.

        Args:
            X (pd.Series): Dataset to fit the imputer.
            y (None): ignored, None to meet requirements of base class

        Returns:
            self. Instance of the class.
        """

        # get the moments for the normal distribution of feature X
        _not_num_series(self.strategy, X)
        moments = (X.mean(), X.std())
        self.statistics_ = {"param": moments, "strategy": self.strategy}
        return self

    def impute(self, X):
        """Perform imputations using the statistics generated from fit.

        The transform method handles the actual imputation. It constructs a
        normal distribution using the sample mean and variance from fit.
        It then imputes missing values with a random draw from the respective
        distribution.

        Args:
            X (pd.Series): Dataset to impute missing data from fit.

        Returns:
            np.array -- imputed dataset.
        """

        # check if fitted and identify location of missingness
        check_is_fitted(self, "statistics_")
        _not_num_series(self.strategy, X)
        ind = X[X.isnull()].index

        # create normal distribution and sample from it
        imp_mean, imp_std = self.statistics_["param"]
        imp = norm(imp_mean, imp_std).rvs(size=len(ind))
        return imp

    def fit_impute(self, X, y):
        """Convenience method to perform fit and imputation in one go."""
        return self.fit(X, y=None).impute(X)
