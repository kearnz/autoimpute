"""This module implements norm imputation via the NormImputer.

The NormImputer constructs a normal distribution with mean and variance
determined from observed values. Missing values are imputed with random draws
from the resulting normal distribution. Right now, this imputer supports
imputation on Series only. Use SingleImputer(strategy="norm") to broadcast
the imputation strategy across multiple columns of a DataFrame.
"""

from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
from autoimpute.imputations.errors import _not_num_series
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=unnecessary-pass

class NormImputer(BaseEstimator):
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
        """Fit Imputer to dataset and calculate mean and sample variance.

        Args:
            X (pd.Series): Dataset to fit the imputer

        Returns:
            self. Instance of the class.
        """

        # get the moments for the normal distribution of feature X
        _not_num_series(self.strategy, X)
        moments = (X.mean(), X.var()/(len(X.index)-1))
        self.statistics_ = {"param": moments, "strategy": self.strategy}
        return self

    def impute(self, X):
        """Perform imputations using the statistics generated from fit.

        The transform method handles the actual imputation. Transform
        constructs a normal distribution for each feature using the mean
        and sample variance from fit. It then imputes missing values with a
        random draw from the respective distribution.

        Args:
            X (pd.Series): Dataset to fit the imputer

        Returns:
            pd.Series -- imputed dataset
        """
        # check if fitted and identify location of missingness
        check_is_fitted(self, "statistics_")
        _not_num_series(self.strategy, X)
        ind = X[X.isnull()].index

        # create normal distribution and sample from it
        imp_mean, imp_var = self.statistics_["param"]
        imp = norm(imp_mean, imp_var).rvs(size=len(ind))
        return imp

    def fit_impute(self, X):
        """Helper method to perform fit and imputation in one go."""
        return self.fit(X).impute(X)
