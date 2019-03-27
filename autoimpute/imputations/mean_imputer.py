"""This module implements mean imputation via the MeanImputer.

The MeanImputer computes the mean of observed values then imputes missing data
with the computed mean. Mean imputation is univariate. Right now, this imputer
supports imputation on Series only. Use SingleImputer(strategy="mean") to
broadcast the imputation strategy across multiple columns of a DataFrame.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=unnecessary-pass

class MeanImputer(BaseEstimator, TransformerMixin):
    """Techniques to impute the mean for missing values within a dataset.

    More complex autoimpute Imputers delegate work to the MeanImputer if mean
    is a specified strategy for a given Series. That being said, MeanImputer
    is a stand-alone class and valid sklearn transformer. It can be used
    directly, but such behavior is discouraged because this imputer supports
    Series only. MeanImputer does not have the flexibility or robustness of
    more complex imputers, nor is its behavior identical.
    Instead, use SingleImputer(strategy="mean").
    """
    # class variables
    strategy = methods.MEAN

    def __init__(self):
        """Create an instance of the MeanImputer class."""
        pass

    def fit(self, X):
        """Fit the Imputer to the dataset and calculate the mean.

        Args:
            X (pd.Series): Dataset to fit the imputer

        Returns:
            self. Instance of the class.
        """
        mu = X.mean()
        self.statistics_ = {"param": mu, "strategy": self.strategy}
        return self

    def transform(self, X):
        """Perform imputations using the statistics generated from fit.

        The transform method handles the actual imputation. Missing values
        in a given dataset are replaced with the respective mean from fit.

        Args:
            X (pd.Series): Dataset to fit the imputer

        Returns:
            pd.Series -- imputed dataset
        """
        # check if fitted then impute with mean
        check_is_fitted(self, "statistics_")
        imp = self.statistics_["param"]
        X.fillna(imp, inplace=True)
        return X
