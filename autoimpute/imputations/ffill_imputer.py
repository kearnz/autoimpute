"""This module implements forward & backward imputation via two Imputers.

The LOCFImputer carries the last observation forward (locf) to impute missing
data in a time series. NOCBImputer carries the next observation backward (nocb)
to impute missing data in a time series. Both methods are univariate. Right
now, these imputers support imputation on Series only. Use
TimeSeriesImputer(strategy="locf") or TimeSeries(strategy="nocb") to broadcast
forward or backward fill across multiple columns of a DataFrame.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=unnecessary-pass

class LOCFImputer(BaseEstimator, TransformerMixin):
    """Techniques to carry last observation forward to impute missing data.

    More complex autoimpute Imputers delegate work to the LOCFImputer if locf
    is a specified strategy for a given Series. That being said, LOCFImputer
    is a stand-alone class and valid sklearn transformer. It can be used
    directly, but such behavior is discouraged because this imputer
    supports Series only. LOCFImputer does not have the flexibility or
    robustness of more complex imputers, nor is its behavior identical.
    Instead, use TimeSeriesImputer(strategy="locf").
    """
    # class variables
    strategy = methods.LOCF

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
        # check if fitted then impute with mean if first value
        # or impute with observation carried forward otherwise
        check_is_fitted(self, "statistics_")
        imp = self.statistics_["param"]
        first = X.index[0]
        if pd.isnull(first):
            X.loc[first] = imp
        X.fillna(method="ffill", inplace=True)
        return X

class NOCBImputer(BaseEstimator, TransformerMixin):
    """Techniques to carry next observation backward to impute missing data.

    More complex autoimpute Imputers delegate work to the NOCBImputer if nocb
    is a specified strategy for a given Series. That being said, NOCBImputer
    is a stand-alone class and valid sklearn transformer. It can be used
    directly, but such behavior is discouraged because this imputer
    supports Series only. NOCBImputer does not have the flexibility or
    robustness of more complex imputers, nor is its behavior identical.
    Instead, use TimeSeriesImputer(strategy="nocb").
    """
    # class variables
    strategy = methods.NOCB

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
        # check if fitted then impute with mean if first value
        # or impute with observation carried backward otherwise
        check_is_fitted(self, "statistics_")
        imp = self.statistics_["param"]
        last = X.index[-1]
        if pd.isnull(last):
            X.loc[last] = imp
        X.fillna(method="bfill", inplace=True)
        return X
