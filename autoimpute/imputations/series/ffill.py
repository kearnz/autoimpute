"""This module implements forward & backward imputation via two Imputers.

The LOCFImputer carries the last observation forward (locf) to impute missing
data in a time series. NOCBImputer carries the next observation backward (nocb)
to impute missing data in a time series. Both methods are univariate. Right
now, these imputers support imputation on Series only. Use
TimeSeriesImputer(strategy="locf") or TimeSeriesImputer(strategy="nocb") to
broadcast forward or backward fill across multiple columns of a DataFrame.
"""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=unnecessary-pass
# pylint:disable=unused-argument

class LOCFImputer(BaseEstimator):
    """Impute missing values by carrying the last observation forward.

    LOCFImputer carries the last observation forward to impute missing data.
    The imputer can be used directly, but such behavior is discouraged because
    the imputer supports Series only. LOCFImputer does not have the
    flexibility or robustness of more complex imputers, nor is its behavior
    identical. Instead, use TimeSeriesImputer(strategy="locf").
    """
    # class variables
    strategy = methods.LOCF

    def __init__(self, start=None):
        """Create an instance of the LOCFImputer class.

        Args:
            start (any, optional): can be any value to impute first if first
                is missing. Default is None, which ends up taking first
                observed value found. Can also use "mean" to start with mean
                of the series.

        Returns:
            self. Instance of class.
        """
        self.start = start

    def _handle_start(self, v, X):
        "private method to handle start values."
        if v is None:
            v = X.loc[X.first_valid_index()]
        if v == "mean":
            v = X.mean()
        return v

    def fit(self, X):
        """Fit the Imputer to the dataset.

        Args:
            X (pd.Series): Dataset to fit the imputer.

        Returns:
            self. Instance of the class.
        """
        self.statistics_ = {"param": None, "strategy": self.strategy}
        return self

    def impute(self, X):
        """Perform imputations using the statistics generated from fit.

        The impute method handles the actual imputation. Missing values
        in a given dataset are replaced with the last observation carried
        forward.

        Args:
            X (pd.Series): Dataset to impute missing data from fit.

        Returns:
            np.array -- imputed dataset.
        """
        # check if fitted then impute with mean if first value
        # or impute with observation carried forward otherwise
        check_is_fitted(self, "statistics_")

        # handle start...
        if pd.isnull(X.iloc[0]):
            X.iloc[0] = self._handle_start(self.start, X)
        return X.fillna(method="ffill", inplace=False).values

    def fit_impute(self, X):
        """Convenience method to perform fit and imputation in one go."""
        return self.fit(X).impute(X)

class NOCBImputer(BaseEstimator):
    """Impute missing data by carrying the next observation backward.

    NOCBImputer carries the next observation backward to impute missing data.
    The imputer can be used directly, but such behavior is discouraged because
    the imputer supports Series only. NOCBImputer does not have the
    flexibility or robustness of more complex imputers, nor is its behavior
    identical. Instead, use TimeSeriesImputer(strategy="nocb").
    """
    # class variables
    strategy = methods.NOCB

    def __init__(self, end=None):
        """Create an instance of the NOCBImputer class.

        Args:
            end (any, optional): can be any value to impute end if end
                is missing. Default is None, which ends up taking last
                observed value found. Can also use "mean" to end with
                mean of the series.

        Returns:
            self. Instance of class.
        """
        self.end = end

    def _handle_end(self, v, X):
        "private method to handle end values."
        if v is None:
            v = X.loc[X.last_valid_index()]
        if v == "mean":
            v = X.mean()
        return v

    def fit(self, X):
        """Fit the Imputer to the dataset and calculate the mean.

        Args:
            X (pd.Series): Dataset to fit the imputer

        Returns:
            self. Instance of the class.
        """
        self.statistics_ = {"param": None, "strategy": self.strategy}
        return self

    def impute(self, X):
        """Perform imputations using the statistics generated from fit.

        The impute method handles the actual imputation. Missing values
        in a given dataset are replaced with the next observation carried
        backward.

        Args:
            X (pd.Series): Dataset to impute missing data from fit.

        Returns:
            np.array -- imputed dataset.
        """
        # check if fitted then impute with mean if first value
        # or impute with observation carried backward otherwise
        check_is_fitted(self, "statistics_")

        # handle end...
        if pd.isnull(X.iloc[-1]):
            X.iloc[-1] = self._handle_end(self.end, X)
        return X.fillna(method="ffill", inplace=False).values

    def fit_impute(self, X):
        """Convenience method to perform fit and imputation in one go."""
        return self.fit(X).impute(X)
