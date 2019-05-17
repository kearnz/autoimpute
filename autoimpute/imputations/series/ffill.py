"""This module implements forward & backward imputation via two Imputers.

The LOCFImputer carries the last observation forward (locf) to impute missing
data in a time series. NOCBImputer carries the next observation backward (nocb)
to impute missing data in a time series. Dataframe imputers utilize these
classes when each's strategy is requested. Use SingleImputer or MultipleImputer
with strategy = `locf` or `nocb` to broadcast either strategy across all the
columns in a dataframe, or specify either strategy for a given column.
"""

import pandas as pd
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
from .base import ISeriesImputer
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=unnecessary-pass
# pylint:disable=unused-argument

class LOCFImputer(ISeriesImputer):
    """Impute missing values by carrying the last observation forward.

    LOCFImputer carries the last observation forward to impute missing data.
    The imputer can be used directly, but such behavior is discouraged.
    LOCFImputer does not have the flexibility / robustness of dataframe
    imputers, nor is its behavior identical. Preferred use is
    MultipleImputer(strategy="locf").
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

    def fit(self, X, y=None):
        """Fit the Imputer to the dataset.

        Args:
            X (pd.Series): Dataset to fit the imputer.
            y (None): ignored, None to meet requirements of base class


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
            ix = X.head(1).index[0]
            X.fillna(
                {ix: self._handle_start(self.start, X)}, inplace=True
            )
        return X.fillna(method="ffill", inplace=False)

    def fit_impute(self, X, y=None):
        """Convenience method to perform fit and imputation in one go."""
        return self.fit(X, y).impute(X)

class NOCBImputer(ISeriesImputer):
    """Impute missing data by carrying the next observation backward.

    NOCBImputer carries the next observation backward to impute missing data.
    The imputer can be used directly, but such behavior is discouraged.
    NOCBImputer does not have the flexibility / robustness of dataframe
    imputers, nor is its behavior identical. Preferred use is
    MultipleImputer(strategy="nocb").
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

    def fit(self, X, y=None):
        """Fit the Imputer to the dataset and calculate the mean.

        Args:
            X (pd.Series): Dataset to fit the imputer
            y (None): ignored, None to meet requirements of base class


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
            ix = X.tail(1).index[0]
            X.fillna(
                {ix: self._handle_end(self.end, X)}, inplace=True
            )
        return X.fillna(method="bfill", inplace=False)

    def fit_impute(self, X, y=None):
        """Convenience method to perform fit and imputation in one go."""
        return self.fit(X, y).impute(X)
