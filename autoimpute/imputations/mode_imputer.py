"""This module implements mode imputation via the ModeImputer.

The ModeImputer computes the mode of observed values then imputes missing
data with the computed mode. Mode imputation is univariate. Right now,
this imputer supports imputation on Series only. Use
SingleImputer(strategy="mode") to broadcast the imputation strategy across
multiple columns of a DataFrame.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
methods = method_names
# pylint:disable=attribute-defined-outside-init

class ModeImputer(BaseEstimator, TransformerMixin):
    """Techniques to impute the mode for missing values within a dataset.

    More complex autoimpute Imputers delegate work to the ModeImputer if
    mode is a specified strategy for a given Series. That being said,
    ModeImputer is a stand-alone class and valid sklearn transformer. It can
    be used directly, but such behavior is discouraged because this imputer
    supports Series only. ModeImputer does not have the flexibility or
    robustness of more complex imputers, nor is its behavior identical.
    Instead, use SingleImputer(strategy="mode").
    """
    # class variables
    strategy = methods.MODE
    fill_strategies = (None, "first", "last", "random")

    def __init__(self, fill_strategy=None):
        """Create an instance of the ModeImputer class.

        Args:
            fill_strategy (str, Optional): strategy to pick mode, if multiple.
                Default is None, which means first mode taken.
        """
        self.fill_strategy = fill_strategy

    @property
    def fill_strategy(self):
        """Property getter to return the value of fill_strategy property."""
        return self._fill_strategy

    @fill_strategy.setter
    def fill_strategy(self, fs):
        """Validate the scaler property and set default parameters.

        Args:
            s (scaler): if None, implement the xgboost classifier

        Raises:
            ValueError: classifier does not implement `fit_transform`
        """
        if fs not in self.fill_strategies:
            err = f"{fs} not a valid fill strategy for ModeImputer"
            raise ValueError(err)
        self._fill_strategy = fs

    def fit(self, X):
        """Fit the Imputer to the dataset and calculate the mode.

        Args:
            X (pd.Series): Dataset to fit the imputer

        Returns:
            self. Instance of the class.
        """
        mode = X.mode().values
        self.statistics_ = {"param": mode, "strategy": self.strategy}
        return self

    def transform(self, X):
        """Perform imputations using the statistics generated from fit.

        The transform method handles the actual imputation. Missing values
        in a given dataset are replaced with the mode observed from fit.
        Note that there can be more than one mode. If this is the case, there
        are two possibilities based on the "fill_strategy" parameter. If
        fill_strategy=None, use the first mode. This is the default. If
        fill_strategy="random", randomly sample from the modes and impute.

        Args:
            X (pd.Series): Dataset to fit the imputer

        Returns:
            pd.Series -- imputed dataset

        Raises:
            ValueError: fill_strategy not valid. Must be None or random
        """
        # check is fitted and identify locations of missingness
        check_is_fitted(self, "statistics_")
        ind = X[X.isnull()].index

        # get the number of modes
        imp = self.statistics_["param"]

        # default imputation is to pick first, such as scipy does
        if self.fill_strategy is None:
            fills = imp[0]

        # picking the first of the modes when fill_strategy = first
        if self.fill_strategy == "first":
            fills = imp[0]

        # picking the last of the modes when fill_strategy = last
        if self.fill_strategy == "last":
            fills = imp[-1]

        # sampling when strategy is random
        if self.fill_strategy == "random":
            num_modes = len(imp)
            # check if more modes
            if num_modes == 1:
                fills = imp[0]
            else:
                samples = np.random.choice(imp, len(ind))
                fills = pd.Series(samples, index=ind)

        # finally, fill in the right fill values for missing X
        X.fillna(fills, inplace=True)
        return X
