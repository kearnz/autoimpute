"""This module implements mode imputation via the ModeImputer.

The ModeImputer uses the mode of observed data to impute missing values.
Dataframe imputers utilize this class when its strategy is requested. Use
SingleImputer or MultipleImputer with strategy = `mode` to broadcast the
strategy across all the columns in a dataframe, or specify this strategy
for a given column.
"""

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
from .base import ISeriesImputer
methods = method_names
# pylint:disable=attribute-defined-outside-init

class ModeImputer(ISeriesImputer):
    """Impute missing values with the mode of the observed data.

    The mode imputer calculates the mode of the observed dataset and uses
    it to impute missing observations. In the case where there are more than
    one mode, the user can supply a `fill_strategy` to choose the mode.
    The imputer can be used directly, but such behavior is discouraged.
    ModeImputer does not have the flexibility / robustness of dataframe
    imputers, nor is its behavior identical. Preferred use is
    MultipleImputer(strategy="mode").
    """
    # class variables
    strategy = methods.MODE
    fill_strategies = (None, "first", "last", "random")

    def __init__(self, fill_strategy=None):
        """Create an instance of the ModeImputer class.

        Args:
            fill_strategy (str, Optional): strategy to pick mode, if multiple.
                Default is None, which means first mode taken.
                Options include None, first, last, random.
                First, None -> select first of modes.
                Last -> select the last of modes.
                Random -> randomly sample from modes with replacement.
        """
        self.fill_strategy = fill_strategy

    @property
    def fill_strategy(self):
        """Property getter to return the value of fill_strategy property."""
        return self._fill_strategy

    @fill_strategy.setter
    def fill_strategy(self, fs):
        """Validate the fill_strategy property and set default parameters.

        Args:
            fs (str, None): if None, use first mode.

        Raises:
            ValueError: not a valid fill strategy for ModeImputer.
        """
        if fs not in self.fill_strategies:
            err = f"{fs} not a valid fill strategy for ModeImputer"
            raise ValueError(err)
        self._fill_strategy = fs

    def fit(self, X, y=None):
        """Fit the Imputer to the dataset and calculate the mode.

        Args:
            X (pd.Series): Dataset to fit the imputer.
            y (None): ignored, None to meet requirements of base class

        Returns:
            self. Instance of the class.
        """
        mode = X.mode().values
        self.statistics_ = {"param": mode, "strategy": self.strategy}
        return self

    def impute(self, X):
        """Perform imputations using the statistics generated from fit.

        This method handles the actual imputation. Missing values in a given
        dataset are replaced with the mode observed from fit. Note that there
        can be more than one mode. If more than one mode, use the
        fill_strategy to determine how to use the modes.

        Args:
            X (pd.Series): Dataset to impute missing data from fit.

        Returns:
            float or np.array -- imputed dataset.
        """
        # check is fitted and identify locations of missingness
        check_is_fitted(self, "statistics_")
        ind = X[X.isnull()].index

        # get the number of modes
        imp = self.statistics_["param"]

        # default imputation is to pick first, such as scipy does
        if self.fill_strategy is None:
            imp = imp[0]

        # picking the first of the modes when fill_strategy = first
        if self.fill_strategy == "first":
            imp = imp[0]

        # picking the last of the modes when fill_strategy = last
        if self.fill_strategy == "last":
            imp = imp[-1]

        # sampling when strategy is random
        if self.fill_strategy == "random":
            num_modes = len(imp)
            # check if more modes
            if num_modes == 1:
                imp = imp[0]
            else:
                samples = np.random.choice(imp, len(ind))
                imp = pd.Series(samples, index=ind).values

        # finally, fill in the right fill values for missing X
        return imp

    def fit_impute(self, X, y=None):
        """Convenience method to perform fit and imputation in one go."""
        return self.fit(X, y).impute(X)
