"""This module implements interpolation methods via the InterpolateImputer.

InterpolateImputer imputes missing data using some interpolation strategies
suppoted by pd.Series.interpolate. Linear is the default strategy, although a
number of additional strategies exist. Interpolation is transductive, so the
fit method simply returns the interpolation method but no fit statistic. All
interpolation is performed in transform. Right now, this imputer
supports imputation on Series only. Use TimeSeriesImputer with specified
strategy to broadcast interpolation across multiple columns of a DataFrame.
Note that most interpolation strategies are valid for SingleImputer as well.
"""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=unnecessary-pass
# pylint:disable=unused-argument

class InterpolateImputer(BaseEstimator):
    """Techniques to impute missing values using interpolation techniques.

    More complex autoimpute Imputers delegate work to the InterpolateImputer
    if an intepolation is a specified strategy for a given Series. That being
    said, InterpolateImputer is a stand-alone class and valid sklearn
    transformer. It can be used directly, but such behavior is discouraged
    because this imputer supports Series only. InterpolateImputer does not
    have the flexibility or robustness of more complex imputers, nor is its
    behavior identical. Instead, use TimeSeriesImputer or SingleImputer
    depending on use case.
    """
    # class variables
    strategy = methods.INTERPOLATE
    fill_strategies = (
        "linear", "time", "quadratic", "cubic",
        "spline", "barycentric", "polynomial"
    )

    def __init__(self, fill_strategy="linear",
                 start=None, end=None, order=None):
        """Create an instance of the InterpolateImputer class.

        Args:
            fill_strategy (str, Optional): type of interpolation to perform
                Default is linear. check fill_strategies are supported.
            start (int, Optional): value to impute if first number in
                Series is missing. Default is None, but first valid used
                when required for quadratic, cubic, polynomial
            end (int, Optional): value to impute if last number in
                Series is missing. Default is None, but last valid used
                when required for quadratic, cubic, polynomial
            order (int, Optional): if strategy is spline or polynomial,
                order must be number. Otherwise not considered.

        Returns:
            self. Instance of the class
        """
        self.fill_strategy = fill_strategy
        self.start = start
        self.end = end
        self.order = order

    @property
    def fill_strategy(self):
        """Property getter to return the value of fill_strategy property."""
        return self._fill_strategy

    @fill_strategy.setter
    def fill_strategy(self, fs):
        """Validate the fill_strategy property and set default parameters.

        Args:
            fs (str, Optional): if None, use linear.

        Raises:
            ValueError: not a valid fill strategy for InterpolateImputer
        """
        if fs not in self.fill_strategies:
            err = f"{fs} not a valid fill strategy for InterpolateImputer"
            raise ValueError(err)
        self._fill_strategy = fs

    def _handle_start(self, v, X):
        "private method to handle start values."
        if v is None:
            v = X.loc[X.first_valid_index()]
        if v == "mean":
            v = X.mean()
        return v

    def _handle_end(self, v, X):
        "private method to handle end values."
        if v is None:
            v = X.loc[X.last_valid_index()]
        if v == "mean":
            v = X.mean()
        return v

    def fit(self, X):
        """Fit the Imputer to the dataset. Nothing to calculate.

        Args:
            X (pd.Series): Dataset to fit the imputer

        Returns:
            self. Instance of the class.
        """
        self.statistics_ = {"param": self.fill_strategy,
                            "strategy": self.strategy}
        return self

    def impute(self, X):
        """Perform imputations using the statistics generated from fit.

        The transform method handles the actual imputation. Missing values
        in a given dataset are replaced with results from interpolation.

        Args:
            X (pd.Series): Dataset to fit the imputer

        Returns:
            pd.Series -- imputed dataset
        """
        # check if fitted then impute with interpolation strategy
        check_is_fitted(self, "statistics_")
        imp = self.statistics_["param"]

        # setting defaults if no value passed for start and last
        # quadratic, cubic, and polynomial require first and last
        if imp in ("quadratic", "cubic", "polynomial"):
            # handle start and end...
            if pd.isnull(X.iloc[0]):
                X.iloc[0] = self._handle_start(self.start, X)
            if pd.isnull(X.iloc[-1]):
                X.iloc[-1] = self._handle_end(self.end, X)

        # handling for methods that need order
        num_observed = min(6, X.count())
        if imp in ("polynomial", "spline"):
            if self.order is None or self.order >= num_observed:
                err = f"Order must be between 1 and {num_observed-1}"
                raise ValueError(err)

        # finally, perform interpolation
        return X.interpolate(method=imp,
                             limit=None,
                             limit_direction="both",
                             inplace=False,
                             order=self.order)

    def fit_impute(self, X):
        """Helper method to perform fit and imputation in one go."""
        return self.fit(X).impute(X)
