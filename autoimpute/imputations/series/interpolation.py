"""This module implements interpolation methods via the InterpolateImputer.

InterpolateImputer imputes missing data using some interpolation strategies
suppoted by pd.Series.interpolate. Linear is the default strategy, although a
number of additional strategies exist. Dataframe imputers utilize this class
when its strategy is requested. Use SingleImputer or MultipleImputer with
strategy = `interpolate` to broadcast the strategy across all the columns in a
dataframe, or specify this strategy for a given column.
"""

import pandas as pd
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
from .base import ISeriesImputer
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=unnecessary-pass
# pylint:disable=unused-argument

class InterpolateImputer(ISeriesImputer):
    """Impute missing values using interpolation techniques.

    The InterpolateImputer imputes missing values uses a valid pd.Series
    interpolation strategy. See __init__ method docs for supported strategies.
    The imputer can be used directly, but such behavior is discouraged.
    InterpolateImputer does not have the flexibility / robustness of dataframe
    imputers, nor is its behavior identical. Preferred use is
    MultipleImputer(strategy="interpolate").
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
                Default is linear. Other strategies supported include:
                `time`, `quadratic`, `cubic`, `spline`, `barycentric`,
                `polynomial`.
            start (int, Optional): value to impute if first number in
                Series is missing. Default is None, but first valid used
                when required for quadratic, cubic, polynomial.
            end (int, Optional): value to impute if last number in
                Series is missing. Default is None, but last valid used
                when required for quadratic, cubic, polynomial.
            order (int, Optional): if strategy is spline or polynomial,
                order must be number. Otherwise not considered.

        Returns:
            self. Instance of the class.
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

    def fit(self, X, y=None):
        """Fit the Imputer to the dataset. Nothing to calculate.

        Args:
            X (pd.Series): Dataset to fit the imputer.
            y (None): ignored, None to meet requirements of base class

        Returns:
            self. Instance of the class.
        """
        self.statistics_ = {"param": self.fill_strategy,
                            "strategy": self.strategy}
        return self

    def impute(self, X):
        """Perform imputations using the statistics generated from fit.

        The impute method handles the actual imputation. Missing values
        in a given dataset are replaced with results from interpolation.

        Args:
            X (pd.Series): Dataset to impute missing data from fit.

        Returns:
            np.array -- imputed dataset.
        """
        # check if fitted then impute with interpolation strategy
        check_is_fitted(self, "statistics_")
        imp = self.statistics_["param"]

        # setting defaults if no value passed for start and last
        # quadratic, cubic, and polynomial require first and last
        if imp in ("quadratic", "cubic", "polynomial"):
            # handle start and end...
            if pd.isnull(X.iloc[0]):
                ix = X.head(1).index[0]
                X.fillna(
                    {ix: self._handle_start(self.start, X)}, inplace=True
                )
            if pd.isnull(X.iloc[-1]):
                ix = X.tail(1).index[0]
                X.fillna(
                    {ix: self._handle_end(self.end, X)}, inplace=True
                )

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

    def fit_impute(self, X, y=None):
        """Convenience method to perform fit and imputation in one go."""
        return self.fit(X, y).impute(X)
