"""This module implements least squares imputation via the LeastSquaresImputer.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LinearRegression
from autoimpute.imputations import method_names
from autoimpute.imputations.errors import _not_num_series
from autoimpute.utils.helpers import _get_observed
methods = method_names
# pylint:disable=attribute-defined-outside-init

class LeastSquaresImputer(BaseEstimator, TransformerMixin):
    """Techniques to impute the best fit prediction for missing values.
    """
    # class variables
    strategy = methods.LS

    def __init__(self, verbose, **kwargs):
        """Create an instance of the LeastSquaresImputer class."""
        self.verbose = verbose
        self.lm = LinearRegression(**kwargs)

    def fit(self, X, y):
        """Fit the Imputer to the dataset by fitting linear model.

        Args:
            X (pd.Series): Dataset to fit the imputer

        Returns:
            self. Instance of the class.
        """
        _not_num_series(self.strategy, y)
        X_, y_ = _get_observed(
            self.strategy, X, y, self.verbose
        )
        self.lm.fit(X_, y_)
        self.statistics_ = {"strategy": self.strategy}
        return self

    def transform(self, X, new_preds):
        """Perform imputations using the statistics generated from fit.

        The transform method handles the actual imputation. Missing values
        in a given dataset are replaced with the predictions from the least
        squares regression line of best fit.

        Args:
            X (pd.Series): Dataset to fit the imputer

        Returns:
            pd.Series -- imputed dataset
        """
        # check if fitted then predict with least squares
        check_is_fitted(self, "statistics_")
        _not_num_series(self.strategy, X)
        ind = X[X.isnull()].index
        predictions = self.lm.predict(new_preds)
        imp = pd.Series(predictions, index=ind)
        X.fillna(imp, inplace=True)
        return X
