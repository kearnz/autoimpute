"""This module implements least squares imputation."""

from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LinearRegression
from autoimpute.imputations import method_names
from autoimpute.imputations.errors import _not_num_series
from autoimpute.utils.helpers import _get_observed
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:

class LeastSquaresImputer:
    """Class to impute missing values using least squares regression.

    The LeastSquaresImputer produces imputations using the least squares
    methodology. The PredictiveImputer delegates work to this class when
    the specified imputation strategy is 'least squares'. To implement least
    squares, the imputer wraps the sklearn LinearRegression class. The
    LeastSquaresImputer is a stand-alone class that will generate imputations
    for missing values in a series, but its direct use is discouraged.
    Instead, use PredictiveImputer with strategy = 'least squares'. The
    Predictive imputer performs a number of important checks and error
    handling procedures to ensure the data the LeastSquaresImputer
    receives is formatted correctly for proper imputation.
    """
    # class variables
    strategy = methods.LS

    def __init__(self, verbose, **kwargs):
        """Create an instance of the LeastSquaresImputer class.

        Args:
            verbose (bool): print information to the console
            **kwargs: keyword arguments passed to LinearRegression

        """
        self.verbose = verbose
        self.lm = LinearRegression(**kwargs)

    def fit(self, X, y):
        """Fit the Imputer to the dataset by fitting linear model.

        Args:
            X (pd.Dataframe): dataset to fit the imputer
            y (pd.Series): response, which is eventually imputed

        Returns:
            self. Instance of the class.
        """
        # linear model fit on observed values only
        _not_num_series(self.strategy, y)
        X_, y_ = _get_observed(
            self.strategy, X, y, self.verbose
        )
        self.lm.fit(X_, y_)
        self.statistics_ = {"strategy": self.strategy}
        return self

    def impute(self, X):
        """Generate imputations using predictions from the fit linear model.

        The transform method returns the values for imputation. Missing values
        in a given dataset are replaced with the predictions from the least
        squares regression line of best fit. This transform method returns
        those predictions.

        Args:
            X (pd.DataFrame): predictors to determine imputed values

        Returns:
            np.array: imputations from transformation
        """
        # check if fitted then predict with least squares
        check_is_fitted(self, "statistics_")
        imp = self.lm.predict(X)
        return imp

    def fit_impute(self, X, y):
        """Fit transform method to generate imputations where y is missing.

        Args:
            X (pd.Dataframe): predictors in the dataset.
            y (pd.Series): response w/ missing values to impute

        Returns:
            np.array: imputations from transformation
        """
        # transform occurs with records from X where y is missing
        miss_y_ix = y[y.isnull()].index
        return self.fit(X, y).impute(X.loc[miss_y_ix])
