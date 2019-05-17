"""This module implements least squares and stochastic imputation.

This module contains the LeastSquaresImputer and the StochasticImputer. Both
use least squares to find a line of best fit and fill imputations with the
predictions from the line. Stochastic adds random error to each prediction.
Dataframe imputers utilize this class when its strategy is requested. Use
SingleImputer or MultipleImputer with strategy = `least squares` to broadcast
the strategy across all the columns in a dataframe, or specify this strategy
for a given column.
"""

from numpy import sqrt
from scipy.stats import norm
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from autoimpute.imputations import method_names
from autoimpute.imputations.errors import _not_num_series
from .base import ISeriesImputer
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:

class LeastSquaresImputer(ISeriesImputer):
    """Impute missing values using predictions from least squares regression.

    The LeastSquaresImputer produces predictions using the least squares
    methodology. The prediction from the line of best fit given a set of
    predictors become the imputations. To implement least squares, the imputer
    wraps the sklearn LinearRegression class. The imputer can be used
    directly, but such behavior is discouraged. LeastSquaresImputer does not
    have the flexibility / robustness of dataframe imputers, nor is its
    behavior identical. Preferred use is
    MultipleImputer(strategy="least squares").
    """
    # class variables
    strategy = methods.LS

    def __init__(self, **kwargs):
        """Create an instance of the LeastSquaresImputer class.

        Args:
            **kwargs: keyword arguments passed to LinearRegression

        """
        self.lm = LinearRegression(**kwargs)

    def fit(self, X, y):
        """Fit the Imputer to the dataset by fitting linear model.

        Args:
            X (pd.Dataframe): dataset to fit the imputer.
            y (pd.Series): response, which is eventually imputed.

        Returns:
            self. Instance of the class.
        """
        _not_num_series(self.strategy, y)
        self.lm.fit(X, y)
        self.statistics_ = {"strategy": self.strategy}
        return self

    def impute(self, X):
        """Generate imputations using predictions from the fit linear model.

        The impute method returns the values for imputation. Missing values
        in a given dataset are replaced with the predictions from the least
        squares regression line of best fit. This transform method returns
        those predictions.

        Args:
            X (pd.DataFrame): predictors to determine imputed values.

        Returns:
            np.array: imputed dataset.
        """
        # check if fitted then predict with least squares
        check_is_fitted(self, "statistics_")
        imp = self.lm.predict(X)
        return imp

    def fit_impute(self, X, y):
        """Fit impute method to generate imputations where y is missing.

        Args:
            X (pd.Dataframe): predictors in the dataset.
            y (pd.Series): response w/ missing values to impute.

        Returns:
            np.array: imputed dataset.
        """
        # transform occurs with records from X where y is missing
        miss_y_ix = y[y.isnull()].index
        return self.fit(X, y).impute(X.loc[miss_y_ix])

class StochasticImputer(ISeriesImputer):
    """Impute missing values adding error to least squares regression preds.

    The StochasticImputer predicts using the least squares methodology. The
    imputer then samples from the regression's error distribution and adds the
    random draw to the prediction. This draw adds the stochastic element to
    the imputations. The imputer can be used directly, but such behavior is
    discouraged. StochasticImputer does not have the flexibility / robustness
    of dataframe imputers, nor is its behavior identical. Preferred use is
    MultipleImputer(strategy="stochastic").
    """
    # class variables
    strategy = methods.STOCHASTIC

    def __init__(self, **kwargs):
        """Create an instance of the StochasticImputer class.

        Args:
            **kwargs: keyword arguments passed to LinearRegression.

        """
        self.lm = LinearRegression(**kwargs)

    def fit(self, X, y):
        """Fit the Imputer to the dataset by fitting linear model.

        The fit step also generates predictions on the observed data. These
        predictions are necessary to derive the mean_squared_error, which is
        passed as a parameter to the impute phase. The MSE is used to create
        the normal error distribution from which the imptuer draws.

        Args:
            X (pd.Dataframe): dataset to fit the imputer.
            y (pd.Series): response, which is eventually imputed.

        Returns:
            self. Instance of the class.
        """
        _not_num_series(self.strategy, y)
        self.lm.fit(X, y)
        preds = self.lm.predict(X)
        mse = mean_squared_error(y, preds)
        self.statistics_ = {"param": mse, "strategy": self.strategy}
        return self

    def impute(self, X):
        """Generate imputations using predictions from the fit linear model.

        The impute method returns the values for imputation. Missing values
        in a given dataset are replaced with the predictions from the least
        squares regression line of best fit plus a random draw from the normal
        error distribution.

        Args:
            X (pd.DataFrame): predictors to determine imputed values.

        Returns:
            np.array: imputed dataset.
        """
        # check if fitted then predict with least squares
        check_is_fitted(self, "statistics_")
        mse = self.statistics_["param"]
        preds = self.lm.predict(X)

        # add random draw from normal dist w/ mean squared error
        # from observed model. This makes lm stochastic
        mse_dist = norm.rvs(loc=0, scale=sqrt(mse), size=len(preds))
        imp = preds + mse_dist
        return imp

    def fit_impute(self, X, y):
        """Fit impute method to generate imputations where y is missing.

        Args:
            X (pd.Dataframe): predictors in the dataset.
            y (pd.Series): response w/ missing values to impute

        Returns:
            np.array: imputated dataset.
        """
        # transform occurs with records from X where y is missing
        miss_y_ix = y[y.isnull()].index
        return self.fit(X, y).impute(X.loc[miss_y_ix])
