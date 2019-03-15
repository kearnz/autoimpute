"""This module uses available information in a dataset to predict imputations.

This module contains one class - the PredictiveImputer. Use this class to
predict imputations for each Series within a DataFrame using all or a subset
of the other available features. This class extends the behavior of the
SingleImputer. Unlike the SingleImputer, the supported methods in this class
are multivariate - they use more than just the series itself to determine the
best estimated values for imputaiton.

Todo:
    * class specification
    * outline strategies planned for implementation
    * create multivariate methods module with predictive strategies
"""

from sklearn.base import BaseEstimator, TransformerMixin
from autoimpute.utils import check_nan_columns
from autoimpute.imputations import BaseImputer, predictive_methods
pm = predictive_methods
# pylint:disable=attribute-defined-outside-init
# pylint:disable=arguments-differ
# pylint:disable=protected-access
# pylint:disable=too-many-arguments, too-many-locals

class PredictiveImputer(BaseImputer, BaseEstimator, TransformerMixin):
    """Techniques to impute Series with missing values through learning."""

    strategies = {
        "linear": pm._fit_linear_reg,
        "binary logistic": pm._fit_binary_logistic,
        "multinomial logistic": pm._fit_multi_logistic,
        "default": pm._predictive_default
    }

    def __init__(self, strategy="default", predictors="all",
                 fill_predictors=False, scaler=None, verbose=None):
        """Create an instance of the PredictiveImputer class."""
        BaseImputer.__init__(
            self,
            scaler=scaler,
            verbose=verbose
        )
        self.strategy = strategy
        self.predictors = predictors
        self.fill_predictors = fill_predictors

    @property
    def strategy(self):
        """Property getter to return the value of the strategy property"""
        return self._strategy

    @strategy.setter
    def strategy(self, s):
        """Validate the strategy property to ensure it's Type and Value.

        Class instance only possible if strategy is proper type, as outlined
        in the init method. Passes supported strategies and user arg to
        helper method, which performs strategy checks.

        Args:
            s (str, iter, dict): Strategy passed as arg to class instance.

        Raises:
            ValueError: Strategies not valid (not in allowed strategies).
            TypeError: Strategy must be a string, tuple, list, or dict.
            Both errors raised through helper method `check_strategy_allowed`.
        """
        strat_names = self.strategies.keys()
        self._strategy = self.check_strategy_allowed(strat_names, s)

    def _fit_strategy_validator(self, X):
        """Internal helper method to validate strategies appropriate for fit.

        Checks whether strategies match with type of column they are applied
        to. If not, error is raised through `check_strategy_fit` method.
        """
        # remove nan columns and store colnames
        cols = X.columns.tolist()
        self._strats = self.check_strategy_fit(self.strategy, cols)
        self._preds = self.check_predictors_fit(self.predictors, cols)
        # next, prep the categorical / numerical split
        self._prep_fit_dataframe(X)
        return X

    @check_nan_columns
    def fit(self, X, **kwargs):
        """Fit placeholder."""
        # first, prep columns we plan to use and make sure they are valid
        self._fit_strategy_validator(X)
        self.statistics_ = {}

        # header print statement if verbose = true
        if self.verbose:
            st = "Strategies & Predictors used to fit each column:"
            print(f"{st}\n{'-'*len(st)}")

        # perform fit on each column, depending on that column's strategy
        # note - because we use predictors, logic more involved than single
        for col_name, func_name in self._strats.items():
            f = self.strategies[func_name]
            x, _ = self._prep_pred_cols(col_name, self._preds)
            if x.ndim == 1:
                x = x.values.reshape().reshape(-1, 1)
            y = X[col_name]
            fit_param, fit_name = f(x, y, self.verbose, **kwargs)
            self.statistics_[col_name] = {"param": fit_param,
                                          "strategy": fit_name}
            # print strategies if verbose
            if self.verbose:
                resp = f"Response: {col_name}"
                preds = f"Predictors: {self._preds[col_name]}"
                strat = f"Strategy {fit_name}"
                print(f"{resp}\n{preds}\n{strat}\n{'-'*len(st)}")
        return self
