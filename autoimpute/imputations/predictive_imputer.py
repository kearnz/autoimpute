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
from sklearn.utils.validation import check_is_fitted
from autoimpute.utils import check_nan_columns
from autoimpute.imputations import BaseImputer, SingleImputer
from autoimpute.imputations import predictive_methods
pm = predictive_methods
# pylint:disable=attribute-defined-outside-init
# pylint:disable=arguments-differ
# pylint:disable=protected-access
# pylint:disable=too-many-arguments, too-many-locals

class PredictiveImputer(BaseImputer, BaseEstimator, TransformerMixin):
    """Techniques to impute Series with missing values through learning."""

    strategies = {
        "least squares": pm._fit_least_squares_reg,
        "binary logistic": pm._fit_binary_logistic_reg,
        "multinomial logistic": pm._fit_multi_logistic_reg,
        "stochastic": pm._fit_stochastic_reg,
        "default": pm._predictive_default
    }

    def __init__(self, strategy="default", strategy_args=None,
                 predictors="all", scaler=None, verbose=None, copy=True):
        """Create an instance of the PredictiveImputer class."""
        BaseImputer.__init__(
            self,
            scaler=scaler,
            verbose=verbose
        )
        self.strategy = strategy
        self.strategy_args = strategy_args
        self.predictors = predictors
        self.copy = copy

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
    def fit(self, X):
        """Fit placeholder."""
        # first, prep columns we plan to use and make sure they are valid
        self._fit_strategy_validator(X)
        self.statistics_ = {}
        if not self.scaler is None:
            self._scaler_fit()

        # header print statement if verbose = true
        if self.verbose:
            st = "Strategies & Predictors used to fit each column:"
            print(f"{st}\n{'-'*len(st)}")

        # perform fit on each column, depending on that column's strategy
        # note - because we use predictors, logic more involved than single
        for col_name, func_name in self._strats.items():
            f = self.strategies[func_name]
            x, _ = self._prep_predictor_cols(col_name, self._preds)
            y = X[col_name]
            fit_param, fit_name = f(x, y, self.verbose)
            self.statistics_[col_name] = {"param": fit_param,
                                          "strategy": fit_name}
            # print strategies if verbose
            if self.verbose:
                resp = f"Response: {col_name}"
                preds = f"Predictors: {self._preds[col_name]}"
                strat = f"Strategy {fit_name}"
                print(f"{resp}\n{preds}\n{strat}\n{'-'*len(st)}")
        return self

    @check_nan_columns
    def transform(self, X):
        """Transform placeholder."""
        # initial checks before transformation
        check_is_fitted(self, "statistics_")
        if self.copy:
            X = X.copy()

        # check columns
        X_cols = X.columns.tolist()
        fit_cols = set(self._strats.keys())
        diff_fit = set(fit_cols).difference(X_cols)
        if diff_fit:
            err = "Same columns that were fit must appear in transform."
            raise ValueError(err)

        # transformation logic
        self.imputed_ = {}
        for col_name, fit_data in self.statistics_.items():
            strat = fit_data["strategy"]
            fill = fit_data["param"]
            imp_ix = X[col_name][X[col_name].isnull()].index
            self.imputed_[col_name] = imp_ix.tolist()
            if self.verbose:
                print(f"Transforming {col_name} with strategy '{strat}'")
                print(f"Numer of imputations to perform: {len(imp_ix)}")
            # continue if there are no imputations to make
            if imp_ix.empty:
                continue
            x, _ = self._prep_predictor_cols(col_name, self._preds)
            x = x.loc[imp_ix, :]
            # may abstract SingleImputer in future for flexibility
            x = SingleImputer().fit_transform(x)
            # fill missing values based on the method selected
            # note that default picks a method below depending on col
            # -------------------------------------------------------
            # linear regression imputation
            if strat == "least squares":
                pm._imp_least_squares_reg(X, col_name, x, fill, imp_ix)
            if strat in ("binary logistic", "multinomial logistic"):
                pm._imp_logistic_reg(X, col_name, x, fill, imp_ix)
            if strat == "stochastic":
                pm._imp_stochastic_reg(X, col_name, x, fill, imp_ix)
            # no imputation if strategy is none
            if strat == "none":
                pass
        return X
