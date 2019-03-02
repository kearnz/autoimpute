"""Single imputation lib"""

import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from autoimpute.utils.checks import check_missingness
from autoimpute.utils.checks import _check_strategy, _check_fit_strat
from autoimpute.utils.helpers import _nan_col_dropper, _mode_output
from autoimpute.imputations.methods import _mean, _median, _mode
from autoimpute.imputations.methods import _single_default, _random
# pylint:disable=attribute-defined-outside-init
# pylint:disable=arguments-differ

class SingleImputer(BaseEstimator, TransformerMixin):
    """Techniques to Impute missing values once"""

    strategies = {
        "mean": _mean,
        "median": _median,
        "mode":  _mode,
        "default": _single_default,
        "random": _random
    }

    def __init__(self, strategy="default", fill_value=None,
                 verbose=False, copy=True):
        self.strategy = strategy
        self.fill_value = fill_value
        self.verbose = verbose
        self.copy = copy

    @property
    def strategy(self):
        """return the strategy property"""
        return self._strategy

    @strategy.setter
    def strategy(self, s):
        """validate the strategy property"""
        strat_names = self.strategies.keys()
        self._strategy = _check_strategy(strat_names, s)

    def _fit_strategy_validator(self, X):
        """helper method to ensure right number of strategies"""
        # remove nan columns and store colnames
        ocols = X.columns.tolist()
        X, self._nc = _nan_col_dropper(X)
        ncols = X.columns.tolist()
        self._strats = _check_fit_strat(self.strategy, self._nc, ocols, ncols)
        # print strategies if verbose
        if self.verbose:
            st = "Strategies used to fit each column:"
            print(f"{st}\n{'-'*len(st)}")
            for k, v in self._strats.items():
                print(f"Column: {k}, Strategy: {v}")
        return X

    @check_missingness
    def fit(self, X):
        """Fit method for single imputer"""
        # copy, validate, and create statistics if validated
        if self.copy:
            X = X.copy()
        self._fit_strategy_validator(X)
        self.statistics_ = {}

        # perform fit on each column, depending on that column's strategy
        for col_name, func_name in self._strats.items():
            f = self.strategies[func_name]
            try:
                fit_param, fit_name = f(X[col_name])
            except TypeError as te:
                typ = X[col_name].dtype
                err = f"{func_name} not appropriate for column with type {typ}"
                raise TypeError(err) from te
            self.statistics_[col_name] = {"param":fit_param,
                                          "strategy": fit_name}
            if self.verbose:
                print(f"{col_name} has {func_name} equal to {fit_param}")
        return self

    @check_missingness
    def transform(self, X):
        """Transform method for a single imputer"""
        # initial checks before transformation
        check_is_fitted(self, 'statistics_')
        if self.copy:
            X = X.copy()
            if self._nc:
                wrn = f"Dropping {self._nc} since they were not fit."
                warnings.warn(wrn)
                try:
                    X.drop(self._nc, axis=1, inplace=True)
                except ValueError as ve:
                    err = "Same columns must appear in fit and transform."
                    raise ValueError(err) from ve

        # check columns
        X_cols = X.columns.tolist()
        fit_cols = set(self._strats.keys())
        diff_X = set(X_cols).difference(fit_cols)
        diff_fit = set(fit_cols).difference(X_cols)
        if diff_X or diff_fit:
            err = "Same columns must appear in fit and transform."
            raise ValueError(err)

        # transformation logic
        for col_name, fit_data in self.statistics_.items():
            strat = fit_data["strategy"]
            fill_val = fit_data["param"]
            if strat == "mode":
                _mode_output(X[col_name], fill_val, self.fill_value)
            elif strat == "random":
                ind = X[col_name][X[col_name].isnull()].index
                fills = np.random.choice(fill_val, len(ind))
                X.loc[ind, col_name] = fills
            else:
                X[col_name].fillna(fill_val, inplace=True)
        return X
