"""Single imputation lib"""

import numpy as np
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, TransformerMixin
# pylint:disable=attribute-defined-outside-init
# pylint:disable=arguments-differ

def _mean(series):
    """helper mean"""
    return series.mean(), "mean"

def _median(series):
    """helper median"""
    return series.median(), "median"

def _mode(series):
    """helper mode"""
    return series.mode(), "mode"

def _default(series):
    """helper function for default"""
    if is_numeric_dtype(series):
        return _mean(series)
    if is_string_dtype(series):
        return _mode(series)

def _mode_helper(series, mode, strategy):
    """helper function for mode"""
    num_modes = len(mode)
    if num_modes == 1:
        return series.fillna(mode[0], inplace=True)
    else:
        if strategy is None:
            return series.fillna(mode[0], inplace=True)
        elif strategy == "random":
            ind = series[series.isnull()].index
            fills = np.random.choice(mode, len(ind))
            series.loc[ind] = fills
        else:
            err = f"{strategy} not accepted for mode imputation"
            raise ValueError(err)

class SingleImputer(BaseEstimator, TransformerMixin):
    """Techniques to Impute missing values once"""

    strategies = {
        "mean": _mean,
        "median": _median,
        "mode":  _mode,
        "default": _default
    }

    def __init__(self, strategy="default", fill_value=None,
                 verbose=False, copy=True):
        self.strategy = strategy
        self.fill_value = fill_value
        self.verbose = verbose
        self.copy = copy
        self.statistics_ = {}
        self._strats = {}
        self._fit = True

    @property
    def strategy(self):
        """return the strategy property"""
        return self._strategy

    @strategy.setter
    def strategy(self, s):
        """validate the strategy property"""
        strat_names = self.strategies.keys()
        err_op = f"Strategies must be one of {strat_names}."
        if isinstance(s, str):
            if s in strat_names:
                self._strategy = s
            else:
                err = f"Strategy {s} not a valid imputation method.\n"
                raise ValueError(f"{err} {err_op}")
        elif isinstance(s, (list, tuple, dict)):
            if isinstance(s, dict):
                ss = set(s.values())
            else:
                ss = set(s)
            sdiff = ss.difference(strat_names)
            if not sdiff:
                self._strategy = s
            else:
                err = f"Strategies {sdiff} in {s} not valid for imputation.\n"
                raise ValueError(f"{err} {err_op}")
        else:
            raise ValueError("Strategy must be string, tuple, list, or dict.")

    def _fit_strategy_validator(self, X):
        """helper method to ensure right number of strategies"""
        cols = X.columns.tolist()
        c_l = len(cols)
        s_l = len(self.strategy)

        # if strategy is string, extend strategy to all cols
        if isinstance(self.strategy, str):
            self._strats = {c:self.strategy for c in cols}

        # if list or tuple, ensure same number of cols in X as strategies
        if isinstance(self.strategy, (list, tuple)):
            if c_l == s_l:
                self._strats = {c[0]:c[1] for c in zip(cols, self.strategy)}
            else:
                err = f"# columns ({c_l}) not equal to # strategies ({s_l})"
                raise ValueError(err)

        # if strategy is dict, ensure keys in strategy match cols in X
        if isinstance(self.strategy, dict):
            keys = set(self.strategy.keys())
            cols = set(cols)
            kdiff = keys.difference(cols)
            cdiff = cols.difference(keys)
            if kdiff or cdiff:
                err = f"Keys for strategies don't match columns of X.\n"
                err_k = f"Keys missing: {kdiff}\n"
                err_c = f"Columns missing: {cdiff}"
                raise ValueError(f"{err}{err_k}{err_c}")
            else:
                self._strats = self.strategy

        # print strategies if verbose
        if self.verbose:
            st = "Strategies used to fit each column:"
            print(f"{st}\n{'-'*len(st)}")
            for k, v in self._strats.items():
                print(f"Column: {k}, Strategy: {v}")

    def fit(self, X):
        """Fit method for single imputer"""
        self._fit = False
        self._fit_strategy_validator(X)
        for col_name, func_name in self._strats.items():
            f = self.strategies[func_name]
            fit_param, fit_name = f(X[col_name])
            self.statistics_[col_name] = {"param":fit_param,
                                          "strategy": fit_name}
            if self.verbose:
                print(f"{col_name} has {func_name} equal to {fit_param}")
        self._fit = True
        return self

    def transform(self, X):
        """Transform method for a single imputer"""
        # initial checks before transformation
        if self.copy:
            X = X.copy()
        if not self._fit:
            s = self.__class__.__name__
            err = f"Must fit {s} to data before performing transformation."
            raise NotFittedError(err)
        # check columns
        X_cols = X.columns.tolist()
        fit_cols = set(self._strats.keys())
        diff_X = set(X_cols).difference(fit_cols)
        diff_fit = set(fit_cols).difference(X_cols)
        if diff_X or diff_fit:
            raise ValueError("Same columns must appear in fit and transform.")

        # transformation logic
        for col_name, fit_data in self.statistics_.items():
            strat = fit_data["strategy"]
            fill_val = fit_data["param"]
            if strat != "mode":
                X[col_name].fillna(fill_val, inplace=True)
            else:
                _mode_helper(X[col_name], fill_val, self.fill_value)
        return X
