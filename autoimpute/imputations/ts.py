"""Time Series Imputation Module"""

from sklearn.base import BaseEstimator, TransformerMixin
from autoimpute.utils.helpers import _nan_col_dropper
from autoimpute.utils.checks import check_missingness
from autoimpute.utils.checks import _check_strategy, _check_fit_strat
from autoimpute.imputations.methods import _mean, _median, _mode, _random
from autoimpute.imputations.methods import _linear, _none
# pylint:disable=all

class TimeSeriesImputer(BaseEstimator, TransformerMixin):
    """Techniques to impute time series data"""
    
    strategies = {
        "mean": _mean,
        "median": _median,
        "mode":  _mode,
        "random": _random,
        "linear": _linear,
        "none": _none
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

    def _fit_strategy_validator(self, X, ind=None):
        """helper method to ensure right number of strategies"""
        ts = X.select_dtypes(include=[np.datetime64])
        ts_c = len(ts.columns)
        ts_ix = X.index
        if ts_c == 0:
            if not isinstance(ts_ix, pd.DatetimeIndex):
                err = "DataFrame must have time series index or a time series column"
                raise ValueError(err)
        if ts_c == 1:
            if not isinstance(ts_ix, pd.DatetimeIndex):
                X = X.set_index(ts_c[0], drop=True)
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
