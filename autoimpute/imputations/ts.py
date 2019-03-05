"""This module performs imputations for Series with a time-based index.

This module contains one class - TimeSeriesImputer. Use this class to perform
imputations for each Series within a DataFrame that has a time-based index.
The data within each Series should have logical ordering, even though not all
the imputation methods supported in this module require a time series.

Todo:
    * Support additional time series imputation methods.
    * Add examples of imputations and how the class works in pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from autoimpute.utils.helpers import _nan_col_dropper
from autoimpute.utils.checks import check_missingness
from autoimpute.utils.checks import _check_strategy, _check_fit_strat
from autoimpute.imputations import simple_methods
sm = simple_methods
# pylint:disable=attribute-defined-outside-init
# pylint:disable=arguments-differ
# pylint:disable=protected-access

class TimeSeriesImputer(BaseEstimator, TransformerMixin):
    """Techniques to impute Series with a logical ordering and time component.

    The TimeSeriesImputer class takes a DataFrame and performs imputations
    on each Series within the DataFrame. The TimeSeriesImputer requires the
    DataFrame contain a DateTimeindex OR at least one datetime column. The
    selected datetime column is sorted and becomes the ordering for which
    time series imputation obeys.

    The class is a valid transformer that can be used in an sklearn pipeline
    because it inherits from the TransformerMixin and implements both fit and
    transform methods.

    Some of the imputers are inductive (i.e. fit and transform for new data).
    Others are transductive (i.e. fit_transform only). Transductive methods
    return None during the "fitting" stage. This behavior is a bit odd, but
    it allows inductive and transductive methods within the same Imputer.

    Attributes:
        strategies (dict): dictionary of supported imputation methods.
            Key = imputation name; Value = function to perform imputation.
            `default` imputes linear for numerical, mode for categorical.
            `mean` imputes missing values with the average of the series.
            `median` imputes missing values with the median of the series.
            `mode` imputes missing values with the mode of the series.
                Method handles more than one mode (see _mode_helper method).
            `random` immputes w/ random choice from set of Series unique vals.
            `norm` imputes series using random draws from normal distribution.
                Mean and std calculated from observed values of the Series.
            `categorical` imputes series using random draws from pmf.
                Proportions calculated from non-missing category instances.
            `linear` imputes series using linear interpolation.
            `time` imputes series using time-weighted interpolation.
            `locf` imputes series using last observation carried forward.
                Uses series mean if the first value is missing.
            `nocb` imputes series using next observation carried backward.
                Uses series mean if the last value is missing.
            `none` does not impute the series. Mainly used for time series.
    """

    strategies = {
        "default": sm._fit_ts_default,
        "mean": sm._fit_mean,
        "median": sm._fit_median,
        "mode":  sm._fit_mode,
        "random": sm._fit_random,
        "norm": sm._fit_norm,
        "categorical": sm._fit_categorical,
        "linear": sm._fit_linear,
        "time": sm._fit_time,
        "locf": sm._fit_locf,
        "nocb": sm._fit_nocb,
        "none": sm._fit_none,
    }

    def __init__(self, strategy="default", fill_value=None,
                 index_column=None, verbose=False):
        """Create an instance of the TimeSeriesImputer class.

        As with sklearn classes, all arguments take default values. Therefore,
        TimeSeriesImputer() creates a valid class instance. The instance is
        used to set up an imputer and perform checks on arguments.

        Args:
            strategy (str, iter, dict; optional): strategies for imputation.
                Default value is str -> "default". I.e. default imputation.
                If str, single strategy broadcast to all series in DataFrame.
                If iter, must provide 1 strategy per column. Each method within
                iterator applies to column with same index value in DataFrame.
                If dict, must provide key = column name, value = imputer.
                Dict the most flexible and PREFERRED way to create custom
                imputation strategies if not using the default. Dict does not
                require method for every column; just those specified as keys.
            fill_value (str, optional): fill val when strategy needs more info.
                Right now, fill_value ignored for everything except mode.
                If strategy = mode, fill_value = None or `random`. If None,
                first mode is used (default strategy of SciPy). If `random`,
                imputer will select 1 of n modes at random.
            index_column (str, optional): the name of the column to index.
                Defaults to None. If None, first column with datetime dtype
                selected as the index.
            verbose (bool, optional): print more information to console.
                Default value is False.
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.index_column = index_column
        self.verbose = verbose

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
            Both errors raised through helper method `_check_strategy`
        """
        strat_names = self.strategies.keys()
        self._strategy = _check_strategy(strat_names, s)

    def _fit_strategy_validator(self, X):
        """Internal helper method to validate strategies appropriate for fit.

        Checks whether strategies match with type of column they are applied
        to. If not, error is raised through `_check_fit_strat` method.
        """
        # first, make sure there is at least one datetime column
        ts = X.select_dtypes(include=[np.datetime64])
        ts_c = len(ts.columns)
        ts_ix = X.index
        if not isinstance(ts_ix, pd.DatetimeIndex):
            if ts_c == 0:
                err = "Must have DatetimeIndex or column with type datetime."
                raise ValueError(err)

        # next, strategy check with existing columns passed
        ocols = X.columns.tolist()
        X, self._nc = _nan_col_dropper(X)
        ncols = X.columns.tolist()
        self._strats = _check_fit_strat(self.strategy, self._nc, ocols, ncols)

    def _transform_strategy_validator(self, X):
        """Internal helper to validate strategy before transformation.

        Checks whether differences in columns, and ensures that datetime
        column exists to set as index before imputation methods take place.
        """

        # check columns
        X_cols = X.columns.tolist()
        fit_cols = set(self._strats.keys())
        diff_fit = set(fit_cols).difference(X_cols)
        if diff_fit:
            err = "Same columns that were fit must appear in transform."
            raise ValueError(err)

        # identify if time series columns
        ts = X.select_dtypes(include=[np.datetime64])
        ts_c = len(ts.columns)
        ts_ix = X.index

        # attempt to reindex the right time column
        if not isinstance(ts_ix, pd.DatetimeIndex):
            fts = ts.columns[0]
            if ts_c == 1:
                self.statistics_.pop(fts, None)
                X = X.set_index(fts, drop=True)
            else:
                ic = self.index_column
                if ic is None:
                    self.statistics_.pop(fts, None)
                    X = X.set_index(fts, drop=True)
                else:
                    if ic in ts:
                        self.statistics_.pop(ic, None)
                        X = X.set_index(ic, drop=True)
                    else:
                        err = f"{ic} can't be set as DatetimeIndex."
                        raise KeyError(err)
        # sort and return X
        X.sort_index(ascending=True, inplace=True)
        return X

    @check_missingness
    def fit(self, X):
        """Fit imputation methods to each column within a DataFrame.

        The fit method calclulates the `statistics` necessary to later
        transform a dataset (i.e. perform actual imputatations). Inductive
        methods (mean, mode, median, etc.) calculate statistic on the fit
        data, then impute new missing data with that value. Transductive
        methods (linear) don't calculate anything during fit, as they
        apply imputation during transformation phase only.

        Args:
            X (pd.DataFrame): pandas DataFrame on which imputer is fit.

        Returns:
            self: instance of the TimeSeriesImputer class.
        """
        self._fit_strategy_validator(X)
        self.statistics_ = {}

        if self.verbose:
            st = "Strategies used to fit each column:"
            print(f"{st}\n{'-'*len(st)}")
        # perform fit on each column, depending on that column's strategy
        for col_name, func_name in self._strats.items():
            f = self.strategies[func_name]
            fit_param, fit_name = f(X[col_name])
            self.statistics_[col_name] = {"param":fit_param,
                                          "strategy": fit_name}
            # print strategies if verbose
            if self.verbose:
                print(f"Column: {col_name}, Strategy: {fit_name}")
        return self

    @check_missingness
    def transform(self, X):
        """Impute each column within a DataFrame using fit imputation methods.

        The transform step performs the actual imputations. Given a dataset
        previously fit, `transform` imputes each column with it's respective
        imputed values from fit (in the case of inductive) or performs new fit
        and transform in one sweep (in the case of transductive).

        Args:
            X (pd.DataFrame): fit DataFrame to impute.

        Returns:
            X (pd.DataFrame): new DataFrame with time-series index.

        Raises:
            ValueError: same columns must appear in fit and transform.
        """
        # initial checks before transformation
        check_is_fitted(self, 'statistics_')

        # create dataframe index then proceed
        X = self._transform_strategy_validator(X)
        # transformation logic
        self.imputed_ = {}
        for col_name, fit_data in self.statistics_.items():
            strat = fit_data["strategy"]
            fill_val = fit_data["param"]
            imp_ind = X[col_name][X[col_name].isnull()].index
            self.imputed_[col_name] = imp_ind.tolist()
            if self.verbose:
                print(f"Transforming {col_name} with strategy '{strat}'")
                print(f"Numer of imputations to perform: {len(imp_ind)}")
            # fill missing values based on the method selected
            # note that default picks a method below depending on col
            # -------------------------------------------------------
            # mean and median imputation
            if strat in ("mean", "median"):
                sm._imp_central(X, col_name, fill_val)
            # mode imputation
            if strat == "mode":
                sm._imp_mode(X, col_name, fill_val, self.fill_value)
            # imputatation w/ random value from observed data
            if strat == "random":
                sm._imp_random(X, col_name, fill_val, imp_ind)
            # linear interpolation imputation
            if strat in ("linear", "time"):
                sm._imp_interp(X, col_name, strat)
            # normal distribution imputatinon
            if strat == "norm":
                sm._imp_norm(X, col_name, fill_val, imp_ind)
            # categorical distribution imputation
            if strat == "categorical":
                sm._imp_categorical(X, col_name, fill_val, imp_ind)
            # last observation carried forward
            if strat == "locf":
                sm._imp_locf(X, col_name, fill_val)
            # next observation carried backward
            if strat == "nocb":
                sm._imp_nocb(X, col_name, fill_val)
            # no imputation if strategy is none
            if strat == "none":
                pass
        return X
