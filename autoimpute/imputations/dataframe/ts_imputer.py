"""This module performs imputations for Series with a time-based index.

This module contains one class - TimeSeriesImputer. Use this class to perform
imputations for each Series within a DataFrame that has a time-based index.
The data within each Series should have logical ordering, even though not all
the imputation methods supported in this module require a time series.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from autoimpute.utils import check_nan_columns
from autoimpute.imputations import method_names
from .base_imputer import BaseImputer
from ..series import DefaultTimeSeriesImputer
from ..series import MeanImputer, MedianImputer, ModeImputer
from ..series import NormImputer, CategoricalImputer
from ..series import RandomImputer, InterpolateImputer
from ..series import LOCFImputer, NOCBImputer
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=arguments-differ
# pylint:disable=protected-access
# pylint:disable=too-many-arguments
# pylint:disable=too-many-instance-attributes

class TimeSeriesImputer(BaseImputer, BaseEstimator, TransformerMixin):
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
            `interpolate` imputes series using chosen interpolation method.
                Default is linear. See InterpolateImputer for more info.
            `locf` imputes series using last observation carried forward.
                Uses series mean if the first value is missing.
            `nocb` imputes series using next observation carried backward.
                Uses series mean if the last value is missing.
    """

    strategies = {
        methods.DEFAULT: DefaultTimeSeriesImputer,
        methods.MEAN: MeanImputer,
        methods.MEDIAN: MedianImputer,
        methods.MODE:  ModeImputer,
        methods.RANDOM: RandomImputer,
        methods.NORM: NormImputer,
        methods.CATEGORICAL: CategoricalImputer,
        methods.INTERPOLATE: InterpolateImputer,
        methods.LOCF: LOCFImputer,
        methods.NOCB: NOCBImputer
    }

    def __init__(self, strategy="default", imp_kwgs=None,
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
            imp_kwgs (dict, optional): keyword arguments for each imputer.
                Default is None, which means default imputer created to match
                specific strategy. imp_kwgs keys can be either columns or
                strategies. If strategies, each column given that strategy is
                instantiated with same arguments.
            index_column (str, optional): the name of the column to index.
                Defaults to None. If None, first column with datetime dtype
                selected as the index.
            verbose (bool, optional): print more information to console.
                Default value is False.
        """
        BaseImputer.__init__(
            self,
            imp_kwgs=imp_kwgs,
            scaler=None,
            verbose=verbose
        )
        self.strategy = strategy
        self.imp_kwgs = imp_kwgs
        self.index_column = index_column

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
        # first, make sure there is at least one datetime column
        ts = X.select_dtypes(include=[np.datetime64])
        ts_c = len(ts.columns)
        ts_ix = X.index
        if not isinstance(ts_ix, pd.DatetimeIndex):
            if ts_c == 0:
                err = "Must have DatetimeIndex or column with type datetime."
                raise ValueError(err)

        # next, strategy check with existing columns passed
        s = self.strategy
        cols = X.columns.tolist()
        self._strats = self.check_strategy_fit(s, cols)

    def _transform_strategy_validator(self, X):
        """Internal helper to validate strategy before transformation.

        Checks whether differences in columns, and ensures that datetime
        column exists to set as index before imputation methods take place.
        """
        # initial checks before transformation
        check_is_fitted(self, "statistics_")

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

    @check_nan_columns
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
            ft = "FITTING IMPUTATION METHODS TO DATA..."
            st = "Strategies used to fit each column:"
            print(f"{ft}\n{st}\n{'-'*len(st)}")

        # perform fit on each column, depending on that column's strategy
        # note that right now, operations are COLUMN-by-COLUMN, iteratively
        # in the future, we should handle univar methods in parallel
        for column, method in self._strats.items():
            imp = self.strategies[method]
            imp_params = self._fit_init_params(column, method, self.imp_kwgs)
            try:
                imputer = imp() if imp_params is None else imp(**imp_params)
                imputer.fit(X[column])
                self.statistics_[column] = imputer
                # print strategies if verbose
                if self.verbose:
                    print(f"Column: {column}, Strategy: {method}")
            except TypeError as te:
                name = imp.__name__
                err = f"Invalid arguments passed to {name} __init__ method."
                raise ValueError(err) from te
        return self

    @check_nan_columns
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
        # create dataframe index then proceed
        X = self._transform_strategy_validator(X)
        if self.verbose:
            trans = "PERFORMING IMPUTATIONS ON DATA BASED ON FIT..."
            print(f"{trans}\n{'-'*len(trans)}")

        # transformation logic
        # same applies, should be able to handel in parallel
        self.imputed_ = {}
        for column, imputer in self.statistics_.items():
            imp_ix = X[column][X[column].isnull()].index
            self.imputed_[column] = imp_ix.tolist()
            if self.verbose:
                strat = imputer.statistics_["strategy"]
                print(f"Transforming {column} with strategy '{strat}'")
                print(f"Numer of imputations to perform: {len(imp_ix)}")

            # perform imputation given the specified imputer
            X.loc[imp_ix, column] = imputer.impute(X[column])
        return X
