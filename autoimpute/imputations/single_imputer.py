"""This module performs single imputations for cross-sectional Series.

This module contains one class - the SingleImputer. Use this class to perform
one imputation for each Series within a DataFrame. The methods available are
all univariate - they do not use any other features to perform a given Series'
imputation. Rather, they rely on the Series itself.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from autoimpute.utils import check_nan_columns
from autoimpute.imputations import BaseImputer
from autoimpute.imputations import method_names, single_methods, ts_methods
methods = method_names
sm = single_methods
tm = ts_methods
# pylint:disable=attribute-defined-outside-init
# pylint:disable=arguments-differ
# pylint:disable=protected-access
# pylint:disable=too-many-arguments
# pylint:disable=too-many-instance-attributes


class SingleImputer(BaseImputer, BaseEstimator, TransformerMixin):
    """Techniques to impute Series with missing values one time.

    The SingleImputer class takes a DataFrame and performs single imputations
    on each Series within the DataFrame. The SingleImputer does one pass for
    each column, and it supports univariate methods only.

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
            `default` imputes mean for numerical, mode for categorical.
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
            `none` does not impute the series. Mainly used for time series.
    """

    strategies = {
        methods.DEFAULT: sm._fit_single_default,
        methods.MEAN: sm._fit_mean,
        methods.MEDIAN: sm._fit_median,
        methods.MODE:  sm._fit_mode,
        methods.RANDOM: sm._fit_random,
        methods.NORM: sm._fit_norm,
        methods.CATEGORICAL: sm._fit_categorical,
        methods.LINEAR: tm._fit_linear,
        methods.NONE: sm._fit_none
    }

    def __init__(self, strategy="default", fill_value=None,
                 copy=True, scaler=None, verbose=False):
        """Create an instance of the SingleImputer class.

        As with sklearn classes, all arguments take default values. Therefore,
        SingleImputer() creates a valid class instance. The instance is used to
        set up an imputer and perform checks on arguments.

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
                See details of individual strategies for more info.
            verbose (bool, optional): print more information to console.
                Default value is False.
            copy (bool, optional): create copy of DataFrame or operate inplace.
                Default value is True. Copy created.
        """
        BaseImputer.__init__(
            self,
            scaler=scaler,
            verbose=verbose
        )
        self.strategy = strategy
        self.fill_value = fill_value
        self.copy = copy

    @property
    def strategy(self):
        """Property getter to return the value of the strategy property"""
        return self._strategy

    @strategy.setter
    def strategy(self, s):
        """Validate the strategy property to ensure it's type and value.

        Class instance only possible if strategy is proper type, as outlined
        in the init method. Passes supported strategies and user-defined
        strategy to helper method, which performs strategy checks.

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
        s = self.strategy
        cols = X.columns.tolist()
        self._strats = self.check_strategy_fit(s, cols)

        # scale if necessary
        if self.scaler:
            self._scaler_fit()
            self._scaler_transform()

    def _transform_strategy_validator(self, X, new_data):
        """Private method to validate before transformation phase."""
        # initial checks before transformation
        check_is_fitted(self, "statistics_")

        # check columns
        X_cols = X.columns.tolist()
        fit_cols = set(self._strats.keys())
        diff_fit = set(fit_cols).difference(X_cols)
        if diff_fit:
            err = "Same columns that were fit must appear in transform."
            raise ValueError(err)

        # scaler transform if necessary
        if new_data and self.scaler:
            self._scaler_transform()

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
            self: instance of the SingleImputer class.
        """
        # create statistics if validated
        self._fit_strategy_validator(X)
        self.statistics_ = {}

        # header print statement if verbose = true
        if self.verbose:
            ft = "FITTING IMPUTATION METHODS TO DATA..."
            st = "Strategies used to fit each column:"
            print(f"{ft}\n{st}\n{'-'*len(st)}")

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

    @check_nan_columns
    def transform(self, X, new_data=True):
        """Impute each column within a DataFrame using fit imputation methods.

        The transform step performs the actual imputations. Given a dataset
        previously fit, `transform` imputes each column with it's respective
        imputed values from fit (in the case of inductive) or performs new fit
        and transform in one sweep (in the case of transductive).

        Args:
            X (pd.DataFrame): fit DataFrame to impute.
            new_data (bool, Optional): whether or not new data is used.
                Default is False.

        Returns:
            X (pd.DataFrame): imputed in place or copy of original.

        Raises:
            ValueError: same columns must appear in fit and transform.
        """
        if self.copy:
            X = X.copy()
        self._transform_strategy_validator(X, new_data)
        if self.verbose:
            trans = "PERFORMING IMPUTATIONS ON DATA BASED ON FIT..."
            print(f"{trans}\n{'-'*len(trans)}")

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
            # fill missing values based on the method selected
            # note that default picks a method below depending on col
            # -------------------------------------------------------
            # mean and median imputation
            if strat in (methods.MEAN, methods.MEDIAN):
                sm._imp_central(X, col_name, fill)
            # mode imputation
            if strat == methods.MODE:
                sm._imp_mode(X, col_name, fill, self.fill_value)
            # imputatation w/ random value from observed data
            if strat == methods.RANDOM:
                sm._imp_random(X, col_name, fill, imp_ix)
            # linear interpolation imputation
            if strat == methods.LINEAR:
                tm._imp_interp(X, col_name, strat)
            # normal distribution imputatinon
            if strat == methods.NORM:
                sm._imp_norm(X, col_name, fill, imp_ix)
            # categorical distribution imputation
            if strat == methods.CATEGORICAL:
                sm._imp_categorical(X, col_name, fill, imp_ix)
            # no imputation if strategy is none
            if strat == methods.NONE:
                pass
        return X

    def fit_transform(self, X):
        """Convenience method to fit then transform the same dataset."""
        return self.fit(X).transform(X, False)
