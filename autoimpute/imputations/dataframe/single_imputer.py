"""This module performs one imputation of missing features in a dataset.

This module contains one class - the SingleImputer. Use this class to
impute each Series within a DataFrame one time. This class makes numerous
imputation methods available - both univariate and multivatiate. Each method
runs once on its specified column. When one pass through the columns is
complete, the SingleImputer returns the single imputed dataset.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from autoimpute.utils import check_nan_columns, check_predictors_fit
from autoimpute.utils import check_strategy_fit
from autoimpute.utils.helpers import _one_hot_encode
from autoimpute.imputations.helpers import _get_observed
from .base_imputer import BaseImputer
from ..series import DefaultUnivarImputer

# pylint:disable=attribute-defined-outside-init
# pylint:disable=arguments-differ
# pylint:disable=protected-access
# pylint:disable=too-many-arguments
# pylint:disable=too-many-locals
# pylint:disable=too-many-instance-attributes
# pylint:disable=unused-argument


class SingleImputer(BaseImputer, BaseEstimator, TransformerMixin):
    """Techniques to impute Series with missing values one time.

    The SingleImputer class takes a DataFrame and performs imputations on
    each Series within the DataFrame. The Imputer does one pass for each
    column, and it supports numerous imputation methods for each column.

    The SingleImputer delegates imputation to respective SeriesImputers,
    each of which maps to a specific strategy supported by the SingleImputer.
    Most of the SeriesImputers are inductive (fit and transform for new data).
    Transductive SeriesImputers (such as InterpolateImputer) still perform a
    "mock" fit stage but do all the imputation work in the transform step. The
    fit stage is performed to remain consistent with the sklearn API. The
    class is a valid sklearn transformer that can be used in an sklearn
    Pipeline because it inherits from the TransformerMixin and implements both
    fit and transform methods.
    """

    def __init__(self, strategy="default predictive", predictors="all",
                 imp_kwgs=None, copy=True, seed=None, visit="default"):
        """Create an instance of the SingleImputer class.

        As with sklearn classes, all arguments take default values. Therefore,
        SingleImputer() creates a valid class instance. The instance is
        used to set up a SingleImputer and perform checks on arguments.

        Args:
            strategy (str, iter, dict; optional): strategy for single imputer.
                Default value is str --> `predictive default`.
                See BaseImputer for all available strategies.
                If str, single strategy broadcast to all series in DataFrame.
                If iter, must provide 1 strategy per column. Each method w/in
                iterator applies to column with same index value in DataFrame.
                If dict, must provide key = column name, value = imputer.
                Dict the most flexible and PREFERRED way to create custom
                imputation strategies if not using the default. Dict does not
                require method for every column; just those specified as keys.
            predictors (str, iter, dict, optional): defaults to `all`, i.e.
                use all predictors. If `all`, every column will be used for
                every class prediction. If a list, subset of columns used for
                all predictions. If a dict, specify which columns to use as
                predictors for each imputation. Columns not specified in dict
                but present in `strategy` receive `all` other cols as preds.
                Note predictors are IGNORED for univariate imputation methods,
                so specifying is meaningless unless strategy is predictive.
            imp_kwgs (dict, optional): keyword args for each SeriesImputer.
                Default is None, which means default imputer created to match
                specific strategy. `imp_kwgs` keys can be either columns or
                strategies. If strategies, each column given that strategy is
                instantiated with same arguments. When strategy is `default`,
                `imp_kwgs` is ignored.
            copy (bool, optional): create copy of DataFrame or operate inplace.
                Default value is True. Copy created.
            seed (int, optional): seed setting for reproducible results.
                Defualt is None. No validation, but values should be integer.
        """
        BaseImputer.__init__(
            self,
            strategy=strategy,
            imp_kwgs=imp_kwgs,
            visit=visit
        )
        self.strategy = strategy
        self.predictors = predictors
        self.copy = copy
        self.seed = seed

    def _fit_strategy_validator(self, X):
        """Private method to validate strategies appropriate for fit.

        Checks whether strategies match with type of column they are applied
        to. If not, error is raised through `check_strategy_fit` method.
        """

        # remove nan columns and store colnames
        cols = X.columns.tolist()
        self._strats = check_strategy_fit(self.strategy, cols)
        self._preds = check_predictors_fit(self.predictors, cols)

    def _transform_strategy_validator(self, X):
        """Private method to prep and validate before transformation."""

        # initial checks before transformation and check columns are the same
        check_is_fitted(self, "statistics_")
        X_cols = X.columns.tolist()
        fit_cols = set(self._strats.keys())
        diff_fit = set(fit_cols).difference(X_cols)
        if diff_fit:
            err = "Same columns that were fit must appear in transform."
            raise ValueError(err)

    @check_nan_columns
    def fit(self, X, y=None, imp_ixs=None):
        """Fit specified imputation methods to each column within a DataFrame.

        The fit method calculates the `statistics` necessary to later
        transform a dataset (i.e. perform actual imputations). Inductive
        methods calculate statistic on the fit data, then impute new missing
        data with that value. Most currently supported methods are inductive.

        It's important to note that we have to fit X regardless of whether any
        data is missing. Transform step may have missing data if new data is
        used, so fit each column that appears in the given strategies.

        Args:
            X (pd.DataFrame): pandas DataFrame on which imputer is fit.
            y (pd.Series, pd.DataFrame Optional): response. Default is None.
                Determined interally in fit method. Arg is present to remain
                compatible with sklearn Pipelines.
            imp_ixs (dict): Dictionary of lists of indices that indicate which
                data elements to impute per column or None to identify from
                missing elements per column

        Returns:
            self: instance of the SingleImputer class.

        Raises:
            ValueError: error in specification of strategies. Raised through
                `check_strategy_fit`. See its docstrings for more info.
            ValueError: error in specification of predictors. Raised through
                `check_predictors_fit`. See its docstrings for more info.
        """

        # first, prep columns we plan to use and make sure they are valid
        self._fit_strategy_validator(X)
        self.statistics_ = {}

        # perform fit on each column, depending on that column's strategy
        # note that right now, operations are COLUMN-by-COLUMN, iteratively
        if self.seed is not None:
            np.random.seed(self.seed)
        for column, method in self._strats.items():
            imp = self.strategies[method]
            imp_params = self._fit_init_params(column, method, self.imp_kwgs)

            # try to create an instance of the imputer, given the args
            try:
                if imp_params is None:
                    imputer = imp()
                else:
                    imputer = imp(**imp_params)
            except TypeError as te:
                name = imp.__name__
                err = f"Invalid arguments passed to {name} __init__ method."
                raise ValueError(err) from te

            # identify the column for imputation
            ys = X[column]

            # the fit depends on what type of strategy we use.
            # first, fit univariate methods, which are straightforward.
            if method in self.univariate_strategies:
                imputer.fit(ys, None)

            # now, fit on predictive methods, which are more complex.
            if method in self.predictive_strategies:
                preds = self._preds[column]
                if preds == "all":
                    xs = X.drop(column, axis=1)
                else:
                    xs = X[preds]

                if imp_ixs is not None:
                    ys[imp_ixs[column]] = np.nan

                # fit the data on observed values only.
                x_, y_ = _get_observed(xs, ys)

                # before imputing, need to encode categoricals
                x_ = _one_hot_encode(x_)

                imputer.fit(x_, y_)

            # finally, store imputer for each column as statistics
            self.statistics_[column] = imputer
        return self

    @check_nan_columns
    def transform(self, X, imp_ixs=None, **trans_kwargs):
        """Impute each column within a DataFrame using fit imputation methods.

        The transform step performs the actual imputations. Given a dataset
        previously fit, `transform` imputes each column with it's respective
        imputed values from fit (in the case of inductive) or performs new fit
        and transform in one sweep (in the case of transductive).

        Args:
            X (pd.DataFrame): DataFrame to impute (same as fit or new data).
            imp_ixs (dict): Dictionary of lists of indices that indicate which
                data elements to impute per column or None to identify from
                missing elements per column
            **trans_kwargs: dict, optional args for bayesian.

        Returns:
            X (pd.DataFrame): imputed in place or copy of original.

        Raises:
            ValueError: same columns must appear in fit and transform.
                Raised through _transform_strategy_validator.
        """

        # copy the dataset if necessary, then prep predictors
        if self.copy:
            X = X.copy()
        self._transform_strategy_validator(X)

        # transformation logic
        self.imputed_ = {}
        if self.seed is not None:
            np.random.seed(self.seed)
        for column, imputer in self.statistics_.items():
            if imp_ixs is None:
                imp_ix = X[column][X[column].isnull()].index
            else:
                imp_ix = pd.Index(imp_ixs[column])
            self.imputed_[column] = imp_ix.tolist()

            # continue if there are no imputations to make
            if imp_ix.empty:
                continue

            # implement transform logic for univariate
            if imputer.strategy in self.univariate_strategies:
                x_ = X[column]

            # implement transform logic for predictive
            if imputer.strategy in self.predictive_strategies:
                preds = self._preds[column]
                if preds == "all":
                    x_ = X.drop(column, axis=1)
                else:
                    x_ = X[preds]

                # isolate missingness
                if isinstance(x_, pd.Series):
                    x_ = x_.to_frame()
                    x_ = x_.loc[imp_ix]
                else:
                    x_ = x_.loc[imp_ix, :]

                # default univariate impute for missing covariates
                mis_cov = pd.isnull(x_).sum()
                mis_cov = mis_cov[mis_cov > 0]
                if any(mis_cov):
                    x_m = mis_cov.index
                    for col in x_m:
                        d = DefaultUnivarImputer()
                        if mis_cov[col] == x_.shape[0]:
                            d_imps = 0
                        else:
                            d_imps = d.fit_impute(x_[col], None)
                        x_null = x_[col][x_[col].isnull()].index
                        x_.loc[x_null, col] = d_imps

                # handling encoding again for prediction of imputations
                x_ = _one_hot_encode(x_)

            # perform imputation given the specified imputer and value for x_
            # this fix below checks for strategies that need k if Mice used
            # right now, that's  just bayesian strategies
            # k defaults to None, which works for non Mice related imputation
            if imputer.strategy in (
                "bayesian binary logistic",
                "bayesian least squares"
            ):
                k = trans_kwargs.get("k")
                X.loc[imp_ix, column] = imputer.impute(x_, k=k)
            else:
                X.loc[imp_ix, column] = imputer.impute(x_)
        return X

    def fit_transform(self, X, y=None, **trans_kwargs):
        """Convenience method to fit then transform the same dataset.

        Args:
            X (pd.DataFrame): DataFrame used for fit and transform steps.
            y (pd.DataFrame, pd.Series, Optional): response. Default is None.
                Set internally by `fit` method.
            **trans_kwargs: dict, optional args for bayesian.

        Returns:
            X (pd.DataFrame): imputed in place or copy of original.
        """
        return self.fit(X, y).transform(X, **trans_kwargs)
