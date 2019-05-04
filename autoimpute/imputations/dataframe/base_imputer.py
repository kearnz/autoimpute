"""Module for BaseImputer - a base class for classifiers/dataframe imputers.

This module contains the `BaseImputer`, which is used to abstract away
functionality in both missingness classifiers and dataframe imputers.
"""

import warnings
import itertools
import numpy as np
import pandas as pd
from sklearn.base import clone
from autoimpute.utils import check_strategy_allowed
from autoimpute.imputations import method_names
from ..series import DefaultUnivarImputer, DefaultPredictiveImputer
from ..series import DefaultTimeSeriesImputer
from ..series import MeanImputer, MedianImputer, ModeImputer
from ..series import NormImputer, CategoricalImputer
from ..series import RandomImputer, InterpolateImputer
from ..series import LOCFImputer, NOCBImputer
from ..series import LeastSquaresImputer, StochasticImputer
from ..series import PMMImputer, LRDImputer
from ..series import BinaryLogisticImputer, MultinomialLogisticImputer
from ..series import BayesianLeastSquaresImputer
from ..series import BayesianBinaryLogisticImputer
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=too-many-arguments
# pylint:disable=too-many-instance-attributes
# pylint:disable=inconsistent-return-statements


class BaseImputer:
    """Building blocks for more advanced imputers and missingness classifiers.

    The BaseImputer is not a stand-alone class and thus serves no purpose
    other than as a parent to Imputers and MissingnessClassifiers. Therefore,
    the BaseImputer should not be used directly unless creating an Imputer.
    That being said, all dataframe Imputers should inherit from BaseImputer.
    It contains base functionality for any new dataframe Imputer, and it holds
    the set of strategies that make up this imputation library.

    Attributes:
        univariate_strategies (dict): univariate imputation methods.
            Key = imputation name; Value = function to perform imputation.
            `univariate default` mean for numerical, mode for categorical.
            `time default` interpolate for numerical, mode for categorical.
            `mean` imputes missing values with the average of the series.
            `median` imputes missing values with the median of the series.
            `mode` imputes missing values with the mode of the series.
                Method handles more than one mode (see ModeImputer for info).
            `random` imputes w/ random choice from set of series unique vals.
            `norm` imputes series using random draws from normal distribution.
                Mean and std calculated from observed values of the series.
            `categorical` imputes series using random draws from pmf.
                Proportions calculated from non-missing category instances.
            `interpolate` imputes series using chosen interpolation method.
                Default is linear. See InterpolateImputer for more info.
            `locf` imputes series carrying last observation moving forward.
            `nocb` imputes series carrying next observation moving backward.
        predictive_strategies (dict): predictive imputation methods.
            Key = imputation name; Value = function to perform imputation.
            `predictive default` pmm for numerical, logistic for categorical.
            `least squares` predict missing values from linear regression.
            `binary logistic` predict missing values with 2 classes.
            `multinomial logistic` predict missing values with multiclass.
            `stochastic` linear regression + random draw from norm w/ mse std.
            `bayesian least squares` draw from the posterior predictive
                distribution for each missing value, using OLS model.
            `bayesian binary logistic` draw from the posterior predictive
                distribution for each missing value, using logistic model.
            `pmm` imputes series using predictive mean matching. PMM is a
                semi-supervised method using bayesian & hot-deck imputation.
            `lrd` imputes series using local residual draws. LRD is a
                semi-supervised method using bayesian & hot-deck imputation.
        strategies (dict): univariate and predictive strategies merged.
        visit_sequences: tuple of supported sequences for visiting columns.
            Right now, default = left-to-right. Only sequence supported.
    """
    univariate_strategies = {
        methods.DEFAULT_UNIVAR: DefaultUnivarImputer,
        methods.DEFAULT_TIME: DefaultTimeSeriesImputer,
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

    predictive_strategies = {
        methods.DEFAULT_PRED: DefaultPredictiveImputer,
        methods.LS: LeastSquaresImputer,
        methods.STOCHASTIC: StochasticImputer,
        methods.BINARY_LOGISTIC: BinaryLogisticImputer,
        methods.MULTI_LOGISTIC: MultinomialLogisticImputer,
        methods.BAYESIAN_LS: BayesianLeastSquaresImputer,
        methods.BAYESIAN_BINARY_LOGISTIC: BayesianBinaryLogisticImputer,
        methods.PMM: PMMImputer,
        methods.LRD: LRDImputer
    }

    strategies = {**predictive_strategies, **univariate_strategies}

    visit_sequences = (
        "default",
        "left-to-right"
    )

    def __init__(self, strategy, imp_kwgs, scaler, verbose, visit):
        """Initialize the BaseImputer.

        Args:
            strategy (str, iter, dict; optional): strategies for imputation.
                Default value is str -> `predictive default`.
                If str, single strategy broadcast to all series in DataFrame.
                If iter, must provide 1 strategy per column. Each method w/in
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
            scaler (sklearn scaler, optional): A scaler supported by sklearn.
                Default to None. Otherwise, must be sklearn-compliant scaler.
            verbose (bool, optional): Print information to the console.
                Defaults to False.
            visit (str, None): order to visit columns for imputation.
                Default is `default`, which implements `left-to-right`.
                More strategies (random, monotone, etc.) TBD.
        """
        self.strategy = strategy
        self.imp_kwgs = imp_kwgs
        self.scaler = scaler
        self.verbose = verbose
        self.visit = visit

    @property
    def strategy(self):
        """Property getter to return the value of the strategy property."""
        return self._strategy

    @strategy.setter
    def strategy(self, s):
        """Validate the strategy property to ensure it's type and value.

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
        self._strategy = check_strategy_allowed(strat_names, s)

    @property
    def imp_kwgs(self):
        """Property getter to return the value of imp_kwgs."""
        return self._imp_kwgs

    @imp_kwgs.setter
    def imp_kwgs(self, kwgs):
        """Validate the imp_kwgs and set default properties.

        The BaseImputer validates the `imp_kwgs` argument. `imp_kwgs` contain
        optional keyword arguments for an imputers' strategies or columns. The
        argument is optional, and its default is None.

        Args:
            kwgs (dict, None): None or dictionary of keywords.

        Raises:
            ValueError: imp_kwgs not correctly specified as argument.
        """
        if not isinstance(kwgs, (type(None), dict)):
            err = "imp_kwgs must be dict of args used to instantiate Imputer."
            raise ValueError(err)
        self._imp_kwgs = kwgs

    @property
    def scaler(self):
        """Property getter to return the value of the scaler property."""
        return self._scaler

    @scaler.setter
    def scaler(self, s):
        """Validate the scaler property and set default parameters.

        The BaseImputer provides the option to scale data from within the
        imputer. The scaler is optional, and the default value is None. If
        a scaler is passed, the scaler must be a valid sklearn transformer.
        Therefore, it must implement the `fit_transform` method.

        Args:
            s (scaler): if None, no scaler used, nothing to verify.

        Raises:
            ValueError: scaler does not implement `fit_transform` method.
        """
        if s is None:
            self._scaler = s
        else:
            m = "fit_transform"
            if not hasattr(s, m):
                raise ValueError(f"Scaler must implement {m} method.")
            self._scaler = s

    @property
    def visit(self):
        """Property getter to return the value of the visit property."""
        return self._visit

    @visit.setter
    def visit(self, v):
        """Validate the visit property to ensure it's type and value.

        Class instance only possible if visit is proper type, as outlined in
        the init method. Visit property must be one of valid sequences in the
        `visit_sequences` variable.

        Args:
            v (str): Visit sequence passed as arg to class instance.

        Raises:
            TypeError: visit sequence must be a string.
            ValueError: visit sequenece not in `visit_sequences`.
        """

        # deal with type first
        if not isinstance(v, str):
            err = "visit must be a string specifying visit sequence to use."
            raise TypeError(err)

        # deal with value next
        if v not in self.visit_sequences:
            err = f"visit not valid. Must be one of {self.visit_sequences}"
            raise ValueError(err)

        # otherwise, set property for visit
        self._visit = v

    def _scaler_fit(self):
        """Private method to scale data based on scaler provided."""

        # scale numerical data and dummy data if it exists
        if self._len_num > 0:
            sc = clone(self.scaler)
            self._scaled_num = sc.fit(self._data_num.values)
        else:
            self._scaled_num = None
        if self._len_dum > 0:
            sc = clone(self.scaler)
            self._scaled_dum = sc.fit(self._data_dum.values)
        else:
            self._scaled_dum = None

    def _scaler_transform(self):
        """Private method to transform data using scaled fit."""
        if self._scaled_num:
            sn = self._scaled_num.transform(self._data_num.values)
            self._data_num = pd.DataFrame(sn, columns=self._cols_num)
        if self._scaled_dum:
            sd = self._scaled_dum.transform(self._data_dum.values)
            self._data_dum = pd.DataFrame(sd, columns=self._cols_dum)

    def _scaler_fit_transform(self):
        """Private method to perform fit and transform of scaler"""
        self._scaler_fit()
        self._scaler_transform()

    def _fit_init_params(self, column, method, kwgs):
        """Private method to supply imputation model fit params if any."""

        # first, handle easy case when no kwargs given
        if kwgs is None:
            final_params = kwgs

        # next, check if any kwargs for a given Imputer method type
        # then, override those parameters if specific column kwargs supplied
        if isinstance(kwgs, dict):
            initial_params = kwgs.get(method, None)
            final_params = kwgs.get(column, initial_params)

        # final params must be None or a dictionary of kwargs
        # this additional validation step is crucial to dictionary unpacking
        if not isinstance(final_params, (type(None), dict)):
            err = "Additional params must be dict of args used to init model."
            raise ValueError(err)
        return final_params

    def _check_if_single_dummy(self, col, X):
        """Private method to check if encoding results in single cat."""
        cats = X.columns.tolist()
        if len(cats) == 1:
            c = cats[0]
            msg = f"{c} only category for feature {col}."
            cons = f"Consider removing {col} from dataset."
            warnings.warn(f"{msg} {cons}")

    def _update_dataframes(self, X):
        """Private method to update processed dataframes."""

        # note that this method can be further optimized
        # numerical columns first
        self._data_num = X.select_dtypes(include=(np.number,))
        self._cols_num = self._data_num.columns.tolist()
        self._len_num = len(self._cols_num)

        # datetime columns next
        self._data_time = X.select_dtypes(include=(np.datetime64,))
        self._cols_time = self._data_time.columns.tolist()
        self._len_time = len(self._cols_time)

        # check categorical columns last
        # right now, only support for one-hot encoding
        orig_dum = X.select_dtypes(include=(np.object,))
        self._orig_dum = orig_dum.columns.tolist()
        if not orig_dum.columns.tolist():
            self._dum_dict = {}
            self._data_dum = pd.DataFrame()
        else:
            dummies = []
            self._dum_dict = {}
            self._data_dum = pd.DataFrame()
            for col in orig_dum:
                col_dum = pd.get_dummies(orig_dum[col], prefix=col)
                self._dum_dict[col] = col_dum.columns.tolist()
                self._check_if_single_dummy(col, col_dum)
                dummies.append(col_dum)
            ld = len(dummies)
            if ld == 1:
                self._data_dum = dummies[0]
            else:
                self._data_dum = pd.concat(dummies, axis=1)
        self._cols_dum = self._data_dum.columns.tolist()
        self._len_dum = len(self._cols_dum)

    def _prep_fit_dataframe(self, X):
        """Private method to process numeric & categorical data for fit."""
        self._X_idx = X.index
        self.data_mi = pd.isnull(X)*1
        if self.verbose:
            prep = "PREPPING DATAFRAME FOR IMPUTATION ANALYSIS..."
            print(f"{prep}\n{'-'*len(prep)}")

        # call the update, which sets initial columns for fitting
        self._update_dataframes(X)

        # print categorical and numeric columns if verbose true
        if self.verbose:
            nm = "Number of numeric columns in X: "
            cm = "Number of categorical columns after one-hot encoding: "
            print(f"{nm}{self._len_num}")
            print(f"{cm}{self._len_dum}")

    def _use_all_cols(self, c):
        """Private method to pedict using all columns."""

        # set numerical columns first
        if c in self._cols_num:
            num_cols = self._data_num.drop(c, axis=1)
        else:
            num_cols = self._data_num

        # set categorical columns second
        if c in self._orig_dum:
            d_c = [v for k, v in self._dum_dict.items() if k != c]
            d_fc = list(itertools.chain.from_iterable(d_c))
            d = [k for k in self._data_dum.columns if k in d_fc]
            dum_cols = self._data_dum[d]
        else:
            dum_cols = self._data_dum

        # return all predictors and target for predictor
        return num_cols, dum_cols, self._data_time

    def _use_iter_cols(self, c, preds):
        """Private method to predict using some columns."""

        # set numerical columns first
        if c in self._cols_num:
            cn = self._data_num.drop(c, axis=1)
        else:
            cn = self._data_num
        cols = list(set(preds).intersection(cn.columns.tolist()))
        num_cols = cn[cols]

        # set categorical columns second
        if c in self._orig_dum:
            d_c = [v for k, v in self._dum_dict.items()
                   if k != c and k in preds]
        else:
            d_c = [v for k, v in self._dum_dict.items()
                   if k in preds]
        d_fc = list(itertools.chain.from_iterable(d_c))
        d = [k for k in self._data_dum.columns
             if k in d_fc]
        dum_cols = self._data_dum[d]

        # set the time columns last
        ct = list(set(preds).intersection(self._data_time.columns.tolist()))
        time_cols = self._data_time[ct]

        return num_cols, dum_cols, time_cols

    def _prep_predictor_cols(self, c, predictors):
        """Private method to prep cols for prediction."""
        preds = predictors[c]
        if isinstance(preds, str):
            if preds == "all":
                if self.verbose:
                    print(f"No predictors given for {c}, using all columns.")
                num, dum, time = self._use_all_cols(c)
            else:
                if self.verbose:
                    print(f"Using single column {preds} to predict {c}.")
                num, dum, time = self._use_iter_cols(c, [preds])
        if isinstance(preds, (list, tuple)):
            if self.verbose:
                print(f"Using {preds} as covariates for {c}.")
            num, dum, time = self._use_iter_cols(c, preds)

        # error handling and printing to console
        predictors = [num, dum, time]
        predictor_str = list(map(lambda df: df.columns.tolist(), predictors))
        if not any(predictor_str):
            err = f"Need at least one predictor column to fit {c}."
            raise ValueError(err)
        if self.verbose:
            print(f"Columns used for {c}:")
            print(f"Numeric: {predictor_str[0]}")
            print(f"Categorical: {predictor_str[1]}")
            print(f"Datetime: {predictor_str[2]}")

        # final columns to return for x and y
        predictors = [p for p in predictors if p.size > 0]
        x = pd.concat(predictors, axis=1)
        y = self.data_mi[c]
        return x, y
