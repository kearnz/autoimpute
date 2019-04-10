"""Module for BaseImputer - a base class for classifiers/predictive imputers.

This module contains the `BaseImputer`, which is used to abstract away
functionality in both missingness classifiers and predictive imputers.
"""

import warnings
import itertools
import numpy as np
import pandas as pd
from sklearn.base import clone
from autoimpute.imputations import method_names
from ..series import DefaultSingleImputer, DefaultPredictiveImputer
from ..series import MeanImputer, MedianImputer, ModeImputer
from ..series import NormImputer, CategoricalImputer
from ..series import RandomImputer, InterpolateImputer
from ..series import LeastSquaresImputer, StochasticImputer, PMMImputer
from ..series import BinaryLogisticImputer, MultiLogisticImputer
from ..series import BayesLeastSquaresImputer, BayesBinaryLogisticImputer
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=too-many-arguments
# pylint:disable=too-many-instance-attributes
# pylint:disable=inconsistent-return-statements


class BaseImputer:
    """Building blocks for more advanced imputers and missingness classifiers.

    The `BaseImputer` is not a stand-alone class and thus serves no purpose
    other than as a Parent to Imputers and MissingnessClassifiers. Therefore,
    the BaseImputer should not be used directly unless creating an Imputer.
    That being said, all dataframe Imputers should inherit from BaseImputer.
    It contains base functionality for any new dataframe imputer, and it holds
    the set of strategies that make up this imputation library.

    Attributes:
        univariate_strategies (dict): univariate imputation methods.
            Key = imputation name; Value = function to perform imputation.
            `univariate default` mean for numerical, mode for categorical.
            `mean` imputes missing values with the average of the series.
            `median` imputes missing values with the median of the series.
            `mode` imputes missing values with the mode of the series.
                Method handles more than one mode (see ModeImputer for info).
            `random` imputes w/ random choice from set of Series unique vals.
            `norm` imputes series using random draws from normal distribution.
                Mean and std calculated from observed values of the Series.
            `categorical` imputes series using random draws from pmf.
                Proportions calculated from non-missing category instances.
            `interpolate` imputes series using chosen interpolation method.
                Default is linear. See InterpolateImputer for more info.
        predictive_strategies (dict): predictive imputation methods.
            Key = imputation name; Value = function to perform imputation.
            `default_pred` pmm for numerical, logistic for categorical.
            `least squares` predict missing values from linear regression.
            `binary logistic` predict missing values with 2 classes.
            `multinomial logistic` predict missing values with multiclass.
            `stochastic` linear regression + random draw from norm w/ mse std.
            `bayesian least squares` draw from the posterior predictive
                distribution for each missing value, using underlying OLS.
            `bayesian binary logistic` draw from the posterior predictive
                distribution for each missing value, using underling logistic.
            `pmm` imputes series using predictive mean matching. PMM is a
                semi-supervised method using bayesian & hot-deck imputation.
        strategies (dict): univariate and predictive strategies merged.
    """
    univariate_strategies = {
        methods.DEFAULT_UNIVAR: DefaultSingleImputer,
        methods.MEAN: MeanImputer,
        methods.MEDIAN: MedianImputer,
        methods.MODE:  ModeImputer,
        methods.RANDOM: RandomImputer,
        methods.NORM: NormImputer,
        methods.CATEGORICAL: CategoricalImputer,
        methods.INTERPOLATE: InterpolateImputer
    }

    predictive_strategies = {
        methods.DEFAULT_PRED: DefaultPredictiveImputer,
        methods.LS: LeastSquaresImputer,
        methods.STOCHASTIC: StochasticImputer,
        methods.BINARY_LOGISTIC: BinaryLogisticImputer,
        methods.MULTI_LOGISTIC: MultiLogisticImputer,
        methods.BAYESIAN_LS: BayesLeastSquaresImputer,
        methods.BAYESIAN_BINARY_LOGISTIC: BayesBinaryLogisticImputer,
        methods.PMM: PMMImputer
    }

    strategies = {**predictive_strategies, **univariate_strategies}

    def __init__(self, imp_kwgs, scaler, verbose):
        """Initialize the BaseImputer.

        Args:
            imp_kwgs (dict, optional): keyword arguments for each imputer.
                Default is None, which means default imputer created to match
                specific strategy. imp_kwgs keys can be either columns or
                strategies. If strategies, each column given that strategy is
                instantiated with same arguments.
            scaler (sklearn scaler, optional): A scaler supported by sklearn.
                Default to None. Otherwise, must be sklearn-compliant scaler.
            verbose (bool, optional): Print information to the console.
                Defaults to False.
        """
        self.imp_kwgs = imp_kwgs
        self.scaler = scaler
        self.verbose = verbose

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
