"""Module for BaseImputer - a base class for classifier/predictive imputers.
This module contains the `BaseImputer`, which is used to abstract away
functionality in both missingness classifiers and predictive imputers.

Todo:
    * Determine how column specification fits in.
    * Determine how column "using" for specification fits in.
"""

import warnings
import itertools
import numpy as np
import pandas as pd
from sklearn.base import clone
from autoimpute.utils.helpers import _nan_col_dropper
# pylint:disable=attribute-defined-outside-init
# pylint:disable=too-many-instance-attributes

class BaseImputer:
    """Building blocks for more advanced imputers and missingness classifiers.

    The `BaseImputer` is not a stand-alone class and thus serves no purpose
    other than as a Parent to Imputers and MissingnessClassifiers.
    """

    def __init__(self, scaler, verbose):
        """Initialize the BaseImputer.

        Args:
            strategy (str, iter, dict): strategies for imputation.
            scaler (sklearn scaler, optional): A scaler supported by sklearn.
                Defaults to None, so no scaling of fit/transform data.
            verbose (bool, optional): Print information to the console.
                Defaults to False.
        """
        self.scaler = scaler
        self.verbose = verbose

    @property
    def scaler(self):
        """Property getter to return the value of the scaler property."""
        return self._scaler

    @scaler.setter
    def scaler(self, s):
        """Validate the scaler property and set default parameters.

        Args:
            s (scaler): if None, implement the xgboost classifier

        Raises:
            ValueError: classifier does not implement `fit_transform`
        """
        if s is None:
            self._scaler = s
        else:
            m = "fit_transform"
            if not hasattr(s, m):
                raise ValueError(f"Scaler must implement {m} method.")
            self._scaler = s

    def _scaler_fit(self):
        """Private method to scale data based on scaler provided."""
        # if scaler used, must be from sklearn library
        if self._len_num > 0:
            sc = clone(self.scaler)
            self._scaled_num = sc.fit(self._data_num.values)
        if self._len_dum > 0:
            sc = clone(self.scaler)
            self._scaled_dum = sc.fit(self._data_dum.values)

    def _scaler_transform(self):
        """Private method to transform data using scaled fit."""
        if not self._scaled_num is None:
            cn = self._data_num.columns.tolist()
            sn = self._scaled_num.transform(self._data_num.values)
            self._data_num = pd.DataFrame(sn, columns=cn)
        if not self._scaled_dum is None:
            cd = self._data_dum.columns.tolist()
            sd = self._scaled_dum.transform(self._data_dum.values)
            self._data_dum = pd.DataFrame(sd, columns=cd)

    def check_strategy_allowed(self, strat_names, s):
        """Logic to determine if the strategy passed for imputation is valid.

        Imputer Classes in this library have a very flexible strategy argument.
        The arg can be a string, an iterator, or a dictionary. In each case,
        the method(s) passed are checked against method(s) allowed, which are
        generally stored in a class variable of the given Imputer.

        Args:
            strat_names (iterator): strategies allowed by the Imputer class
            strategy (any): strategies passed as arguments

        Returns:
            strategy (any): if string, iterator, or dictionary

        Raises:
            ValueError: Strategies not valid (not in allowed strategies).
            TypeError: Strategy must be a string, tuple, list, or dict.
        """
        err_op = f"Strategies must be one of {list(strat_names)}."
        if isinstance(s, str):
            if s not in strat_names:
                err = f"Strategy {s} not a valid imputation method.\n"
                raise ValueError(f"{err} {err_op}")
        elif isinstance(s, (list, tuple, dict)):
            if isinstance(s, dict):
                ss = set(s.values())
            else:
                ss = set(s)
            sdiff = ss.difference(strat_names)
            if sdiff:
                err = f"Strategies {sdiff} in {s} not valid imputation.\n"
                raise ValueError(f"{err} {err_op}")
        else:
            raise TypeError("Strategy must be string, tuple, list, or dict.")
        return s

    def check_strategy_fit(self, s, nc, o_cols, cols):
        """Check whether strategies of imputer make sense given data passed.

        An Imputer takes strategies to use for imputation. Those strategies
        are validated when an instance is created. When fitting actual data,
        strategies must be validated again to verify they make sense given
        the columns in the dataset passed. For example, "mean" is fine
        when instance created, but "mean" will not work for a categorical
        column. This check validates strategy used for given column each
        strategy assigned to.

        Args:
            strategy (str, iter, dict): strategies passed for columns.
                String = 1 strategy, broadcast to all columns.
                Iter = multiple strategies, must match col index and length.
                Dict = multiple strategies, must match col name, but not all
                columns are mandatory. Will simply impute based on name.
            nc(set): any columns removed because they are fully missing.
            o_cols: original columns before nc determined.
            cols: columns remaining after nc determined.

        Raises:
            ValueError (iter): length of columns and strategies must be equal.
            ValueError (dict): keys of strategies and columns must match.
        """
        o_l = len(o_cols)
        # if strategy is string, extend strategy to all cols
        if isinstance(s, str):
            return {c:s for c in cols}

        # if list or tuple, ensure same number of cols in X as strategies
        # note that list/tuple must have strategy specified for every column
        if isinstance(s, (list, tuple)):
            s_l = len(s)
            if s_l != o_l:
                err = f"Original columns ({o_l}) must equal strategies ({s_l})"
                raise ValueError(err)
            if nc:
                i = 0
                for ind, name in enumerate(o_cols):
                    if name in nc:
                        del s[ind-i]
                        i += 1
            return {c[0]:c[1] for c in zip(cols, s)}

        # if strategy is dict, ensure keys in strategy match cols in X
        # note that dict is preferred way to impute SOME columns and not all
        if isinstance(s, dict):
            if nc:
                for k in nc:
                    s.pop(k, None)
            diff_s = set(s.keys()).difference(cols)
            if diff_s:
                err = f"Keys of strategies and column names must match."
                raise ValueError(err)
            return s

    def check_predictors_fit(self, predictors, nc, o_cols, cols):
        """Checked predictors used for fitting each column.

        Args:
            predictors (str, iter, dict): predictors passed for columns.
                String = "all" or raises error
                Iter = multiple strategies, must match col index and length.
                Dict = multiple strategies, must match col name, but not all
                columns are mandatory. Will simply impute based on name.
            nc(set): any columns removed because they are fully missing.
            o_cols: original columns before nc determined.
            cols: columns remaining after nc determined.

        Returns:
            predictors

        Raises:
            ValueError (str): string not equal to all.
            ValueError (iter): items in `predictors` not in columns of X.
            ValueError (dict, keys): keys of response must be columns in X.
            ValueError (dict, vals): vals of responses must be columns in X.
        """
        # if string, value must be `all`, or else raise an error
        if isinstance(predictors, str):
            if predictors != "all":
                err = f"Must pass list or dict unless using all columns."
                raise ValueError(err)
            return {c:predictors for c in cols}

        # if list or tuple, remove nan cols and check col names
        if isinstance(predictors, (list, tuple)):
            for i, pred in enumerate(predictors):
                if pred not in o_cols:
                    err = f"{pred} in predictors not a valid column in X."
                    raise ValueError(err)
                if pred in nc:
                    predictors.pop(i)
            return {c:predictors for c in cols}

        # if dictionary, remove nan cols and check col names
        if isinstance(predictors, dict):
            # check the keys first...
            if nc:
                for k in nc:
                    predictors.pop(k, None)
            diff_s = set(predictors.keys()).difference(cols)
            if diff_s:
                err = "Keys of predictors and column names must match."
                raise ValueError(err)
            # then check the values of each key
            for k, preds in predictors.items():
                if isinstance(preds, str):
                    if preds != "all":
                        if preds in nc or preds not in cols:
                            err = f"Invalid column as only predictor for {k}."
                            raise ValueError(err)
                elif isinstance(preds, (tuple, list)):
                    predictor_vals = []
                    for i, p in enumerate(preds):
                        if p not in cols:
                            err = f"Invalid column as predictor for {k}."
                            raise ValueError(err)
                        if p in nc:
                            preds.pop(i)
                        else:
                            predictor_vals.append(p)
                    p_len = len(predictor_vals)
                    if not p_len:
                        err = f"All predictor columns for {k} are invalid."
                        raise ValueError(err)
                else:
                    err = "Values in predictor must be str, list, or tuple."
                    raise ValueError(err)
            # finally, create predictors dict
            for c in cols:
                if c not in predictors:
                    predictors[c] = "all"
            return predictors

    def _check_if_single_dummy(self, col, X):
        """Private method to check if encoding results in single cat."""
        cats = X.columns.tolist()
        if len(cats) == 1:
            c = cats[0]
            msg = f"{c} only category for feature {col}."
            cons = f"Consider removing {col} from dataset."
            warnings.warn(f"{msg} {cons}")

    def _prep_fit_dataframe(self, X):
        """Private method to process numeric & categorical data for fit."""
        X, self._nc = _nan_col_dropper(X)
        self.data_mi = pd.isnull(X)*1
        self._data_num = X.select_dtypes(include=(np.number,))
        self._len_num = len(self._data_num.columns)

        # right now, only support for one-hot encoding
        orig_dum = X.select_dtypes(include=(np.object,))
        if not orig_dum.columns.tolist():
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
        self._len_dum = len(self._data_dum.columns)

        # print categorical and numeric columns if verbose true
        if self.verbose:
            print(f"Number of numeric columns: {self._len_num}")
            print(f"Number of categorical columns: {self._len_dum}")

    def _use_all_cols(self, X, i, c):
        """Private method to pedict using all columns."""
        # dealing with a numeric column...
        if X[c].dtype == np.number:
            if self._len_num > 1:
                num_cols = self._data_num.drop(c, axis=1)
                num_str = num_cols.columns.tolist()
                if self._len_dum > 0:
                    dummy_str = self._data_dum.columns.tolist()
                    cl = [num_cols.values, self._data_dum.values]
                    x = np.concatenate(cl, axis=1)
                else:
                    dummy_str = None
                    x = num_cols.values
            else:
                num_str = None
                if self._len_dum > 0:
                    dummy_str = self._data_dum.columns.tolist()
                    x = self._data_dum.values
                else:
                    raise ValueError("Need at least one predictor column.")
            if self.verbose:
                print(f"Columns used for {i} - {c}:")
                print(f"Numeric: {num_str}")
                print(f"Categorical: {dummy_str}")

        # dealing with categorical columns...
        else:
            d_c = [v for k, v in self._dum_dict.items() if k != c]
            d_fc = list(itertools.chain.from_iterable(d_c))
            d = [k for k in self._data_dum.columns if k in d_fc]
            len_d = len(d)
            if len_d > 0:
                dummy_cols = self._data_dum[d].values
                dummy_str = self._data_dum[d].columns.tolist()
                if self._len_num > 0:
                    num_str = self._data_num.columns.tolist()
                    cl = [self._data_num.values, dummy_cols]
                    x = np.concatenate(cl, axis=1)
                else:
                    num_str = None
                    x = dummy_cols
            else:
                dummy_str = None
                if self._len_num > 0:
                    num_str = self._data_num.columns.tolist()
                    x = self._data_num.values
                else:
                    raise ValueError("Need at least one predictor column.")
            if self.verbose:
                print(f"Columns used for {i} - {c}:")
                print(f"Numeric: {num_str}")
                print(f"Categorical: {dummy_str}")

        # return all predictors and target for predictor
        y = self.data_mi[c].values
        return x, y

    def _prep_cols(self, X, i, c, preds):
        """Private method to prep cols for prediction."""
        if preds == "all":
            if self.verbose:
                print("No predictors specified, using all available.")
            return self._use_all_cols(X, i, c)
