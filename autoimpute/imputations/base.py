"""Module for BaseImputer - a base class for classifier/predictive imputers.
This module contains the `BaseImputer`, which is used to abstract away
functionality in both missingness classifiers and predictive imputers.

Todo:
    * Determine how column specification fits in.
    * Determine how column "using" for specification fits in.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from autoimpute.utils.helpers import _nan_col_dropper
# pylint:disable=attribute-defined-outside-init

class BaseImputer(object):
    """Building blocks for more advanced imputers and missingness classifiers.

    The `BaseImputer` is not a stand-alone class and thus serves no purpose
    other than as a Parent to Imputers and MissingnessClassifiers.
    """

    def __init__(self, scaler=None, verbose=False):
        """Initialize the BaseImputer.

        Args:
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
            else:
                self._scaler = s

    def _check_if_single_dummy(self, col, X):
        """Private method to check if encoding results in single cat."""
        cats = X.columns.tolist()
        if len(cats) == 1:
            c = cats[0]
            msg = f"{c} only category for feature {col}."
            cons = f"Consider removing {col} from dataset."
            warnings.warn(f"{msg} {cons}")

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

    def _prep_dataframes(self, X):
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
