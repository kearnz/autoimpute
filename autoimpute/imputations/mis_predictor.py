"""MissingnessPredictor Class used to generate test sets"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from autoimpute.imputations.predictors import xgb_model
from autoimpute.utils.checks import check_missingness

class MissingnessPredictor(BaseEstimator, TransformerMixin):
    """
    Predicts the likelihood of missingness for a given dataset
    Default method uses xgboost, although other predictors are supported
    """
    def __init__(self, predictor=xgb_model, scaler=None, verbose=False):
        """Create an instance of the MissingnessPredictor"""
        self.predictor = predictor
        self.scaler = scaler
        self.verbose = verbose
        self.fit_ = False
        self.data_mi = None
        self.data_numeric = None
        self.data_dummy = None
        self.preds_df = None

    def _vprint(self, statement):
        """Printer for verbosity"""
        if self.verbose:
            print(statement)

    @check_missingness
    def fit(self, X):
        """Get everything that the transform step needs to make predictions"""
        self.data_mi = pd.isnull(X)*1
        self.data_numeric = X[[col for col in X if X[col].dtype
                               in (np.dtype('int64'), np.dtype('float64'))]]

        # right now, only support for one-hot encoding
        dummies = [pd.get_dummies(X[col], prefix=col)
                   for col in X if X[col].dtype == np.dtype('object')]
        len_numeric = len(self.data_numeric.columns)
        len_dummies = len(dummies)
        if len_dummies == 0:
            self.data_dummy = pd.DataFrame()
        elif len_dummies == 1:
            self.data_dummy = dummies[0]
        else:
            self.data_dummy = pd.concat(dummies, axis=1)

        # if scaler used, must be from sklearn library
        if self.scaler is not None:
            if hasattr(self.scaler, "fit_transform"):
                if len_numeric > 0:
                    cn = self.data_numeric.columns.tolist()
                    dn = self.scaler.fit_transform(self.data_numeric.values)
                    self.data_numeric = pd.DataFrame(dn, columns=cn)
                if len_dummies > 0:
                    cd = self.data_dummy.columns.tolist()
                    dd = self.scaler.fit_transform(self.data_dummy.values)
                    self.data_dummy = pd.DataFrame(dd, columns=cd)
            else:
                raise ValueError("Scalers must be from scikit-learn.")
        self.fit_ = True
        self._vprint(f"Number of numeric columns: {len_numeric}")
        self._vprint(f"Number of categorical columns: {len_dummies}")
        return self

    @check_missingness
    def transform(self, X):
        """Transform values and predict missingness"""
        if not self.fit_:
            raise ValueError("Need to fit data first before transformation.")
        preds_mi = []
        len_numeric = len(self.data_numeric.columns)
        len_dummies = len(self.data_dummy.columns)

        # iterative missingness predictor using all remaining columns
        for i, c in enumerate(self.data_mi):
            # dealing with a numeric column...
            if X[c].dtype != np.dtype('object'):
                # if more than 1 numeric column...
                if len_numeric > 1:
                    # drop the current column of interest...
                    num_cols = self.data_numeric.drop(c, axis=1)
                    num_str = num_cols.columns.tolist()
                    # concat values from cat cols or use just numerical
                    if len_dummies > 1:
                        dummy_str = self.data_dummy.columns.tolist()
                        cl = [num_cols.values, self.data_dummy.values]
                        x = np.concatenate(cl, axis=1)
                    else:
                        dummy_str = None
                        x = num_cols.values
                # if only 1 or no numeric columns...
                else:
                    num_str = None
                    # use categorical columns or throw error
                    if len_dummies > 1:
                        dummy_str = self.data_dummy.columns.tolist()
                        x = self.data_dummy.values
                    else:
                        raise ValueError("Need at least one predictor column.")
                self._vprint(f"Columns used for {i} - {c}:")
                self._vprint(f"Numeric: {num_str}")
                self._vprint(f"Categorical: {dummy_str}")
            # dealing with categorical columns
            else:
                d = [k for k in self.data_dummy.columns if not k.startswith(c)]
                len_d = len(d)
                # and that dummy is not y...
                if len_d > 0:
                    dummy_cols = self.data_dummy[d].values
                    dummy_str = self.data_dummy[d].columns.tolist()
                    # check if any numeric columns...
                    if len_numeric > 0:
                        num_str = self.data_numeric.columns.tolist()
                        cl = [self.data_numeric.values, dummy_cols]
                        x = np.concatenate(cl, axis=1)
                    else:
                        num_str = None
                        x = dummy_cols
                else:
                    dummy_str = None
                    if len_numeric > 0:
                        num_str = self.data_numeric.columns.tolist()
                        x = self.data_numeric.values
                    else:
                        raise ValueError("Need at least one predictor column.")
                self._vprint(f"Columns used for {i} - {c}:")
                self._vprint(f"Numeric: {num_str}")
                self._vprint(f"Categorical: {dummy_str}")
            # target for predictor
            y = self.data_mi[c].values
            # make predictions and append to list for pred df
            preds = self.predictor(x, y)
            preds_mi.append(preds)

        # preparing final dataframe after transformation
        preds_mi = np.array(preds_mi).T
        self.data_mi.columns = [f"{c}_mis" for c in X.columns]
        pred_cols = [f"{c}_pred" for c in self.data_mi.columns]
        self.preds_df = pd.DataFrame(preds_mi, columns=pred_cols)
        return self

    def fit_transform(self, X):
        """Convenience method for fit and transformation"""
        return self.fit(X).transform(X)
