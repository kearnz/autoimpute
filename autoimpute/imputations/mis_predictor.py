"""MissingnessPredictor Class used to generate test sets"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from .predictors import xgb_model

class MissingnessPredictor(BaseEstimator, TransformerMixin):
    """
    Predicts the likelihood of missingness for a given dataset
    Default method uses xgboost, although other predictors are supported
    """
    def __init__(self, predictor=xgb_model, verbose=False):
        """Create an instance of the MissingnessPredictor"""
        self.predictor = predictor
        self.verbose = verbose
        self.data_mi = None
        self.data_numeric = None
        self.data_dummy = None
        self.preds_df = None

    def fit(self, X):
        """Get everything that the transform step needs to make predictions"""
        self.data_mi = pd.isnull(X)*1
        self.data_numeric = X[[col for col in X if X[col].dtype
                               in (np.dtype('int64'), np.dtype('float64'))]]
        dummies = [pd.get_dummies(X[col], prefix=col)
                   for col in X if X[col].dtype == np.dtype('object')]
        self.data_dummy = pd.concat(dummies, axis=1)
        return self

    def transform(self, X):
        """Transform method for the MissingnessPredictor Class"""
        preds_mi = []
        for i, c in enumerate(self.data_mi):
            if X[c].dtype != np.dtype('object'):
                if self.verbose:
                    num_ = self.data_numeric.drop(c, axis=1).columns.tolist()
                    print(f"Columns used for {i} - {c}:")
                    print(f"Numeric: {num_}")
                    print(f"Dummy: {self.data_dummy.columns.tolist()}")
                x = np.concatenate([self.data_numeric.drop(c, axis=1).values,
                                    self.data_dummy.values], axis=1)
            else:
                d = [k for k in self.data_dummy.columns if not k.startswith(c)]
                if self.verbose:
                    print(f"Columns used for {i} - {c}:")
                    print(f"Numeric: {self.data_numeric.columns.tolist()}")
                    print(f"Dummy: {self.data_dummy[d].columns.tolist()}")
                x = np.concatenate([self.data_numeric.values,
                                    self.data_dummy[d].values], axis=1)
            y = self.data_mi[c].values
            preds = self.predictor(x, y)
            preds_mi.append(preds)
        preds_mi = np.array(preds_mi).T
        self.data_mi.columns = [f"{c}_mis" for c in X.columns]
        pred_cols = [f"{c}_pred" for c in self.data_mi.columns]
        self.preds_df = pd.DataFrame(preds_mi, columns=pred_cols)
        return self

    def fit_transform(self, X):
        """convenience method to fit and transform"""
        return self.fit(X).transform(X)
