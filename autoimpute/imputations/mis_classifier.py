"""MissingnessClassifier Class used to generate test sets"""

import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from autoimpute.utils.checks import check_missingness

class MissingnessClassifier(BaseEstimator, TransformerMixin):
    """
    Predicts the likelihood of missingness for a given dataset
    Default method uses xgboost, although other predictors are supported
    """
    def __init__(self, classifier=None, scaler=None, verbose=False):
        """Create an instance of the MissingnessPredictor"""
        self.classifier = classifier
        self.scaler = scaler
        self.verbose = verbose
        self.fit_ = False
        self.data_mi = None
        self.data_mi_preds = None
        self.data_numeric = None
        self.data_dummy = None
        self.single_dummy = []
        self.preds_mi_fit = {}

    @property
    def classifier(self):
        """return the classifier property"""
        return self._classifier

    @property
    def scaler(self):
        """return the scaler property"""
        return self._scaler

    @classifier.setter
    def classifier(self, c):
        """Validate the classifier property and set default param"""
        if c is None:
            self._classifier = XGBClassifier()
        else:
            m = "predict_proba"
            if not hasattr(c, m):
                raise TypeError(f"Classifier must implement {m} method.")
            else:
                self._classifier = c

    @scaler.setter
    def scaler(self, s):
        """Validate the scaler property and confirm default param"""
        if s is None:
            self._scaler = s
        else:
            m = "fit_transform"
            if not hasattr(s, m):
                raise TypeError(f"Scaler must implement {m} method.")
            else:
                self._scaler = s

    def _vprint(self, statement):
        """Printer for verbosity"""
        if self.verbose:
            print(statement)

    def _single_dummy(self, X):
        """Detect if single category present for a one-hot enocded feature"""
        cats = X.columns.tolist()
        if len(cats) == 1:
            c = cats[0]
            cf = c.split('_')[0]
            self.single_dummy.append(c)
            msg = f"{c} only category for feature {cf}."
            cons = f"Consider removing {cf} from dataset."
            warnings.warn(f"{msg} {cons}")

    def _prep_dataframes(self, X):
        """Prepare numeric and categorical data for fit method"""
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
            self._single_dummy(dummies[0])
        else:
            for each_dummy in dummies:
                self._single_dummy(each_dummy)
            self.data_dummy = pd.concat(dummies, axis=1)

        # if scaler used, must be from sklearn library
        if not self.scaler is None:
            if len_numeric > 0:
                sc = clone(self.scaler)
                cn = self.data_numeric.columns.tolist()
                dn = sc.fit_transform(self.data_numeric.values)
                self.data_numeric = pd.DataFrame(dn, columns=cn)
            if len_dummies > 0:
                sc = clone(self.scaler)
                cd = self.data_dummy.columns.tolist()
                dd = sc.fit_transform(self.data_dummy.values)
                self.data_dummy = pd.DataFrame(dd, columns=cd)

        # print categorical and numeric columns if verbose true
        self._vprint(f"Number of numeric columns: {len_numeric}")
        self._vprint(f"Number of categorical columns: {len_dummies}")

    def _prep_classifier_cols(self, X, i, c):
        """perpare the data for each classifier"""
        # get dataframe lengths for numerical and categorical
        len_numeric = len(self.data_numeric.columns)
        len_dummies = len(self.data_dummy.columns)

        # dealing with a numeric column...
        if X[c].dtype != np.dtype('object'):
            # if more than 1 numeric column...
            if len_numeric > 1:
                # drop the current column of interest...
                num_cols = self.data_numeric.drop(c, axis=1)
                num_str = num_cols.columns.tolist()
                # concat values from cat cols or use just numerical
                if len_dummies > 0:
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
                if len_dummies > 0:
                    dummy_str = self.data_dummy.columns.tolist()
                    x = self.data_dummy.values
                else:
                    raise ValueError("Need at least one predictor column.")
            self._vprint(f"Columns used for {i} - {c}:")
            self._vprint(f"Numeric: {num_str}")
            self._vprint(f"Categorical: {dummy_str}")
        # dealing with categorical columns
        else:
            d = [k for k in self.data_dummy.columns
                 if not k.startswith(f"{c}_")]
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
        return x, y

    @check_missingness
    def fit(self, X):
        """Get everything that the transform step needs to make predictions"""
        self._prep_dataframes(X)
        # iterative missingness predictor using all remaining columns
        for i, c in enumerate(self.data_mi):
            x, y = self._prep_classifier_cols(X, i, c)
            # make predictions and append to list for pred df
            clf = clone(self.classifier)
            cls_fit = clf.fit(x, y)
            self.preds_mi_fit[c] = cls_fit
        self.fit_ = True
        return self

    @check_missingness
    def transform(self, X):
        """Transform a dataset with the same columns"""
        if not self.fit_:
            return ValueError("Must fit a dataset before transforming.")
        X_cols = X.columns.tolist()
        mi_cols = self.data_mi.columns.tolist()
        diff_X = set(X_cols).difference(mi_cols)
        diff_mi = set(mi_cols).difference(X_cols)
        if not diff_X and not diff_mi:
            preds_mat = []
            for i, c in enumerate(self.data_mi):
                x, _ = self._prep_classifier_cols(X, i, c)
                cls_fit = self.preds_mi_fit[c]
                y_pred = cls_fit.predict_proba(x)[:, 1]
                preds_mat.append(y_pred)
            preds_mat = np.array(preds_mat).T
            pred_cols = [f"{cl}_pred" for cl in X.columns]
            self.data_mi_preds = pd.DataFrame(preds_mat, columns=pred_cols)
        else:
            raise ValueError("Same columns must appear in fit and transform.")
        return self

    def fit_transform(self, X):
        """Convenience method for fit and transformation"""
        return self.fit(X).transform(X)
