"""Module to predict missingness in a dataset and generate test cases.

This module contains the MissingnessClassifier, which is used to predict
missingness within a dataset using information derived from other features.

Todo:
    * Allow for flexible column specification used in predictor.
    * Alllow for basic imputation methods before classification.
    * Update docstrings for class initialization and instance methods.
    * Add examples of proper usage for the class and its instance methods.
"""

import warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from autoimpute.utils.checks import check_missingness
from autoimpute.utils.helpers import _nan_col_dropper
# pylint:disable=attribute-defined-outside-init
# pylint:disable=arguments-differ

class MissingnessClassifier(BaseEstimator, ClassifierMixin):
    """Classify values as missing or not, based on missingness patterns.

    The class has has numerous use cases. First, it fits columns of a DataFrame
    and predicts whether or not an observation is missing, based on all
    available information in other columns. The class supports both class
    prediction and class probabilities.

    Second, the class can generate test cases for imputation analysis. Test
    cases are values that are truly observed but have a high probability of
    being missing. These cases make imputation process supervised as opposed
    to unsupervised. A user never knows the true value of missing data but can
    verify imputation methods on test cases for which the true value is known.
    """
    def __init__(self, classifier=None, scaler=None, verbose=False):
        """Create an instance of the MissingnessClassifier.

        Args:
            classifier (classifier, optional): valid classifier from sklearn.
                If None, default is xgboost. Note that classifier must
                conform to sklearn style. This means it must implement the
                `predict_proba` method and act as a porper classifier.
            scaler (scaler, optional): valid scaler from sklearn.
                If None, default is None. Note that scaler must conform to
                sklearn style. This means it must implement the `transform`
                method and act as a proper scaler.
            verbose (bool, optional): print information to the console.
                Default is False.
        """
        self.classifier = classifier
        self.scaler = scaler
        self.verbose = verbose

    @property
    def classifier(self):
        """Property getter to return the value of the classifier property"""
        return self._classifier

    @property
    def scaler(self):
        """Property getter to return the value of the scaler property"""
        return self._scaler

    @classifier.setter
    def classifier(self, c):
        """Validate the classifier property and set default parameters.

        Args:
            c (classifier): if None, implement the xgboost classifier

        Raises:
            ValueError: classifier does not implement `predict_proba`
        """
        if c is None:
            self._classifier = XGBClassifier()
        else:
            m = "predict_proba"
            if not hasattr(c, m):
                raise ValueError(f"Classifier must implement {m} method.")
            else:
                self._classifier = c

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

    def _check_if_single_dummy(self, X):
        """Private method to check if encoding results in single cat."""
        cats = X.columns.tolist()
        if len(cats) == 1:
            c = cats[0]
            cf = c.split('_')[0]
            msg = f"{c} only category for feature {cf}."
            cons = f"Consider removing {cf} from dataset."
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
        dummies = [pd.get_dummies(X[col], prefix=col)
                   for col in X.select_dtypes(include=(np.object,))]
        ld = len(dummies)
        if ld == 0:
            self._data_dum = pd.DataFrame()
        elif ld == 1:
            self._data_dum = dummies[0]
            self._check_if_single_dummy(self._data_dum)
        else:
            self._data_dum = pd.concat(dummies, axis=1)
            for each_dum in dummies:
                self._check_if_single_dummy(each_dum)
        self._len_dum = len(self._data_dum.columns)

        # print categorical and numeric columns if verbose true
        if self.verbose:
            print(f"Number of numeric columns: {self._len_num}")
            print(f"Number of categorical columns: {self._len_dum}")

    def _prep_classifier_cols(self, X, i, c):
        """Private method to perpare the data for each classifier."""
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
            d = [k for k in self._data_dum.columns
                 if not k.startswith(f"{c}_")]
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

    def _prep_predictor(self, X, new_data):
        """Private method to prep for prediction."""
        # initial checks before transformation
        check_is_fitted(self, 'statistics_')

        # remove columns in transform if they were removed in fit
        if self._nc:
            wrn = f"{self._nc} dropped in transform since they were not fit."
            warnings.warn(wrn)
            X.drop(self._nc, axis=1, inplace=True)

        # check dataset features are the same for both fit and transform
        X_cols = X.columns.tolist()
        mi_cols = self.data_mi.columns.tolist()
        diff_X = set(X_cols).difference(mi_cols)
        diff_mi = set(mi_cols).difference(X_cols)
        if diff_X or diff_mi:
            raise ValueError("Same columns must appear in fit and predict.")

        # if not error, check if new data
        if new_data:
            self._prep_dataframes(X)
        if not self.scaler is None:
            self._scaler_transform()

    @check_missingness
    def fit(self, X, **kwargs):
        """Fit an individual classifier for each column in the DataFrame.

        For each feature in the DataFrame, a classifier (default: xgboost) is
        fit with the feature as the response (y) and all other features as
        covariates (X). The resulting classifiers are stored in the class
        instance statistics. One `fit` for each column in the dataset. Column
        specification will be supported as well.

        Args:
            X (pd.DataFrame): DataFrame on which to fit classifiers
            **kwargs: keyword arguments used by classifiers

        Returns:
            self: instance of MissingnessClassifier
        """
        self._prep_dataframes(X)
        self.statistics_ = {}
        if not self.scaler is None:
            self._scaler_fit()
        if self.verbose:
            print("FITTING...")
        # iterate missingness fit using classifier and all remaining columns
        for i, c in enumerate(self.data_mi):
            x, y = self._prep_classifier_cols(X, i, c)
            clf = clone(self.classifier)
            cls_fit = clf.fit(x, y, **kwargs)
            self.statistics_[c] = cls_fit
        return self

    @check_missingness
    def predict(self, X, new_data=True, **kwargs):
        """Predict class of each feature. 1 for missing; 0 for not missing.

        First checks to ensure data has been fit. If fit, `predict` method
        uses the respective classifier of each feature (stored in statistics)
        and predicts class membership for each observation of each feature.
        1 = missing; 0 = not missing. Prediction is binary, as class membership
        is hard. If probability deesired, use `predict_proba` method.

        Args:
            X (pd.DataFrame): DataFrame used to create predictions.
            new_data (bool, optional): whether or not new data is used.
                Default is False.
            kwargs: kewword arguments. Used by the classifer.

        Returns:
            pd.DataFrame: DataFrame with class prediction for each observation.
        """
        # predictions for each column using respective fit classifier
        self._prep_predictor(X, new_data)
        if self.verbose:
            print("PREDICTING CLASS MEMBERSHIP...")
        preds_mat = []
        for i, c in enumerate(self.data_mi):
            x, _ = self._prep_classifier_cols(X, i, c)
            cls_fit = self.statistics_[c]
            y_pred = cls_fit.predict(x, **kwargs)
            preds_mat.append(y_pred)

        # store the predictor matrix class membership as a dataframe
        preds_mat = np.array(preds_mat).T
        pred_cols = [f"{cl}_pred" for cl in X.columns]
        self.data_mi_preds = pd.DataFrame(preds_mat, columns=pred_cols)
        return self.data_mi_preds

    @check_missingness
    def predict_proba(self, X, new_data=True, **kwargs):
        """Predict probability of missing class membership of each feature.

        First checks to ensure data has been fit. If fit, `predict_proba`
        method uses the respsective classifier of each feature (in statistics)
        and predicts probability of missing class membership for each
        observation of each feature. Prediction is probability of missing.
        Therefore, probability of not missing is 1-P(missing). For hard class
        membership prediction, use `predict`.

        Args:
            X (pd.DataFrame): DataFrame used to create probabilities.
            new_data (bool, Optional): whether or not new data is used.
                Default is False.

        Returns:
            pd.DataFrame: DataFrame with probability of missing class for
                each observation.
        """
        self._prep_predictor(X, new_data)
        if self.verbose:
            print("PREDICTING CLASS PROBABILITY...")
        preds_mat = []
        for i, c in enumerate(self.data_mi):
            x, _ = self._prep_classifier_cols(X, i, c)
            cls_fit = self.statistics_[c]
            y_pred = cls_fit.predict_proba(x, **kwargs)[:, 1]
            preds_mat.append(y_pred)

        # store the predictor matrix probabilities as a dataframe
        preds_mat = np.array(preds_mat).T
        pred_cols = [f"{cl}_pred" for cl in X.columns]
        self.data_mi_proba = pd.DataFrame(preds_mat, columns=pred_cols)
        return self.data_mi_proba

    def fit_predict(self, X, new_data=False):
        """Convenience method for fit and class prediction.

        Args:
            X (pd.DataFrame): DataFrame to fit classifier and predict class.
            new_data (bool, optional): Whether or not new data used.
                Default is False.

        Returns:
            pd.DataFrame: DataFrame of class predictions.
        """
        return self.fit(X).predict(X, new_data)

    def fit_predict_proba(self, X, new_data=False):
        """Convenience method for fit and class probability prediction.

        Args:
            X (pd.DataFrame): DataFrame to fit classifier and prredict prob.
            new_data (bool, optional): Whether or not new data used.
                Default is False.

        Returns:
            pd.DataFrame: DataFrame of class probability predictions.
        """
        return self.fit(X).predict_proba(X, new_data)

    @check_missingness
    def gen_test_indices(self, X, thresh=0.5, new_data=False, use_exist=False):
        """Generate indices of false positives for each fitted column.

        Method generates the locations (indices) of false positives returned
        from classifiers. These are instances that have a high probability of
        being missing even though true value is observed. Use this method to
        get indices without mutating the actual DataFrame. To set the values
        to missing for the actual DataFrame, use `gen_test_df`.

        Args:
            X (pd.DataFrame): DataFrame from which test indices generated.
                Data first goes through `fit_predict_proba`.
            thresh (float, optional): Threshhold for generating false positive.
                If raw value is observed and P(missing) >= thresh, then the
                observation is considered a false positive and index is stored.
            new_data (bool, optional): Whether or not new data is used.
                Default is False.
            use_exist (bool, optional): Whether or not to use existing fit and
                classifiers. Default is False.

        Returns:
            self: test_indice available from `self.test_indices`
        """
        # ALWAYS fit_transform with dataset, as test vals can change
        self.test_indices = {}
        if not use_exist:
            self.fit_predict_proba(X, new_data)

        # loop through missing data indicators, eval new set for missing
        for c in self.data_mi:
            mi_c = self.data_mi[c]
            not_mi = mi_c[mi_c == 0].index
            pred_not_mi = self.data_mi_proba.loc[not_mi, f"{c}_pred"]
            pred_wrong = pred_not_mi[pred_not_mi > thresh].index
            self.test_indices[c] = pred_wrong
            if self.verbose:
                print(f"Test indices for {c}:\n{pred_wrong.values.tolist()}")
        return self

    def gen_test_df(self, X, thresh=0.5, m=0.05, inplace=False,
                    new_data=False, use_exist=False):
        """Generate new DatFrame with value of false positives set to missing.

        Method generates new DataFrame with the locations (indices) of false
        positives set to missing. Utilizes `gen_test_indices` to detect index
        of false positives.

        Args:
            X (pd.DataFrame): DataFrame from which test indices generated.
                Data first goes through `fit_predict_proba`.
            thresh (float, optional): Threshhold for generating false positive.
                If raw value is observed and P(missing) >= thresh, then the
                observation is considered a false positive and index is stored.
            m (float, optional): % false positive threshhold for warning.
                If % <= m, issue warning with % of test cases.
            new_data (bool, optional): Whether or not new data is used.
                Default is False.
            use_exist (bool, optional): Whether or not to use existing fit and
                classifiers. Default is False.

        Returns:
            pd.DataFrame: DataFrame with false positives set to missing.
        """
        if not inplace:
            X = X.copy()

        self.gen_test_indices(X, thresh, new_data, use_exist)
        min_num = np.floor(m*len(X.index))
        for c in X:
            ix_ = self.test_indices[c]
            if len(ix_) <= min_num:
                w = f"Fewer than {m*100}% set to missing for {c}"
                warnings.warn(w)
            if X[c].dtype == np.number:
                X.loc[ix_, c] = np.nan
            else:
                X.loc[ix_, c] = None
        return X
