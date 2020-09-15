"""Module to predict missingness in data and generate imputation test cases.

This module contains the MissingnessClassifier, which is used to predict
missingness within a dataset using information derived from other features.
The MissingnessClassifier also generates test cases for imputation. Often,
we do not and will never have the true value of a missing data point,
so its challenging to validate an imputation model's performance.
The MissingnessClassifer generates missing "test" samples from observed
that have high likelihood of being missing, which a user can then "impute".
This practice is useful to validate models that contain truly missing data.
"""

import warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from autoimpute.utils import check_nan_columns, check_predictors_fit

# pylint:disable=attribute-defined-outside-init
# pylint:disable=arguments-differ
# pylint:disable=too-many-arguments
# pylint:disable=too-many-instance-attributes

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
    def __init__(self, classifier=None, predictors="all"):
        """Create an instance of the MissingnessClassifier.

        The MissingnessClassifier inherits from sklearn BaseEstimator and
        ClassifierMixin. This inheritence and this class' implementation
        ensure that the MissingnessClassifier is a valid classifier that will
        work in an sklearn pipeline.

        Args:
            classifier (classifier, optional): valid classifier from sklearn.
                If None, default is xgboost. Note that classifier must
                conform to sklearn style. This means it must implement the
                `predict_proba` method and act as a porper classifier.
            predictors (str, iter, dict, optiona): defaults to all, i.e.
                use all predictors. If all, every column will be used for
                every class prediction. If a list, subset of columns used for
                all predictions. If a dict, specify which columns to use as
                predictors for each imputation. Columns not specified in dict
                will receive `all` by default.
        """
        self.classifier = classifier
        self.predictors = predictors

    @property
    def classifier(self):
        """Property getter to return the value of the classifier property"""
        return self._classifier

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
            self._classifier = c

    def _fit_strategy_validator(self, X):
        """Internal helper method to validate behavior appropriate for fit."""

        # remove nan columns and store colnames
        cols = X.columns.tolist()
        self._preds = check_predictors_fit(self.predictors, cols)

        # next, prep the categorical / numerical split
        # only necessary for classes that use other features
        # wont see this requirement in the single imputer
        self.data_mi = X.isnull().astype(int)

    def _predictor_strategy_validator(self, X):
        """Private method to prep for prediction."""

        # initial checks before transformation
        check_is_fitted(self, "statistics_")

        # check dataset features are the same for both fit and transform
        X_cols = X.columns.tolist()
        mi_cols = self.data_mi.columns.tolist()
        diff_X = set(X_cols).difference(mi_cols)
        diff_mi = set(mi_cols).difference(X_cols)
        if diff_X or diff_mi:
            raise ValueError("Same columns must appear in fit and predict.")

    @check_nan_columns
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

        # start with fit checks
        self._fit_strategy_validator(X)
        self.statistics_ = {}

        # iterate missingness fit using classifier and all remaining columns
        for column in self.data_mi:
            # only fit non time-based columns...
            if not np.issubdtype(self.data_mi[column].dtype, np.datetime64):
                y = self.data_mi[column]
                preds = self._preds[column]
                if preds == "all":
                    x = X.drop(column, axis=1)
                else:
                    x = X[preds]
                clf = clone(self.classifier)
                cls_fit = clf.fit(x.values, y.values, **kwargs)
                self.statistics_[column] = cls_fit
        return self

    @check_nan_columns
    def predict(self, X, **kwargs):
        """Predict class of each feature. 1 for missing; 0 for not missing.

        First checks to ensure data has been fit. If fit, `predict` method
        uses the respective classifier of each feature (stored in statistics)
        and predicts class membership for each observation of each feature.
        1 = missing; 0 = not missing. Prediction is binary, as class membership
        is hard. If probability deesired, use `predict_proba` method.

        Args:
            X (pd.DataFrame): DataFrame used to create predictions.
            kwargs: kewword arguments. Used by the classifer.

        Returns:
            pd.DataFrame: DataFrame with class prediction for each observation.
        """

        # predictions for each column using respective fit classifier
        self._predictor_strategy_validator(X)
        preds_mat = []
        for column in self.data_mi:
            if not np.issubdtype(self.data_mi[column].dtype, np.datetime64):
                preds = self._preds[column]
                if preds == "all":
                    x = X.drop(column, axis=1)
                else:
                    x = X[preds]
                cls_fit = self.statistics_[column]
                y_pred = cls_fit.predict(x.values, **kwargs)
                preds_mat.append(y_pred)
            else:
                y_pred = np.zeros(len(self.data_mi.index))
                preds_mat.append(y_pred)

        # store the predictor matrix class membership as a dataframe
        preds_mat = np.array(preds_mat).T
        pred_cols = [f"{cl}_pred" for cl in X.columns]
        self.data_mi_preds = pd.DataFrame(preds_mat, columns=pred_cols)
        return self.data_mi_preds

    @check_nan_columns
    def predict_proba(self, X, **kwargs):
        """Predict probability of missing class membership of each feature.

        First checks to ensure data has been fit. If fit, `predict_proba`
        method uses the respsective classifier of each feature (in statistics)
        and predicts probability of missing class membership for each
        observation of each feature. Prediction is probability of missing.
        Therefore, probability of not missing is 1-P(missing). For hard class
        membership prediction, use `predict`.

        Args:
            X (pd.DataFrame): DataFrame used to create probabilities.

        Returns:
            pd.DataFrame: DataFrame with probability of missing class for
                each observation.
        """
        self._predictor_strategy_validator(X)
        preds_mat = []
        for column in self.data_mi:
            if not np.issubdtype(self.data_mi[column].dtype, np.datetime64):
                preds = self._preds[column]
                if preds == "all":
                    x = X.drop(column, axis=1)
                else:
                    x = X[preds]
                cls_fit = self.statistics_[column]
                y_pred = cls_fit.predict_proba(x.values, **kwargs)[:, 1]
                preds_mat.append(y_pred)
            else:
                y_pred = np.zeros(len(self.data_mi.index))
                preds_mat.append(y_pred)

        # store the predictor matrix probabilities as a dataframe
        preds_mat = np.array(preds_mat).T
        pred_cols = [f"{cl}_pred" for cl in X.columns]
        self.data_mi_proba = pd.DataFrame(preds_mat, columns=pred_cols)
        return self.data_mi_proba

    def fit_predict(self, X):
        """Convenience method for fit and class prediction.

        Args:
            X (pd.DataFrame): DataFrame to fit classifier and predict class.

        Returns:
            pd.DataFrame: DataFrame of class predictions.
        """
        return self.fit(X).predict(X)

    def fit_predict_proba(self, X):
        """Convenience method for fit and class probability prediction.

        Args:
            X (pd.DataFrame): DataFrame to fit classifier and prredict prob.

        Returns:
            pd.DataFrame: DataFrame of class probability predictions.
        """
        return self.fit(X).predict_proba(X)

    @check_nan_columns
    def gen_test_indices(self, X, thresh=0.5, use_exist=False):
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
            use_exist (bool, optional): Whether or not to use existing fit and
                classifiers. Default is False.

        Returns:
            self: test_indice available from `self.test_indices`
        """

        # always fit_transform with dataset, as test vals can change
        self.test_indices = {}
        if not use_exist:
            self.fit_predict_proba(X)

        # loop through missing data indicators, eval new set for missing
        for c in self.data_mi:
            mi_c = self.data_mi[c]
            not_mi = mi_c[mi_c == 0].index
            pred_not_mi = self.data_mi_proba.loc[not_mi, f"{c}_pred"]
            pred_wrong = pred_not_mi[pred_not_mi > thresh].index
            self.test_indices[c] = pred_wrong
        return self

    def gen_test_df(self, X, thresh=0.5, m=0.05,
                    inplace=False, use_exist=False):
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
            use_exist (bool, optional): Whether or not to use existing fit and
                classifiers. Default is False.

        Returns:
            pd.DataFrame: DataFrame with false positives set to missing.
        """
        if not inplace:
            X = X.copy()

        self.gen_test_indices(X, thresh, use_exist)
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
