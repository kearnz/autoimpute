"""This module implements logistic regression imputation.

This module contains the BinaryLogisticImputer and the MultiLogisticImputer.
Both use logistic regression to generate class predictions that become values
for imputations of missing data. Binary is optimized to deal with two classes,
while Multi is optimized to deal with multiple classes. Right now, each
imputer supports imputation on Series only. Use the PredictiveImputer with
strategy = "binary logistic" or "multinomial logistic" to broadcast the
strategies across all the columns in a dataframe.
"""

import warnings
from pandas import Series
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LogisticRegression
from autoimpute.imputations import method_names
from autoimpute.utils.helpers import _get_observed
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:

class BinaryLogisticImputer(BaseEstimator):
    """Impute missing values w/ predictions from binary logistic regression.

    The BinaryLogisticImputer produces predictions using logsitic regression
    with two classes. The class predictions given a set of predictors become
    the imputations. To implement logistic regression, the imputer wraps the
    sklearn LogisticRegression class with a default solver (liblinear). The
    imputer can be used directly, but such behavior is discouraged because the
    imputer supports Series only. BinaryLogisticImputer does not have the
    flexibility or robustness of more complex imputers, nor is its behavior
    identical. Instead, use PredictiveImputer(strategy="binary logistic").
    """
    # class variables
    strategy = methods.BINARY_LOGISTIC

    def __init__(self, verbose, **kwargs):
        """Create an instance of the BinaryLogisticImputer class.

        Args:
            verbose (bool): print information to the console.
            **kwargs: keyword arguments passed to LogisticRegresion.

        """
        self.verbose = verbose
        self.solver = kwargs.pop("solver", "liblinear")
        self.glm = LogisticRegression(solver=self.solver, **kwargs)

    def fit(self, X, y):
        """Fit the Imputer to the dataset by fitting logistic model.

        Args:
            X (pd.Dataframe): dataset to fit the imputer.
            y (pd.Series): response, which is eventually imputed.

        Returns:
            self. Instance of the class.
        """
        # logistic model fit on observed values only
        X_, y_ = _get_observed(
            self.strategy, X, y, self.verbose
        )
        y_ = y_.astype("category").cat
        y_cat_l = len(y_.codes.unique())
        if y_cat_l > 2:
            err = "Binary requires 2 categories. Use multinomial instead."
            raise ValueError(err)
        self.glm.fit(X_, y_.codes)
        self.statistics_ = {"param": y_.categories, "strategy": self.strategy}
        return self

    def impute(self, X):
        """Generate imputations using predictions from the fit logistic model.

        The impute method returns the values for imputation. Missing values
        in a given dataset are replaced with the predictions from the logistic
        regression class specification.

        Args:
            X (pd.DataFrame): predictors to determine imputed values.

        Returns:
            np.array: imputed dataset.
        """
        # check if fitted then predict with logistic
        check_is_fitted(self, "statistics_")
        labels = self.statistics_["param"]
        preds = self.glm.predict(X)

        # map category codes back to actual labels
        # then impute the actual labels to keep categories in tact
        label_dict = {i:j for i, j in enumerate(labels.values)}
        imp = Series(preds).replace(label_dict, inplace=False)
        return imp.values

    def fit_impute(self, X, y):
        """Fit impute method to generate imputations where y is missing.

        Args:
            X (pd.Dataframe): predictors in the dataset.
            y (pd.Series): response w/ missing values to impute.

        Returns:
            np.array: imputed dataset.
        """
        # transform occurs with records from X where y is missing
        miss_y_ix = y[y.isnull()].index
        return self.fit(X, y).impute(X.loc[miss_y_ix])

class MultiLogisticImputer(BaseEstimator):
    """Impute missing values w/ preds from multinomial logistic regression.

    The MultiLogisticImputer produces predictions using logsitic regression
    with more than two classes. Class predictions given a set of predictors
    become the imputations. To implement logistic regression, the imputer
    wraps the sklearn LogisticRegression class with a default solver (saga)
    and default `multi_class` set to multinomial. The imputer can be used
    directly, but such behavior is discouraged because the imputer supports
    Series only. MultiLogisticImputer does not have the flexibility or
    robustness of more complex imputers, nor is its behavior identical.
    Instead, use PredictiveImputer(strategy="multinomial logistic").
    """
    # class variables
    strategy = methods.MULTI_LOGISTIC

    def __init__(self, verbose, **kwargs):
        """Create an instance of the MultiLogisticImputer class.

        Args:
            verbose (bool): print information to the console.
            **kwargs: keyword arguments passed to LogisticRegression.

        """
        self.verbose = verbose
        self.solver = kwargs.pop("solver", "saga")
        self.multiclass = kwargs.pop("multi_class", "multinomial")
        self.glm = LogisticRegression(
            solver=self.solver,
            multi_class=self.multiclass,
            **kwargs
        )

    def fit(self, X, y):
        """Fit the Imputer to the dataset by fitting logistic model.

        Args:
            X (pd.Dataframe): dataset to fit the imputer.
            y (pd.Series): response, which is eventually imputed.

        Returns:
            self. Instance of the class.
        """
        # logistic model fit on observed values only
        X_, y_ = _get_observed(
            self.strategy, X, y, self.verbose
        )
        y_ = y_.astype("category").cat
        y_cat_l = len(y_.codes.unique())
        if y_cat_l == 2:
            w = "Multiple categories (c) expected. Use binary instead if c=2."
            warnings.warn(w)
        self.glm.fit(X_, y_.codes)
        self.statistics_ = {"param": y_.categories, "strategy": self.strategy}
        return self

    def impute(self, X):
        """Generate imputations using predictions from the fit logistic model.

        The impute method returns the values for imputation. Missing values
        in a given dataset are replaced with the predictions from the logistic
        regression class specification.

        Args:
            X (pd.DataFrame): predictors to determine imputed values.

        Returns:
            np.array: imputed dataset.
        """
        # check if fitted then predict with logistic
        check_is_fitted(self, "statistics_")
        labels = self.statistics_["param"]
        preds = self.glm.predict(X)

        # map category codes back to actual labels
        # then impute the actual labels to keep categories in tact
        label_dict = {i:j for i, j in enumerate(labels.values)}
        imp = Series(preds).replace(label_dict, inplace=False)
        return imp.values

    def fit_impute(self, X, y):
        """Fit impute method to generate imputations where y is missing.

        Args:
            X (pd.Dataframe): predictors in the dataset.
            y (pd.Series): response w/ missing values to impute.

        Returns:
            np.array: imputed dataset.
        """
        # transform occurs with records from X where y is missing
        miss_y_ix = y[y.isnull()].index
        return self.fit(X, y).impute(X.loc[miss_y_ix])
