"""This module implements logistic regression imputation.

This module contains the BinaryLogisticImputer and the MultiLogisticImputer.
Both use logistic regression to generate class predictions that become values
for imputations of missing data. Binary is optimized to deal with two classes,
while Multi is optimized to deal with multiple classes. Dataframe imputers
utilize these classes when each's strategy is requested. Use SingleImputer or
MultipleImputer with strategy = `binary logistic` or `multinomial logistic`
to broadcast either strategy across all the columns in a dataframe, or specify
either strategy for a given column.
"""

import warnings
from pandas import Series
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LogisticRegression
from autoimpute.imputations import method_names
from .base import ISeriesImputer
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:

class BinaryLogisticImputer(ISeriesImputer):
    """Impute missing values w/ predictions from binary logistic regression.

    The BinaryLogisticImputer produces predictions using logsitic regression
    with two classes. The class predictions given a set of predictors become
    the imputations. To implement logistic regression, the imputer wraps the
    sklearn LogisticRegression class with a default solver (liblinear). The
    imputer can be used directly, but such behavior is discouraged.
    BinaryLogisticImputer does not have the flexibility / robustness of
    dataframe imputers, nor is its behavior identical. Preferred use is
    MultipleImputer(strategy="binary logistic").
    """
    # class variables
    strategy = methods.BINARY_LOGISTIC

    def __init__(self, **kwargs):
        """Create an instance of the BinaryLogisticImputer class.

        Args:
            **kwargs: keyword arguments passed to LogisticRegresion.

        """
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
        y = y.astype("category").cat
        y_cat_l = len(y.codes.unique())
        if y_cat_l > 2:
            err = "Binary requires 2 categories. Use multinomial instead."
            raise ValueError(err)
        self.glm.fit(X, y.codes)
        self.statistics_ = {"param": y.categories, "strategy": self.strategy}
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

class MultinomialLogisticImputer(ISeriesImputer):
    """Impute missing values w/ preds from multinomial logistic regression.

    The MultinomialLogisticImputer produces predictions w/ logsitic regression
    with more than two classes. Class predictions given a set of predictors
    become the imputations. To implement logistic regression, the imputer
    wraps the sklearn LogisticRegression class with a default solver (saga)
    and default `multi_class` set to multinomial. The imputer can be used
    directly, but such behavior is discouraged. MultinomialLogisticImputer
    does not have the flexibility / robustness of dataframe imputers, nor is
    its behavior identical. Preferred use is
    MultipleImputer(strategy="multinomial logistic").
    """
    # class variables
    strategy = methods.MULTI_LOGISTIC

    def __init__(self, **kwargs):
        """Create an instance of the MultiLogisticImputer class.

        Args:
            **kwargs: keyword arguments passed to LogisticRegression.

        """
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
        y = y.astype("category").cat
        y_cat_l = len(y.codes.unique())
        if y_cat_l == 2:
            w = "Multiple categories (c) expected. Use binary instead if c=2."
            warnings.warn(w)
        self.glm.fit(X, y.codes)
        self.statistics_ = {"param": y.categories, "strategy": self.strategy}
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
