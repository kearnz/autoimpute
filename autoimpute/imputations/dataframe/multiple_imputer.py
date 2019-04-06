"""This module implements multiple imputation through the other imputers.

This module contains one class - the MultipleImputer. Use this class to
perform multiple imputations for each Series within a DataFrame using methods
from either the SingleImputer or PredictiveImputer. This class retains each
imputation as its own dataset. It allows for flexible logic between each
imputation. More details about the options appear in the class itself.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from . import BaseImputer

class MultipleImputer(BaseImputer, BaseEstimator, TransformerMixin):
    """Techniques to impute Series with missing values multiple times.

    The MultipleImputer class implements multiple imputation. It leverages the
    methods found in the SingleImputers and PredictiveImputers. This imputer
    passes all imputation work to the Single and Predictive imputer, but it
    controls the arguments each imputer receives, which are flexible depending
    on what the user specifies for each imputation.
    """

    def __init__(self, n=5, strategy="default", predictors="all",
                 imp_kwgs=None, copy=True, scaler=None, verbose=False):
        """Create an instance of the MultipleImputer class.

        As with sklearn classes, all arguments take default values. Therefore,
        MultipleImputer() creates a valid class instance. The instance is
        used to set up an imputer and perform checks on arguments.

        Args:
            n (int, optional): number of imputations to perform. Default is 5.
                Value must be greater than or equal to 1.
            strategy (str, iter, dict; optional): strategies for imputation.
                Default value is str -> "default". I.e. default imputation.
                If str, single strategy broadcast to all series in DataFrame.
                If iter, must provide 1 strategy per column. Each method within
                iterator applies to column with same index value in DataFrame.
                If dict, must provide key = column name, value = imputer.
                Dict the most flexible and PREFERRED way to create custom
                imputation strategies if not using the default. Dict does not
                require method for every column; just those specified as keys.
            predictors (str, iter, dict, optional): defaults to all, i.e.
                use all predictors. If all, every column will be used for
                every class prediction. If a list, subset of columns used for
                all predictions. If a dict, specify which columns to use as
                predictors for each imputation. Columns not specified in dict
                but present in `strategy` receive `all` other cols as preds.
            imp_kwgs (dict, optional): keyword arguments for each imputer.
                Default is None, which means default imputer created to match
                specific strategy. imp_kwgs keys can be either columns or
                strategies. If strategies, each column given that strategy is
                instantiated with same arguments.
            copy (bool, optional): create copy of DataFrame or operate inplace.
                Default value is True. Copy created.
            scaler (scaler, optional): scale variables before transformation.
                Default is None, although StandardScaler recommended.
            verbose (bool, optional): print more information to console.
                Default value is False.
        """
        BaseImputer.__init__(
            self,
            imp_kwgs=imp_kwgs,
            scaler=scaler,
            verbose=verbose
        )
        self.n = n
        self.strategy = strategy
        self.predictors = predictors
        self.imp_kwgs = imp_kwgs
        self.copy = copy
        self.scaler = scaler
        self.verbose = verbose
