"""This module implements multiple imputation through the other imputers.

This module contains one class - the MultipleImputer. Use this class to
perform multiple imputations for each Series within a DataFrame using methods
from either the SingleImputer or PredictiveImputer. This class retains each
imputation as its own dataset. It allows for flexible logic between each
imputation. More details about the options appear in the class itself.
"""

# pylint:disable=unused-import
from sklearn.base import BaseEstimator, TransformerMixin
from autoimpute.imputations import method_names
from autoimpute.utils import check_nan_columns
from .base_imputer import BaseImputer
from ..series import DefaultPredictiveImputer
from ..series import LeastSquaresImputer, StochasticImputer, PMMImputer
from ..series import BinaryLogisticImputer, MultiLogisticImputer
from ..series import BayesLeastSquaresImputer, BayesBinaryLogisticImputer
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=protected-access
# pylint:disable=too-many-arguments
# pylint:disable=unused-argument

class MultipleImputer(BaseImputer, BaseEstimator, TransformerMixin):
    """Techniques to impute Series with missing values multiple times.

    The MultipleImputer class implements multiple imputation. It leverages the
    methods found in the PredictiveImputer. This imputer passes all imputation
    work to the PredictiveImputer, but it controls the arguments each imputer
    receives, which are flexible depending on what the user specifies for each
    imputation.

    Note that the imputer allows for one imputation method per column only.
    Therefore, the behavior of `strategy` is the exact same as other classes.
    But the predictors and the seed are allowed to change for each imputation.
    """

    strategies = {
        methods.DEFAULT: DefaultPredictiveImputer,
        methods.LS: LeastSquaresImputer,
        methods.STOCHASTIC: StochasticImputer,
        methods.BINARY_LOGISTIC: BinaryLogisticImputer,
        methods.MULTI_LOGISTIC: MultiLogisticImputer,
        methods.BAYESIAN_LS: BayesLeastSquaresImputer,
        methods.BAYESIAN_BINARY_LOGISTIC: BayesBinaryLogisticImputer,
        methods.PMM: PMMImputer
    }

    visit_sequences = (
        "default",
        "left-to-right"
    )

    def __init__(self, n=5, strategy="default", predictors="all",
                 imp_kwgs=None, copy=True, scaler=None, verbose=False,
                 seed=None, visit="default", parallel=False):
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
            seed (int, optional): seed setting for reproducible results.
                Defualt is None. No validation, but values should be integer.
            visit (str, optional): order to visit columns for imputation.
                Default is "default", which is left-to-right. Options include:
                - "default", "left-to-right" -> visit in order of columns.
                - TBD: "random" -> shulffe columns and visit.
                - TBD: "most missing" -> in order of most missing to least.
                - TBD: "least missing" -> in order of least missing to most.
            parallel (bool, optional): run n imputations in parallel or
                sequentially. Default is False to start, but will be True.
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
        self.copy = copy
        self.seed = seed
        self.visit = visit
        self.parallel = parallel

    @property
    def n(self):
        """Property getter to return the value of the n property."""
        return self._n

    @n.setter
    def n(self, n_):
        """Validate the n property to ensure it's Type and Value.

        Args:
            n_ (int): n passed as arg to class instance.

        Raises:
            TypeError: n must be an integer.
            ValueError: n must be greater than zero.
        """
        # deal with type first
        if not isinstance(n_, int):
            err = "n must be an integer specifying number of imputations."
            raise TypeError(err)

        # then check the value is greater than zero
        if n_ < 1:
            err = "n > 0. Cannot perform fewer than 1 imputation."
            raise ValueError(err)

        # otherwise set the property value for n
        self._n = n_

    @property
    def strategy(self):
        """Property getter to return the value of the strategy property."""
        return self._strategy

    @strategy.setter
    def strategy(self, s):
        """Validate the strategy property to ensure it's Type and Value.

        Class instance only possible if strategy is proper type, as outlined
        in the init method. Passes supported strategies and user arg to
        helper method, which performs strategy checks.

        Args:
            s (str, iter, dict): Strategy passed as arg to class instance.

        Raises:
            ValueError: Strategies not valid (not in allowed strategies).
            TypeError: Strategy must be a string, tuple, list, or dict.
            Both errors raised through helper method `check_strategy_allowed`.
        """
        strat_names = self.strategies.keys()
        self._strategy = self.check_strategy_allowed(strat_names, s)

    @property
    def visit(self):
        """Property getter to return the value of the visit property."""
        return self._visit

    @visit.setter
    def visit(self, v):
        """Validate the visit property to ensure it's Type and Value.

        Class instance only possible if visit is proper type, as outlined in
        the init method. Visit property must be one of valid sequences in the
        `visit_sequences` variable.

        Args:
            v (str): Visit sequence passed as arg to class instance.

        Raises:
            TypeError: visit sequence must be a string.
            ValueError: visit sequenece not in `visit_sequences`.
        """
        # deal with type first
        if not isinstance(v, str):
            err = "visit must be a string specifying visit sequence to use."
            raise TypeError(err)

        # deal with value next
        if v not in self.visit_sequences:
            err = f"visit not valid. Must be one of {self.visit_sequences}"
            raise ValueError(err)

        # otherwise, set property for visit
        self._visit = v

    def _fit_strategy_validator(self, X):
        """Internal helper method to validate strategies appropriate for fit.

        Checks whether strategies match with type of column they are applied
        to. If not, error is raised through `check_strategy_fit` method.
        """
        # remove nan columns and store colnames
        cols = X.columns.tolist()
        self._strats = self.check_strategy_fit(self.strategy, cols)

        # if predictors is a list...
        if isinstance(self.predictors, (tuple, list)):
            # and it is not the same list of predictors for every iteration...
            if not all([isinstance(x, str) for x in self.predictors]):
                len_pred = len(self.predictors)
                # raise error if not the correct length
                if len_pred != self.n:
                    err = f"Predictors has {len_pred} items. Need {self.n}"
                    raise ValueError(err)
                # check predictors for each in list
                self._preds = [
                    self.check_predictors_fit(p, cols)
                    for p in self.predictors
                ]
            # if it is a list, but not a list of objects...
            else:
                # broadcast predictors
                self._preds = self.check_predictors_fit(self.predictors, cols)
                self._preds = [self._preds]*self.n
        # if string or dictionary...
        else:
            # broadcast predictors
            self._preds = self.check_predictors_fit(self.predictors, cols)
            self._preds = [self._preds]*self.n

    @check_nan_columns
    def fit(self, X, y=None):
        """Fit imputation methods to each column within a DataFrame.

        The fit method calclulates the `statistics` necessary to later
        transform a dataset (i.e. perform actual imputatations). Inductive
        methods calculate statistic on the fit data, then impute new missing
        data with that value. All currently supported methods are inductive.

        Args:
            X (pd.DataFrame): pandas DataFrame on which imputer is fit.

        Returns:
            self: instance of the PredictiveImputer class.
        """
        # first, prep columns we plan to use and make sure they are valid
        self._fit_strategy_validator(X)
        self.statistics_ = {}
