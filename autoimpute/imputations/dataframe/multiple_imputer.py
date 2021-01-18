"""This module performs multiple imputations of missing features in a dataset.

This module contains one class - the MultipleImputer. Use this class to
impute each Series within a DataFrame multiple times. This class makes
numerous imputation methods available - both univariate and multivatiate. Each
method runs `n` times on its specified column. When `n` passes through the
columns are complete, the MultipleImputer returns the `n` imputed datasets.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
from autoimpute.utils import check_nan_columns, check_predictors_fit
from autoimpute.utils import check_strategy_fit
from .base_imputer import BaseImputer
from .single_imputer import SingleImputer
methods = method_names

# pylint:disable=attribute-defined-outside-init
# pylint:disable=protected-access
# pylint:disable=too-many-arguments
# pylint:disable=unused-argument
# pylint:disable=too-many-instance-attributes
# pylint:disable=arguments-differ


class MultipleImputer(BaseImputer, BaseEstimator, TransformerMixin):
    """Techniques to impute Series with missing values multiple times.

    The MultipleImputer class applies imputation multiple times. It leverages the
    methods found in the BaseImputer. This imputer passes all the work for
    each imputation to the SingleImputer, but it controls the arguments
    each imputer receives. The args are flexible depending on what the user
    specifies for each imputation.

    Note that the Imputer allows for one imputation method per column only.
    Therefore, the behavior of `strategy` is the same as the SingleImputer,
    but the predictors are allowed to change for each imputation.
    """

    def __init__(self, n=5, strategy="default predictive", predictors="all",
                 imp_kwgs=None, seed=None, visit="default", return_list=False):
        """Create an instance of the MultipleImputer class.

        As with sklearn classes, all arguments take default values. Therefore,
        MultipleImputer() creates a valid class instance. The instance is
        used to set up an imputer and perform checks on arguments.

        Args:
            n (int, optional): number of imputations to perform. Default is 5.
                Value must be greater than or equal to 1.
            strategy (str, iter, dict; optional): strategy for single imputer.
                Default value is str --> `predictive default`.
                See BaseImputer for all available strategies.
                If str, single strategy broadcast to all series in DataFrame.
                If iter, must provide 1 strategy per column. Each method w/in
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
                instantiated with same arguments. When strategy is `default`,
                imp_kwgs is ignored.
            seed (int, optional): seed setting for reproducible results.
                Defualt is None. No validation, but values should be integer.
            return_list (bool, optional): return m as list or generator.
                Default is False. m imputations returned as generator. More
                memory efficient. return as list if return_list=True
        """
        BaseImputer.__init__(
            self,
            strategy=strategy,
            imp_kwgs=imp_kwgs,
            visit=visit
        )
        self.n = n
        self.predictors = predictors
        self.seed = seed
        self.return_list = return_list
        self.copy = True

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

    def _fit_strategy_validator(self, X):
        """Internal helper method to validate strategies appropriate for fit.

        Checks whether strategies match with type of column they are applied
        to. If not, error is raised through `check_strategy_fit` method.
        """

        # remove nan columns and store colnames
        cols = X.columns.tolist()
        self._strats = check_strategy_fit(self.strategy, cols)

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
                    check_predictors_fit(p, cols)
                    for p in self.predictors
                ]
            # if it is a list, but not a list of objects...
            else:
                # broadcast predictors
                self._preds = check_predictors_fit(self.predictors, cols)
                self._preds = [self._preds]*self.n
        # if string or dictionary...
        else:
            # broadcast predictors
            self._preds = check_predictors_fit(self.predictors, cols)
            self._preds = [self._preds]*self.n

    def _transform_strategy_validator(self):
        """Private method to prep for prediction."""
        check_is_fitted(self, "statistics_")

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

        # first, run the fit strategy validator, then create statistics
        self._fit_strategy_validator(X)
        self.statistics_ = {}

        # deal with potentially setting seed for each individual predictor
        if self.seed is not None:
            self._seeds = [self.seed + i for i in range(1, self.n*13, 13)]
        else:
            self._seeds = [None]*self.n

        # create PredictiveImputers. sequentially only right now
        for i in range(1, self.n+1):
            imputer = SingleImputer(
                strategy=self.strategy,
                predictors=self._preds[i-1],
                imp_kwgs=self.imp_kwgs,
                copy=self.copy,
                seed=self._seeds[i-1],
                visit=self.visit
            )
            imputer.fit(X)
            self.statistics_[i] = imputer

        return self

    @check_nan_columns
    def transform(self, X, **trans_kwargs):
        """Impute each column within a DataFrame using fit imputation methods.

        The transform step performs the actual imputations. Given a dataset
        previously fit, `transform` imputes each column with it's respective
        imputed values from fit (in the case of inductive) or performs new fit
        and transform in one sweep (in the case of transductive).

        Args:
            X (pd.DataFrame): fit DataFrame to impute.
            **trans_kwargs: dict, optional args for bayesian.

        Returns:
            X (pd.DataFrame): imputed in place or copy of original.

        Raises:
            ValueError: same columns must appear in fit and transform.
        """

        # call transform strategy validator before applying transform
        self._transform_strategy_validator()

        # make it easy to access the location of the imputed values
        self.imputed_ = {}
        for column in self._strats.keys():
            imp_ix = X[column][X[column].isnull()].index
            self.imputed_[column] = imp_ix.tolist()

        # right now, return a generator by default
        # sequential only for now
        imputed = ((i[0], i[1].transform(X, **trans_kwargs))
                   for i in self.statistics_.items())
        if self.return_list:
            imputed = list(imputed)
        return imputed

    def fit_transform(self, X, y=None, **trans_kwargs):
        """Convenience method to fit then transform the same dataset."""
        return self.fit(X, y).transform(X, **trans_kwargs)
