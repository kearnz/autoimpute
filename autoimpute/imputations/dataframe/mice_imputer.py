"""This module performs a series of multiple imputations of missing features
in a dataset.

This module contains one class - the MiceImputer. Use this class to
impute each Series within a DataFrame multiple times using an iteration of fits
and transformations to reach a stable state of imputation each time. This
extension of MultipleImputer makes the same imputation methods available as its
parent class - both univariate and multivatiate. Each method runs `n` times on
its specified column. When `n` passes through the columns are complete, the
MultipleImputer returns the `n` imputed datasets. For each of these `n`
imputations, the method (re)fits and applies imputation to the dataset `k`
times. Typically `k` should be at least 3 to reach a stable state.
Its functioning is based upon the R package MICE
(https://cran.r-project.org/web/packages/mice/)
"""

from autoimpute.utils import check_nan_columns
from .multiple_imputer import MultipleImputer


class MiceImputer(MultipleImputer):
    """Techniques to impute Series with missing values multiple times using
    repeated fits and applications to reach a stable imputation.

    The MiceImputer class implements multiple imputation, i.e., a series
    or repetition of applications of imputation to reach a stable imputation,
    similar to the functioning of the R package MICE.
    It leverages the methods found in the BaseImputer. This imputer passes
    all the work for each imputation to the SingleImputer, but it controls
    the arguments each imputer receives. The args are flexible depending on
    what the user specifies for each imputation.

    Note that the Imputer allows for one imputation method per column only.
    Therefore, the behavior of `strategy` is the same as the SingleImputer,
    but the predictors are allowed to change for each imputation.
    """

    def __init__(self, k=3, n=5,
                 strategy="default predictive", predictors="all",
                 imp_kwgs=None, seed=None, visit="default",
                 return_list=False):
        """Create an instance of the SeriesImputer class.

        As with sklearn classes, all arguments take default values. Therefore,
        SeriesImputer() creates a valid class instance. The instance is
        used to set up an imputer and perform checks on arguments.

        Args:
            k (int, optional): number of repeated fits and transformations to
                apply to reach a stable impution. Default is 3.
                Value must be greater than or equal to 1.
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
        self.k = k
        MultipleImputer.__init__(
            self,
            n=n,
            strategy=strategy,
            predictors=predictors,
            imp_kwgs=imp_kwgs,
            seed=seed,
            visit=visit,
            return_list=return_list
        )

    @property
    def k(self):
        """Property getter to return the value of the k property."""
        return self._k

    @k.setter
    def k(self, k_):
        """Validate the k property to ensure it's Type and Value.

        Args:
            k_ (int): k passed as arg to class instance.

        Raises:
            TypeError: k must be an integer.
            ValueError: k must be greater than zero.
        """

        # deal with type first
        if not isinstance(k_, int):
            err = """
                k must be an int specifying number of repeated fits
                and transformations in a series of imputations.
            """
            raise TypeError(err)

        # then check the value is greater than zero
        if k_ < 1:
            err = "k > 0. Cannot perform fewer than 1 imputation."
            raise ValueError(err)

        # otherwise set the property value for k
        self._k = k_

    def _iterate_imputation(self, X, imp):
        """Helper function that iterates self.k times to create a stable imputation
        by repeated application and retraining of the imputation models.
        Used by transform()

        Args:
            X (pd.DataFrame): fit DataFrame to impute
            imp (Imputer): Imputer to apply to X
        """
        X2 = imp.transform(X, imp_ixs=self.imputed_)
        for k in range(self.k - 1):
            imp.fit(X2, imp_ixs=self.imputed_)
            X2 = imp.transform(X, imp_ixs=self.imputed_, k=k)
        return X2

    @check_nan_columns
    def transform(self, X):
        """Impute each column within a DataFrame using fit imputation methods.

        The transform step performs the actual imputations. Given a dataset
        previously fit, `transform` imputes each column with it's respective
        imputed values from fit (in the case of inductive) or performs new fit
        and transform in one sweep (in the case of transductive).
        The transformations and fits are repeatedly applied and refitted self.k
        times to reach a stable imputation.

        Args:
            X (pd.DataFrame): fit DataFrame to impute.

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
        imputed = ((i[0], self._iterate_imputation(X, i[1]))
                   for i in self.statistics_.items())

        if self.return_list:
            imputed = list(imputed)
        return imputed
