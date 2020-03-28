from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
from autoimpute.utils import check_nan_columns, check_predictors_fit
from autoimpute.utils import check_strategy_fit
from .base_imputer import BaseImputer
from .single_imputer import SingleImputer

from .multiple_imputer import MultipleImputer


class SeriesImputer(MultipleImputer):
    def __init__(self, k=3, n=5, strategy="default predictive", predictors="all",
                 imp_kwgs=None, seed=None, visit="default",
                 return_list=False):
        self.k = k
        super(SeriesImputer, self).__init__(n=n, strategy=strategy, predictors=predictors, imp_kwgs = imp_kwgs, seed=seed, visit=visit, return_list=return_list)


    @check_nan_columns
    def transform(self, X):
        """Impute each column within a DataFrame using fit imputation methods.

        TODO

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

        imputed = []
        for i in self.statistics_.items():
            X2 = i[1].transform(X, imp_ixs=self.imputed_)
            for k in range(self.k-1):
                i[1].fit(X2, imp_ixs=self.imputed_)
                X2 = i[1].transform(X, imp_ixs=self.imputed_)
            imputed.append((i[0], X2))

        if self.return_list:
            imputed = list(imputed)
        return imputed
