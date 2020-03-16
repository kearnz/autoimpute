"""This module implements normal imputation with constant unit variance single imputation 
via the NormUnitVarianceImputer.

The NormUnitVarianceImputer imputes missing data assuming that the
single column is normally distributed with a-priori known constant  unit
variance. Use SingleImputer or MultipleImputer with strategy=`norm_const_variance`
to broadcast the strategy across all the columns in a dataframe, 
or specify this strategy for a given column.
"""

from scipy import stats
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
from autoimpute.imputations.errors import _not_num_series
from .base import ISeriesImputer
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=unnecessary-pass

class NormUnitVarianceImputer(ISeriesImputer):
    """Impute missing values assuming normally distributed 
    data with unknown mean and *known* variance.
    """
    # class variables
    strategy = methods.NORM_UNIT_VARIANCE

    def __init__(self):
        """Create an instance of the NormUnitVarianceImputer class."""
        pass

    def fit(self, X, y):
        """Fit the Imputer to the dataset and calculate the mean.

        Args:
            X (pd.Series): Dataset to fit the imputer.
            y (None): ignored, None to meet requirements of base class

        Returns:
            self. Instance of the class.
        """
        _not_num_series(self.strategy, X)
        mu = X.mean()  # mean of observed data
        self.statistics_ = {"param": mu, "strategy": self.strategy}
        return self

    def impute(self, X):
        """Perform imputations using the statistics generated from fit.

        The impute method handles the actual imputation. Missing values
        in a given dataset are replaced with the respective mean from fit.

        Args:
            X (pd.Series): Dataset to impute missing data from fit.

        Returns:
            np.array -- imputed dataset.
        """
        # check if fitted then impute with mean
        check_is_fitted(self, "statistics_")
        _not_num_series(self.strategy, X)
        omu = self.statistics_["param"] # mean of observed data
        idx = X.isnull()                # missing data
        nO = sum(~idx)                  # number of observed
        m = sum(idx)                    # number to impute
        muhatk = stats.norm(omu,np.sqrt(1/nO))
        # imputation cross-terms *NOT* uncorrelated
        Ymi=stats.multivariate_normal(np.ones(m)*muhatk.rvs(),
                                      np.ones((m,m))/nO+np.eye(m)).rvs()
        out = X.copy()
        out[idx] = Ymi
        return out

    def fit_impute(self, X, y=None):
        """Convenience method to perform fit and imputation in one go."""
        return self.fit(X, y).impute(X)

if __name__ == '__main__':
    from autoimpute.imputations import SingleImputer
    si=SingleImputer('normal unit variance')
    Yo=stats.norm(0,1).rvs(100)
    df = pd.DataFrame(columns=['Yo'],index=range(200),dtype=float)
    df.loc[range(100),'Yo'] = Yo
    si.fit_transform(df)
