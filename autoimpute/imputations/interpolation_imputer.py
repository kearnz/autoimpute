"""This module implements mean imputation via the InterpolateImputer.

The InterpolateImputer imputes missing data using an interpolation strategy
suppoted by pd.Series.interpolate. Linear is the default strategy, although a
number of additional strategies exist. Interpolation is transductive, so the
fit method simply returns the interpolation method but no fit statistic. All
interpolation is performed in transform. Right now, this imputer
supports imputation on Series only. Use TimeSeriesImputer with specified
strategy to broadcast interpolation across multiple columns of a DataFrame.
Note that some interpolation strategies are valid for SingleImputer as well.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=unnecessary-pass
# pylint:disable=unused-argument

class InterpolateImputer(BaseEstimator, TransformerMixin):
    """Techniques to impute missing values using interpolation techniques.

    More complex autoimpute Imputers delegate work to the InterpolateImputer
    if an intepolation is a specified strategy for a given Series. That being
    said, InterpolateImputer is a stand-alone class and valid sklearn
    transformer. It can be used directly, but such behavior is discouraged
    because this imputer supports Series only. InterpolateImputer does not
    have the flexibility or robustness of more complex imputers, nor is its
    behavior identical. Instead, use TimeSeriesImputer or SingleImputer
    depending on use case.
    """
    def __init__(self, strategy="linear"):
        """Create an instance of the InterpolateImputer class.

        Args:
            strategy (str, Optional): type of interpolation to perform
                Default is linear. Time also supported for now.

        Returns:
            self. Instance of the class
        """
        self.strategy = strategy

    def fit(self, X):
        """Fit the Imputer to the dataset. Nothing to calculate.

        Args:
            X (pd.Series): Dataset to fit the imputer

        Returns:
            self. Instance of the class.
        """
        self.statistics_ = {"param": None, "strategy": self.strategy}
        return self

    def transform(self, X):
        """Perform imputations using the statistics generated from fit.

        The transform method handles the actual imputation. Missing values
        in a given dataset are replaced with the respective mean from fit.

        Args:
            X (pd.Series): Dataset to fit the imputer

        Returns:
            pd.Series -- imputed dataset
        """
        # check if fitted then impute with interpolation strategy
        # Note here some of the default argumens should be **kwargs
        check_is_fitted(self, "statistics_")
        imp = self.statistics_["strategy"]
        X.interpolate(method=imp,
                      limit=None,
                      limit_direction="both",
                      inplace=True)
        return X
