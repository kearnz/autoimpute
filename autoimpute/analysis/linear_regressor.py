"""Module containing linear regression for multiply imputed datasets."""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted
from statsmodels.api import OLS
from autoimpute.utils import check_nan_columns
from .base_regressor import MiBaseRegressor

# pylint:disable=attribute-defined-outside-init
# pylint:disable=too-many-locals

class MiLinearRegression(MiBaseRegressor, BaseEstimator):
    """Linear Regression wrapper for multiply imputed datasets.

    The MiLinearRegression class wraps the sklearn and statsmodels libraries
    to extend linear regression to multiply imputed datasets. The class wraps
    statsmodels as well as sklearn because sklearn alone does not provide
    sufficient functionality to pool estimates under Rubin's rules. sklearn is
    for machine learning; therefore, important inference capabilities are
    lacking, such as easily calculating std. error estimates for parameters.
    If users want inference from regression analysis of multiply imputed
    data, utilze the statsmodels implementation in this class instead.

    Attributes:
        linear_models (dict): linear models used by supported python libs.
    """

    linear_models = {
        "type": "linear",
        "statsmodels": OLS,
        "sklearn": LinearRegression
    }

    def __init__(self, mi=None, model_lib="statsmodels", mi_kwgs=None,
                 model_kwgs=None):
        """Create an instance of the Autoimpute MiLinearRegression class.

        Args:
            mi (MiceImputer, Optional): An instance of a MiceImputer.
                Default is none. Can create one through `mi_kwgs` instead.
            model_lib (str, Optional): library the regressor will use to
                implement regression. Options are sklearn and statsmodels.
                Default is statsmodels.
            mi_kwgs (dict, Optional): keyword args to instantiate
                MiceImputer. Default is None. If valid MiceImputer
                passed as mi argument, then mi_kwgs ignored.
            model_kwgs (dict, Optional): keyword args to instantiate
                regressor. Default is None.

        Returns:
            self. Instance of the class.
        """
        MiBaseRegressor.__init__(
            self,
            mi=mi,
            model_lib=model_lib,
            mi_kwgs=mi_kwgs,
            model_kwgs=model_kwgs
        )

    @check_nan_columns
    def fit(self, X, y):
        """Fit model specified to multiply imputed dataset.

        Fit a linear regression on multiply imputed datasets. The method first
        creates multiply imputed data using the MiceImputer instantiated
        when creating an instance of the class. It then runs a linear model on
        each m datasets. The linear model comes from sklearn or statsmodels.
        Finally, the fit method calculates pooled parameters from the m linear
        models. Note that variance for pooled parameters using Rubin's rules
        is available for statsmodels only. sklearn does not implement
        parameter inference out of the box. Autoimpute sklearn pooling TBD.

        Args:
            X (pd.DataFrame): predictors to use. can contain missingness.
            y (pd.Series, pd.DataFrame): response. can contain missingness.

        Returns:
            self. Instance of the class
        """

        # retain columns incase encoding occurs
        self.fit_X_columns = X.columns.tolist()

        # generate the imputation datasets from multiple imputation
        # then fit the analysis models on each of the imputed datasets
        self.models_ = self._apply_models_to_mi_data(
            self.linear_models, X, y
        )

        # generate the fit statistics from each of the m models
        self.statistics_ = self._get_stats_from_models(self.models_)

        # still return an instance of the class
        return self

    @check_nan_columns
    def predict(self, X):
        """Make predictions using statistics generated from fit.

        The regression uses the pooled parameters from each of the imputed
        datasets to generate a set of single predictions. The pooled params
        come from multiply imputed datasets, but the predictions themselves
        follow the same rules as an ordinary linear regression.

        Args:
            X (pd.DataFrame): data to make predictions using pooled params.

        Returns:
            np.array: predictions.
        """

        # validation before prediction
        X = self._predict_strategy_validator(self, X)

        # get the alpha and betas, then create linear equation for predictions
        alpha = self.statistics_["coefs"].values[0]
        betas = self.statistics_["coefs"].values[1:]
        preds = alpha + betas.dot(X.T)
        return preds

    def summary(self):
        """Provide a summary for model parameters, variance, and metrics.

        The summary method brings together the statistics generated from fit
        as well as the variance ratios, if available. The statistics are far
        more valuable when using statsmodels than sklearn.

        Returns:
            pd.DataFrame: summary statistics
        """

        # only possible once we've fit a model with statsmodels
        check_is_fitted(self, "statistics_")
        sdf = pd.DataFrame(self.statistics_)
        sdf.rename(columns={"lambda_": "lambda"}, inplace=True)
        return sdf
