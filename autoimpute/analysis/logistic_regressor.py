"""Module containing logistic regression for multiply imputed datasets."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
from statsmodels.discrete.discrete_model import Logit
from autoimpute.utils import check_nan_columns
from .base_regressor import BaseRegressor
# pylint:disable=attribute-defined-outside-init
# pylint:disable=too-many-locals

class MiLogisticRegression(BaseRegressor):
    """Logistic Regression wrapper for multiply imputed datasets.

    The MiLogisticRegression class wraps the sklearn and statsmodels libraries
    to extend logistic regression to multiply imputed datasets. The class wraps
    statsmodels as well as sklearn because sklearn alone does not provide
    sufficient functionality to pool estimates under Rubin's rules. sklearn is
    for machine learning; therefore, important inference capabilities are
    lacking, such as easily calculating std. error estimates for parameters.
    If users want inference from regression analysis of multiply imputed
    data, utilze the statsmodels implementation in this class instead.
    """

    def __init__(self, model_lib="statsmodels", mi_kwgs=None,
                 model_kwgs=None):
        """Create an instance of the AutoImpute MiLogisticRegression class.

        Args:
            model_lib (str, Optional): library the regressor will use to
                implement regression. Options are sklearn and statsmodels.
                Default is statsmodels.
            mi_kwgs (dict, Optional): keyword args to instantiate
                MultipleImputer. Default is None.
            model_kwgs (dict, Optional): keyword args to instantiate
                regressor. Default is None.

        Returns:
            self. Instance of the class.
        """
        BaseRegressor.__init__(
            self,
            model_lib=model_lib,
            mi_kwgs=mi_kwgs,
            model_kwgs=model_kwgs
        )

    def _fit_strategy_validator(self, X, y):
        """Private method to validate data before fitting model."""

        # y must be a series or dataframe
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            err = "y must be a Series or DataFrame"
            raise ValueError(err)

        # y and X must have the same number of rows
        if X.shape[0] != y.shape[0]:
            err = "y and X must have the same number of records"
            raise ValueError(err)

        # y must have a name if series.
        if isinstance(y, pd.Series):
            self._yn = y.name
            if self._yn is None:
                err = "series y must have a name"
                raise ValueError(err)

        # y must have one column if dataframe.
        if isinstance(y, pd.DataFrame):
            yc = y.shape[1]
            if yc != 1:
                err = "y should only have one column"
                raise ValueError(err)
            self._yn = y.columns.tolist()[0]

        # if no errors thus far, add y to X for imputation
        X[self._yn] = y
        return self.mi.fit_transform(X)

    def _predict_strategy_validator(self, X):
        """Private method to validate before prediction."""

        # first check that model is fitted, then check columns are the same
        check_is_fitted(self, "statistics_")
        X_cols = X.columns.tolist()
        fit_cols = set(self.statistics_["coefficient"].index.tolist()[1:])
        diff_fit = set(fit_cols).difference(X_cols)
        if diff_fit:
            err = "Same columns that were fit must appear in predict."
            raise ValueError(err)

    @check_nan_columns
    def fit(self, X, y, add_constant=True):
        """Fit model specified to multiply imputed dataset.

        Fit a logistic regression on multiply imputed datasets. The method
        creates multiply imputed data using the MultipleImputer instantiated
        when creating an instance of the class. It then runs a logistic model
        on m datasets. The logistic model comes from sklearn or statsmodels.
        Finally, the fit method calculates pooled parameters from m logistic
        models. Note that variance for pooled parameters using Rubin's rules
        is available for statsmodels only. sklearn does not implement
        parameter inference out of the box.

        Args:
            X (pd.DataFrame): predictors to use. can contain missingness.
            y (pd.Series, pd.DataFrame): response. can contain missingness.
            add_constant (bool, Optional): whether or not to add a constant.
                Default is True. Applies to statsmodels only. If sklearn used,
                `add_constant` is ignored.

        Returns:
            self. Instance of the class
        """

        # setup and validation
        mi_data = self._fit_strategy_validator(X, y)
        self.models_ = {}
        self.statistics_ = {}

        # sequential only for now. multiple processing later.
        for dataset in mi_data:
            ind, X = dataset
            y = X.pop(self._yn)
            if self.model_lib == "sklearn":
                model = self._fit_sklearn(LogisticRegression, X, y)
            if self.model_lib == "statsmodels":
                model = self._fit_statsmodels(Logit, X, y, add_constant)
            self.models_[ind] = model

        # pooling phase: sklearn - coefficients only, no variance
        items = self.models_.items()
        li = len(items)
        if self.model_lib == "sklearn":
            alpha = sum([j.intercept_ for i, j in items]) / li
            params = sum([j.coef_ for i, j in items]) / li
            coefs = pd.Series(np.insert(params, 0, alpha))
            coefs.index = ["const"] + X.columns.tolist()
            self.statistics_["coefficient"] = coefs

        # pooling phase: statsmodels - coefficients and variance possible
        if self.model_lib == "statsmodels":
            params = [j.params for i, j in items]
            bses = [j.bse for i, j in items]
            coefs = sum(params)/li

            # variance metrics
            vw = sum(map(lambda x: x*x, bses)) / li
            vb = sum(map(lambda p: (p - coefs)**2, params)) / (li - 1)
            vt = vw + vb + (vb / li)
            self.statistics_["coefficient"] = coefs
            self.statistics_["var_within"] = vw
            self.statistics_["var_between"] = vb
            self.statistics_["var_total"] = vt

        # still return an instance of the class
        return self

    @check_nan_columns
    def predict(self, X):
        """Make predictions using statistics generated from fit.

        The regression uses the pooled parameters from each of the imputed
        datasets to generate a set of single predictions. The pooled params
        come from multiply imputed datasets, but the predictions themselves
        follow the same rules as an logistic regression.

        Args:
            X (pd.DataFrame): data to make predictions using pooled params.

        Returns:
            np.array: predictions.
        """
        # validation before prediction
        self._predict_strategy_validator(X)

        # get the alpha and betas, then create linear equation for predictions
        alpha = self.statistics_["coefficient"].values[0]
        betas = self.statistics_["coefficient"].values[1:]
        preds = alpha + betas.dot(X.T)
        return preds

    def _var_error_handle(self):
        """Private method to handle error for variance ratios."""

        # only possible once we've fit a model with statsmodels
        check_is_fitted(self, "statistics_")
        if self.model_lib != "statsmodels":
            err = f"Variance ratios not available unless using statsmodels."
            raise ValueError(err)

    def variance_from_missing(self):
        """Calculate the variance attributable to missing data.

        Variance attributable to missing data is the amount of variation that
        we can attribute to the need for imputation. Lambda represents this
        ratio. If lambda is 0, no variation comes from missing data. If
        the variation is 1, then all variation comes from missing data. More
        likely are values between 0 and 1. The higher the value, the more
        influence the imputation model has than the complete data model that
        generated the data in the first place.

        Returns:
            lambda: ratio of variation attributable to missing data

        Raises:
            ValueError: Variance ratios not available unless statsmodels
        """
        self._var_error_handle()
        m = self.mi.n
        b = self.statistics_["var_between"]
        t = self.statistics_["var_total"]
        lambda_ = self._var_ratios(m, b, t)
        return lambda_

    def relative_increase_in_variance(self):
        """Calculate the relative increase in variance due to nonresponse.

        Relative Increase in variance explains how much variance increased
        or decreases relative to what it was without multiple imputation.
        The metric is closely linked to `variance_from_missing`. In fact,
        r_ = lambda_ / (1-lambda_). The higher amount of variance attributable
        to the imputation model, the greater the relative increase in variance
        because of multiple imputation. Logically, this should make sense.

        Returns:
            r: relative increase due to nonresponse

        Raises:
            ValueError: Variance ratios not available unless statsmodels
        """
        self._var_error_handle()
        m = self.mi.n
        b = self.statistics_["var_between"]
        u = self.statistics_["var_within"]
        r_ = self._var_ratios(m, b, u)
        return r_

    def degrees_freedom(self):
        """Calculate the degrees of freedom, as found in Van Buuren.

        Returns:
            v: adjusted degrees of freedom

        Raises:
            ValueError: Variance ratios not available unless statsmodels
        """
        self._var_error_handle()
        m = self.mi.n
        l = self.variance_from_missing()
        # include the coefficient for degrees freedom
        n = self.statistics_["coefficient"].index.size
        # all models same # obs, but can't be sure there's more than 1 model
        k = self.models_[1].nobs
        v = self._degrees_freedom(m, l, n, k)
        return v

    def fraction_missing_info(self):
        """Calculate fraction of missing information, as found in Van Buuren.

        Returns:
            fmi: fraction of missing info

        Raises:
            ValueError: Variance ratios not available unless statsmodels
        """
        r = self.relative_increase_in_variance()
        v = self.degrees_freedom()
        fmi = (r+2/(v+3))/1+r
        return fmi

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

        # add variance ratios if dealing with statsmodels
        if self.model_lib == "statsmodels":
            sdf["var_missing"] = self.variance_from_missing()
            sdf["var_rel_increase"] = self.relative_increase_in_variance()
            sdf["df"] = self.degrees_freedom()
            sdf["fmi"] = self.fraction_missing_info()
        return sdf
