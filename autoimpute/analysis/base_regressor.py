"""Module to set up Autoimpute regressors for multiply imputed analysis."""

from collections import OrderedDict
import numpy as np
import pandas as pd
from statsmodels.api import add_constant
from sklearn.utils.validation import check_is_fitted
from autoimpute.utils.helpers import _one_hot_encode
from autoimpute.imputations import MiceImputer

# pylint:disable=attribute-defined-outside-init

class MiBaseRegressor:
    """Building blocks to create an Autoimpute regressor.

    Every Autoimpute regressor inherits from the MiBaseRegressor. The class
    provides the functionality necessary for Autoimpute regressors to wrap
    sklearn or statsmodels libraries and apply them to multiply imputed
    datasets. It also creates the MiceImputer used to impute data if the
    user does not specify a custom MiceImputer during instantiation.

    Attributes:
        model_libs (tuple): libraries supported by Autoimpute regressors.
    """

    model_libs = ("sklearn", "statsmodels")

    def __init__(self, mi, model_lib, mi_kwgs, model_kwgs):
        """Create an instance of the MiBaseRegressor class.

        The MiBaseRegressor class is not a stand-alone class and should not be
        used other than as a parent class to an Autoimpute regressor. An
        Autoimpute regressor wraps either sklearn or statsmodels regressors to
        apply them on multiply imputed datasets. The MiBaseRegressor contains
        the logic Autoimpute regressors share. In addition, it creates an
        instance of the MiceImputer to impute missing data.

        Args:
            mi (MiceImputer): An instance of a MiceImputer. If `mi`
                passed explicitly, this `mi` will be used for MultipleImptuer.
                Can use `mi_kwgs` instead, although `mi` is cleaner/preferred.
            model_lib (str): library the regressor will use to implement
                regression. Options are sklearn and statsmodels.
                Default is statsmodels.
            mi_kwgs (dict): keyword args to instantiate MiceImputer.
                If valid MiceImputer passed to `mi`, model_kwgs ignored.
                If `mi_kwgs` is None and `mi` is None, MiBaseRegressor creates
                default instance of MiceImputer.
            model_kwgs (dict): keyword args to instantiate regressor. Arg is
                passed along to either sklearn or statsmodels regressor. If
                `model_kwgs` is None, default instance of regressor created.

        Returns:
            self. Instance of MiBaseRegressor class.
        """
        # Order Important. `mi_kwgs` validation first b/c it's used in `mi`
        # also note - encoder is not argument, b/c one-hot only right now
        self.mi_kwgs = mi_kwgs
        self.mi = mi
        self.model_kwgs = model_kwgs
        self.model_lib = model_lib

    @property
    def mi_kwgs(self):
        """Property getter to return the value of mi_kwgs."""
        return self._mi_kwgs

    @mi_kwgs.setter
    def mi_kwgs(self, kwgs):
        """Validate the mi_kwgs and set default properties.

        The MiBaseRegressor validates mi_kwgs argument. mi_kwgs contain
        optional keyword arguments to create a MiceImputer. The argument
        is optional, and its default is None.

        Args:
            kwgs (dict, None): None or dictionary of keywords.

        Raises:
            ValueError: mi_kwgs not correctly specified as argument.
        """
        if not isinstance(kwgs, (type(None), dict)):
            err = "mi_kwgs must be None or dict of args for MiceImputer."
            raise ValueError(err)
        self._mi_kwgs = kwgs

    @property
    def mi(self):
        """Property getter to return the value of mi."""
        return self._mi

    @mi.setter
    def mi(self, m):
        """Validate mi and set default properties.

        The MiBaseRegressor validates the mi argument. mi must be a valid
        instance of a MiceImputer. It can also be None. If None, the
        MiBaseRegressor will create a MiceImputer on its own, either by
        default or with any key values passed to the mi_kwgs args dict.

        Args:
            m (MiceImputer, None): Instance of a MiceImputer.

        Raises:
            ValueError: mi is not an instance of a MiceImputer.
        """

        # check if m is None or a MiceImputer
        if not isinstance(m, (type(None), MiceImputer)):
            err = f"{m} must be None or a valid instance of MiceImputer."
            raise ValueError(err)

        # handle each case if None or MiceImputer
        if m is not None:
            self._mi = m
        else:
            # handle whether or not mi_kwgs should be passed
            if self.mi_kwgs:
                self._mi = MiceImputer(**self.mi_kwgs)
            else:
                self.mi = MiceImputer()

    @property
    def model_kwgs(self):
        """Property getter to return the value of model_kwargs."""
        return self._model_kwgs

    @model_kwgs.setter
    def model_kwgs(self, kwgs):
        """Validate the model_kwgs and set default properties.

        The MiBaseRegressor validates the model_kwgs argument. model_kwgs
        contain optional keyword arguments pased to a regression model. The
        argument is optional, and its default is None.

        Args:
            kwgs (dict, None): None or dictionary of keywords.

        Raises:
            ValueError: model_kwgs not correctly specified as argument.
        """
        if not isinstance(kwgs, (type(None), dict)):
            err = "model_kwgs must be dict of args used to instantiate model."
            raise ValueError(err)
        self._model_kwgs = kwgs

    @property
    def model_lib(self):
        """Property getter to return the value of model_lib."""
        return self._model_lib

    @model_lib.setter
    def model_lib(self, lib):
        """Validate model_lib and set default properties.

        The MiBaseRegressor validates the model_lib argument. model_lib should
        be in the MiBaseRegressor.model_libs tuple, which contains the libs to
        use for regression of multiply imputed datasets. The library chosen is
        important. Only statsmodels (the default) provides proper parameter
        pooling using Rubin's rules. sklearn provides mean estimate pooling
        only. sklearn variance parameter pooling and diagnostics in dev, TBD.

        Args:
            lib (iter): library to use

        Raises:
            ValueError: lib not a valid library to use.
        """
        if lib not in self.model_libs:
            err = f"{lib} not valid `model_lib`. Must be {self.model_libs}."
            raise ValueError(err)
        self._model_lib = lib

    def _fit_strategy_validator(self, X, y):
        """Private method to validate data before fitting model."""

        # y must be a series or dataframe
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            err = "y must be a Series or DataFrame"
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
            y = y.iloc[:, 0]
            self._yn = y.name

        # y and X must have the same number of rows
        if X.shape[0] != y.shape[0]:
            err = "y and X must have the same number of records"
            raise ValueError(err)

        # if no errors thus far, add y to X for imputation
        X[self._yn] = y

        # return the multiply imputed datasets
        return self.mi.fit_transform(X)

    def _fit_model(self, model_type, regressor, X, y):
        """Private method to fit a model using sklearn or statsmodels."""

        # encoding for predictor variable
        # we enforce that predictors were imputed in imputation phase.
        X = _one_hot_encode(X)
        self.new_X_columns = X.columns.tolist()

        # encoding for response variable
        if model_type == "logistic":
            ycat = y.astype("category").cat
            y = ycat.codes
            self._response_categories = ycat.categories

        # statsmodels fit case, which requires different logic than sklearn
        if self.model_lib == "statsmodels":
            X = add_constant(X)
            if self.model_kwgs:
                model = regressor(y, X, **self.model_kwgs)
            else:
                model = regressor(y, X)
            model = model.fit()

        # sklearn fit case, which requires different logic than statsmodels
        if self.model_lib == "sklearn":
            if self.model_kwgs:
                model = regressor(**self.model_kwgs)
            else:
                model = regressor()
            # sklearn doesn't need encoding for response
            model.fit(X, y)

        # return the model after fitting it to a given dataset
        return model

    def _apply_models_to_mi_data(self, model_dict, X, y):
        """Private method to apply analysis model to multiply imputed data."""

        # find regressor based on model lib, then get mutliply imputed data
        model_type = model_dict["type"]
        regressor = model_dict[self.model_lib]
        mi_data = self._fit_strategy_validator(X, y)
        models = {}

        # then preform analysis models. Sequential only right now.
        for dataset in mi_data:
            ind, X = dataset
            y = X.pop(self._yn)
            model = self._fit_model(model_type, regressor, X, y)
            models[ind] = model

        # returns a dictionary: k=imp #; v=analysis model applied to imp #
        return models

    def _predict_strategy_validator(self, instance, X):
        """Private method to validate before prediction."""

        # first check that model is fitted, then check columns are the same
        check_is_fitted(instance, "statistics_")
        X_cols = X.columns.tolist()
        fit_cols = set(instance.fit_X_columns)
        diff_fit = set(fit_cols).difference(X_cols)
        if diff_fit:
            err = "Same columns that were fit must appear in predict."
            raise ValueError(err)

        # encoding for predictor variable
        # we enforce that predictors were imputed in imputation phase.
        if X.isnull().sum().any():
            me = "Data passed to make predictions can't contain missingness."
            raise ValueError(me)
        X = _one_hot_encode(X)
        return X

    def _var_ratios(self, imps, num, denom):
        """Private method for the variance ratios."""
        return (num+(num/imps))/denom

    def _degrees_freedom(self, imps, lambda_, v_com):
        """Private method to calculate degrees of freedom for estimates."""

        # note we nudge lambda if zero b/c need lambda for other stats
        # see source code barnard.rubin.R from MICE for more
        lambda_ = np.maximum(1e-04, lambda_)
        v_old = (imps-1)/lambda_**2
        v_obs = ((v_com+1)/(v_com+3))*v_com*(1-lambda_)
        v = (v_old*v_obs)/(v_old+v_obs)
        return v

    def _get_stats_from_models(self, models):
        """Private method to generate statistics given on model lib chosen."""

        # initial setup - get items from models and get number of models
        items = models.items()
        m = self.mi.n

        # pooling phase: sklearn - coefficients only, no variance
        if self.model_lib == "sklearn":

            # find basic parameters, but can't return much more than coeff
            # sklearn does not implement inference out of the box
            # will have to write methods to do so from stratch, so TBD
            self.mi_alphas_ = [j.intercept_ for i, j in items]
            self.mi_params_ = [j.coef_ for i, j in items]
            alpha = sum(self.mi_alphas_) / m
            params = sum(self.mi_params_) / m
            coefs = pd.Series(np.insert(params, 0, alpha))
            coefs.index = ["const"] + self.new_X_columns
            statistics = OrderedDict(
                coefs=coefs
            )

        # pooling phase: statsmodels - coefficients and variance possible
        if self.model_lib == "statsmodels":

            # data and model parameters
            self.mi_params_ = [j.params for i, j in items]
            self.mi_std_errors_ = [j.bse for i, j in items]
            coefs = sum(self.mi_params_)/ m
            k = coefs.index.size
            n = list(items)[0][1].nobs
            df_com = n-k

            # variance metrics (See VB Ch 2.3)
            vw = sum(map(lambda x: x**2, self.mi_std_errors_))/m
            vb = sum(map(lambda p: (p-coefs)**2, self.mi_params_))/max(1, m-1)
            vt = vw + vb + (vb / m)
            stdt = np.sqrt(vt)

            # variance ratios (See VB Ch 2.3)
            # efficiency as specified in stats manual
            lambda_ = self._var_ratios(m, vb, vt)
            r_ = self._var_ratios(m, vb, vw)
            v_ = self._degrees_freedom(m, lambda_, df_com)
            fmi_ = ((v_+1)/(v_+3))*lambda_ + 2/(v_+3)
            eff_ = (1+(np.maximum(1e-04, fmi_)/m))**-1

            # create statistics with pooled metrics from above
            statistics = OrderedDict(
                coefs=coefs,
                std=stdt,
                vw=vw,
                vb=vb,
                vt=vt,
                dfcom=df_com,
                dfadj=v_,
                lambda_=lambda_,
                riv=r_,
                fmi=fmi_,
                eff=eff_
            )

        # finally, return dictionary with stats from fit used in transform
        return statistics
