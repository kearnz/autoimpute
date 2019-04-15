"""Module sets up AutoImpute regressors for multiply imputed data analysis."""

from statsmodels.api import add_constant
from autoimpute.imputations import MultipleImputer
# pylint:disable=attribute-defined-outside-init

class BaseRegressor:
    """Building blocks to create an AutoImpute regressor.

    Every AutoImpute regressor inherits from the BaseRegressor. The class
    provides the functionality necessary for AutoImpute regressors to wrap
    sklearn or statsmodels libraries and apply them to multiply imputed
    datasets. It also creates the MultipleImputer used to impute data.

    Attributes:
        model_libs (tuple): libraries supported by AutoImpute regressors.
    """

    model_libs = ("sklearn", "statsmodels")

    def __init__(self, model_lib, mi_kwgs, model_kwgs):
        """Create an instance of the BaseRegressor class.

        The BaseRegressor class is not a stand-alone class and should not be
        used other than as a parent class to an AutoImpute regressor. An
        AutoImpute regressor wraps either sklearn or statsmodels regressors to
        apply them on multiply imputed datasets. The BaseRegressor contains
        the logic AutoImpute regressors share. In addition, it creates an
        instance of the MultipleImputer to impute missing data.

        Args:
            model_lib (str): library the regressor will use to implement
                regression. Options are sklearn and statsmodels.
                Default is statsmodels.
            mi_kwgs (dict): keyword args to instantiate MultipleImputer.
            model_kwgs (dict): keyword args to instantiate regressor.

        Returns:
            self. Instance of BaseRegressor class.
        """
        self.mi_kwgs = mi_kwgs
        self.model_kwgs = model_kwgs
        self.model_lib = model_lib
        if self.mi_kwgs:
            self.mi = MultipleImputer(**self.mi_kwgs)
        else:
            self.mi = MultipleImputer()

    @property
    def mi_kwgs(self):
        """Property getter to return the value of mi_kwgs."""
        return self._mi_kwgs

    @mi_kwgs.setter
    def mi_kwgs(self, kwgs):
        """Validate the mi_kwgs and set default properties.

        The BaseRegressor validates the `mi_kwgs` argument. `mi_kwgs` contain
        optional keyword arguments to create a MultipleImputer. The argument
        is optional, and its default is None.

        Args:
            kwgs (dict, None): None or dictionary of keywords.

        Raises:
            ValueError: mi_kwgs not correctly specified as argument.
        """
        if not isinstance(kwgs, (type(None), dict)):
            err = "mi_kwgs must be dict of args used to instantiate Imputer."
            raise ValueError(err)
        self._mi_kwgs = kwgs

    @property
    def model_kwgs(self):
        """Property getter to return the value of model_kwargs."""
        return self._model_kwgs

    @model_kwgs.setter
    def model_kwgs(self, kwgs):
        """Validate the model_kwgs and set default properties.

        The BaseRegressor validates the `model_kwgs` argument. `model_kwgs`
        contain optional keyword arguments to create a regression model. The
        argument is optional, and its default is None

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

        The BaseRegressor validates the `model_lib` argument. `model_lib`
        should be in the BaseRegressor.model_libs tuple, which contains the
        possible libs to use for regression of multiply imputed datasets. The
        library chosen is important. Only statsmodels (the default) provides
        proper parameter pooling using Rubin's rules. sklearn provides mean
        estimate pooling only.

        Args:
            lib (iter): library to use

        Raises:
            ValueError: `lib` not a valid library to use.
        """

        # error handling for improper data types
        if lib not in self.model_libs:
            err = f"{lib} not valid `model_lib`. Must be {self.model_libs}."
            raise ValueError(err)
        self._model_lib = lib

    def _fit_sklearn(self, m, X, y):
        """Private method to fit a model using sklearn."""
        model = m(**self.model_kwgs) if self.model_kwgs else m()
        model.fit(X, y)
        return model

    def _fit_statsmodels(self, m, X, y, const):
        """Private method to fit a model using statsmodels."""

        # add a constant if necessary
        if const:
            X = add_constant(X)
        model = m(y, X, **self.model_kwgs) if self.model_kwgs else m(y, X)
        model = model.fit()
        return model
