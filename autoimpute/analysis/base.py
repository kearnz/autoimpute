"""Module sets up AutoImpute regressors for multiply imputed data analysis."""

from types import GeneratorType
# pylint:disable=attribute-defined-outside-init

class BaseRegressor:
    """Building blocks to create an AutoImpute regressor.

    Every AutoImpute regressor inherits from the BaseRegressor. The class
    provides the functionality necessary for AutoImpute regressors to wrap
    sklearn or statsmodels libraries and apply them to multiply imputed
    datasets.

    Attributes:
        model_libs (tuple): libraries supported by AutoImpute regressors.
    """

    model_libs = ("sklearn", "statsmodels")

    def __init__(self, mi_data, model_lib):
        """Create an instance of the BaseRegressor class.

        The BaseRegressor class is not a stand-alone class and should not be
        used other than as a parent class to an AutoImpute regressor. An
        AutoImpute regressor wraps either sklearn or statsmodels regressors to
        apply them on multiply imputed datasets. The BaseRegressor contains
        the logic AutoImpute regressors share.

        Args:
            mi_data (iter): an iterator containing m imputed datasets.
                If directly from `fit_transform` method, will be a generator.
                Also acceptable for mi_data to be tuple or a list.
            model_lib (str): library the regressor will use to implement
                regression. Options are sklearn and statsmodels.

        Returns:
            self. Instance of BaseRegressor class.
        """
        self.mi_data = mi_data
        self.model_lib = model_lib

    @property
    def mi_data(self):
        """Property getter to return the value of mi_data."""
        return self._mi_data

    @mi_data.setter
    def mi_data(self, data):
        """Validate mi_data and set default properties.

        The BaseRegressor validates the `mi_data` argument. `mi_data` contains
        m imputed datasets, where m>=1. `mi_data` should come directly from
        `fit_transform` method of the MultipleImputer class.

        Args:
            data (iter): multiply imputed dataset with m DataFrames.

        Raises:
            ValueError: `mi_data` not a valid iterator of dataframes.
        """

        # error handling for improper data types
        if not isinstance(data, (list, tuple, GeneratorType)):
            err = "mi_data must be iterator of imupted dataframes."
            raise ValueError(err)
        self._mi_data = data

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
