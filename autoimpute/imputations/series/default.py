"""This module implements default imputers used for series Imputer classes.

These Imputer classes serve as defaults within more advanced imputers. They
are flexible, and they allow users to quickly run imputations without getting
a runtime error as they would in sklearn if the data types in a dataset are
mixed. There are three default imputers at the moment: DefaultUnivarImputer,
DefaultTimeSeriesImputer and DefaultPredictiveImputer. Default imputers
inherit from DefaultBaseImputer.
"""

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations import method_names
from .pmm import PMMImputer
from .mean import MeanImputer
from .mode import ModeImputer
from .interpolation import InterpolateImputer
from .logistic_regression import MultinomialLogisticImputer
from .base import ISeriesImputer
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=unnecessary-pass
# pylint:disable=dangerous-default-value
# pylint:disable=too-many-instance-attributes

class DefaultBaseImputer(ISeriesImputer):
    """Building blocks for the default imputers.

    The DefaultBaseImputer is not a stand-alone class and thus serves no
    purpose other than as a Parent to DefaultImputers. Therefore, the
    DefaultBaseImputer should not be used directly unless creating a new
    version of a DefaultImputer.
    """
    def __init__(self, num_imputer, cat_imputer, num_kwgs, cat_kwgs):
        """Initialize the DefaultBaseImputer.

        Args:
            num_imputer (Imputer): valid Imputer for numerical data.
            cat_imputer (Imputer): valid Imputer for categorical data.
            num_kwgs (dict): Keyword args for numerical imputer.
            cat_kwgs (dict): keyword args for categorical imputer.

        Returns:
            self. Instance of the class
        """
        # INSTANCE ATTRIBUTES MUST BE IN ORDER THEY ARE VALIDATED WITH GET/SET
        # --------------------------------------------------------------------
        # Position of arguments in __init__ is essentially arbitrary
        # But attribute must appear in proper order if using getters/setters
        self.num_kwgs = num_kwgs
        self.cat_kwgs = cat_kwgs
        self.num_imputer = num_imputer
        self.cat_imputer = cat_imputer

    @property
    def num_kwgs(self):
        """Property getter to return the value of num_kwgs."""
        return self._num_kwgs

    @property
    def cat_kwgs(self):
        """Property getter to return the value of cat_kwgs."""
        return self._cat_kwgs

    @num_kwgs.setter
    def num_kwgs(self, kwgs):
        """Validate the num_kwgs and set default properties."""
        if not isinstance(kwgs, (type(None), dict)):
            err = "num_kwgs must be dict of args used to init num_imputer."
            raise ValueError(err)
        self._num_kwgs = kwgs

    @cat_kwgs.setter
    def cat_kwgs(self, kwgs):
        """Validate the cat_kwgs and set default properties."""
        if not isinstance(kwgs, (type(None), dict)):
            err = "cat_kwgs must be dict of args used to init cat_imputer."
            raise ValueError(err)
        self._cat_kwgs = kwgs

    @property
    def num_imputer(self):
        """Property getter to return the value of the num imputer."""
        return self._num_imputer

    @property
    def cat_imputer(self):
        """Property getter to return the value of the cat imputer."""
        return self._cat_imputer

    @num_imputer.setter
    def num_imputer(self, imp):
        """Validate the num imputer and set default parameters.

        Args:
            imp (Imputer): must be a valid autoimpute Imputer

        Raises:
            ValueError: any Imputer class must end in Imputer
            ValueError: Imputer must implement fit_impute
            ValueError: argument not an instance of an Imputer
        """
        # try necessary because imp may not have __base__ attribute
        try:
            # once imp confirmed class, error handling
            cls_ = imp.__name__.endswith("Imputer")
            if not cls_:
                err = f"{imp} must be a class ending in Imputer"
                raise ValueError(err)
            # valid imputers must have `fit_impute` method
            m = "fit_impute"
            if not hasattr(imp, m):
                err = f"Imputer must implement {m} method."
                raise ValueError(err)
            # if valid imputer, instantiate it with kwargs
            # if kwargs contains improper args, imp will handle error
            if self.num_kwgs is None:
                self._num_imputer = imp()
            else:
                self._num_imputer = imp(**self.num_kwgs)
        # deal with imp that doesn't have __base__ attribute
        except AttributeError as ae:
            err = f"{imp} is not an instance of an Imputer"
            raise ValueError(err) from ae

    @cat_imputer.setter
    def cat_imputer(self, imp):
        """Validate the cat imputer and set default parameters.

        Args:
            imp (Imputer): must be a valid autoimpute imputer

        Raises:
            ValueError: any imputer class must end in Imputer
            ValueError: imputer must implement fit_impute
            ValueError: argument not an instance of an Imputer
        """
        # try necessary because imp could initially be anything
        try:
            # once imp confirmed class, error handling
            cls_ = imp.__name__.endswith("Imputer")
            if not cls_:
                err = f"{imp} must be an Imputer class from autoimpute"
                raise ValueError(err)
            # valid imputers must have `fit_impute` method
            m = "fit_impute"
            if not hasattr(imp, m):
                err = f"Imputer must implement {m} method."
                raise ValueError(err)
            # if valid imputer, instantiate it with kwargs
            # if kwargs contains improper args, imp will handle error
            if self.cat_kwgs is None:
                self._cat_imputer = imp()
            else:
                self._cat_imputer = imp(**self.cat_kwgs)
        except AttributeError as ae:
            err = f"{imp} is not a valid Imputer"
            raise ValueError(err) from ae

    def fit(self, X, y):
        """Fit the Imputer to the dataset and determine the right approach.

        Args:
            X (pd.Series): Dataset to fit the imputer, or predictors
            y (pd.Series): None, or dataset to fit predictors

        Returns:
            self. Instance of the class.
        """
        # start off with stats blank
        stats = {"param": None, "strategy": None}

        # if y is None, fitting simply X. univariate method.
        if y is None:
            if is_numeric_dtype(X):
                stats = {"param": self.num_imputer.fit(X, y),
                         "strategy": self.num_imputer.strategy}
            if is_string_dtype(X):
                stats = {"param": self.cat_imputer.fit(X, y),
                         "strategy": self.cat_imputer.strategy}

        # if y is not None, fitting X to y. predictive method.
        if not y is None:
            if is_numeric_dtype(y):
                stats = {"param": self.num_imputer.fit(X, y),
                         "strategy": self.num_imputer.strategy}
            if is_string_dtype(y):
                stats = {"param": self.cat_imputer.fit(X, y),
                         "strategy": self.cat_imputer.strategy}

        # return final stats
        self.statistics_ = stats
        return self

    def impute(self, X):
        """Perform imputations using the statistics generated from fit.

        The impute method handles the actual imputation. Missing values
        in a given dataset are replaced with the respective mean from fit.

        Args:
            X (pd.Series): Dataset to impute missing data from fit.

        Returns:
            pd.Series -- imputed dataset.
        """
        # check is fitted and delegate transformation to respective imputer
        check_is_fitted(self, "statistics_")
        imp = self.statistics_["param"]

        # ensure that param is not none, which indicates time series column
        if imp:
            X_ = imp.impute(X)
            return X_

    def fit_impute(self, X, y):
        """Convenience method to perform fit and imputation in one go."""
        return self.fit(X, y).impute(X)

class DefaultUnivarImputer(DefaultBaseImputer):
    """Impute missing data using default methods for univariate imputation.

    This imputer is the default for univariate imputation. The imputer
    determines how to impute based on the column type of each column in a
    dataframe. The imputer can be used directly, but such behavior is
    discouraged. DefaultUnivarImputer does not have the flexibility /
    robustness of more complex imputers, nor is its behavior identical.
    Preferred use is MultipleImputer(strategy="default univariate").
    """
    # class variables
    strategy = methods.DEFAULT_UNIVAR

    def __init__(
            self,
            num_imputer=MeanImputer,
            cat_imputer=ModeImputer,
            num_kwgs=None,
            cat_kwgs={"fill_strategy": "random"}
        ):
        """Create an instance of the DefaultUnivarImputer class.

        The dataframe imputers delegate work to the DefaultUnivarImputer if
        strategy="default univariate" The DefaultUnivarImputer then determines
        how to impute numerical and categorical columns by default. It does so
        by passing its arguments to the DefaultBaseImputer, which handles
        validation and instantiation of numerical and categorical imputers.

        Args:
            num_imputer (Imputer, Optional): valid Imputer for numerical data.
                Default is MeanImputer.
            cat_imputer (Imputer, Optional): valid Imputer for categorical
                data. Default is ModeImputer.
            num_kwgs (dict, optional): Keyword args for numerical imputer.
                Default is None.
            cat_kwgs (dict, optional): keyword args for categorical imputer.
                Default is {"fill_strategy": "random"}.

        Returns:
            self. Instance of class.
        """
        # delegate to DefaultBaseImputer
        DefaultBaseImputer.__init__(
            self,
            num_imputer=num_imputer,
            cat_imputer=cat_imputer,
            num_kwgs=num_kwgs,
            cat_kwgs=cat_kwgs
        )

    def fit(self, X, y=None):
        """Defer fit to the DefaultBaseImputer."""
        super().fit(X, y)
        return self

    def impute(self, X):
        """Defer transform to the DefaultBaseImputer."""
        X_ = super().impute(X)
        return X_

class DefaultTimeSeriesImputer(DefaultBaseImputer):
    """Impute missing data using default methods for time series.

    This imputer is the default imputer for time series imputation. The
    imputer determines how to impute based on the column type of each column
    in a dataframe. The imputer can be used directly, but such behavior is
    discouraged. DefaultTimeSeriesImputer does not have the flexibility /
    robustness of more complex imputers, nor is its behavior identical.
    Preferred use is MultipleImputer(strategy="default time").
    """
    # class variables
    strategy = methods.DEFAULT_TIME

    def __init__(
            self,
            num_imputer=InterpolateImputer,
            cat_imputer=ModeImputer,
            num_kwgs={"fill_strategy": "linear"},
            cat_kwgs={"fill_strategy": "random"}
        ):
        """Create an instance of the DefaultTimeSeriesImputer class.

        The dataframe imputers delegate work to the DefaultTimeSeriesImputer
        if strategy="default time". The DefaultTimeSeriesImputer then
        determines how to impute numerical and categorical columns by default.
        It does so by passing its arguments to the DefaultBaseImputer, which
        handles validation and instantiation of default numerical and
        categorical imputers.

        Args:
            num_imputer (Imputer, Optional): valid Imputer for numerical data.
                Default is InterpolateImputer.
            cat_imputer (Imputer, Optional): valid Imputer for categorical
                data. Default is ModeImputer.
            num_kwgs (dict, optional): Keyword args for numerical imputer.
                Default is {"strategy": "linear"}.
            cat_kwgs (dict, optional): keyword args for categorical imputer.
                Default is {"fill_strategy": "random"}.

        Returns:
            self. Instance of class.
        """
        DefaultBaseImputer.__init__(
            self,
            num_imputer=num_imputer,
            cat_imputer=cat_imputer,
            num_kwgs=num_kwgs,
            cat_kwgs=cat_kwgs
        )

    def fit(self, X, y=None):
        """Defer fit to the DefaultBaseImputer."""
        super().fit(X, y)
        return self

    def impute(self, X):
        """Defer transform to the DefaultBaseImputer."""
        X_ = super().impute(X)
        return X_

class DefaultPredictiveImputer(DefaultBaseImputer):
    """Impute missing data using default methods for prediction.

    This imputer is the default imputer for the MultipleImputer class. When
    an end-user does not supply a strategy, the DefaultPredictiveImputer
    determines how to impute based on the column type of each column in a
    dataframe. The imputer can be used directly, but such behavior is
    discouraged. DefaultPredictiveImputer does not have the flexibility /
    robustness of more complex imputers, nor is its behavior identical.
    Preferred use is MultipleImputer(strategy="default predictive").
    """
    # class variables
    strategy = methods.DEFAULT_PRED

    def __init__(
            self,
            num_imputer=PMMImputer,
            cat_imputer=MultinomialLogisticImputer,
            num_kwgs=None,
            cat_kwgs=None
        ):
        """Create an instance of the DefaultPredictiveImputer class.

        The dataframe imputers delegate work to DefaultPredictiveImputer if
        strategy="default predictive" or no strategy given when class is
        instantiated. The DefaultPredictiveImputer determines how to impute
        numerical and categorical columns by default. It does so by passing
        its arguments to the DefaultBaseImputer, which handles validation and
        instantiation of default numerical and categorical imputers.

        Args:
            num_imputer (Imputer, Optional): valid Imputer for numerical data.
                Default is PMMImputer.
            cat_imputer (Imputer, Optional): valid Imputer for categorical
                data. Default is MultiLogisticImputer.
            num_kwgs (dict, optional): Keyword args for numerical imputer.
                Default is None.
            cat_kwgs (dict, optional): keyword args for categorical imputer.
                Default is None.

        Returns:
            self. Instance of class.
        """
        # delegate to DefaultBaseImputer
        DefaultBaseImputer.__init__(
            self,
            num_imputer=num_imputer,
            cat_imputer=cat_imputer,
            num_kwgs=num_kwgs,
            cat_kwgs=cat_kwgs
        )

    def fit(self, X, y):
        """Defer fit to the DefaultBaseImputer."""
        super().fit(X, y)
        return self

    def impute(self, X):
        """Defer transform to the DefaultBaseImputer."""
        X_ = super().impute(X)
        return X_
