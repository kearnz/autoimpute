"""This module implements default imputers used for multiple Imputer classes.

The default imputers serve as defaults within more advanced imputers. They
are flexible, and they allow users to quickly run imputations without getting
a runtime error as they would in sklearn if the data types in a dataset are
mixed. There are three default imputers at the moment: DefaultSingleImputer,
DefaultTimeSeriesImputer, and DefaultPredictiveImputer. They all inherit from
the DefaultBaseImputer.
"""

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from autoimpute.imputations.mean_imputer import MeanImputer
from autoimpute.imputations.mode_imputer import ModeImputer
from autoimpute.imputations.interpolation_imputer import InterpolateImputer
from autoimpute.imputations import method_names
methods = method_names
# pylint:disable=attribute-defined-outside-init
# pylint:disable=unnecessary-pass
# pylint:disable=dangerous-default-value

class DefaultBaseImputer:
    """Building blocks for the default Single, Time, and Predictive Imputers.

    The DefaultBaseImputer is not a stand-alone class and thus serves no
    purpose other than as a Parent to DefaultImputers. Therefore, the
    DefaultBaseImputer should not be used directly unless creating a new
    version of a DefaultImputer.
    """
    def __init__(self, num_imputer, cat_imputer, num_kwgs, cat_kwgs):
        """Initialize the DefaultBaseImputer.

        Args:
            num_imputer (Imputer): valid Imputer class for numerical data.
            cat_imputer (Imputer): valid Imputer class for categorical data.
            num_kwgs (dict, optional): Keyword args for numerical imputer.
            cat_kwgs (dict, optional): keyword args for categorical imputer.

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
            ValueError: Imputer must implement fit_transform
            ValueError: argument not an instance of an Imputer
        """
        # try necessary because imp may not have __base__ attribute
        try:
            # once imp confirmed class, error handling
            cls_ = imp.__name__.endswith("Imputer")
            if not cls_:
                err = f"{imp} must be a class ending in Imputer"
                raise ValueError(err)
            m = "fit_transform"
            if not hasattr(imp, m):
                err = f"Imputer must implement {m} method."
                raise ValueError(err)
            # if valid imputer, instantiate it with kwargs
            # if kwargs contains improper args, imp will handle error
            init_imp = imp() if self.num_kwgs is None else imp(**self.num_kwgs)
            self._num_imputer = init_imp
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
            ValueError: imputer must implement fit_transform
            ValueError: argument not an instance of an Imputer
        """
        # try necessary because imp could initially be anything
        try:
            # once imp confirmed class, error handling
            cls_ = imp.__name__.endswith("Imputer")
            if not cls_:
                err = f"{imp} must be an Imputer class from autoimpute"
                raise ValueError(err)
            m = "fit_transform"
            if not hasattr(imp, m):
                err = f"Imputer must implement {m} method."
                raise ValueError(err)
            # if valid imputer, instantiate it with kwargs
            # if kwargs contains improper args, imp will handle error
            init_imp = imp() if self.cat_kwgs is None else imp(**self.cat_kwgs)
            self._cat_imputer = init_imp
        except AttributeError as ae:
            err = f"{imp} is not a valid Imputer"
            raise ValueError(err) from ae

    def fit(self, X):
        """Fit the Imputer to the dataset and determine the right approach.

        Args:
            X (pd.Series): Dataset to fit the imputer

        Returns:
            self. Instance of the class.
        """
        # delegate numeric features to the num imputer
        if is_numeric_dtype(X):
            stats = {"param": self.num_imputer.fit(X),
                     "strategy": self.num_imputer.strategy}
        # delegate categorical features to the cat imputer
        elif is_string_dtype(X):
            stats = {"param": self.cat_imputer.fit(X),
                     "strategy": self.cat_imputer.strategy}
        # time series does not need imputation, as we require it full
        else:
            stats = {"param": None, "strategy": None}
        self.statistics_ = stats
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
        # check is fitted and delegate transformation to respective imputer
        check_is_fitted(self, "statistics_")
        imp = self.statistics_["param"]

        # ensure that param is not none, which indicates time series column
        if imp:
            imp.transform(X)
        return X

class DefaultSingleImputer(DefaultBaseImputer, BaseEstimator,
                           TransformerMixin):
    """Techniques to impute cross sectional dataset when no strategy given.

    More complex autoimpute Imputers delegate work to the DefaultSingleImputer
    if no strategy is specified from the SingleImputer. The imputer picks a
    basic strategy appropriate for the data type of the column it's imputing.
    That being said, it is a stand-alone class and valid sklearn transformer.
    It can be used directly, but such behavior is discouraged because it
    supports Series only. It does not have the flexibility or robustness of
    more complex imputers, nor is its behavior identical.
    Instead, use SingleImputer(strategy="default").
    """
    # class variables
    strategy = methods.DEFAULT

    def __init__(
            self,
            num_imputer=MeanImputer,
            cat_imputer=ModeImputer,
            num_kwgs=None,
            cat_kwgs={"fill_strategy": "random"}):
        """Create an instance of the DefaultSingleImputer class."""
        DefaultBaseImputer.__init__(
            self,
            num_imputer=num_imputer,
            cat_imputer=cat_imputer,
            num_kwgs=num_kwgs,
            cat_kwgs=cat_kwgs
        )

    def fit(self, X):
        """Defer fit to the DefaultBaseImputer."""
        super().fit(X)
        return self

    def transform(self, X):
        """Defer transform to the DefaultBaseImputer."""
        super().transform(X)
        return X


class DefaultTimeSeriesImputer(DefaultBaseImputer, BaseEstimator,
                               TransformerMixin):
    """Techniques to impute time-based dataset when no strategy given.

    More complex autoimpute Imputers delegate work to DefaultTimeSeriesImputer
    if no strategy is specified from the TimeSeriesImputer. The imputer picks a
    basic strategy appropriate for the data type of the column it's imputing.
    That being said, it is a stand-alone class and valid sklearn transformer.
    It can be used directly, but such behavior is discouraged because it
    supports Series only. It does not have the flexibility or robustness of
    more complex imputers, nor is its behavior identical.
    Instead, use TimeSeriesImputer(strategy="default").
    """
    # class variables
    strategy = methods.DEFAULT

    def __init__(
            self,
            num_imputer=InterpolateImputer,
            cat_imputer=ModeImputer,
            num_kwgs={"strategy": "linear"},
            cat_kwgs={"fill_strategy": "random"}
        ):
        """Create an instance of the DefaultTimeSeriesImputer class."""
        DefaultBaseImputer.__init__(
            self,
            num_imputer=num_imputer,
            cat_imputer=cat_imputer,
            num_kwgs=num_kwgs,
            cat_kwgs=cat_kwgs
        )

    def fit(self, X):
        """Defer fit to the DefaultBaseImputer."""
        super().fit(X)
        return self

    def transform(self, X):
        """Defer transform to the DefaultBaseImputer."""
        super().transform(X)
        return X
