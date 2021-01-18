"""Module for BaseImputer - a base class for DataFrame imputers.

This module contains the `BaseImputer`, which is used to abstract away
functionality in both DataFrame imputers. The `BaseImputer` also holds the
methods available for imputation analysis.
"""

import warnings
from autoimpute.utils import check_strategy_allowed
from autoimpute.imputations import method_names
from ..series import DefaultUnivarImputer, DefaultPredictiveImputer
from ..series import DefaultTimeSeriesImputer
from ..series import MeanImputer, MedianImputer, ModeImputer
from ..series import NormImputer, CategoricalImputer
from ..series import RandomImputer, InterpolateImputer
from ..series import LOCFImputer, NOCBImputer
from ..series import LeastSquaresImputer, StochasticImputer
from ..series import PMMImputer, LRDImputer
from ..series import BinaryLogisticImputer, MultinomialLogisticImputer
from ..series import BayesianLeastSquaresImputer
from ..series import BayesianBinaryLogisticImputer
from ..series import NormUnitVarianceImputer
methods = method_names

# pylint:disable=attribute-defined-outside-init
# pylint:disable=too-many-arguments
# pylint:disable=too-many-instance-attributes
# pylint:disable=inconsistent-return-statements

class BaseImputer:
    """Building blocks for more advanced DataFrame imputers.

    The BaseImputer is not a stand-alone class and thus serves no purpose
    other than as a parent to Imputers. Therefore, the BaseImputer should not
    be used directly unless creating an Imputer. That being said, all
    DataFrame Imputers should inherit from BaseImputer. It contains base
    functionality for any new DataFrame Imputer, and it holds the set of
    strategies that make up this imputation library.

    Attributes:
        univariate_strategies (dict): univariate imputation methods.
            |  Key = imputation name; Value = function to perform imputation.
            |  `univariate default` mean for numerical, mode for categorical.
            |  `time default` interpolate for numerical, mode for categorical.
            |  `mean` imputes missing values with the average of the series.
            |  `median` imputes missing values with the median of the series.
            |  `mode` imputes missing values with the mode of the series.
            |     Method handles more than one mode (see ModeImputer for info).
            |  `random` imputes random choice from set of series unique vals.
            |  `norm` imputes series w/ random draws from normal distribution.
            |     Mean and std calculated from observed values of the series.
            |  `categorical` imputes series using random draws from pmf.
            |     Proportions calculated from non-missing category instances.
            |  `interpolate` imputes series using chosen interpolation method.
            |     Default is linear. See InterpolateImputer for more info.
            |  `locf` imputes series carrying last observation moving forward.
            |  `nocb` imputes series carrying next observation moving backward.
            |  `normal unit variance` imputes using unit variance w/ norm.
        predictive_strategies (dict): predictive imputation methods.
            |  Key = imputation name; Value = function to perform imputation.
            |  `predictive default` pmm for numerical,logistic for categorical.
            |  `least squares` predict missing values from linear regression.
            |  `binary logistic` predict missing values with 2 classes.
            |  `multinomial logistic` predict missing values with multiclass.
            |  `stochastic` linear regression+random draw from norm w/ mse std.
            |  `bayesian least squares` draw from the posterior predictive
            |     distribution for each missing value, using OLS model.
            |  `bayesian binary logistic` draw from the posterior predictive
            |     distribution for each missing value, using logistic model.
            |  `pmm` imputes series using predictive mean matching. PMM is a
            |     semi-supervised method using bayesian & hot-deck imputation.
            |  `lrd` imputes series using local residual draws. LRD is a
            |     semi-supervised method using bayesian & hot-deck imputation.
    """
    univariate_strategies = {
        methods.DEFAULT_UNIVAR: DefaultUnivarImputer,
        methods.DEFAULT_TIME: DefaultTimeSeriesImputer,
        methods.MEAN: MeanImputer,
        methods.MEDIAN: MedianImputer,
        methods.MODE:  ModeImputer,
        methods.RANDOM: RandomImputer,
        methods.NORM: NormImputer,
        methods.CATEGORICAL: CategoricalImputer,
        methods.INTERPOLATE: InterpolateImputer,
        methods.LOCF: LOCFImputer,
        methods.NOCB: NOCBImputer,
        methods.NORM_UNIT_VARIANCE: NormUnitVarianceImputer,
    }

    predictive_strategies = {
        methods.DEFAULT_PRED: DefaultPredictiveImputer,
        methods.LS: LeastSquaresImputer,
        methods.STOCHASTIC: StochasticImputer,
        methods.BINARY_LOGISTIC: BinaryLogisticImputer,
        methods.MULTI_LOGISTIC: MultinomialLogisticImputer,
        methods.BAYESIAN_LS: BayesianLeastSquaresImputer,
        methods.BAYESIAN_BINARY_LOGISTIC: BayesianBinaryLogisticImputer,
        methods.PMM: PMMImputer,
        methods.LRD: LRDImputer
    }

    strategies = {**predictive_strategies, **univariate_strategies}

    visit_sequences = (
        "default",
        "left-to-right"
    )

    def __init__(self, strategy, imp_kwgs, visit):
        """Initialize the BaseImputer.

        Args:
            strategy (str, iter, dict; optional): strategies for imputation.
                Default value is str -> `predictive default`.
                If str, single strategy broadcast to all series in DataFrame.
                If iter, must provide 1 strategy per column. Each method w/in
                iterator applies to column with same index value in DataFrame.
                If dict, must provide key = column name, value = imputer.
                Dict the most flexible and PREFERRED way to create custom
                imputation strategies if not using the default. Dict does not
                require method for every column; just those specified as keys.
            imp_kwgs (dict, optional): keyword arguments for each imputer.
                Default is None, which means default imputer created to match
                specific strategy. imp_kwgs keys can be either columns or
                strategies. If strategies, each column given that strategy is
                instantiated with same arguments.
            visit (str, None): order to visit columns for imputation.
                Default is `default`, which implements `left-to-right`.
                More strategies (random, monotone, etc.) TBD.
        """
        self.strategy = strategy
        self.imp_kwgs = imp_kwgs
        self.visit = visit

    @property
    def strategy(self):
        """Property getter to return the value of the strategy property."""
        return self._strategy

    @strategy.setter
    def strategy(self, s):
        """Validate the strategy property to ensure it's type and value.

        Class instance only possible if strategy is proper type, as outlined
        in the init method. Passes supported strategies and user arg to
        helper method, which performs strategy checks.

        Args:
            s (str, iter, dict): Strategy passed as arg to class instance.

        Raises:
            ValueError: Strategies not valid (not in allowed strategies).
            TypeError: Strategy must be a string, tuple, list, or dict.
            Both errors raised through helper method `check_strategy_allowed`.
        """
        strat_names = self.strategies.keys()
        self._strategy = check_strategy_allowed(strat_names, s)

    @property
    def imp_kwgs(self):
        """Property getter to return the value of imp_kwgs."""
        return self._imp_kwgs

    @imp_kwgs.setter
    def imp_kwgs(self, kwgs):
        """Validate the imp_kwgs and set default properties.

        The BaseImputer validates the `imp_kwgs` argument. `imp_kwgs` contain
        optional keyword arguments for an imputers' strategies or columns. The
        argument is optional, and its default is None.

        Args:
            kwgs (dict, None): None or dictionary of keywords.

        Raises:
            ValueError: imp_kwgs not correctly specified as argument.
        """
        if not isinstance(kwgs, (type(None), dict)):
            err = "imp_kwgs must be dict of args used to instantiate Imputer."
            raise ValueError(err)
        self._imp_kwgs = kwgs

    @property
    def visit(self):
        """Property getter to return the value of the visit property."""
        return self._visit

    @visit.setter
    def visit(self, v):
        """Validate the visit property to ensure it's type and value.

        Class instance only possible if visit is proper type, as outlined in
        the init method. Visit property must be one of valid sequences in the
        `visit_sequences` variable.

        Args:
            v (str): Visit sequence passed as arg to class instance.

        Raises:
            TypeError: visit sequence must be a string.
            ValueError: visit sequenece not in `visit_sequences`.
        """

        # deal with type first
        if not isinstance(v, str):
            err = "visit must be a string specifying visit sequence to use."
            raise TypeError(err)

        # deal with value next
        if v not in self.visit_sequences:
            err = f"visit not valid. Must be one of {self.visit_sequences}"
            raise ValueError(err)

        # otherwise, set property for visit
        self._visit = v

    def _fit_init_params(self, column, method, kwgs):
        """Private method to supply imputation model fit params if any."""

        # first, handle easy case when no kwargs given
        if kwgs is None:
            final_params = kwgs

        # next, check if any kwargs for a given Imputer method type
        # then, override those parameters if specific column kwargs supplied
        if isinstance(kwgs, dict):
            initial_params = kwgs.get(method, None)
            final_params = kwgs.get(column, initial_params)

        # final params must be None or a dictionary of kwargs
        # this additional validation step is crucial to dictionary unpacking
        if not isinstance(final_params, (type(None), dict)):
            err = "Additional params must be dict of args used to init model."
            raise ValueError(err)
        return final_params

    def _check_if_single_dummy(self, col, X):
        """Private method to check if encoding results in single cat."""
        cats = X.columns.tolist()
        if len(cats) == 1:
            c = cats[0]
            msg = f"{c} only category for feature {col}."
            cons = f"Consider removing {col} from dataset."
            warnings.warn(f"{msg} {cons}")
