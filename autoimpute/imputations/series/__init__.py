"""Manage the series imputations folder from the autoimpute package.

This module handles imports from the series imputations folder that should be
accessible whenever someone imports autoimpute.imputations.series. Although
these imputers are stand-alone classes, their direct use is discouraged.
More robust imputers from the dataframe folder delegate work to these imputers
whenever their respective strategies are requested.

This module handles `from autoimpute.imputations.series import *` with the
__all__ variable below. This command imports the main public classes and
methods from autoimpute.imputations.series.
"""

from .default import DefaultUnivarImputer
from .default import DefaultTimeSeriesImputer
from .default import DefaultPredictiveImputer
from .mean import MeanImputer
from .median import MedianImputer
from .mode import ModeImputer
from .random import RandomImputer
from .norm import NormImputer
from .categorical import CategoricalImputer
from .ffill import NOCBImputer, LOCFImputer
from .interpolation import InterpolateImputer
from .linear_regression import LeastSquaresImputer, StochasticImputer
from .logistic_regression import BinaryLogisticImputer
from .logistic_regression import MultinomialLogisticImputer
from .bayesian_regression import BayesianLeastSquaresImputer
from .bayesian_regression import BayesianBinaryLogisticImputer
from .pmm import PMMImputer
from .lrd import LRDImputer
from .norm_unit_variance import NormUnitVarianceImputer

__all__ = [
    "DefaultUnivarImputer",
    "DefaultTimeSeriesImputer",
    "DefaultPredictiveImputer",
    "MeanImputer",
    "MedianImputer",
    "ModeImputer",
    "RandomImputer",
    "NormImputer",
    "CategoricalImputer",
    "NOCBImputer",
    "LOCFImputer",
    "InterpolateImputer",
    "LeastSquaresImputer",
    "StochasticImputer",
    "BinaryLogisticImputer",
    "MultinomialLogisticImputer",
    "BayesianLeastSquaresImputer",
    "BayesianBinaryLogisticImputer",
    "PMMImputer",
    "LRDImputer",
    "NormUnitVarianceImputer",
]
