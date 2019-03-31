"""Manage the series imputations folder from the autoimpute package.

This module handles imports from the series imputations folder that should be
accessible whenever someone imports autoimpute.imputations.series. The list
below specifies the methods and classes that are available on import.

Imported:
    DefaultSingleImputer
    DefaultTimeSeriesImputer
    MeanImputer
    MedianImputer
    RandomImputer
    NormImputer
    CategoricalImputer
    NOCBImputer
    LOCFImputer
    InterpolateImputer

This module handles `from autoimpute.imputations.series import *` with the
__all__ variable below. This command imports the main public classes and
methods from autoimpute.imputations.series.
"""

from .default import DefaultSingleImputer, DefaultTimeSeriesImputer
from .mean import MeanImputer
from .median import MedianImputer
from .mode import ModeImputer
from .random import RandomImputer
from .norm import NormImputer
from .categorical import CategoricalImputer
from .ffill import NOCBImputer, LOCFImputer
from .interpolation import InterpolateImputer
from .linear_regression import LeastSquaresImputer, StochasticImputer
from .logistic_regression import BinaryLogisticImputer, MultiLogisticImputer
from .bayesian_regression import BayesLeastSquaresImputer

__all__ = [
    "DefaultSingleImputer",
    "DefaultTimeSeriesImputer",
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
    "MultiLogisticImputer",
    "BayesLeastSquaresImputer"
]
