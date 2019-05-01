"""Manage the analysis folder from the autoimpute package.

This module handles imports from the analysis folder that should be accessible
whenever someone imports autoimpute.analysis. The list below specifies the
methods and classes that are available on import.

This module handles `from autoimpute.analysis import *` with the __all__
variable below. This command imports the public classes and methods from
autoimpute.analysis.
"""

from .base_regressor import MiBaseRegressor
from .linear_regressor import MiLinearRegression
from .logistic_regressor import MiLogisticRegression
from .metrics import raw_bias, percent_bias

__all__ = [
    "MiBaseRegressor",
    "MiLinearRegression",
    "MiLogisticRegression",
    "raw_bias",
    "percent_bias"
]
