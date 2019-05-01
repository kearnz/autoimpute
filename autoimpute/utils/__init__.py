"""Manage the utils lib from the autoimpute package.

This module handles imports from the utils directory that should be accessible
whenever someone imports autoimpute.utils. The imports include methods for
checks & validations as well as functions to explore patterns in missing data.

This module handles `from autoimpute.utils import *` with the __all__ variable
below. This command imports the main public methods from autoimpute.utils.
"""

from .checks import check_data_structure, check_missingness
from .checks import check_nan_columns, check_strategy_allowed
from .checks import check_strategy_fit, check_predictors_fit
from .patterns import md_pairs, md_pattern, md_locations
from .patterns import inbound, outbound, influx, outflux, flux
from .patterns import proportions, nullility_cov, nullility_corr

__all__ = [
    "check_data_structure",
    "check_missingness",
    "check_nan_columns",
    "check_strategy_allowed",
    "check_strategy_fit",
    "check_predictors_fit",
    "md_pairs",
    "md_pattern",
    "md_locations",
    "inbound",
    "outbound",
    "influx",
    "outflux",
    "flux",
    "proportions",
    "nullility_cov",
    "nullility_corr"
]
