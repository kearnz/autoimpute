"""Manage the utils lib from the autoimpute package.

This module handles imports from the utils library that should be accessible
whenever someone imports autoimpute.utils. The methods included are the check
decorators as well as the major patterns / exploratory data mechanisms.

The module also overrides the `from utils import *` with the __all__ var
"""

from autoimpute.utils.checks import check_data_structure, check_missingness
from autoimpute.utils.checks import remove_nan_columns
from autoimpute.utils.patterns import md_pairs, md_pattern, md_locations
from autoimpute.utils.patterns import inbound, outbound, influx, outflux, flux
from autoimpute.utils.patterns import proportions, feature_cov, feature_corr

__all__ = ["check_data_structure", "check_missingness", "remove_nan_columns",
           "md_locations", "md_pairs", "md_pattern",
           "feature_corr", "feature_cov", "proportions",
           "inbound", "outbound", "influx", "outflux", "flux"]
