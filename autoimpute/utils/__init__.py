"""Manage the utils lib from the autoimpute package.

This module handles imports from the utils directory that should be accessible
whenever someone imports autoimpute.utils. The methods included are the check
decorators as well as the major patterns and exploratory data methods. The list
below specifies the methods and classes that are currently available on import.

Imported:
    check_data_structure
    check_missingness
    remove_nan_columns
    md_pairs
    md_pattern
    md_locations
    inbound
    outbound
    influx
    outflux
    flux
    proportions
    feature_cov
    feature_corr

This module handles `from autoimpute.utils import *` with the __all__ variable
below. This command imports the main public methods from autoimpute.utils.
"""

from autoimpute.utils.checks import check_data_structure, check_missingness
from autoimpute.utils.checks import check_nan_columns
from autoimpute.utils.patterns import md_pairs, md_pattern, md_locations
from autoimpute.utils.patterns import inbound, outbound, influx, outflux, flux
from autoimpute.utils.patterns import proportions, feature_cov, feature_corr

__all__ = ["check_data_structure", "check_missingness", "check_nan_columns",
           "md_pairs", "md_pattern", "md_locations",
           "inbound", "outbound", "influx", "outflux", "flux",
           "proportions", "feature_cov", "feature_corr"]
