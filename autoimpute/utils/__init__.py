"""Manage utils lib"""

# import the main functions users should use directly
from autoimpute.utils.checks import check_data_structure, check_missingness
from autoimpute.utils.patterns import md_pairs, md_pattern, md_locations
from autoimpute.utils.patterns import inbound, outbound, influx, outflux, flux
from autoimpute.utils.patterns import proportions, feature_cov, feature_corr

# override from utils import * with main functions
__all__ = ["check_data_structure", "check_missingness",
           "md_locations", "md_pairs", "md_pattern",
           "feature_corr", "feature_cov", "proportions",
           "inbound", "outbound", "influx", "outflux", "flux"]
