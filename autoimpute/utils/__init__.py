"""Manage utils lib"""

# import the main functions users should use directly
from .checks import check_data_structure, check_dimensions
from .patterns import md_pairs, md_pattern, proportions
from .patterns import inbound, outbound, influx, outflux, flux

# override from utils import * with main functions
__all__ = ["check_data_structure", "check_dimensions",
           "md_pairs", "md_pattern", "proportions",
           "inbound", "outbound", "influx", "outflux", "flux"]
