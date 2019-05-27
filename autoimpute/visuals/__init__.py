"""Manage the visuals lib from the autoimpute package.

This module handles imports from the visuals directory that should be
accessible whenever someone imports autoimpute.visuals. The imports include
methods for visual analysis of missing data, from exploration to analysis.

This module handles `from autoimpute.visuals import *` with the __all__
variable below. This command imports the main public methods from
autoimpute.visuals.
"""

from .utils import plot_md_locations, plot_md_percent
from .utils import plot_nullility_corr, plot_nullility_dendogram
from .imputations import plot_imp_scatter, plot_imp_dists, plot_imp_boxplots
from .imputations import plot_imp_swarm, plot_imp_strip

__all__ = [
    "plot_md_locations",
    "plot_md_percent",
    "plot_nullility_corr",
    "plot_nullility_dendogram",
    "plot_imp_scatter",
    "plot_imp_dists",
    "plot_imp_boxplots",
    "plot_imp_swarm",
    "plot_imp_strip"
]
