"""Manage the autoimpute package.

This module handles imports that should be accessible from the top level.
Because this package contains specific directories for imputation, diagnostics,
exploratory data analysis, and visualizations, no classes or methods should be
accessible from autoimpute itself. Users of this package should import specific
classes or functions they need from the appropriate folder.

Examples of correctly specified imports:
    - import autoimupte.imputations as ai
    - from autoimpute.imputations import MissingnessClassifier
    - from autoimpute.utils import md_pairs, md_pattern, flux

Examples of incorrectly specified imports:
    - import autoimpute as ai (gives folder access only)
    - from autoimpute import * (wildcard imports discouraged and overridden)

This module handles `from autoimpute import *` with the __all__ variable below.
This command imports the major directories from autoimpute.
"""

from .__version__ import __version__

__all__ = [
    "utils",
    "visuals",
    "imputations",
    "analysis"
]
