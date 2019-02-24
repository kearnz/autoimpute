"""Manage the autoimpute package.

This module handles imports that should be accessible from the top level.
B/c this pacakge performs imputations, diagnostics, EDA, and visualizations,
no classes or methods are accessible from autoimpute itself. Users of this
package should import specific classes or functions they need from the
appropriate folder.

Examples:
    from autoimpute.imputations import MissingnessClassifier
    from autoimpute.utils import md_pairs, md_pattern, flux

The module also overrides the `from autoimpute import *` with the __all__ var.
"""

__all__ = ["utils", "visuals", "imputations"]
