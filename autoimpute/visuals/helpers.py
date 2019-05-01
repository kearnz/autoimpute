"""Helper functions used throughout other methods in automipute.visuals."""

import pandas as pd

def _fully_complete(data):
    """Private method to exit plotting and raise error if no data missing."""
    if not pd.isnull(data).sum().any():
        err = "No data is missing in any column. Cannot generate plot."
        raise ValueError(err)

def _default_plot_args(**kwargs):
    """Private method to set up the default plot style arguments."""
    defaults = {}
    defaults["figure.figsize"] = kwargs.pop("figsize", (12, 8))
    return defaults
