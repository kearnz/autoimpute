"""Visualizations to explore missingness within a dataset.

This module is a wrapper around the excellent missingno library, which
provides a number of plots to explore missingness within a dataset. This
wrapper handles some basic plot style setting and error handling for the user
that missingno handles differently. The reason we wrap missingno is to fine
tune the package and apply it directly to autoimpute.
"""

import missingno as msno
from autoimpute.utils import check_data_structure
from .helpers import _fully_complete, _default_plot_args

@check_data_structure
def plot_md_locations(data, **kwargs):
    """Plot the locations where data is missing within a DataFrame.

    Args:
        data (pd.DataFrame): DataFrame to plot.
        **kwargs: Keyword arguments for plot. Passed to missingno.matrix.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: missingness location plot.

    Raises:
        TypeError: if data is not a DataFrame. Error raised through decorator.
    """
    _default_plot_args(**kwargs)
    msno.matrix(data, **kwargs)

@check_data_structure
def plot_md_percent(data, **kwargs):
    """Plot the percentage of missing data by column within a DataFrame.

    Args:
        data (pd.DataFrame): DataFrame to plot.
        **kwargs: Keyword arguments for plot. Passed to missingno.bar.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: missingness percent plot.

    Raises:
        TypeError: if data is not a DataFrame. Error raised through decorator.
    """
    _default_plot_args(**kwargs)
    msno.bar(data, **kwargs)

@check_data_structure
def plot_nullility_corr(data, **kwargs):
    """Plot the nullility correlation of missing data within a DataFrame.

    Args:
        data (pd.DataFrame): DataFrame to plot.
        **kwargs: Keyword arguments for plot. Passed to missingno.heatmap.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: nullility correlation plot.

    Raises:
        TypeError: if data is not a DataFrame. Error raised through decorator.
        ValueError: dataset fully observed. Raised through helper method.
    """
    _fully_complete(data)
    _default_plot_args(**kwargs)
    msno.heatmap(data, **kwargs)

@check_data_structure
def plot_nullility_dendogram(data, **kwargs):
    """Plot the nullility dendogram of missing data within a DataFrame.

    Args:
        data (pd.DataFrame): DataFrame to plot.
        **kwargs: Keyword arguments for plot. Passed to missingno.dendogram.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: nullility dendogram plot.

    Raises:
        TypeError: if data is not a DataFrame. Error raised through decorator.
        ValueError: dataset fully observed. Raised through helper method.
    """
    _fully_complete(data)
    _default_plot_args(**kwargs)
    msno.dendrogram(data, **kwargs)
