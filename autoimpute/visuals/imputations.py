"""Visualizations to explore imputations of an incomplete dataset."""

import matplotlib.pylab as plt
import seaborn as sns
from .helpers import _validate_data, _validate_kwgs, _get_observed, _melt_df
from .helpers import _default_plot_args, _plot_imp_dists_helper

#pylint:disable=unused-variable
#pylint:disable=too-many-arguments

def plot_imp_dists(d, mi, imp_col, include_observed=True,
                   separate_observed=True, side_by_side=False,
                   hist_observed=False, hist_imputed=False,
                   gw=(.5, .5), gh=(.5, .5), **plot_kwgs):
    """Plot the density between imputations for a given column.

    Use this method to plot the density of a given column after multiple
    imputation. The function allows the user to also plot the observed data
    from the column prior to imputation taking place. Further, the user can
    specify whether the observed should be separated into its own plot or not.

    Args:
        d (list): dataset returned from multiple imputation.
        mi (MultipleImputer): multiple imputer used to generate d.
        imp_col (str): column to plot. Should be a column with imputations
        include_observed (bool, Optional): whether or not to include observed
            data in the plot. Default is True. If False, observed data for
            imp_col will not be included as a distribution for density.
        separate_observed (bool, Optional): whether or not to separate the
            observed data when plotting against imputed. Default is True. If
            False, observed data distribution will be plotted on same plot
            as the imputed data distribution. Note, this attribute matters if
            and only if `include_observed=True`.
        side_by_side (bool, Optional): whether columns should be plotted next
            to each other or stacked vertically. Default is False. If True,
            plots will be plotted side-by-side. Note, this attribute matters
            if and only if `include_observed=True`.
        hist_observed (bool, Optional): whether histogram should be plotted
            along with the density for observed values. Default is False.
            Note, this attribute matters if and only if
            `include_observed=True`.
        hist_imputed (bool, Optional): whether histogram should be plotted
            along with the density for imputed values. Default is False. Note,
            this attribute matters if and only if `include_observed=True`.
        gw (tuple, Optional): if side-by-side plot, the width ratios for each
            plot. Default is (.5, .5), so each plot will be same width.
            Matters if and only if `include_observed=True` and
            `side_by_side=True`.
        gh (tuple, Optional): if stacked plot, the height ratios for each plot.
            Default is (.5, .5), so each plot will be the same height.
            Matters if and only if `include_observed=True` and
            `side_by_side=False`.
        **plot_kwgs: keyword arguments used by sns.set.

    Returns:
        sns.distplot: densityplot for observed and/or imputed data

    Raises:
        ValueError: see _validate_data method
    """

    # start by setting plot kwgs
    sns.set(rc=_default_plot_args(**plot_kwgs))

    # define the functionality if observed should be included
    if include_observed:
        obs = _get_observed(d, mi, imp_col)
        obs = d[0][1].loc[obs, imp_col]

        # define the functionality if separate observed
        if separate_observed:
            g = {}
            g["w"] = {"width_ratios": gw}
            g["h"] = {"height_ratios": gh}

            # define the functionality if side by side or not
            if side_by_side:
                f, ax = plt.subplots(1, 2, gridspec_kw=g["w"])
            else:
                f, ax = plt.subplots(2, 1, gridspec_kw=g["h"])
            sns.distplot(obs, hist=hist_observed, ax=ax[0], label="Observed")
            _plot_imp_dists_helper(d, hist_imputed, imp_col, ax[1])

        # handle case where not separated
        else:
            sns.distplot(obs, hist=hist_observed, label="Observed")
            _plot_imp_dists_helper(d, hist_imputed, imp_col)

    # handle case where not observed
    else:
        _validate_data(d, mi, imp_col)
        _plot_imp_dists_helper(d, hist_imputed, imp_col)
    plt.legend()

def plot_imp_boxplots(d, mi, imp_col, side_by_side=False,
                      obs_kwgs=None, imp_kwgs=None, **plot_kwgs):
    """Plot the boxplots between observed and imputations for a given column.

    Use this method to plot the boxplots of a given column after multiple
    imputation. The function also plots the boxplot of the observed data from
    the column prior to imputation taking place. Further, the user can specify
    additional arguments to tailor the design of the plots themselves.

    Args:
        d (list): dataset returned from multiple imputation.
        mi (MultipleImputer): multiple imputer used to generate d.
        imp_col (str): column to plot. Should be a column with imputations.
        side_by_side (bool, Optional): whether columns should be plotted next
            to each other or stacked vertically. Default is False. If True,
            plots will be plotted side-by-side.
        obs_kwgs (dict, Optional): dictionary of arguments to unpack for
            observed boxplot. Default is None, so no additional tailoring.
        imp_kwgs (dict, Optional): dictionary of arguments to unpack for
            imputed boxplots. Default is None, so no additional tailoring.
        **plot_kwgs: keyword arguments used by sns.set.
    """

    # set plot type and define names necessary
    sns.set(rc=_default_plot_args(**plot_kwgs))
    obs = _get_observed(d, mi, imp_col)
    obs_ = d[0][1].loc[obs, imp_col].copy().to_frame()
    obs_["obs"] = "obs"
    n = len(d)
    ratio = 1/(n+1)
    g = (ratio, 1-ratio)
    datasets_merged = _melt_df(d, mi, imp_col)

    # validate obs_kwgs, imp_kwgs
    _validate_kwgs(obs_kwgs)
    _validate_kwgs(imp_kwgs)

    # deal with plotting side by side
    if side_by_side:
        xo = "obs"
        yo = imp_col
        yi = imp_col
        xi = "imp_num"
        f, ax = plt.subplots(
            1, 2, gridspec_kw={"width_ratios": (ratio, 1-ratio)}
        )
    else:
        xo = imp_col
        yo = "obs"
        yi = "imp_num"
        xi = imp_col
        f, ax = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": (ratio, 1-ratio)}
        )

    # dealing with plotting with or without kwgs
    if not obs_kwgs is None:
        sns.boxplot(
            x=xo, y=yo, data=obs_, ax=ax[0], **obs_kwgs
        ).set(xlabel="Observed")
    else:
        sns.boxplot(
            x=xo, y=yo, data=obs_, ax=ax[0]
        ).set(xlabel="Observed")
    if not imp_kwgs is None:
        sns.boxplot(
            x=xi, y=yi, data=datasets_merged, ax=ax[1], **imp_kwgs
        ).set(xlabel="Imputed")
    else:
        sns.boxplot(
            x=xi, y=yi, data=datasets_merged, ax=ax[1]
        ).set(xlabel="Imputed")
