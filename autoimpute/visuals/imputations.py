"""Visualizations to explore imputations of an incomplete dataset."""

import matplotlib.pylab as plt
import seaborn as sns
from autoimpute.utils import check_data_structure
from autoimpute.imputations import SingleImputer
from .helpers import _validate_data, _validate_kwgs, _get_observed, _melt_df
from .helpers import _default_plot_args, _plot_imp_dists_helper

#pylint:disable=unused-variable
#pylint:disable=too-many-arguments
#plyint:disable=too-many-locals

@check_data_structure
def plot_imp_scatter(d, x, y, strategy, color=None,
                     title="Jointplot after Imputation",
                     h=8.27, imp_kwgs=None, a=0.5, marginals=None,
                     obs_color="navy", imp_color="red", **plot_kwgs):
    """Plot the joint scatter and density plot after single imputation.

    Use this method to visualize a scatterplot between two features, x and y,
    where y is imputed and x is a predictor used to impute y. This method
    performs single imputation and is useful to determine how an imputation
    method looks under the hood.

    Args:
        d (pd.DataFrame): DataFrame with data to impute and plot.
        x (str): column to plot on x axis.
        y (str): column to plot on y axis and set color for imputation.
        strategy (str): imputation method for SingleImputer.
        color (str, Optional): which variable to color with imputations.
            Deafult is none, which means y is colored. Other option is to
            color "x". Color should be the same as "x" or "y".
        title (str, Optional): title of plot.
            "Defualt is Jointplot after Imputation".
        h (float, Optional): height of the jointplot. Default is 8.27
        imp_kwgs (dict, Optional): imp kwgs for SingleImputer procedure.
            Default is None.
        a (float, Optional): alpha for plot color. Default is 0.5
        marginals (dict, Optional): dictionary of marginal plot args.
            Default is None, configured in code below.
        obs_color (str, Optional): color of observed. Default is navy.
        imp_color (str, Optional): color of imputations. Default is red.
        **plot_kwgs: keyword arguments used by sns.set.

    Raises:
        ValueError: x and y must be names of columns in data
    """

    # plot setup and arg validation
    _default_plot_args(**plot_kwgs)
    _validate_kwgs(marginals)
    _validate_kwgs(imp_kwgs)
    if marginals is None:
        marginals = dict(rug=True, kde=True)

    # validate x and y selection
    if not x in d.columns or not y in d.columns:
        err = "x and y must be names of columns in data"
        raise ValueError(err)

    # create imputer with strategy and optional imp kwgs
    if imp_kwgs is None:
        imp = SingleImputer(strategy=strategy)
    else:
        imp = SingleImputer(strategy=strategy, imp_kwgs=imp_kwgs)

    # handling the color configuration
    if color is None:
        color = y
    else:
        if color == y:
            color = y
        elif color == x:
            color = x
        else:
            err = "color must be the same as `y` or `x`"
            raise ValueError(err)

    # configure and apply the imputer
    impute = imp.fit_transform(d)
    impute["colors"] = obs_color
    impute.loc[imp.imputed_[color], "colors"] = imp_color
    joints_color = impute["colors"]

    # create the joint plot
    joint_kws = dict(facecolor=joints_color, edgecolor=joints_color)
    g = sns.jointplot(x=x, y=y, data=impute, alpha=a, height=h,
                      joint_kws=joint_kws, marginal_kws=marginals)

    # final plot config and title
    plt.subplots_adjust(top=0.925)
    g.fig.suptitle(title)

def plot_imp_dists(d, mi, imp_col, title="Distributions after Imputation",
                   include_observed=True, separate_observed=True,
                   side_by_side=False, hist_observed=False,
                   hist_imputed=False, gw=(.5, .5), gh=(.5, .5), **plot_kwgs):
    """Plot the density between imputations for a given column.

    Use this method to plot the density of a given column after multiple
    imputation. The function allows the user to also plot the observed data
    from the column prior to imputation taking place. Further, the user can
    specify whether the observed should be separated into its own plot or not.

    Args:
        d (list): dataset returned from multiple imputation.
        mi (MultipleImputer): multiple imputer used to generate d.
        imp_col (str): column to plot. Should be a column with imputations.
        title (str, Optional): title of plot. Default is
            "Distributions after Imputation".
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
    _default_plot_args(**plot_kwgs)

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

    # plot title and legend
    plt.suptitle(title)
    plt.legend()

def plot_imp_boxplots(d, mi, imp_col, side_by_side=False,
                      title="Observed vs. Imputed Boxplots",
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
        title (str, Optional): title of boxplots. Default is
            "Observed vs. Imputed Boxplots."
        obs_kwgs (dict, Optional): dictionary of arguments to unpack for
            observed boxplot. Default is None, so no additional tailoring.
        imp_kwgs (dict, Optional): dictionary of arguments to unpack for
            imputed boxplots. Default is None, so no additional tailoring.
        **plot_kwgs: keyword arguments used by sns.set.

    Returns:
        sns.distplot: boxplots for observed and imputed data

    Raises:
        ValueError: see _validate_data method.
    """

    # set plot type and define names necessary
    _default_plot_args(**plot_kwgs)
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
        ).set(xlabel="", ylabel="")
    else:
        sns.boxplot(
            x=xo, y=yo, data=obs_, ax=ax[0]
        ).set(xlabel="", ylabel="")
    if not imp_kwgs is None:
        sns.boxplot(
            x=xi, y=yi, data=datasets_merged, ax=ax[1], **imp_kwgs
        ).set(xlabel="", ylabel="")
    else:
        sns.boxplot(
            x=xi, y=yi, data=datasets_merged, ax=ax[1]
        ).set(xlabel="", ylabel="")

    # plot title
    plt.suptitle(title)

def plot_imp_swarm(d, mi, imp_col, palette=None,
                   title="Imputation Swarm", **plot_kwgs):
    """Create the swarm plot for multiply imputed data.

    Args:
        d (list): dataset returned from multiple imputation.
        mi (MultipleImputer): multiple imputer used to generate d.
        imp_col (str): column to plot. Should be a column with imputations.
        title (str, Optional): title of plot. Default is "Imputation Swarm".
        palette (list, tuple, Optional): colors for the imps and observed.
            Default is None. if None, colors default to ["r","c"].
        **plot_kwgs: keyword arguments used by sns.set.

    Returns:
        sns.distplot: swarmplot for imputed data

    Raises:
        ValueError: see _validate_data method.
    """

    # set plot type, validate, and define names necessary
    _default_plot_args(**plot_kwgs)
    _validate_data(d, mi, imp_col)
    datasets_merged = _melt_df(d, mi, imp_col)
    if palette is None:
        palette = ["r", "c"]

    # swarmplot example
    sns.swarmplot(
        x="imp_num", y=imp_col, hue="imputed", palette=palette,
        data=datasets_merged, hue_order=["yes", "no"]
    ).set(xlabel="Imputation Number", title=title)

def plot_imp_strip(d, mi, imp_col, palette=None,
                   title="Imputation Strip", **plot_kwgs):
    """Create the strip plot for multiply imputed data.

    Args:
        d (list): dataset returned from multiple imputation.
        mi (MultipleImputer): multiple imputer used to generate d.
        imp_col (str): column to plot. Should be a column with imputations.
        title (str, Optional): title of plot. Default is "Imputation Strip".
        palette (list, tuple, Optional): colors for the imps and observed.
            Default is None. if None, colors default to ["r","c"].
        **plot_kwgs: keyword arguments used by sns.set.

    Returns:
        sns.distplot: stripplot for imputed data

    Raises:
        ValueError: see _validate_data method.
    """

    # set plot type, validate, and define names necessary
    _default_plot_args(**plot_kwgs)
    _validate_data(d, mi, imp_col)
    datasets_merged = _melt_df(d, mi, imp_col)
    if palette is None:
        palette = ["r", "c"]

    # stripplot example
    sns.stripplot(
        x="imp_num", y=imp_col, hue="imputed", palette=palette,
        data=datasets_merged, jitter=True, hue_order=["yes", "no"], dodge=True
    ).set(xlabel="Imputation Number", title=title)
