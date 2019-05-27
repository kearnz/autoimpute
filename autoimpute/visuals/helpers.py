"""Helper functions used throughout other methods in automipute.visuals."""

import pandas as pd
import seaborn as sns
from autoimpute.imputations import MultipleImputer

#pylint:disable=unnecessary-lambda

def _fully_complete(data):
    """Private method to exit plotting and raise error if no data missing."""
    if not pd.isnull(data).sum().any():
        err = "No data is missing in any column. Cannot generate plot."
        raise ValueError(err)

def _default_plot_args(**kwargs):
    """Private method to set up the default plot style arguments."""
    rc = {}
    rc["figure.figsize"] = kwargs.pop("figsize", (12, 8))
    context = kwargs.pop("context", "talk")
    sns.set(context=context, rc=rc)

def _validate_data(d, mi, imp_col=None):
    """Private helper method to validate data vs multiple imputations.

    Args:
        d (list): dataset returned from multiple imputation.
        mi (MultipleImputer): multiple imputer used to generate d.
        imp_col (str): column to plot. Should be a column with imputations.

    Raises:
        ValueError: d should be list of tuples returned from mi transform.
        ValueError: mi should be instance of MultipleImputer used to produce d.
        ValueError: mi should have imputed_ attribute after transformation.
        ValueError: Number of imputations should equal length of the dataset.
        ValueError: Columns in each imputed data should be the same.
        ValueError: Colums in each imputed data should be same as mi.imputed_.
        ValueError: imp_col must be in both datasets and mi.imputed_ keys.
    """

    if not isinstance(d, list):
        err = "d should be list of tuples returned from mi transform."
        raise ValueError(err)

    if not isinstance(mi, MultipleImputer):
        err = "mi should be instance of MultipleImputer used to produce d."
        raise ValueError(err)

    if not hasattr(mi, "imputed_"):
        err = "mi should have imputed_ attribute after transformation."
        raise ValueError(err)

    # names of values from expressions we need to test
    num_imps = len(d)
    num_mi = mi.n
    imp_cols = set(mi.imputed_.keys())
    sets_ = [set(d[i][1].columns.tolist()) for i in range(len(d))]
    diff_d = any(list(map(lambda i: sets_[0].difference(i), sets_[1:])))
    diff_m = len(sets_[0].difference(imp_cols))

    if num_imps != num_mi:
        err = "Number of imputations should equal length of the dataset."
        raise ValueError(err)

    if diff_d:
        err = "Columns in each imputed dataset should be the same."
        raise ValueError(err)

    if diff_m > 0:
        err = "Colums w/in each imputed dataset should be same as mi.imputed_"
        raise ValueError(err)

    if not imp_col is None:
        imp_in_d = imp_col in sets_[0]
        imp_in_mi = imp_col in imp_cols
        if not all([imp_in_d, imp_in_mi]):
            err = "imp_col must be in both datasets and mi.imputed_ keys."
            raise ValueError(err)

def _get_observed(d, mi, imp_col):
    """Private helper method to get observed data after imputation."""
    _validate_data(d, mi, imp_col)
    obs = set(d[0][1].index).difference(mi.imputed_[imp_col])
    return list(obs)

def _plot_imp_dists_helper(d, hist_imputed, imp_col, ax=None, l="Imputed"):
    """Private helper method to plot distribution of imputed data."""
    for each in d:
        sns.distplot(
            each[1][imp_col], hist=hist_imputed, ax=ax,
            label=f"{imp_col} Imp {each[0]}"
        ).set(xlabel=l)

def _validate_kwgs(kwgs):
    """Private helper method to validate kwargs arguments."""
    if not isinstance(kwgs, (type(None), dict)):
        err = "kwgs must be None or a dictionary of keyword arguments"
        raise ValueError(err)

def _melt_df(d, mi, imp_col):
    """Private helper method to melt dataframe vertically."""
    datasets_added = []
    for each in d:
        e = each[1].copy()
        e["imp_num"] = f"Imp {each[0]}"
        e["imputed"] = "no"
        imputed = mi.imputed_[imp_col]
        e.loc[imputed, "imputed"] = "yes"
        datasets_added.append(e)
    datasets_merged = pd.concat(datasets_added)
    return datasets_merged
